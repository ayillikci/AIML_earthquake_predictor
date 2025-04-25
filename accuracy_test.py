# accuracy_test.py - Improved Version with Astro Features and XGBoost

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from astro_quake_predictor import load_earthquake_data
from astro_features import aspect_features_for_date

# Load full dataset
data_path = 'query.csv'
df = load_earthquake_data(data_path)

# Make cutoff dates timezone-aware
utc = pytz.UTC
cutoff_train = utc.localize(datetime(2018, 12, 31))
cutoff_test_start = utc.localize(datetime(2025, 1, 1))
cutoff_test_end = utc.localize(datetime(2025, 4, 24))

# Split into training and testing by date
train_df = df[df['time'] <= cutoff_train]
test_df = df[(df['time'] > cutoff_test_start) & (df['time'] <= cutoff_test_end)]

# Train on historical data until end of 2024
quake_dates = train_df['time'].dt.date.unique()
quake_labels = [1] * len(quake_dates)

all_train_dates = pd.date_range(train_df['time'].min(), train_df['time'].max()).date
non_quake_dates = list(set(all_train_dates) - set(quake_dates))
sampled_non_quakes = np.random.choice(non_quake_dates, size=len(quake_dates), replace=False)
non_quake_labels = [0] * len(sampled_non_quakes)

X_train_dates = list(quake_dates) + list(sampled_non_quakes)
y_train = quake_labels + non_quake_labels

# --- Feature engineering with astro aspects ---
def create_aspect_matrix(dates):
    data = [aspect_features_for_date(date) for date in dates]
    return pd.DataFrame(data)

X_train = create_aspect_matrix(X_train_dates)

# Use XGBoost classifier
model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
joblib.dump(model, 'earthquake_model_test.pkl')
print("✅ Model trained and saved with XGBoost using planetary aspects")

# Evaluate on test period: Jan 1, 2025 – Apr 24, 2025
quake_test_dates = test_df['time'].dt.date.unique()
quake_test_labels = [1] * len(quake_test_dates)

test_dates = pd.date_range(cutoff_test_start, cutoff_test_end).date
non_quake_test_dates = list(set(test_dates) - set(quake_test_dates))
sampled_non_quake_test = np.random.choice(non_quake_test_dates, size=len(quake_test_dates), replace=False)
non_quake_test_labels = [0] * len(sampled_non_quake_test)

X_test_dates = list(quake_test_dates) + list(sampled_non_quake_test)
y_test = quake_test_labels + non_quake_test_labels

X_test = create_aspect_matrix(X_test_dates)

# Load trained model and evaluate
test_model = joblib.load('earthquake_model_test.pkl')
y_pred = test_model.predict(X_test)
y_probs = test_model.predict_proba(X_test)[:, 1]

print("\n📊 Accuracy Report (Test Period: Jan 2025 – Apr 2025):")
print(classification_report(y_test, y_pred))

# --- Visualization 1: Prediction vs Actual Earthquakes ---
plt.figure(figsize=(14, 6))
plt.plot(X_test_dates, y_probs, label='Predicted Probability', color='blue', alpha=0.7)
plt.scatter(quake_test_dates, [1.05]*len(quake_test_dates), color='red', label='Actual Earthquakes (Mag ≥ 5)', marker='x', s=100)
plt.axhline(y=0.5, color='gray', linestyle='--', label='Decision Threshold (0.5)')
plt.title('Earthquake Probability vs Actual Events (Jan–Apr 2025)')
plt.xlabel('Date')
plt.ylabel('Probability')
plt.xticks(rotation=45)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Visualization 2: Feature Importance ---
plt.figure(figsize=(10, 6))
sns.barplot(x=test_model.feature_importances_, y=X_test.columns)
plt.title('Feature Importance (XGBoost) - Planetary Aspects')
plt.xlabel('Importance')
plt.ylabel('Astrological Feature')
plt.tight_layout()
plt.grid(True)
plt.show()
