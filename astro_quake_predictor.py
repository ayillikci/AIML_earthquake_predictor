# astro_quake_predictor.py

import pandas as pd
import numpy as np
from skyfield.api import load
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TURKEY_BOUNDS = {
    'lat_min': 35.0,
    'lat_max': 43.0,
    'lon_min': 25.0,
    'lon_max': 45.0,
}

PLANETS = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']

# --- LOAD EPHEMERIS ---
eph = load('de421.bsp')
ts = load.timescale()
earth = eph['earth']

# --- FUNCTIONS ---
def load_earthquake_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['time'])
    df = df[(df['latitude'] >= TURKEY_BOUNDS['lat_min']) &
            (df['latitude'] <= TURKEY_BOUNDS['lat_max']) &
            (df['longitude'] >= TURKEY_BOUNDS['lon_min']) &
            (df['longitude'] <= TURKEY_BOUNDS['lon_max']) &
            (df['mag'] >= 5.0)]
    return df

def get_planet_positions(date):
    t = ts.utc(date.year, date.month, date.day)
    positions = {}
    for planet in PLANETS:
        try:
            obj = eph[planet]
            astrometric = earth.at(t).observe(obj).apparent()
            _, lon, _ = astrometric.ecliptic_latlon()
            positions[planet] = lon.degrees
        except Exception as e:
            positions[planet] = np.nan
    return positions

def create_feature_matrix(dates):
    data = []
    for date in tqdm(dates, desc='Calculating planetary positions'):
        pos = get_planet_positions(date)
        data.append([pos[p] for p in PLANETS])
    return pd.DataFrame(data, columns=PLANETS)

def train_model(df):
    quake_dates = df['time'].dt.date.unique()
    quake_labels = [1] * len(quake_dates)

    start = df['time'].min().date()
    end = df['time'].max().date()
    all_dates = pd.date_range(start, end).date
    non_quake_dates = list(set(all_dates) - set(quake_dates))
    sampled_non_quakes = np.random.choice(non_quake_dates, size=len(quake_dates), replace=False)
    non_quake_labels = [0] * len(sampled_non_quakes)

    all_dates_combined = list(quake_dates) + list(sampled_non_quakes)
    labels = quake_labels + non_quake_labels

    X = create_feature_matrix(all_dates_combined)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, 'earthquake_model.pkl')
    print("✅ Model saved to earthquake_model.pkl")

def predict_next_10_years():
    model = joblib.load('earthquake_model.pkl')
    start = datetime.today().date()
    end = start + timedelta(days=365 * 10)
    dates = pd.date_range(start, end).date
    X = create_feature_matrix(dates)
    probs = model.predict_proba(X)[:, 1]

    print(f"\n🔮 Earthquake Forecast for Next 10 Years:\n")
    for d, p in zip(dates, probs):
        risk = "High Risk" if p >= 0.5 else "Low Risk"
        print(f"{d}: {risk} (Prob: {p:.2f})")

    # --- Visualization ---
    plt.figure(figsize=(15, 6))
    plt.plot(dates, probs, label='Probability of Earthquake', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Risk Threshold (0.5)')

    # Annotate peaks over 0.8
    for date, prob in zip(dates, probs):
        if prob > 0.8:
            plt.annotate(f'{date}', xy=(date, prob), xytext=(date, prob + 0.05),
                         textcoords='data', ha='center', fontsize=8,
                         arrowprops=dict(arrowstyle='->', color='gray'))

    plt.title('Earthquake Risk Forecast (Next 10 Years - Turkey)')
    plt.xlabel('Date')
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

# --- AUTOMATIC EXECUTION ---
if __name__ == '__main__':
    data_path = 'query.csv'  # Your uploaded CSV filename
    df = load_earthquake_data(data_path)
    train_model(df)
    predict_next_10_years()
