# astro_quake_predictor.py

# Importing necessary libraries
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
# Define geographic bounds for filtering earthquakes in Turkey
TURKEY_BOUNDS = {
    'lat_min': 35.0,
    'lat_max': 43.0,
    'lon_min': 25.0,
    'lon_max': 45.0,
}

# List of planets to track for astrological features
PLANETS = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']

# --- LOAD EPHEMERIS ---
# Load planetary ephemeris data from Skyfield
eph = load('de421.bsp')
ts = load.timescale()
earth = eph['earth']

# --- FUNCTIONS ---
def load_earthquake_data(file_path):
    """Load and filter earthquake data specific to Turkey with magnitude >= 5.0."""
    df = pd.read_csv(file_path, parse_dates=['time'])
    df = df[(df['latitude'] >= TURKEY_BOUNDS['lat_min']) &
            (df['latitude'] <= TURKEY_BOUNDS['lat_max']) &
            (df['longitude'] >= TURKEY_BOUNDS['lon_min']) &
            (df['longitude'] <= TURKEY_BOUNDS['lon_max']) &
            (df['mag'] >= 5.0)]
    return df

def get_planet_positions(date):
    """Return planetary longitudes (ecliptic) for a given date."""
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
    """Create a DataFrame of planetary positions for a list of dates."""
    data = []
    for date in tqdm(dates, desc='Calculating planetary positions'):
        pos = get_planet_positions(date)
        data.append([pos[p] for p in PLANETS])
    return pd.DataFrame(data, columns=PLANETS)

def train_model(df):
    """Train a Random Forest model to classify earthquake occurrence based on planetary positions."""
    quake_dates = df['time'].dt.date.unique()
    quake_labels = [1] * len(quake_dates)

    # Generate non-earthquake dates for balanced training
    start = df['time'].min().date()
    end = df['time'].max().date()
    all_dates = pd.date_range(start, end).date
    non_quake_dates = list(set(all_dates) - set(quake_dates))
    sampled_non_quakes = np.random.choice(non_quake_dates, size=len(quake_dates), replace=False)
    non_quake_labels = [0] * len(sampled_non_quakes)

    # Combine positive and negative samples
    all_dates_combined = list(quake_dates) + list(sampled_non_quakes)
    labels = quake_labels + non_quake_labels

    X = create_feature_matrix(all_dates_combined)
    y = np.array(labels)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, 'earthquake_model.pkl')
    print("✅ Model saved to earthquake_model.pkl")

def predict_next_10_years():
    """Predict and visualize earthquake probability for the next 10 years based on planetary configurations."""
    model = joblib.load('earthquake_model.pkl')

    # Prompt user for optional custom date range
    use_custom_range = input("Do you want to set a custom date range? (y/n): ").strip().lower()
    if use_custom_range == 'y':
        start_input = input("Enter start date (YYYY-MM-DD): ").strip()
        end_input = input("Enter end date (YYYY-MM-DD): ").strip()
        start = datetime.strptime(start_input, '%Y-%m-%d').date()
        end = datetime.strptime(end_input, '%Y-%m-%d').date()
    else:
        start = datetime.today().date()
        end = start + timedelta(days=365 * 10)

    # Generate date range and predictions
    dates = pd.date_range(start, end).date
    X = create_feature_matrix(dates)
    probs = model.predict_proba(X)[:, 1]

    # Print risk forecast
    print(f"\n🔮 Earthquake Forecast from {start} to {end}:")
    for d, p in zip(dates, probs):
        risk = "High Risk" if p >= 0.5 else "Low Risk"
        print(f"{d}: {risk} (Prob: {p:.2f})")

    # --- Visualization 1 ---
    # Plot probabilities over time and annotate peaks above 0.8
    plt.figure(figsize=(15, 6))
    plt.plot(dates, probs, label='Probability of Earthquake', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Risk Threshold (0.5)')
    for date, prob in zip(dates, probs):
        if prob > 0.8:
            plt.annotate(f'{date}', xy=(date, prob), xytext=(date, prob + 0.05),
                         textcoords='data', ha='center', fontsize=8,
                         arrowprops=dict(arrowstyle='->', color='gray'))
    plt.title('Earthquake Risk Forecast (Custom Range - Turkey)')
    plt.xlabel('Date')
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # --- Visualization 2 (probability > 0.65) ---
    # This second chart is optional and currently commented out
    # Uncomment to view days with probabilities > 0.65

    # high_prob_dates = [d for d, p in zip(dates, probs) if p > 0.65]
    # high_prob_values = [p for p in probs if p > 0.65]
    #
    # plt.figure(figsize=(15, 6))
    # plt.plot(dates, probs, alpha=0.2, label='All Probabilities')
    # plt.scatter(high_prob_dates, high_prob_values, color='orange', label='Prob > 0.65')
    # for d, p in zip(high_prob_dates, high_prob_values):
    #     plt.annotate(f'{d}', xy=(d, p), xytext=(d, p + 0.02), fontsize=7, ha='center')
    # plt.axhline(y=0.65, color='orange', linestyle='--', label='0.65 Threshold')
    # plt.title('Highlighted Earthquake Risk Days (Prob > 0.65)')
    # plt.xlabel('Date')
    # plt.ylabel('Probability')
    # plt.legend()
    # plt.tight_layout()
    # plt.grid(True)
    # plt.show()

# --- AUTOMATIC EXECUTION ---
if __name__ == '__main__':
    # Automatically load data, train model, and forecast when script runs
    data_path = 'query.csv'  # Your uploaded CSV filename
    df = load_earthquake_data(data_path)
    train_model(df)
    predict_next_10_years()
