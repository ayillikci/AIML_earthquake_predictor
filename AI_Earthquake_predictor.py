import argparse
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skyfield.api import load, load_file
from skyfield import almanac

# ---------------------- 1. Load Earthquake Data ----------------------
def fetch_earthquakes_from_csv(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        sys.exit(1)

    df = pd.read_csv(filepath, parse_dates=['time'])
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize("UTC")

    df = df[df['mag'] >= 6.0]
    start_date = pd.Timestamp("1849-12-26", tz="UTC")
    end_date = pd.Timestamp("2150-01-22", tz="UTC")
    df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]

    return df[['time', 'latitude', 'longitude', 'mag']]


def generate_negative_times(n: int, start: datetime, end: datetime) -> pd.DatetimeIndex:
    total_seconds = int((end - start).total_seconds())
    random_seconds = np.random.randint(0, total_seconds, size=n)
    return pd.to_datetime(start) + pd.to_timedelta(random_seconds, unit='s')

# ---------------------- 2. Load Skyfield Data ----------------------
_ts = load.timescale()
de440_path = load.download('de440s.bsp')
_planets = load_file(de440_path)

PLANETS = {
    'mercury': _planets[199],
    'venus': _planets[299],
    'mars': _planets['mars barycenter'],
    'jupiter': _planets['jupiter barycenter'],
    'saturn': _planets['saturn barycenter'],
    'uranus': _planets['uranus barycenter'],
    'neptune': _planets['neptune barycenter'],
}
EARTH = _planets[399]
SUN = _planets[10]

# ---------------------- 3. Celestial Feature Functions ----------------------
def compute_moon_phase_angles(times: pd.Series) -> np.ndarray:
    ts = _ts.from_datetimes(times.tolist())
    phase = almanac.moon_phase(_planets, ts)
    return phase.radians

def compute_planet_longitudes(times: pd.Series) -> pd.DataFrame:
    ts = _ts.from_datetimes(times.tolist())
    df = pd.DataFrame(index=times)
    for name, body in PLANETS.items():
        astrometric = EARTH.at(ts).observe(body)
        lon, lat, dist = astrometric.ecliptic_latlon()
        df[f"{name}_lon"] = lon.radians
    return df

def compute_pairwise_angles(times: pd.Series) -> pd.DataFrame:
    longs = compute_planet_longitudes(times)
    ts = _ts.from_datetimes(times.tolist())
    sun_lon = EARTH.at(ts).observe(SUN).ecliptic_latlon()[0].radians
    longs['sun_lon'] = sun_lon
    df = pd.DataFrame(index=times)
    cols = list(longs.columns)
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            diff = np.abs(longs[a] - longs[b])
            diff = np.minimum(diff, 2 * np.pi - diff)
            df[f"{a}_vs_{b}"] = diff
    return df

# ---------------------- 4. Dataset Prep ----------------------
def prepare_dataset(approach: int, eq_df: pd.DataFrame):
    pos_times = eq_df['time']
    n = len(pos_times)
    neg_times = generate_negative_times(n, start=eq_df['time'].min(), end=eq_df['time'].max())
    all_times = pd.concat([pos_times, pd.Series(neg_times)], ignore_index=True)
    labels = np.array([1] * n + [0] * n)

    feats = []
    if approach in (2, 4):
        feats.append(compute_planet_longitudes(all_times))
    if approach in (3, 4):
        feats.append(compute_pairwise_angles(all_times))

    X = pd.concat(feats, axis=1)
    return X, labels

# ---------------------- 5. Train and Predict ----------------------
def evaluate_approach(approach, eq_df, test_size, n_estimators):
    print(f"\n🚀 Running approach {approach}...")
    X, y = prepare_dataset(approach, eq_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"   ✅ Accuracy: {acc:.3f}")
    return clf, acc

def generate_future_dates(years=5, step_days=5):
    start = datetime.utcnow().replace(tzinfo=pd.Timestamp.now(tz='UTC').tz)
    end = start + timedelta(days=365 * years)
    return pd.date_range(start, end, freq=f"{step_days}D")

def predict_future_events(clf, approach, future_times):
    feats = []
    if approach in (2, 4):
        feats.append(compute_planet_longitudes(future_times))
    if approach in (3, 4):
        feats.append(compute_pairwise_angles(future_times))
    X_future = pd.concat(feats, axis=1)
    probs = clf.predict_proba(X_future)[:, 1]
    return future_times, probs

def plot_predictions(predictions_by_approach):
    for approach, (times, probs) in predictions_by_approach.items():
        plt.figure(figsize=(10, 4))
        plt.plot(times, probs, label=f"Approach {approach}", color='tab:blue')
        plt.title(f"Approach {approach} Predictions")
        plt.xlabel("Date")
        plt.ylabel("Earthquake Probability")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ---------------------- 6. Main CLI Entry ----------------------
def main():
    parser = argparse.ArgumentParser(description="Celestial-based earthquake prediction")
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--n-estimators', type=int, default=100)
    args = parser.parse_args()

    print("📂 Loading earthquake data from CSV…")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "earthquake_datas_1800.csv")
    eq_df = fetch_earthquakes_from_csv(csv_path)

    print("✅ Loaded data. Now evaluating models…")
    classifiers = {}
    results = {}
    for approach in [2, 3, 4]:
        clf, acc = evaluate_approach(approach, eq_df, args.test_size, args.n_estimators)
        classifiers[approach] = clf
        results[approach] = acc

    print("\n🧮 All models trained.")
    for a, acc in results.items():
        print(f" - Approach {a}: {acc:.3f} accuracy")

    print("\n🔮 Let's predict the future!")
    x_years = int(input("Enter number of years into the future to predict: "))
    y_mag = float(input("Enter minimum magnitude to focus on (e.g., 6.0): "))

    print(f"📅 Generating predictions for the next {x_years} years...")
    future_times = generate_future_dates(years=x_years)
    predictions_by_approach = {}
    for approach in [2, 3, 4]:
        print(f"   ⏳ Predicting with approach {approach}...")
        clf = classifiers[approach]
        times, probs = predict_future_events(clf, approach, future_times)
        predictions_by_approach[approach] = (times, probs)

    plot_predictions(predictions_by_approach)

if __name__ == "__main__":
    main()

