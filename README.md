🌍🔮 Earthquake Prediction Based on Planetary Positions 
This project is an experimental application that uses planetary positions (astrological data) in combination with machine learning to estimate the probability of earthquake occurrences in Turkey.

It combines historical earthquake records with the positions of planets as observed from Earth, and applies machine learning to detect patterns and forecast seismic risk for future dates.

🚀 Features
📅 Trains on historical earthquake data in Turkey (magnitude ≥ 5.0)

🌌 Extracts planetary positions (Sun, Moon, Mercury–Neptune) using NASA JPL ephemeris

🧠 Trains a Random Forest classifier on astrological features

🔮 Predicts the probability of earthquakes for any future date

🗓️ Lets users interactively define a date range for predictions

📊 Generates interactive risk plots with labeled high-risk dates (e.g., P > 0.8)

💬 Fully documented, PEP8-compliant, and CLI-driven

⚙️ How It Works
Load earthquake data for Turkey from a CSV file.

For each earthquake date, compute planetary positions (longitude in ecliptic coordinates).

Create a labeled dataset (quake days vs. non-quake days).

Train a Random Forest model using these astrological features.

Predict probabilities for a future range of dates.

Visualize the risk over time, highlight key risk dates.

🧪 Technologies Used
Python

Skyfield (planetary calculations)

Scikit-learn (ML)

Matplotlib (visualization)

Pandas, NumPy, tqdm

CSV-based data ingestion

🔍 Why Astrology?
This is a research-driven, cross-disciplinary exploration — not a scientifically proven method. It blends metaphysical assumptions (planetary influence) with statistical modeling to explore whether any meaningful correlations exist.

📁 Use Cases
Educational / exploratory project for data science, astronomy, and ML

Artistic or conceptual exploration of natural cycles

Experimental forecasting model with visual storytelling

