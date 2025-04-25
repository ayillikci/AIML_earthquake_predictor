# astro_features.py

from skyfield.api import load
import numpy as np

# Load ephemeris and timescale
eph = load('de421.bsp')
ts = load.timescale()
earth = eph['earth']

# Define pairs and aspect angles
ASPECT_ANGLES = {
    'conjunction': 0,
    'opposition': 180,
    'square': 90,
    'trine': 120,
    'sextile': 60,
}

# Allow ± orb (tolerance) in degrees
ORBS = {
    'conjunction': 8,
    'opposition': 8,
    'square': 6,
    'trine': 6,
    'sextile': 4,
}

# Supported planetary pairs using valid names from the kernel
PLANET_PAIRS = [
    ('sun', 'moon'),
    ('mars', 'saturn barycenter'),
    ('jupiter barycenter', 'neptune barycenter')
]

# Compute ecliptic longitude
def get_ecliptic_longitude(body, date):
    t = ts.utc(date.year, date.month, date.day)
    astrometric = earth.at(t).observe(eph[body]).apparent()
    _, lon, _ = astrometric.ecliptic_latlon()
    return lon.degrees

# Determine if two angles are in an aspect relationship
def is_aspect(angle_diff, aspect):
    exact = ASPECT_ANGLES[aspect]
    orb = ORBS[aspect]
    delta = min(abs(angle_diff - exact), abs(360 - angle_diff - exact))
    return int(delta <= orb)

# Create feature dictionary for a date
def aspect_features_for_date(date):
    features = {}
    longitudes = {planet: get_ecliptic_longitude(planet, date) for pair in PLANET_PAIRS for planet in pair}

    for p1, p2 in PLANET_PAIRS:
        angle = abs(longitudes[p1] - longitudes[p2]) % 360
        for aspect in ASPECT_ANGLES:
            key = f"{p1.replace(' ', '_')}_{p2.replace(' ', '_')}_{aspect}"
            features[key] = is_aspect(angle, aspect)

    return features
