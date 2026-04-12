import pandas as pd
import numpy as np

def add_interaction_features(X):
    """Add domain-relevant interaction features for irrigation prediction."""
    X = X.copy()

    # Temperature x Humidity interaction
    if 'Temperature' in X.columns and 'Humidity' in X.columns:
        X['temp_humidity_interaction'] = X['Temperature'] * X['Humidity']

    # Heat index proxy
    if 'Temperature' in X.columns and 'Solar_Radiation' in X.columns:
        X['heat_index'] = X['Temperature'] + 0.33 * X['Solar_Radiation']

    # Soil moisture deficit proxy
    if 'Soil_Moisture' in X.columns and 'Evapotranspiration' in X.columns:
        X['moisture_deficit'] = X['Soil_Moisture'] - X['Evapotranspiration']

    return X

def drop_low_variance_features(X, threshold=0.01):
    """Drop features with variance below threshold."""
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    kept_cols = X.columns[selector.get_support()]
    dropped = set(X.columns) - set(kept_cols)
    if dropped:
        print(f"Dropped low variance features: {dropped}")
    return X[kept_cols]