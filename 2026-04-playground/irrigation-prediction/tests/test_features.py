import pytest
import pandas as pd
import numpy as np
from src.features import add_interaction_features, drop_low_variance_features

@pytest.fixture
def sample_features():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'Temperature': np.random.uniform(15, 40, n),
        'Humidity': np.random.uniform(20, 90, n),
        'Soil_Moisture': np.random.uniform(0.1, 0.9, n),
        'Solar_Radiation': np.random.uniform(100, 800, n),
        'Evapotranspiration': np.random.uniform(0.1, 0.5, n),
        'constant_col': np.ones(n)  # zero variance — should be dropped
    })

def test_interaction_features_added(sample_features):
    result = add_interaction_features(sample_features)
    assert 'temp_humidity_interaction' in result.columns

def test_interaction_does_not_drop_originals(sample_features):
    result = add_interaction_features(sample_features)
    assert 'Temperature' in result.columns
    assert 'Humidity' in result.columns

def test_interaction_values_correct(sample_features):
    result = add_interaction_features(sample_features)
    expected = sample_features['Temperature'] * sample_features['Humidity']
    pd.testing.assert_series_equal(result['temp_humidity_interaction'], expected, check_names=False)

def test_drop_low_variance_removes_constant(sample_features):
    result = drop_low_variance_features(sample_features, threshold=0.01)
    assert 'constant_col' not in result.columns

def test_drop_low_variance_keeps_variable_cols(sample_features):
    result = drop_low_variance_features(sample_features, threshold=0.01)
    assert 'Temperature' in result.columns
    assert 'Humidity' in result.columns