import pandas as pd
import numpy as np
import pytest
from src.data import split_raw_data, prepare_data, val_transformation, submission_transformation



# create a sample dataframe for testing
@pytest.fixture
def sample_df():
    """Minimal fake dataframe that mirrors the real dataset structure."""

    np.random.seed(42)

    n = 100

    return pd.DataFrame({
        'id': range(n),
        'Temperature': np.random.uniform(15, 40, n),
        'Humidity': np.random.uniform(20, 90, n),
        'Soil_Moisture': np.random.uniform(0.1, 0.9, n),
        'Wind_Speed': np.random.uniform(0, 20, n),
        'Solar_Radiation': np.random.uniform(100, 800, n),
        'Crop_Type': np.random.choice(['Wheat', 'Corn', 'Rice'], n),
        'Soil_Type': np.random.choice(['Sandy', 'Clay', 'Loam'], n),
        'Irrigation_Need': np.random.choice(['Low', 'Medium', 'High'], n)
    })

@pytest.fixture
def split_data(sample_df):
    """Returns train and val splits."""
    return split_raw_data(sample_df)

@pytest.fixture
def prepared_data(split_data):
    """Returns X_train, y_train, le."""
    train_df, _ = split_data
    return prepare_data(train_df)


# ── split_raw_data ───────────────────────────────────────────────────────────

def test_split_returns_two_dataframes(split_data):
    train_df, val_df = split_data
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)

def test_split_sizes(sample_df, split_data):
    train_df, val_df = split_data
    assert len(train_df) + len(val_df) == len(sample_df)

def test_split_no_overlap(split_data):
    train_df, val_df = split_data
    train_ids = set(train_df['id'])
    val_ids   = set(val_df['id'])
    assert train_ids.isdisjoint(val_ids)

def test_split_stratified(split_data):
    """Class proportions in train and val should be roughly equal."""
    train_df, val_df = split_data
    train_props = train_df['Irrigation_Need'].value_counts(normalize=True).sort_index()
    val_props   = val_df['Irrigation_Need'].value_counts(normalize=True).sort_index()
    for cls in train_props.index:
        assert abs(train_props[cls] - val_props[cls]) < 0.15

def test_split_index_reset(split_data):
    train_df, val_df = split_data
    assert list(train_df.index) == list(range(len(train_df)))
    assert list(val_df.index)   == list(range(len(val_df)))

# ── prepare_data ─────────────────────────────────────────────────────────────

def test_prepare_returns_correct_types(prepared_data):
    X, y, le = prepared_data
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)

def test_prepare_target_is_1d(prepared_data):
    _, y, _ = prepared_data
    assert y.ndim == 1

def test_prepare_target_classes(prepared_data):
    _, y, _ = prepared_data
    assert set(np.unique(y)).issubset({0, 1, 2})

def test_prepare_no_target_in_features(prepared_data):
    X, _, _ = prepared_data
    assert 'Irrigation_Need' not in X.columns

def test_prepare_no_id_in_features(prepared_data):
    X, _, _ = prepared_data
    assert 'id' not in X.columns

def test_prepare_no_bool_columns(prepared_data):
    X, _, _ = prepared_data
    bool_cols = X.select_dtypes(include=['bool']).columns
    assert len(bool_cols) == 0

def test_prepare_no_object_columns(prepared_data):
    """All categoricals should be encoded — no raw strings left."""
    X, _, _ = prepared_data
    obj_cols = X.select_dtypes(include=['object']).columns
    assert len(obj_cols) == 0

def test_prepare_label_encoder_classes(prepared_data):
    _, _, le = prepared_data
    assert set(le.classes_) == {'Low', 'Medium', 'High'}

# ── val_transformation ───────────────────────────────────────────────────────

def test_val_columns_match_train(split_data, prepared_data):
    _, val_df    = split_data
    X_train, _, le = prepared_data
    X_val, _     = val_transformation(val_df, X_train, le)
    assert list(X_val.columns) == list(X_train.columns)

def test_val_target_is_1d(split_data, prepared_data):
    _, val_df      = split_data
    X_train, _, le = prepared_data
    _, y_val       = val_transformation(val_df, X_train, le)
    assert y_val.ndim == 1

def test_val_no_bool_columns(split_data, prepared_data):
    _, val_df      = split_data
    X_train, _, le = prepared_data
    X_val, _       = val_transformation(val_df, X_train, le)
    assert len(X_val.select_dtypes(include=['bool']).columns) == 0

# ── submission_transformation ────────────────────────────────────────────────

def test_submission_columns_match_train(sample_df, prepared_data):
    X_train, _, _ = prepared_data
    sub_df = sample_df.drop(columns=['Irrigation_Need'])
    X_sub  = submission_transformation(sub_df, X_train)
    assert list(X_sub.columns) == list(X_train.columns)

def test_submission_no_target_column(sample_df, prepared_data):
    X_train, _, _ = prepared_data
    sub_df = sample_df.drop(columns=['Irrigation_Need'])
    X_sub  = submission_transformation(sub_df, X_train)
    assert 'Irrigation_Need' not in X_sub.columns

