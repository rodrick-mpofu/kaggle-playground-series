import os
import xgboost as xgb
import mlflow.xgboost
from functools import lru_cache
from sklearn.preprocessing import LabelEncoder
import numpy as np


class ModelManager:
    """Loads and holds the trained model and label encoder in memory."""

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.model_version = None
        self.feature_columns = None

    def load(self, model_uri: str, label_encoder: LabelEncoder, feature_columns: list, version: str = "1.0.0"):
        """Load model from MLflow artifact URI."""
        self.model = mlflow.xgboost.load_model(model_uri)
        self.label_encoder = label_encoder
        self.feature_columns = feature_columns
        self.model_version = version
        print(f"Model loaded: version {version}")

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, input_df) -> tuple:
        """Run prediction and return (label, class_name)."""
        import pandas as pd
        dmatrix = xgb.DMatrix(input_df, enable_categorical=True)
        label = int(self.model.predict(dmatrix)[0])
        class_name = self.label_encoder.inverse_transform([label])[0]
        return label, class_name


# Single shared instance across the entire app
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    return model_manager