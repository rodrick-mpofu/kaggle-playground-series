import pytest
import numpy as np
from sklearn.metrics import balanced_accuracy_score

def test_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    assert balanced_accuracy_score(y_true, y_pred) == 1.0

def test_worst_predictions():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([1, 2, 0, 2, 0, 1])
    assert balanced_accuracy_score(y_true, y_pred) == 0.0

def test_balanced_accuracy_range():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 2, 0, 2, 0, 1])
    score = balanced_accuracy_score(y_true, y_pred)
    assert 0.0 <= score <= 1.0