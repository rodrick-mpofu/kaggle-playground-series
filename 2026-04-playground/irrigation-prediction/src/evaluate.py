import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def evaluate_model(model, dval, y_val, le):
    """Run full evaluation suite on validation set."""
    import xgboost as xgb

    preds = model.predict(dval).astype(int)

    bal_acc = balanced_accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='weighted')

    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Weighted F1:       {f1:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_val, preds, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, preds))

    return {
        'balanced_accuracy': bal_acc,
        'f1_weighted': f1
    }