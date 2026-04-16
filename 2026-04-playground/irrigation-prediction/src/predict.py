import xgboost as xgb
import pandas as pd
import numpy as np
import mlflow.xgboost
from config import RANDOM_STATE, NUM_CLASSES

def train_final_model(study, X_train, y_train, X_val, y_val, cfg):
    import mlflow
    best_params = {
        **study.best_trial.params,
        'objective': cfg.model.objective,
        'num_class': cfg.data.num_classes,
        'eval_metric': cfg.model.eval_metric,
        'random_state': cfg.random_state,
        'n_jobs': cfg.n_jobs
    }
    best_round = study.best_trial.user_attrs['best_round']

    X_full = pd.concat([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    dfull = xgb.DMatrix(X_full, label=y_full, enable_categorical=True)

    with mlflow.start_run(run_name="final_model"):
        model = xgb.train(best_params, dfull, num_boost_round=best_round)
        mlflow.log_params(best_params)
        mlflow.log_param('best_round', best_round)
        mlflow.xgboost.log_model(model, artifact_path="final_model")

    return model

def generate_submission(model, X_submit, submission_df, le, cfg):
    import mlflow
    dsubmit = xgb.DMatrix(X_submit, enable_categorical=True)
    preds = model.predict(dsubmit).astype(int)
    labels = le.inverse_transform(preds)

    submission = pd.DataFrame({
        'id': submission_df['id'],
        'Irrigation_Need': labels
    })
    path = cfg.paths.submissions + "submission.csv"
    submission.to_csv(path, index=False)
    mlflow.log_artifact(path)
    print(submission['Irrigation_Need'].value_counts())
    return submission