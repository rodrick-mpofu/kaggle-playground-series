import xgboost as xgb
import pandas as pd
import numpy as np
import mlflow.xgboost
import os
import pickle

def train_final_model(study, X_train, y_train, X_val, y_val, cfg, le):
    import mlflow
    best_params = {
        **study.best_trial.params,
        'objective': cfg.model.objective,
        'num_class': cfg.data.num_classes,
        'eval_metric': cfg.model.eval_metric,
        'tree_method': cfg.model.tree_method,
        'device': cfg.model.device,
        'random_state': cfg.random_state,
        'n_jobs': cfg.model.n_jobs
    }
    best_round = study.best_trial.user_attrs['best_round']

    X_full = pd.concat([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    dfull = xgb.DMatrix(X_full, label=y_full, enable_categorical=True)

    with mlflow.start_run(run_name="final_model") as run:
        model = xgb.train(best_params, dfull, num_boost_round=best_round)
        mlflow.log_params(best_params)
        mlflow.log_param('best_round', best_round)
        mlflow.xgboost.log_model(model, artifact_path="final_model")

        # Get the actual MLflow artifact URI for the saved model
        model_uri = f"runs:/{run.info.run_id}/final_model"
        print(f"Model logged to MLflow at: {model_uri}")

        # Save artifacts for API — including the correct model URI
        os.makedirs("models", exist_ok=True)
        artifacts = {
            "label_encoder": le,
            "feature_columns": list(X_train.columns),
            "model_uri": model_uri,  # save the URI so API can find it
            "version": "1.0.0"
        }
        with open("models/artifacts.pkl", "wb") as f:
            pickle.dump(artifacts, f)

        print("Artifacts saved to models/artifacts.pkl")

    return model

    import mlflow
    best_params = {
        **study.best_trial.params,
        'objective': cfg.model.objective,
        'num_class': cfg.data.num_classes,
        'eval_metric': cfg.model.eval_metric,
        'random_state': cfg.random_state,
        'n_jobs': cfg.model.n_jobs
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