import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
from sklearn.metrics import balanced_accuracy_score
from config import RANDOM_STATE, STUDY_NAME, STORAGE, EXPERIMENT_NAME, NUM_CLASSES

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'multi:softmax',
        'num_class': NUM_CLASSES,
        'eval_metric': 'mlogloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 6, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 80),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        evals_result = {}
        model = xgb.train(
            params, dtrain,
            num_boost_round=5000,
            evals=[(dval, 'eval')],
            early_stopping_rounds=100,
            verbose_eval=False,
            evals_result=evals_result
        )

        trial.set_user_attr('best_round', model.best_iteration)
        mlflow.log_metric('best_round', model.best_iteration)

        for step, loss in enumerate(evals_result['eval']['mlogloss']):
            mlflow.log_metric('mlogloss', loss, step=step)

        preds = model.predict(dval).astype(int)
        score = balanced_accuracy_score(y_val, preds)
        mlflow.log_metric('balanced_accuracy', score)
        mlflow.xgboost.log_model(model, artifact_path="model")

    return score

def run_study(X_train, y_train, X_val, y_val, n_trials=100):
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="optuna_study"):
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=STORAGE,
            direction="maximize",
            load_if_exists=True
        )
        print(f"Resuming study with {len(study.trials)} trials completed")

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials
        )

        mlflow.log_metric('best_balanced_accuracy', study.best_trial.value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})

    return study