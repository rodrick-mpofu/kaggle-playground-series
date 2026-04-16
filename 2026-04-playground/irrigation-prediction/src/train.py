import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
from sklearn.metrics import balanced_accuracy_score

def objective(trial, X_train, y_train, X_val, y_val, cfg):
    params = {
        'objective': cfg.model.objective,
        'num_class': cfg.data.num_classes,
        'eval_metric': cfg.model.eval_metric,
        'learning_rate': trial.suggest_float('learning_rate', 
                                             cfg.model.search_space.learning_rate.low, 
                                             cfg.model.search_space.learning_rate.high, log=True),
        'max_depth': trial.suggest_int('max_depth', cfg.model.search_space.max_depth.low, 
                                       cfg.model.search_space.max_depth.high),
        'min_child_weight': trial.suggest_int('min_child_weight', cfg.model.search_space.min_child_weight.low, 
                                             cfg.model.search_space.min_child_weight.high),
        'subsample': trial.suggest_float('subsample', cfg.model.search_space.subsample.low, 
                                         cfg.model.search_space.subsample.high),
        'colsample_bytree': trial.suggest_float('colsample_bytree', cfg.model.search_space.colsample_bytree.low, 
                                               cfg.model.search_space.colsample_bytree.high),
        'reg_alpha': trial.suggest_float('reg_alpha', cfg.model.search_space.reg_alpha.low,
                                           cfg.model.search_space.reg_alpha.high),
        'reg_lambda': trial.suggest_float('reg_lambda', cfg.model.search_space.reg_lambda.low,
                                           cfg.model.search_space.reg_lambda.high),
        'gamma': trial.suggest_float('gamma', cfg.model.search_space.gamma.low, 
                                     cfg.model.search_space.gamma.high),
        'random_state': cfg.random_state,
        'n_jobs': cfg.model.n_jobs
    }

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        evals_result = {}
        model = xgb.train(
            params, dtrain,
            num_boost_round=cfg.model.num_boost_round,
            evals=[(dval, 'eval')],
            early_stopping_rounds=cfg.model.early_stopping_rounds,
            verbose_eval=cfg.model.verbose_eval,
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

def run_study(X_train, y_train, X_val, y_val, cfg, n_trials):
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="optuna_study"):
        study = optuna.create_study(
            study_name=cfg.optuna.study_name,
            storage=cfg.optuna.storage,
            direction=cfg.optuna.direction,
            load_if_exists=True
        )
        print(f"Resuming study with {len(study.trials)} trials completed")

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, cfg),
            n_trials= cfg.optuna.n_trials
        )

        mlflow.log_metric('best_balanced_accuracy', study.best_trial.value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})

    return study