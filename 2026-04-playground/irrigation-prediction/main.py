import pandas as pd
import hydra
from omegaconf import OmegaConf
from src.data import split_raw_data, prepare_data, val_transformation, submission_transformation
from src.train import run_study
from src.predict import train_final_model, generate_submission


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # Print the configuration
    print(OmegaConf.to_yaml(cfg))  # prints full config at start of every run
    # Load data
    df = pd.read_csv(cfg.data.train)
    submission_df = pd.read_csv(cfg.data.test)

    # Prepare data in correct order
    train_df, val_df = split_raw_data(df, cfg)
    X_train, y_train, le = prepare_data(train_df, cfg)
    X_val, y_val = val_transformation(val_df, X_train, le, cfg)
    X_submit = submission_transformation(submission_df, X_train)

    # Run Optuna study
    study = run_study(X_train, y_train, X_val, y_val, n_trials=cfg.optuna.n_trials, cfg)

    # Train final model and generate submission
    model = train_final_model(study, X_train, y_train, X_val, y_val, cfg)
    generate_submission(model, X_submit, submission_df, le, cfg)

if __name__ == "__main__":
    main()