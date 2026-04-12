import pandas as pd
from src.data import split_raw_data, prepare_data, val_transformation, submission_transformation
from src.train import run_study
from src.predict import train_final_model, generate_submission
from config import RAW_TRAIN, RAW_TEST, N_TRIALS

def main():
    # Load data
    df = pd.read_csv(RAW_TRAIN)
    submission_df = pd.read_csv(RAW_TEST)

    # Prepare data in correct order
    train_df, val_df = split_raw_data(df)
    X_train, y_train, le = prepare_data(train_df)
    X_val, y_val = val_transformation(val_df, X_train, le)
    X_submit = submission_transformation(submission_df, X_train)

    # Run Optuna study
    study = run_study(X_train, y_train, X_val, y_val, n_trials=N_TRIALS)

    # Train final model and generate submission
    model = train_final_model(study, X_train, y_train, X_val, y_val)
    generate_submission(model, X_submit, submission_df, le)

if __name__ == "__main__":
    main()