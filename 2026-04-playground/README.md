# Predicting Irrigation Need

**Goal:** Predict the class of irrigation required (Low, Medium, High) under varying environmental conditions.

**Competition:** [Kaggle Playground Series: S6E4](https://kaggle.com/competitions/playground-series-s6e4)

```

## Project Structure
irrigation-prediction/
│
├── data/
│   ├── raw/                  # Original Kaggle files, never modified
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/            # Output of preprocessing scripts
│       ├── X_train.parquet
│       ├── X_val.parquet
│       ├── y_train.npy
│       └── y_val.npy
│
├── src/
│   ├── init.py
│   ├── data.py               # Data splitting, encoding, and transformation
│   ├── features.py           # Feature engineering logic
│   ├── train.py              # Optuna objective function and study runner
│   ├── evaluate.py           # Metrics and evaluation utilities
│   └── predict.py            # Final model training and submission generation
│
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis only
│
├── models/                   # Saved model artifacts
├── submissions/              # Kaggle submission CSVs
├── mlruns/                   # MLflow tracking (auto-generated)
│
├── config.py                 # All constants and configuration
├── main.py                   # Pipeline entry point
├── requirements.txt
├── .gitignore
└── README.md

```

## Approach

- Performed exploratory data analysis (EDA) to understand feature distributions and class balance
- Built a modular Python pipeline with separate modules for data processing, training, and prediction
- Encoded categorical features with `pd.get_dummies` and the target with `LabelEncoder`
- Tuned an XGBoost classifier using Optuna with 100 trials and SQLite-backed study persistence
- Tracked all experiments with MLflow including hyperparameters, metrics, and model artifacts
- Evaluated using balanced accuracy score to align with competition metric

## Results

- Best model achieved a balanced accuracy score of **TBD**

---

## Reproducing Results

```bash
# 1. Clone the repo
git clone https://github.com/your-username/irrigation-prediction.git
cd irrigation-prediction

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add raw Kaggle data to data/raw/

# 5. Run the full pipeline
python main.py

# 6. View MLflow experiment results
mlflow ui
```

---

## Tech Stack

- **Modeling:** XGBoost
- **Tuning:** Optuna
- **Experiment Tracking:** MLflow
- **Data:** pandas, NumPy, scikit-learn
- **Environment:** Python 3.12, VSCode

---

## Citation

```bibtex
@misc{playground-series-s6e4,
    author = {Yao Yan, Walter Reade, Elizabeth Park},
    title = {Predicting Irrigation Need},
    year = {2026},
    howpublished = {\url{https://kaggle.com/competitions/playground-series-s6e4}},
    note = {Kaggle}
}
```