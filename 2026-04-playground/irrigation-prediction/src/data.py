import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import RANDOM_STATE, TEST_SIZE, TARGET, ID_COL

def split_raw_data(df):
    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def prepare_data(train_df):
    df = train_df.copy()
    df = pd.get_dummies(
        df,
        columns=[col for col in df.select_dtypes('object').columns if col != TARGET],
        drop_first=True
    )
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    X = df.drop([TARGET, ID_COL], axis=1)
    bool_cols = X.select_dtypes(include=['bool']).columns
    X[bool_cols] = X[bool_cols].astype(int)
    print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    print("X_train shape:", X.shape)
    return X, y, le

def val_transformation(val_df, X_train, le):
    df = val_df.copy()
    y = le.transform(df[TARGET])
    df = pd.get_dummies(df.drop(columns=[TARGET, ID_COL]), drop_first=True)
    df = df.reindex(columns=X_train.columns, fill_value=0)
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df, y

def submission_transformation(test_df, X_train):
    df = test_df.drop(columns=[ID_COL]).copy()
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=X_train.columns, fill_value=0)
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df