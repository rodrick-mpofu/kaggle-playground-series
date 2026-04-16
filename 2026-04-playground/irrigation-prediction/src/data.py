import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def split_raw_data(df, cfg):
    train_df, val_df = train_test_split(
        df,
        test_size=cfg.data.test_size,
        random_state=cfg.random_state,
        stratify=df[cfg.data.target]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def prepare_data(train_df, cfg):
    df = train_df.copy()
    df = pd.get_dummies(
        df,
        columns=[col for col in df.select_dtypes('object').columns if col != cfg.data.target],
        drop_first=True
    )
    le = LabelEncoder()
    y = le.fit_transform(df[cfg.data.target])
    X = df.drop([cfg.data.target, cfg.data.id_col], axis=1)
    bool_cols = X.select_dtypes(include=['bool']).columns
    X[bool_cols] = X[bool_cols].astype(int)
    print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    print("X_train shape:", X.shape)
    return X, y, le

def val_transformation(val_df, X_train, le, cfg):
    df = val_df.copy()
    y = le.transform(df[cfg.data.target])
    df = pd.get_dummies(df.drop(columns=[cfg.data.target, cfg.data.id_col]), drop_first=True)
    df = df.reindex(columns=X_train.columns, fill_value=0)
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df, y

def submission_transformation(test_df, X_train):
    df = test_df.drop(columns=['id']).copy()
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=X_train.columns, fill_value=0)
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df