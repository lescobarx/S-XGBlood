# -*- coding: utf-8 -*-
"""
Sepsis Prediction Pipeline with Procalcitonin (PCT) Feature

This script loads clinical data including PCT biomarker, preprocesses it,
applies manual oversampling, computes clinical feature ratios,
standardizes features, trains an XGBoost classifier, and evaluates on
a validation cohort.

"""

import json
import random
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import resample


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file, encode categorical variables, and filter missing labels.

    Args:
        file_path: Path to the input CSV file (semicolon-delimited).

    Returns:
        Preprocessed DataFrame with binary Label and encoded Sex.
    """
    df = pd.read_csv(file_path, delimiter=';')
    # Map Sex to binary values
    df['Sex'] = df['Sex'].map({'M': 1, 'W': 0})
    # Drop rows missing Label or Diagnosis
    df.dropna(subset=['Label', 'Diagnosis'], inplace=True)
    # Encode Label: 'Sepsis' -> 1, others -> 0
    df['Label'] = np.where(df['Label'] == 'Sepsis', 1, 0).astype(int)
    return df


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: dict,
    random_seed: int
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier with given parameters and validation monitoring.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        best_params: Hyperparameters for XGBClassifier.
        random_seed: Seed for reproducibility.

    Returns:
        Trained XGBClassifier model.
    """
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)

    model = xgb.XGBClassifier(
        **best_params,
        random_state=random_seed
    )
    # Train model with evaluation on validation set
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    return model


def oversample_manual(
    df: pd.DataFrame,
    label_col: str = 'Label',
    diag_col: str = 'Diagnosis',
    target_diag: str = 'Sepsis',
    oversample_factor: int = 10,
    numeric_cols: list = None,
    group_cols: list = None,
    add_noise: bool = True,
    noise_level: float = 0.01,
    balance_exact: bool = False,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Manually oversample negative class instances for a specific diagnosis.

    Args:
        df: Input DataFrame to oversample.
        label_col: Name of the label column.
        diag_col: Name of the diagnosis column.
        target_diag: Diagnosis to target for oversampling.
        oversample_factor: Number of replications for subset.
        numeric_cols: List of numeric columns to optionally add noise.
        group_cols: Columns to group by before replication.
        add_noise: Whether to add Gaussian noise to numeric features.
        noise_level: Proportion of noise relative to feature standard deviation.
        balance_exact: If True, balance classes exactly by sampling.
        random_state: Seed for random operations.

    Returns:
        DataFrame with added oversampled records.
    """
    np.random.seed(random_state)

    # Select negative class with target diagnosis
    subset = df[(df[label_col] == 0) & (df[diag_col] == target_diag)]

    if group_cols:
        # Oversample within each clinical group
        replicas = []
        for _, group in subset.groupby(group_cols):
            repeated = pd.concat([group] * oversample_factor, ignore_index=True)
            replicas.append(repeated)
        oversampled = pd.concat(replicas, ignore_index=True)
    else:
        # Global replication
        oversampled = pd.concat([subset] * oversample_factor, ignore_index=True)

    if add_noise and numeric_cols:
        # Add slight Gaussian noise to numeric columns
        for col in numeric_cols:
            std = oversampled[col].std()
            noise = np.random.normal(0, noise_level * std, size=len(oversampled))
            oversampled[col] += noise

    if balance_exact:
        # Balance classes exactly if specified
        count_0 = df[df[label_col] == 0].shape[0]
        count_1 = df[df[label_col] == 1].shape[0]
        diff = abs(count_1 - count_0)
        if count_0 < count_1 and diff > 0:
            extra = oversampled.sample(n=diff, replace=True, random_state=random_state)
            return pd.concat([df, extra], ignore_index=True)
        return pd.concat([df, oversampled], ignore_index=True)
    else:
        return pd.concat([df, oversampled], ignore_index=True)


if __name__ == '__main__':
    # Seed for reproducibility
    current_random_seed = 1714400672
    np.random.seed(current_random_seed)
    random.seed(current_random_seed)

    # Load training and validation datasets, filter by PCT availability
    data_le = load_data('./Data/data_le.csv')
    data_le = data_le[~data_le['PCT'].isna()]
    data_le_val = load_data('./Data/data_le_val.csv')
    data_le_val = data_le_val[~data_le_val['PCT'].isna()]

    # Define core features including PCT biomarker
    features = ['Age', 'Sex', 'HGB', 'MCV', 'PLT', 'RBC', 'WBC', 'PCT']

    # Apply manual oversampling on training set
    data_le = oversample_manual(
        df=data_le,
        label_col='Label',
        diag_col='Diagnosis',
        target_diag='Sepsis',
        oversample_factor=10,
        numeric_cols=features,
        group_cols=['Sex'],
        add_noise=True,
        noise_level=0.02,
        balance_exact=True,
        random_state=current_random_seed
    )

    # Compute key feature ratios related to sepsis severity
    for df in [data_le, data_le_val]:
        df['HGB_WBC_ratio'] = df['HGB'] / (df['WBC'] + 1)
        df['HGB_RBC_ratio'] = df['HGB'] / (df['RBC'] + 1)
        df['PLT_WBC_ratio'] = df['PLT'] / (df['WBC'] + 1)

    # Extend feature list with computed ratios
    features.extend(['HGB_WBC_ratio', 'HGB_RBC_ratio', 'PLT_WBC_ratio'])

    # Load best hyperparameters and update evaluation settings
    with open('./Results/best_params_optuna_pct.json', 'r', encoding='utf-8') as f:
        best_params = json.load(f)
    best_params.update({
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'early_stopping_rounds': 30
    })

    # Split features and labels for training and validation
    X_le, y_le = data_le[features], data_le['Label']
    X_le_val, y_le_val = data_le_val[features], data_le_val['Label']

    # Standardize feature distributions
    scaler = StandardScaler()
    X_le = scaler.fit_transform(X_le)
    X_le_val = scaler.transform(X_le_val)

    # Train the XGBoost model with PCT feature
    xgb_model_pct = train_xgboost(
        X_le,
        y_le,
        X_le_val,
        y_le_val,
        best_params,
        current_random_seed
    )

    # Save trained model and scaler artifacts
    joblib.dump(xgb_model_pct, 'Data/xgboost_optimized_CBC_PCT.pkl')
    joblib.dump(scaler, 'Data/scaler_CBC_PCT.pkl')
    print("\nModel saved successfully.")

    # Evaluate model performance on validation set
    y_pred_proba_le_val = xgb_model_pct.predict_proba(X_le_val)[:, 1]
    auc_le_val = roc_auc_score(y_le_val, y_pred_proba_le_val)
    print(f"Leipzig validation AUC: {auc_le_val:.3f}")
