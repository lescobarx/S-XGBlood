# -*- coding: utf-8 -*-
"""
Seeded XGBoost AUC Analysis Across Multiple Cohorts

This script repeatedly trains an XGBoost model with different random seeds,
applies manual oversampling, computes clinical feature ratios, standardizes
features, evaluates ROC AUC on multiple validation sets (Leipzig, Greifswald,
MIMIC), and aggregates AUC statistics.

"""

import json
import random
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


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
    seed: int
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier with specified parameters and seed.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Validation feature matrix.
        y_val: Validation labels.
        best_params: Hyperparameters for XGBClassifier.
        seed: Random seed for reproducibility.

    Returns:
        Trained XGBClassifier model.
    """
    np.random.seed(seed)
    random.seed(seed)

    model = xgb.XGBClassifier(
        **best_params,
        random_state=seed
    )
    # Train with validation set monitoring, verbosity disabled
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
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
        oversample_factor: Number of replications.
        numeric_cols: List of numeric cols to optionally add noise.
        group_cols: Columns to group by before replication.
        add_noise: Whether to add Gaussian noise.
        noise_level: Proportion of noise relative to feature std.
        balance_exact: If True, balance classes exactly.
        random_state: Seed for random operations.

    Returns:
        DataFrame with added oversampled records.
    """
    np.random.seed(random_state)

    # Select negative class with target diagnosis
    subset = df[(df[label_col] == 0) & (df[diag_col] == target_diag)]

    # Replicate subset by group or globally
    if group_cols:
        replicas = []
        for _, group in subset.groupby(group_cols):
            replicas.append(pd.concat([group] * oversample_factor, ignore_index=True))
        oversampled = pd.concat(replicas, ignore_index=True)
    else:
        oversampled = pd.concat([subset] * oversample_factor, ignore_index=True)

    # Optionally add Gaussian noise
    if add_noise and numeric_cols:
        for col in numeric_cols:
            std = oversampled[col].std()
            noise = np.random.normal(0, noise_level * std, size=len(oversampled))
            oversampled[col] += noise

    # Optionally balance classes exactly
    if balance_exact:
        count_0 = df[df[label_col] == 0].shape[0]
        count_1 = df[df[label_col] == 1].shape[0]
        diff = abs(count_1 - count_0)
        if count_0 < count_1 and diff > 0:
            extra = oversampled.sample(n=diff, replace=True, random_state=random_state)
            return pd.concat([df, extra], ignore_index=True)
        return pd.concat([df, oversampled], ignore_index=True)

    return pd.concat([df, oversampled], ignore_index=True)


if __name__ == '__main__':
    # Load best hyperparameters
    with open('./Results/best_params_optuna.json', 'r', encoding='utf-8') as f:
        best_params = json.load(f)

    # Lists to store AUC results
    auc_data_le_val = []
    auc_data_gw = []
    auc_data_mi = []

    # Iterate over a set of random seeds
    seeds = [1969207776, 1969381777, 1969549025, 1969715444,
             1969880797, 1970045850, 1970211320, 1970376985,
             1970541848, 1970708391, 1970871325, 1971035844,
             1971201286, 1971364957, 1971529602, 1971694292,
             1971859146, 1972024133, 1972188575, 1972352809]

    for seed in seeds:
        print(f"\n✅ Running CBC model with seed: {seed}")

        # Load datasets for each cohort
        data_le = load_data('./Data/data_le.csv')
        data_le_val = load_data('./Data/data_le_val.csv')
        data_gw = load_data('./Data/data_gw.csv')
        data_mi = load_data('./Data/data_mi.csv')

        # Define base features
        features = ['Age', 'Sex', 'HGB', 'MCV', 'PLT', 'RBC', 'WBC']

        # Apply manual oversampling to training set
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
            random_state=seed
        )

        # Compute clinical feature ratios for all cohorts
        for df in (data_le, data_le_val, data_gw, data_mi):
            df['HGB_WBC_ratio'] = df['HGB'] / (df['WBC'] + 1)
            df['HGB_RBC_ratio'] = df['HGB'] / (df['RBC'] + 1)
            df['PLT_WBC_ratio'] = df['PLT'] / (df['WBC'] + 1)

        # Extend features list with ratios
        features.extend(['HGB_WBC_ratio', 'HGB_RBC_ratio', 'PLT_WBC_ratio'])

        # Split into feature matrices and labels
        X_le, y_le = data_le[features], data_le['Label']
        X_le_val, y_le_val = data_le_val[features], data_le_val['Label']
        X_gw, y_gw = data_gw[features], data_gw['Label']
        X_mi, y_mi = data_mi[features], data_mi['Label']

        # Standardize features
        scaler = StandardScaler()
        X_le = scaler.fit_transform(X_le)
        X_le_val = scaler.transform(X_le_val)
        X_gw = scaler.transform(X_gw)
        X_mi = scaler.transform(X_mi)

        # Train model
        xgb_model = train_xgboost(
            X_le, y_le,
            X_le_val, y_le_val,
            best_params,
            seed
        )

        # Evaluate and store AUC for each cohort
        for cohort_name, X, y, auc_list in [
            ('Leipzig', X_le_val, y_le_val, auc_data_le_val),
            ('Greifswald', X_gw, y_gw, auc_data_gw),
            ('MIMIC', X_mi, y_mi, auc_data_mi)
        ]:
            y_pred = xgb_model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_pred)
            auc_list.append(auc)
            print(f"\n🚀 AUC on {cohort_name}: {auc:.3f}")

    # Compute and save AUC statistics
    stats = {
        'data_le_val': auc_data_le_val,
        'data_gw': auc_data_gw,
        'data_mi': auc_data_mi
    }
    result_file = 'Results/results_seed_AUC.txt'
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write("\n=== AUC Statistics ===\n")
        for name, values in stats.items():
            f.write(
                f"{name}: Max: {np.max(values):.4f} - Min: {np.min(values):.4f} - Std: {np.std(values):.4f}\n"
            )

    print(f"\n✅ Results saved to: {result_file}")
