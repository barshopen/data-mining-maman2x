import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from typing import Callable, Dict
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# ----------------------- Config and Constraints -----------------------

FILE_PATH: str = "healthcare-dataset-stroke-data.csv"

illegal_conditions: Dict[str, Callable[[pd.Series], pd.Series]] = {
    'age': lambda x: x < 0,
    'gender': lambda x: ~x.isin(['Male', 'Female']),
    'hypertension': lambda x: ~x.isin([0, 1]),
    'heart_disease': lambda x: ~x.isin([0, 1]),
    'ever_married': lambda x: ~x.isin(['Yes', 'No']),
    'work_type': lambda x: ~x.isin(['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed']),
    'Residence_type': lambda x: ~x.isin(['Urban', 'Rural']),
    'avg_glucose_level': lambda x: (x <= 40) | (x > 400),
    'bmi': lambda x: x <= 0,
    'smoking_status': lambda x: ~x.isin(['never smoked', 'formerly smoked', 'smokes', 'Unknown']),
    'stroke': lambda x: ~x.isin([0, 1]),
}

extreme_thresholds: Dict[str, tuple[float, float]] = {
    'age': (0, 110),
    'avg_glucose_level': (40, 400),
    'bmi': (10, 60),
}

expected_types: Dict[str, type] = {
    'age': float,
    'gender': str,
    'hypertension': int,
    'heart_disease': int,
    'ever_married': str,
    'work_type': str,
    'Residence_type': str,
    'avg_glucose_level': float,
    'bmi': float,
    'smoking_status': str,
    'stroke': int
}

# ----------------------------- Load & Clean -----------------------------

def load_data() -> pd.DataFrame:
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "fedesoriano/stroke-prediction-dataset",
        FILE_PATH,
    )
    df.replace(['N/A', 'n/a', 'NA', '', 'na'], np.nan, inplace=True)
    return df

def clean_illegal_values(df: pd.DataFrame) -> pd.DataFrame:
    valid_mask = pd.Series(True, index=df.index)
    for col, condition in illegal_conditions.items():
        if col in df.columns:
            valid_mask &= ~condition(df[col])
    return df[valid_mask].copy()

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if 'bmi' in df.columns:
        median_bmi = df['bmi'].median()
        print(f"\n📈 Imputing {df['bmi'].isnull().sum()} missing 'bmi' values with median: {median_bmi:.2f}")
        df['bmi'] = df['bmi'].fillna(median_bmi).clip(upper=60)
    return df

# --------------------------- Validation & Summary ---------------------------

def analyze_column(df: pd.DataFrame, col: str, illegal_counts, extreme_counts, missing_counts) -> None:
    data = df[col]
    print(f"\n--- {col} ---")
    missing = data.isnull().sum()
    missing_counts[col] = missing
    print(f"Missing values: {missing}")
    if pd.api.types.is_numeric_dtype(data):
        print(f"Range: [{data.min()} - {data.max()}], Mean: {data.mean():.2f}, Std: {data.std():.2f}")
        if col in extreme_thresholds:
            low, high = extreme_thresholds[col]
            extreme = data[(data < low) | (data > high)]
            extreme_counts[col] = len(extreme)
            print(f"Extreme values ({low}–{high}): {len(extreme)}")
        else:
            extreme_counts[col] = 0
    else:
        unique_vals = data.dropna().unique()
        print(f"Unique values: {unique_vals.tolist()}")
        extreme_counts[col] = 0
    if col in illegal_conditions:
        illegal = data[illegal_conditions[col](data)]
        illegal_counts[col] = len(illegal)
        print(f"Illegal values: {len(illegal)}")
    else:
        illegal_counts[col] = 0

def analyze_dataset(df: pd.DataFrame) -> None:
    illegal_counts = defaultdict(int)
    extreme_counts = defaultdict(int)
    missing_counts = defaultdict(int)
    for col in df.columns:
        if col != "id":
            analyze_column(df, col, illegal_counts, extreme_counts, missing_counts)
    print("\n📊 === SUMMARY REPORT ===")
    summary = pd.DataFrame({
        'Illegal Values': pd.Series(illegal_counts),
        'Extreme Values': pd.Series(extreme_counts),
        'Missing Values': pd.Series(missing_counts)
    }).fillna(0).astype(int)
    print(summary)

def conforms_to_type(val: object, expected_type: type) -> bool:
    if pd.isnull(val):
        return False
    try:
        if expected_type == str:
            return isinstance(val, str)
        expected_type(val)
        return True
    except:
        return False

def check_type_conformance(df: pd.DataFrame) -> pd.Series:
    print("\n🔍 Type Conformance Check:")
    valid_mask = pd.Series(True, index=df.index)
    for col, expected_type in expected_types.items():
        conformance = df[col].apply(lambda v: conforms_to_type(v, expected_type))
        invalid_count = (~conformance).sum()
        if invalid_count > 0:
            print(f"Column '{col}' has {invalid_count} invalid {expected_type.__name__} values")
        valid_mask &= conformance
    return valid_mask

# ----------------------------- Modeling -----------------------------

def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df_encoded = df.copy()
    encoders = {}
    for col in df_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
    return df_encoded, encoders

def split_data(df: pd.DataFrame, target_col: str) -> tuple:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    print(f"\n🔁 Applied SMOTE: {len(y_train)} → {len(y_bal)} samples")
    return X_bal, y_bal

def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(name: str, model, X_test, y_test) -> None:
    # 1. Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # 2. Custom threshold
    threshold = 0.3  # Adjust this value as needed
    y_pred = (y_proba >= threshold).astype(int)

    # 3. Evaluation metrics
    print(f"\n📊 {name} Results (Threshold = {threshold}):")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred, zero_division=0))

    # 4. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Stroke', 'Stroke'])
    disp.plot(cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

def plot_decision_tree(model: DecisionTreeClassifier, feature_names: list[str]) -> None:
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=["No Stroke", "Stroke"], filled=True)
    plt.title("CART Decision Tree")
    plt.savefig("cart_tree.png", bbox_inches='tight')
    plt.close()

def plot_sample_random_forest_tree(model: RandomForestClassifier, feature_names: list[str]) -> None:
    plt.figure(figsize=(20, 10))
    plot_tree(model.estimators_[0], feature_names=feature_names, class_names=["No Stroke", "Stroke"], filled=True)
    plt.title("First Tree in Random Forest")
    plt.savefig("random_forest_tree_0.png", bbox_inches='tight')
    plt.close()

def run_tree_models_pipeline(df_cleaned: pd.DataFrame) -> None:
    df_encoded, encoders = encode_categorical(df_cleaned)
    X_train, X_test, y_train, y_test = split_data(df_encoded, target_col='stroke')
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)
    tree_model = train_decision_tree(X_train_bal, y_train_bal)
    rf_model = train_random_forest(X_train_bal, y_train_bal)
    evaluate_model("Decision Tree (CART) with SMOTE", tree_model, X_test, y_test)
    evaluate_model("Random Forest with SMOTE", rf_model, X_test, y_test)
    plot_decision_tree(tree_model, feature_names=X_train.columns.tolist())
    plot_sample_random_forest_tree(rf_model, feature_names=X_train.columns.tolist())

# ----------------------------- Main -----------------------------

def main() -> None:
    df = load_data()
    print("First 5 records:\n", df.head())
    analyze_dataset(df)
    print("\n🧹 Cleaning Report (Illegal Only)")
    original_count = len(df)
    df_cleaned = clean_illegal_values(df)
    cleaned_count = len(df_cleaned)

    print(f"Original: {original_count} rows")
    print(f"After cleaning: {cleaned_count} rows")
    print(f"Removed: {original_count - cleaned_count} rows")
    df_cleaned = clean_dataset(df_cleaned)
    df_cleaned['age_group'] = pd.cut(df_cleaned['age'], bins=[0, 40, 55, 70, 120],  labels=[0, 1, 2, 3])
    analyze_dataset(df_cleaned)
    _ = check_type_conformance(df_cleaned)
    df_cleaned.to_csv("stroke_cleaned.csv", index=False)
    run_tree_models_pipeline(df_cleaned)

if __name__ == "__main__":
    main()
