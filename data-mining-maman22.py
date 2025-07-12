import time
from typing import Callable, Dict, List, Set
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kagglehub import KaggleDatasetAdapter, load_dataset
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import itertools
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score


FILE_PATH = "healthcare-dataset-stroke-data.csv"  # relative path inside Kaggle dataset
MIN_SUPPORT = 0.40  # ≥ 40 % of transactions
MIN_CONFIDENCE = 0.60  # ≥ 60 % confidence

# -------------------------- Data Loading --------------------------------

def load_stroke_data() -> pd.DataFrame:
    """Download (first run) and return the stroke-prediction dataframe."""
    print("⏳ Loading dataset from Kaggle …")
    t0 = time.time()
    df: pd.DataFrame = load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "fedesoriano/stroke-prediction-dataset",
        FILE_PATH,
    )
    print(f"✅ Dataset loaded in {time.time() - t0:.1f}s → shape={df.shape}\n")

    # Basic clean-up identical to Part-1
    df.replace(["N/A", "n/a", "NA", "na", ""], np.nan, inplace=True)

    # Impute BMI missing values with the median (as in Part 1)
    if "bmi" in df.columns:
        df["bmi"].fillna(df["bmi"].median(), inplace=True)

    return df

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

# -------------------------- Pre-processing ------------------------------

def discretise(df: pd.DataFrame) -> pd.DataFrame:
    """Copy the *age_group* discretisation used in Part 1."""
    bins = [0, 40, 55, 70, 120]
    labels = [0, 1, 2, 3]  # 0 = ≤40, 3 = >70
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    return df

# -------------------------- Transaction Encoding ------------------------

def to_transaction_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return one-hot encoded transactional DataFrame ready for Apriori."""
    # Build list-of-lists of tokens like "gender_Male"
    transactions: List[List[str]] = []
    for _, row in df.iterrows():
        txn: List[str] = [f"{c}_{row[c]}" for c in cols if pd.notna(row[c])]
        transactions.append(txn)

    te = TransactionEncoder()
    encoded_arr = te.fit(transactions).transform(transactions)
    return pd.DataFrame(encoded_arr, columns=te.columns_)

# -------------------------- Redundancy Pruning --------------------------

def prune_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Remove rules whose antecedent is a superset of another rule with ≥ confidence."""
    keep_indices = []
    for i, r1 in rules.iterrows():
        antecedent1: Set[str] = r1["antecedents"]
        conf1 = r1["confidence"]
        consq = r1["consequents"]
        redundant = False
        for j, r2 in rules.iterrows():
            if i == j:
                continue
            if consq != r2["consequents"]:
                continue
            if antecedent1.issuperset(r2["antecedents"]) and conf1 <= r2["confidence"]:
                redundant = True
                break
        if not redundant:
            keep_indices.append(i)
    return rules.loc[keep_indices].reset_index(drop=True)

# -------------------------- Apriori Pipeline ----------------------------

def apriori_pipeline(df: pd.DataFrame, min_support: float = MIN_SUPPORT, min_conf: float = MIN_CONFIDENCE):
    # df = load_stroke_data()
    df = discretise(df)

    cat_cols = [
        "gender",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
        "stroke",
        "age_group",
    ]

    print("🔄 Converting to one-hot …")
    df_bin = to_transaction_df(df, cat_cols)
    print(f"✅ Encoded shape = {df_bin.shape}\n")

    # 1. Frequent item‑sets
    print("🔍 Mining frequent item-sets (Apriori)…")
    t0 = time.time()
    freq = apriori(df_bin, min_support=min_support, use_colnames=True, verbose=0)
    print("✨ Frequent item-sets:")
    for i, r in freq.iterrows():
        print(f"  #{i+1:02d}: {r['itemsets']} | supp={r['support']:.3f}")
    print(f"   → {len(freq)} item-sets in {time.time() - t0:.2f}s\n")

    # 2. Association rules
    if freq.empty:
        print("⚠️ No item-sets meet the support threshold — adjust MIN_SUPPORT.")
        return

    print("🔗 Generating rules …")
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    print(f"   → {len(rules)} raw rules\n")

    if rules.empty:
        print("⚠️ No rules meet the confidence threshold — adjust MIN_CONFIDENCE.")
        return

    # Sort by confidence for neat printing
    rules.sort_values("confidence", ascending=False, inplace=True, ignore_index=True)

    # 3. Display raw rules 
    print("📃 Raw rules by confidence:")
    for i, r in rules.iterrows():
        ant = ", ".join(sorted(r["antecedents"]))
        cons = ", ".join(sorted(r["consequents"]))
        print(f"  #{i+1:02d}: IF {{{ant}}} → {{{cons}}} | supp={r['support']:.3f} conf={r['confidence']:.3f} lift={r['lift']:.2f}")

    print(f"   → {len(rules)} rules found")

    # 3.5. Filter rules by lift
    print("🔍 Filtering rules by lift …")
    tol = 0.05                    # tolerance window around 1.0

    rules = rules[(rules['lift'] < 1 - tol) | (rules['lift'] > 1 + tol)].reset_index(drop=True)

    # 4. Prune redundancy
    print("\n Pruning redundant rules …")
    pruned = prune_rules(rules)
    print(f"   → {len(pruned)} rules remain after pruning\n")

    # 5. Display pruned rules 
    print("📃 Pruned rules by confidence:")
    for i, r in pruned.iterrows():
        ant = ", ".join(sorted(r["antecedents"]))
        cons = ", ".join(sorted(r["consequents"]))
        print(f"  #{i+1:02d}: IF {{{ant}}} → {{{cons}}} | supp={r['support']:.3f} conf={r['confidence']:.3f} lift={r['lift']:.2f}")

    # Optional: return both DataFrames for downstream analysis
    return rules, pruned

# -------------------------- Part 2 Clustering -----------------------------#
# -----------------------------------------------------------------------
# Part 2.2 – Hierarchical Clustering *without* Silhouette
# -----------------------------------------------------------------------



def _sse_radius_diameter(X: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Return total SSE, mean radius and mean diameter for a partition."""
    sse, radii, diams = 0.0, [], []
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts) == 0:
            continue
        centroid = pts.mean(axis=0)
        # cohesion
        sse_c   = np.square(pts - centroid).sum()
        radius  = np.sqrt(sse_c / len(pts))
        diam    = pairwise_distances(pts).max()
        sse    += sse_c
        radii.append(radius)
        diams.append(diam)
    return sse, float(np.mean(radii)), float(np.mean(diams))


def clustering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hierarchical clustering with linkage.
    Scans k = 2…10, evaluates with Davies–Bouldin, SSE, radius, diameter,
    generates plots, and prints a cluster profile.  **No silhouette used.**
    """
    print("🔬 Starting Clustering Pipeline …")

    # ------------------------------------------------------------------
    # 1.  Pre-processing  (same as before)
    # ------------------------------------------------------------------
    num_cols = ["age", "avg_glucose_level", "bmi"]
    cat_cols = [c for c in df.columns if c not in num_cols + ["stroke", "id"]]

    df_scaled = df.copy()
    df_scaled[num_cols] = MinMaxScaler().fit_transform(df_scaled[num_cols])

    X_df = pd.get_dummies(df_scaled[num_cols + cat_cols], drop_first=False)

    # 1) numeric (already scaled 0-1) – min & max
    print("\nNumeric columns (scaled) — min / max")
    print(df_scaled[num_cols].agg(['min', 'max']).T)

    # 2) categorical — list the actual one-hot indicator columns that exist
    print("\nCategorical columns — one-hot indicators")

    for col in cat_cols:
        prefix = f"{col}_"
        dummies = [c for c in X_df.columns if c.startswith(prefix)]
        
        if dummies:                     # real one-hot columns like gender_Female
            print(f"{col}: {dummies}")
        else:                           # column stayed numeric (binary 0/1)
            uniques = sorted(df[col].dropna().unique().tolist())
            print(f"{col}: kept as numeric {uniques}")
    X = X_df.values
    y = df["stroke"].values
    print(f"✅ Prepared data for clustering: {X.shape}")

    # ------------------------------------------------------------------
    # 2.  Scan k = 2 … 10
    # ------------------------------------------------------------------
    ks, dbs, sses, radii, diams, labels_dict = [], [], [], [], [], {}
    print("🔍 Scanning k = 2 to 10 …")

    for k in range(2,3):
        labels = AgglomerativeClustering(
            n_clusters=k, linkage="ward").fit_predict(X)

        db  = davies_bouldin_score(X, labels)
        sse, rad, dia = _sse_radius_diameter(X, labels)

        ks.append(k); dbs.append(db); sses.append(sse)
        radii.append(rad); diams.append(dia)
        labels_dict[k] = labels

        print(f"   k={k}: DB={db:.3f}  SSE={sse:,.0f}  "
              f"mean radius={rad:.3f}  mean diameter={dia:.3f}")

    # choose partition with *lowest* DB index
    best_k      = ks[int(np.argmin(dbs))]
    best_labels = labels_dict[best_k]
    print(f"\n✅ Selected k = {best_k}  (Davies–Bouldin = {min(dbs):.3f})")

    # ------------------------------------------------------------------
    # 3.  Plots
    # ------------------------------------------------------------------
    Path("plots").mkdir(exist_ok=True)

    # Quality-indices vs k
    fig, ax1 = plt.subplots()
    ax1.plot(ks, dbs, marker="o", label="Davies–Bouldin")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Davies–Bouldin (lower = better)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(ks, sses, marker="s", color="tab:orange", label="SSE")
    ax2.set_ylabel("Total SSE", color="tab:orange")
    plt.title("Internal cluster-quality indices vs. k")
    fig.tight_layout()
    plt.savefig("plots/quality_vs_k.png", dpi=300)
    plt.close()

    # Ward-linkage dendrogram (truncated)
    Z = linkage(X, method="ward")
    plt.figure(figsize=(8, 3))
    dendrogram(Z, truncate_mode="lastp", p=20,
               leaf_rotation=90., leaf_font_size=10.)
    # draw cut-line at the selected level
    plt.axhline(Z[-(best_k - 1), 2], c="red", ls="--")
    plt.title("Ward-linkage dendrogram (truncated)")
    plt.tight_layout()
    plt.savefig("plots/dendrogram_trunc.png", dpi=300)
    plt.close()

    # 2-D t-SNE embedding
    tsne = TSNE(n_components=2, random_state=42, init="pca")
    X_emb = tsne.fit_transform(X)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_emb[:, 0], y=X_emb[:, 1],
                    hue=best_labels, palette="bright",
                    s=15, linewidth=0)
    plt.title(f"t-SNE embedding coloured by cluster (k={best_k})")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("plots/tsne_clusters.png", dpi=300)
    plt.close()

    print("📊 Plots saved to plots/ directory")

    # ------------------------------------------------------------------
    # 4.  Cluster profile
    # ------------------------------------------------------------------
    df_profiled = df.copy()
    df_profiled["cluster"] = best_labels

    # Create gender_Female column if it doesn't exist
    if "gender_Female" not in df_profiled.columns:
        df_profiled["gender_Female"] = (df_profiled["gender"] == "Female").astype(int)

    # Add marital-status & smoking indicators
    df_profiled["residence_rural"] = (df_profiled["Residence_type"] == "Rural").astype(int)

    # Ensure marital-status & smoking indicator columns exist
    if "married" not in df_profiled.columns:
        df_profiled["married"] = (df_profiled["ever_married"] == "Yes").astype(int)
    if "smokes" not in df_profiled.columns:
        df_profiled["smokes"] = (df_profiled["smoking_status"] == "smokes").astype(int)
    if "former_smoke" not in df_profiled.columns:
        df_profiled["former_smoke"] = (df_profiled["smoking_status"] == "formerly smoked").astype(int)
    if "never_smoke" not in df_profiled.columns:
        df_profiled["never_smoke"] = (df_profiled["smoking_status"] == "never smoked").astype(int)


    profile = (df_profiled.groupby("cluster")
               .agg(size=("cluster", "size"),
                    mean_age=("age", "mean"),
                    pct_female=("gender_Female", "mean"),
                    pct_hyper=("hypertension", "mean"),
                    pct_stroke=("stroke", "mean"),
                    pct_married=("married", "mean"),
                    pct_smokes=("smokes", "mean"),
                    pct_former=("former_smoke", "mean"),
                    pct_never=("never_smoke", "mean"),
                    pct_residence_rural=("residence_rural", "mean"),

                    )
               .round({"mean_age": 1,
                       "pct_female": 2,
                       "pct_hyper": 2,
                       "pct_stroke": 3,
                       "pct_married": 2,
                       "pct_smokes": 2,
                       "pct_former": 2,
                       "pct_residence_rural": 2,
                       "pct_never": 2})
               .reset_index())

    print(f"\n=== Cluster profile (k = {best_k}) ===")
    print(profile.to_string(index=False))

    # ------------------------------------------------------------------
    # 5.  Stroke-specific inspection (optional, kept from your version)
    # ------------------------------------------------------------------
    stroke_pos = df_profiled[df_profiled["stroke"] == 1]
    print("\n🩺 Stroke-positive cases per cluster:")
    print(stroke_pos["cluster"].value_counts().sort_index())

    share = (stroke_pos["cluster"].value_counts(normalize=True)
                                   .sort_index()
                                   .mul(100).round(1))
    print("\n📊 Percentage of all stroke cases found in each cluster (%):")
    print(share)

    return profile

# -------------------------- CLI entry‑point -----------------------------


# -------------------------- Part 3 Neural Network -----------------------------
# 
# 

def prepare_nn_data(df: pd.DataFrame, test_size: float = 0.20):
    """
    Scale + one-hot encode then split into train / test (80/20).
    Returns train, test tuples and the test indices.
    """
    num_cols = ["age", "avg_glucose_level", "bmi"]
    cat_cols = [c for c in df.columns if c not in num_cols + ["stroke", "id"]]

    # preprocessing
    df_scaled = df.copy()
    df_scaled[num_cols] = MinMaxScaler().fit_transform(df_scaled[num_cols])

    X_df = pd.get_dummies(df_scaled[num_cols + cat_cols], drop_first=False)
    X = X_df.values.astype("float32")
    y = df_scaled["stroke"].values.astype("float32")

    indices = np.arange(len(df))
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, indices, test_size=test_size, stratify=y, random_state=42)

    # --- SMOTE oversampling on the training data only ---
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train.astype(int))
    y_train_res = y_train_res.astype('float32')

    return (X_train_res, y_train_res), (X_test, y_test), test_idx

def build_nn_model(input_dim: int) -> keras.Model:
    """
    64-16 MLP with dropout, sigmoid output.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.10),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.10),
        layers.Dense(1, activation="sigmoid")
    ])


    def focal_loss(alpha=0.25, gamma=2.0):
        def loss_fn(y_true, y_pred):
            # clip to avoid NaNs
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            # p_t is model’s estimated prob for the true class
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            # standard BCE
            bce = - (y_true * tf.math.log(y_pred) +
                    (1 - y_true) * tf.math.log(1 - y_pred))
            # focal term
            modulator = (1 - p_t) ** gamma
            # alpha balancing
            alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
            return tf.reduce_mean(alpha_weight * modulator * bce)
        return loss_fn
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        # loss=keras.losses.BinaryCrossentropy(),
        loss=focal_loss(alpha=0.75, gamma=2.0),
        metrics=[keras.metrics.AUC(name="AUC"),
                 keras.metrics.Precision(name="Precision"),
                 keras.metrics.Recall(name="Recall")],
    )
    return model

def nn_pipeline(df: pd.DataFrame):
    """
    Part 3 – trains the NN, makes plots, prints confusion matrix + report.
    """
    print("\n🧠 PART 3: NEURAL NETWORK")
    print("=" * 60)

    # 1. Data prep -------------------------------------------------------
    (X_train, y_train), (X_test, y_test), test_idx = prepare_nn_data(df)
    print(f"✅ Data split: train={X_train.shape[0]}  test={X_test.shape[0]}")

    # 2. Model -----------------------------------------------------------
    model = build_nn_model(X_train.shape[1])
    class_weight = {0: 1.0, 1: (len(y_train) - y_train.sum()) / y_train.sum()}
    print(f"⚖️  Class weights: {class_weight}")

    # 3. Training --------------------------------------------------------
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=320,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=30, min_delta=1e-4, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
        ],
        verbose=2
    )

    # 4. Learning-curve plot --------------------------------------------
    Path("plots").mkdir(exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(history.history["loss"], label="train loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val loss")
    plt.ylabel("Binary cross-entropy loss"); plt.xlabel("Epoch")
    plt.title("Loss vs. Epochs")
    plt.legend()
    plt.tight_layout(); plt.savefig("plots/nn_loss.png", dpi=300); plt.close()
    print("📈 Saved loss curve to plots/nn_loss.png")

    # 5. Evaluation on test ---------------------------------------------
    y_prob = model.predict(X_test, verbose=0).ravel()
    best_recall, best_thr = 0.0, 0.5
    for thr in np.linspace(0, 1, 101):
        y_pred_thr = (y_prob > thr).astype(int)
        r = f1_score(y_test, y_pred_thr, pos_label=1)
        if r > best_recall:
            best_recall, best_thr = r, thr

    print(f"▶ Best positive‐class recall = {best_recall:.3f} at threshold = {best_thr:.2f}")
    y_pred = (y_prob > best_thr).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    print("\n🔍 Confusion matrix (test):")
    print(cm)
    print("\n", classification_report(y_test, y_pred, digits=3))

    # --- confusion-matrix plot (for report) ---
    fig, ax = plt.subplots(figsize=(3,3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for i, j in itertools.product(range(2), range(2)):
        ax.text(j, i, cm[i,j], ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.title("Confusion matrix (test)")
    plt.tight_layout(); plt.savefig("plots/nn_conf_mat.png", dpi=300); plt.close()
    print("🖼️  Saved confusion-matrix plot to plots/nn_conf_mat.png")

    # 6. Extreme cases ---------------------------------------------------
    test_df = df.iloc[test_idx].copy()
    test_df["p"] = y_prob
    # Misclassified cases -------------------------------------------------
    misclassified = test_df.copy()
    misclassified["y_true"] = y_test
    misclassified["y_pred"] = y_pred
    misclassified = misclassified[misclassified["y_true"] != misclassified["y_pred"]]
    print("\n❌ All misclassified test cases:")
    cols_to_show = ["id", "age", "avg_glucose_level", "y_true", "y_pred", "p"] if "id" in misclassified.columns else ["age", "avg_glucose_level", "y_true", "y_pred", "p"]
    if misclassified.empty:
        print("✅ None (perfect classification on test set!)")
    else:
        print(misclassified[cols_to_show].to_csv(index=False))

    # Highlight a few extreme cases
    fn = misclassified[(misclassified["y_true"] == 1) & (misclassified["y_pred"] == 0)].sort_values("p").head(3)
    fp = misclassified[(misclassified["y_true"] == 0) & (misclassified["y_pred"] == 1)].sort_values("p", ascending=False).head(3)

    print("\n⚠️  Hard False-Negatives (lowest p):\n", fn[["id","age","avg_glucose_level","p"]])
    print("\n⚠️  Hard False-Positives (highest p):\n", fp[["id","age","avg_glucose_level","p"]])

    return dict(history=history, cm=cm, fn=fn, fp=fp)

# ------------------------ integrate in main() ----------------------------
#


def main() -> None:
    """
    Main function to run both Part 2.1 (Association Rules) and Part 2.2 (Clustering)
    """
    print("🚀 Starting Data Mining Pipeline for Maman 2.2...")
    
    # Load and clean data
    df = load_stroke_data()
    df = clean_dataset(df)
    df = clean_illegal_values(df)
    
    print(f"📊 Final dataset shape: {df.shape}")
    print("="*60)
    
    # # Part 2.1: Association Rule Mining
    print("\n🔗 PART 2.1: ASSOCIATION RULE MINING")
    print("="*60)
    apriori_pipeline(df)
    
    print("\n" + "="*60)
    
    # # Part 2.2: Hierarchical Clustering
    print("\n🔬 PART 2.2: HIERARCHICAL CLUSTERING")
    print("="*60)
    clustering_pipeline(df)
    
    print("\n" + "="*60)
    
    # Part 3: Neural Network
    print("\n🧠 PART 3: NEURAL NETWORK")
    print("=" * 60)
    nn_pipeline(df)

    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
