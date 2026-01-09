import re
from dataclasses import dataclass
from pathlib import Path
from luar_features import featurize_luar


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV


# Config
@dataclass
class Config:
    data_path: str = "Dataset.xlsx"
    out_dir: str = "results"
    seed: int = 42

    #stylometry baseline
    char_ngram_range: tuple = (3, 5)
    max_features: int = 200000

    # TF-IDF tweaks
    max_df: float = 0.9
    sublinear_tf: bool = True

    # Length filtering (tokens)
    min_tokens: int = 30
    max_tokens: int = 1500

    #Holdout settings
    holdout_min_samples: int = 300
    train_balance: bool = True

    # Saving / analysis
    decision_threshold: float = 0.5
    save_text_in_preds: bool = False 
    top_k_features: int = 30


cfg = Config()


#utilities
def basic_clean(text: str) -> str:
    """Light cleaning that won't destroy stylistic signals."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def standardize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Output label: 1=AI, 0=Human.
    Expects df has 'label_name' column like 'human-written'/'ai-generated'
    """
    df = df.copy()
    df["label_name"] = df["label_name"].astype(str).str.strip().str.lower()

    def map_label(x: str) -> int:
        if "human" in x:
            return 0
        if "ai" in x or "gpt" in x or "generated" in x or "llm" in x or "machine" in x:
            return 1
        raise ValueError(f"Unrecognized label_name: {x}")

    df["label"] = df["label_name"].map(map_label)
    return df


def describe_dataset(df: pd.DataFrame) -> None:
    print("\n=== Dataset head ===")
    print(df.head(2)[["text", "source", "label_name", "label"]])

    print("\n=== Label counts ===")
    print(df["label_name"].value_counts(dropna=False))

    print("\n=== Source counts (top 20) ===")
    print(df["source"].value_counts(dropna=False).head(20))

    print("\n=== Label by source (top sources) ===")
    top_sources = df["source"].value_counts().head(12).index
    print(df[df["source"].isin(top_sources)].groupby("source")["label"].value_counts().unstack(fill_value=0))

    print("\n=== Length summary (tokens) ===")
    print(df["n_tokens"].describe())


def save_split(df_train: pd.DataFrame, df_test: pd.DataFrame, name: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df_train[["source", "label", "n_tokens", "n_chars"]].to_csv(out / f"{name}_train_meta.csv", index=False)
    df_test[["source", "label", "n_tokens", "n_chars"]].to_csv(out / f"{name}_test_meta.csv", index=False)


def safe_setting_name(name: str) -> str:
    return (
        name.replace(":", "__")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
    )


# Splits
def make_splits_random(df: pd.DataFrame, seed: int):
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"]
    )
    return train_df, test_df


def make_splits_holdout_ai_source(df: pd.DataFrame, holdout_ai_source: str, seed: int, train_balance: bool = True):
    """
    Hold out one AI generator/source, but keep BOTH classes in the test set by sampling human examples.
    Optionally balance the training set for clean interpretation.
    """
    ai = df[df["label"] == 1].copy()
    hu = df[df["label"] == 0].copy()

    test_ai = ai[ai["source"] == holdout_ai_source].copy()
    if len(test_ai) == 0:
        raise ValueError(f"No AI samples for source={holdout_ai_source}")

    # Balanced test set
    test_hu = hu.sample(n=len(test_ai), random_state=seed).copy()
    test_df = pd.concat([test_ai, test_hu], ignore_index=True).sample(frac=1.0, random_state=seed)

    # Training pool excludes held-out AI + the sampled human test items
    train_ai = ai[ai["source"] != holdout_ai_source].copy()
    train_hu = hu.drop(index=test_hu.index).copy()

    if train_balance:
        n = min(len(train_ai), len(train_hu))
        train_ai = train_ai.sample(n=n, random_state=seed)
        train_hu = train_hu.sample(n=n, random_state=seed)

    train_df = pd.concat([train_ai, train_hu], ignore_index=True).sample(frac=1.0, random_state=seed)

    return train_df, test_df


# Feature pipelines
def featurize_char_ngrams(train_texts, test_texts, cfg: Config):
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=cfg.char_ngram_range,
        min_df=2,
        max_df=cfg.max_df,
        sublinear_tf=cfg.sublinear_tf,
        max_features=cfg.max_features,
    )
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)
    return X_train, X_test, vec


#Models and evaluation
def fit_lr_calibrated(X_train, y_train, seed: int):
    base = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    cal.fit(X_train, y_train)
    return cal


def fit_predict_proba_lr_calibrated(X_train, y_train, X_test, seed: int, threshold: float):
    cal = fit_lr_calibrated(X_train, y_train, seed)
    proba = cal.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return cal, pred, proba


def fit_predict_proba_rf(X_train, y_train, X_test, seed: int, threshold: float):
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return clf, pred, proba


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def evaluate_full(y_true, y_pred, y_prob):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": brier_score(y_true, y_prob),
    }

    # confusion matrix (TN, FP, FN, TP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    # AUC requires both classes in y_true
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["auc"] = np.nan

    return out


def save_predictions(df_test: pd.DataFrame, y_pred, y_prob, setting_name: str, model_name: str, out_dir: str, cfg: Config):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    preds = df_test.copy()
    preds = preds.reset_index(drop=True)

    preds["pred"] = y_pred
    preds["proba_ai"] = y_prob
    preds["correct"] = (preds["pred"] == preds["label"]).astype(int)

    keep_cols = ["source", "label_name", "label", "pred", "proba_ai", "correct", "n_tokens", "n_chars"]
    if cfg.save_text_in_preds:
        keep_cols = keep_cols + ["text"]

    fname = f"preds__{safe_setting_name(setting_name)}__{model_name}.csv"
    preds[keep_cols].to_csv(out / fname, index=False)


def extract_top_features_lr(vec: TfidfVectorizer, lr: LogisticRegression, top_k: int):
    """
    Returns two DataFrames:
      - top AI-indicative char n-grams (highest positive weights)
      - top Human-indicative char n-grams (most negative weights)
    Assumes label 1 = AI, label 0 = Human.
    """
    feature_names = np.array(vec.get_feature_names_out())
    coefs = lr.coef_.ravel()

    top_ai_idx = np.argsort(coefs)[-top_k:][::-1]
    top_hu_idx = np.argsort(coefs)[:top_k]

    top_ai = pd.DataFrame({"feature": feature_names[top_ai_idx], "weight": coefs[top_ai_idx]})
    top_hu = pd.DataFrame({"feature": feature_names[top_hu_idx], "weight": coefs[top_hu_idx]})
    return top_ai, top_hu


def fit_plain_lr_for_interpretability(X_train, y_train, seed: int):
    """
    Train a single LR on full training data (no calibration, no CV)
    purely for feature importance / coefficient inspection.
    """
    lr = LogisticRegression(max_iter=4000, solver="liblinear", random_state=seed)
    lr.fit(X_train, y_train)
    return lr


# Main experiment 
def run_one_setting(df_train, df_test, setting_name: str, cfg: Config):
    results = []

    y_train = df_train["label"].values
    y_test = df_test["label"].values
    train_texts = df_train["text"].tolist()
    test_texts = df_test["text"].tolist()

    #1) Stylometry: char n-grams
    Xtr, Xte, vec = featurize_char_ngrams(train_texts, test_texts, cfg)

    cal, pred, proba = fit_predict_proba_lr_calibrated(Xtr, y_train, Xte, cfg.seed, cfg.decision_threshold)
    m = evaluate_full(y_test, pred, proba)
    results.append({**m, "setting": setting_name, "features": "char_tfidf_3-5", "model": "LR_calibrated",
                    "n_train": int(len(df_train)), "n_test": int(len(df_test))})
    save_predictions(df_test, pred, proba, setting_name, "LR_calibrated", cfg.out_dir, cfg)

    rf, pred, proba = fit_predict_proba_rf(Xtr, y_train, Xte, cfg.seed, cfg.decision_threshold)
    m = evaluate_full(y_test, pred, proba)
    results.append({**m, "setting": setting_name, "features": "char_tfidf_3-5", "model": "RF",
                    "n_train": int(len(df_train)), "n_test": int(len(df_test))})
    save_predictions(df_test, pred, proba, setting_name, "RF", cfg.out_dir, cfg)

    # Top features for interpretability (char model only)
    lr_plain = fit_plain_lr_for_interpretability(Xtr, y_train, cfg.seed)
    top_ai, top_hu = extract_top_features_lr(vec, lr_plain, cfg.top_k_features)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    top_ai.to_csv(out / f"top_features_ai__{safe_setting_name(setting_name)}.csv", index=False)
    top_hu.to_csv(out / f"top_features_human__{safe_setting_name(setting_name)}.csv", index=False)
    """
    # LUAR embeddings
    Xtr_luar, Xte_luar = featurize_luar(
        train_texts, test_texts,
        model_name="rrivera1849/LUAR-MUD",
        batch_size=8,
        max_length=64
    )


    cal, pred, proba = fit_predict_proba_lr_calibrated(Xtr_luar, y_train, Xte_luar, cfg.seed, cfg.decision_threshold)
    m = evaluate_full(y_test, pred, proba)
    results.append({**m, "setting": setting_name, "features": "luar", "model": "LUAR_LR_calibrated",
                    "n_train": int(len(df_train)), "n_test": int(len(df_test))})
    save_predictions(df_test, pred, proba, setting_name, "LUAR_LR_calibrated", cfg.out_dir, cfg)
    """
    return results



def main():
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    if cfg.data_path.endswith(".xlsx"):
        df = pd.read_excel(cfg.data_path)
    else:
        df = pd.read_csv(cfg.data_path)

    # Expecting columns: text, source, label_name
    df["text"] = df["text"].apply(basic_clean)
    df["source"] = df["source"].astype(str).str.strip().str.lower()

    df = df[df["text"].str.len() > 0].copy()
    df = standardize_labels(df)

    # Length columns + filter
    df["n_chars"] = df["text"].str.len()
    df["n_tokens"] = df["text"].str.split().apply(len)
    df = df[(df["n_tokens"] >= cfg.min_tokens) & (df["n_tokens"] <= cfg.max_tokens)].copy()

    describe_dataset(df)

    all_results = []

    # Setting A: random split
    tr, te = make_splits_random(df, cfg.seed)
    save_split(tr, te, "random_80_20", cfg.out_dir)
    all_results += run_one_setting(tr, te, "random_80_20", cfg)

    # Settings B: hold out multiple AI generators with enough samples
    ai_sources = (
        df[df["label"] == 1]["source"]
        .value_counts()
        .loc[lambda s: s >= cfg.holdout_min_samples]
        .index
        .tolist()
    )

    for holdout_ai in ai_sources:
        tr, te = make_splits_holdout_ai_source(df, holdout_ai, cfg.seed, train_balance=cfg.train_balance)
        save_split(tr, te, f"holdout_ai_{holdout_ai}", cfg.out_dir)
        all_results += run_one_setting(tr, te, f"holdout_ai:{holdout_ai}", cfg)

    res_df = pd.DataFrame(all_results)
    res_path = Path(cfg.out_dir) / "metrics_full.csv"
    res_df.to_csv(res_path, index=False)

    print("\n=== Results (full metrics) ===")
    print(res_df.sort_values(["setting", "features", "model"]).to_string(index=False))
    print(f"\nSaved: {res_path}")
    print(f"Also saved per-setting prediction CSVs and top-feature CSVs in: {cfg.out_dir}/")


if __name__ == "__main__":
    main()
