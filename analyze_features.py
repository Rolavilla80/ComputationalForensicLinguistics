import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

OUT_DIR = Path("results")
DATA_PATH = "Dataset.xlsx"          
SETTING = "random_80_20"            
TOPK = 30

def safe_setting_name(name: str) -> str:
    return name.replace(":", "__").replace("/", "_").replace("\\", "_").replace(" ", "_")

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def bucket_feature(f: str) -> str:
    """Rule-based buckets for char n-grams."""
    if re.search(r"http|www\.|\.com|\.org|@", f):
        return "markup/url"
    if re.search(r"\[\d+\]|\(\d{4}\)|et al", f):
        return "citation"
    if re.search(r"\d", f):
        return "digits"
    if "’" in f or "‘" in f:
        return "curly_apostrophe/quotes"
    if "'" in f or '"' in f:
        return "straight_quotes/apostrophe"
    if re.search(r"n't|n’t|’t| 's|’s", f):
        return "contractions"
    if re.search(r"[.,;:!?]", f):
        return "punctuation"
    if re.search(r"\s{2,}|\n", f):
        return "whitespace/newlines"
    if re.search(r"\b(the|and|to|of|in|for)\b", f):
        return "function_word_fragment"
    return "other"

def feature_presence_by_source(df, vectorizer, features, text_col="text", source_col="source"):
    X = vectorizer.transform(df[text_col].tolist())
    vocab = vectorizer.vocabulary_

    ok = [f for f in features if f in vocab]
    if not ok:
        raise ValueError("None of the selected features exist in TF-IDF vocab.")

    idx = [vocab[f] for f in ok]
    Xsub = X[:, idx]
    pres = (Xsub > 0).astype(int)

    rows = []
    for src, g in df.groupby(source_col):
        r = g.index.to_numpy()
        frac = np.asarray(pres[r].mean(axis=0)).ravel()
        rows.append(pd.Series(frac, index=ok, name=src))
    out = pd.DataFrame(rows).reset_index(names=["source"])
    return out

def find_snippets(text: str, feat: str, window: int = 60, max_snips: int = 2):
    """Return up to max_snips snippets showing feat in context."""
    out = []
    if not isinstance(text, str) or not feat:
        return out
    start = 0
    while len(out) < max_snips:
        i = text.find(feat, start)
        if i == -1:
            break
        a = max(0, i - window)
        b = min(len(text), i + len(feat) + window)
        snippet = text[a:b].replace("\n", " ")
        out.append(snippet)
        start = i + len(feat)
    return out

# ------ Load dataset ----------
df = pd.read_excel(DATA_PATH)
df["text"] = df["text"].apply(basic_clean)
df["source"] = df["source"].astype(str).str.strip().str.lower()
df = df[df["text"].str.len() > 0].copy()
# ---------Load features ----------
ai_path = OUT_DIR / f"top_features_ai__{safe_setting_name(SETTING)}.csv"
hu_path = OUT_DIR / f"top_features_human__{safe_setting_name(SETTING)}.csv"
top_ai = pd.read_csv(ai_path).head(TOPK)
top_hu = pd.read_csv(hu_path).head(TOPK)

top_ai["direction"] = "AI"
top_hu["direction"] = "Human"
feat_df = pd.concat([top_ai, top_hu], ignore_index=True)

feat_df["bucket"] = feat_df["feature"].apply(bucket_feature)
feat_df.sort_values(["direction", "weight"], ascending=[True, False], inplace=True)

bucketed_path = OUT_DIR / f"feature_buckets__{safe_setting_name(SETTING)}.csv"
feat_df.to_csv(bucketed_path, index=False)
print("Saved:", bucketed_path)


vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    max_features=200000,
)
vec.fit(df["text"].tolist())

features = feat_df["feature"].tolist()
pres_df = feature_presence_by_source(df, vec, features)
pres_path = OUT_DIR / f"feature_presence_by_source__{safe_setting_name(SETTING)}.csv"
pres_df.to_csv(pres_path, index=False)
print("Saved:", pres_path)

rows = []
for _, row in feat_df.iterrows():
    f = row["feature"]
    mask = df["text"].str.contains(re.escape(f), regex=True)
    hits = df[mask].head(6)
    for _, r in hits.iterrows():
        snips = find_snippets(r["text"], f, window=70, max_snips=2)
        for s in snips:
            rows.append({
                "feature": f,
                "weight": row["weight"],
                "direction": row["direction"],
                "bucket": row["bucket"],
                "source": r["source"],
                "snippet": s,
            })

snip_df = pd.DataFrame(rows)
snip_path = OUT_DIR / f"feature_snippets__{safe_setting_name(SETTING)}.csv"
snip_df.to_csv(snip_path, index=False)
print("Saved:", snip_path)
