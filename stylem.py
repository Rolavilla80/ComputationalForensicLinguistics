import re
from pathlib import Path
import pandas as pd
import numpy as np

# CONFIG
DATA_PATH = "Dataset.xlsx"     
RESULTS_DIR = Path("results")  
OUT_DIR = RESULTS_DIR / "feature_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_TO_ANALYZE = sorted({
    p.name.split("__", 1)[1].replace(".csv", "")
    for p in RESULTS_DIR.glob("top_features_ai__*.csv")
})

TOPN = 30  


# HELPERS
def load_top_features(setting: str) -> pd.DataFrame:
    ai_path = RESULTS_DIR / f"top_features_ai__{setting}.csv"
    hu_path = RESULTS_DIR / f"top_features_human__{setting}.csv"
    if not ai_path.exists():
        raise FileNotFoundError(f"Missing: {ai_path}")
    if not hu_path.exists():
        raise FileNotFoundError(f"Missing: {hu_path}")

    ai = pd.read_csv(ai_path).head(TOPN).copy()
    hu = pd.read_csv(hu_path).head(TOPN).copy()

    ai["direction"] = "AI"
    hu["direction"] = "HUMAN"

    df = pd.concat([ai, hu], ignore_index=True)
    df["setting"] = setting
    return df


def bucket_feature(feat: str) -> str:
    """
    Heuristic buckets for char n-grams.
    """
    f = str(feat)

    if any(x in f.lower() for x in ["http", "www", ".com", ".org", "html", "href"]):
        return "URL/markup"

    if re.search(r"\d", f) or any(x in f for x in ["[", "]", "(", ")", "{", "}"]):
        return "Digits/brackets"

    if any(x in f for x in ['"', "“", "”", "‘", "’"]):
        return "Quotation style"

    if any(x in f for x in ["n't", "n’t", "'s", "’s", "’t", "'t"]):
        return "Contractions/apostrophes"

    if "’" in f or "'" in f:
        return "Contractions/apostrophes"

    if re.search(r"\s", f) and re.search(r"[.,;:!?]", f):
        return "Punctuation+spacing"

    if re.fullmatch(r"[^\w\s]+", f):
        return "Punctuation-only"

    if any(x in f.lower() for x in [" the", " and", " to", " of", " in", " for", " that", " with"]):
        return "Function-word fragments"

    return "Other/unknown"


def safe_contains(series: pd.Series, substring: str) -> pd.Series:
    pattern = re.escape(substring)
    return series.str.contains(pattern, regex=True, na=False)


def provenance_counts(df_data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        present = safe_contains(df_data["text"], feat)

        by_source = df_data.assign(present=present).groupby("source")["present"].mean()
        by_label = df_data.assign(present=present).groupby("label")["present"].mean()

        hu_rate = float(by_label.get(0, np.nan))
        ai_rate = float(by_label.get(1, np.nan))

        top_sources = by_source.sort_values(ascending=False).head(3)
        top_sources_str = "; ".join([f"{idx}:{val:.3f}" for idx, val in top_sources.items()])

        rows.append({
            "feature": feat,
            "presence_human": hu_rate,
            "presence_ai": ai_rate,
            "delta_ai_minus_human": ai_rate - hu_rate,
            "top_sources_presence": top_sources_str,
        })

    return pd.DataFrame(rows).sort_values("delta_ai_minus_human", ascending=False)


def collect_snippets(df_data: pd.DataFrame, feature: str, n: int = 2, window: int = 45):
    pattern = re.escape(feature)
    out = []
    for _, row in df_data.iterrows():
        text = str(row["text"])
        m = re.search(pattern, text)
        if not m:
            continue
        start = max(0, m.start() - window)
        end = min(len(text), m.end() + window)
        snippet = text[start:end].replace("\n", " ")
        out.append({
            "label": row["label_name"],
            "source": row["source"],
            "snippet": snippet
        })
        if len(out) >= n:
            break
    return out


def build_feature_evidence(df_data: pd.DataFrame, features: list[str], n_each: int = 2) -> pd.DataFrame:
    rows = []
    human_df = df_data[df_data["label"] == 0]
    ai_df = df_data[df_data["label"] == 1]

    for feat in features:
        h_snips = collect_snippets(human_df, feat, n=n_each)
        a_snips = collect_snippets(ai_df, feat, n=n_each)

        rows.append({
            "feature": feat,
            "human_snippets": " || ".join([f'{x["source"]}: {x["snippet"]}' for x in h_snips]),
            "ai_snippets": " || ".join([f'{x["source"]}: {x["snippet"]}' for x in a_snips]),
        })

    return pd.DataFrame(rows)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


#main
def main():
    # Load dataset for provenance + snippets
    df = pd.read_excel(DATA_PATH)
    df["text"] = df["text"].astype(str)
    df["source"] = df["source"].astype(str).str.strip().str.lower()
    df["label_name"] = df["label_name"].astype(str).str.strip().str.lower()

    # label: 1=AI, 0=Human
    def map_label(x: str) -> int:
        return 0 if "human" in x else 1

    df["label"] = df["label_name"].map(map_label)

    # Load top features across settings
    all_top = []
    for setting in SETTINGS_TO_ANALYZE:
        all_top.append(load_top_features(setting))
    all_top = pd.concat(all_top, ignore_index=True)

    # Buckets
    all_top["bucket"] = all_top["feature"].map(bucket_feature)
    all_top.to_csv(OUT_DIR / "top_features_with_buckets.csv", index=False)

    # Bucket summary per setting & direction
    bucket_summary = (
        all_top.groupby(["setting", "direction", "bucket"])
        .size()
        .reset_index(name="count")
        .sort_values(["setting", "direction", "count"], ascending=[True, True, False])
    )
    bucket_summary.to_csv(OUT_DIR / "bucket_summary.csv", index=False)

    # Provenance
    unique_feats = sorted(all_top["feature"].unique().tolist())
    prov = provenance_counts(df, unique_feats)
    prov.to_csv(OUT_DIR / "feature_provenance.csv", index=False)

    # Merge provenance back
    merged = all_top.merge(prov, on="feature", how="left")
    merged.to_csv(OUT_DIR / "top_features_with_buckets_and_provenance.csv", index=False)

    # NEW: Snippet evidence for top features (by strongest AI-minus-human delta)
    top_feats = (
        merged.sort_values("delta_ai_minus_human", ascending=False)
              .drop_duplicates("feature")
              .head(10)["feature"]
              .tolist()
    )
    evidence = build_feature_evidence(df, top_feats, n_each=2)
    evidence.to_csv(OUT_DIR / "feature_snippet_evidence.csv", index=False)

    # Stability / overlap across settings
    overlap_rows = []
    for direction in ["AI", "HUMAN"]:
        sets = {}
        for setting in SETTINGS_TO_ANALYZE:
            feats = set(
                all_top[(all_top["setting"] == setting) & (all_top["direction"] == direction)]["feature"].tolist()
            )
            sets[setting] = feats

        settings = list(sets.keys())
        for i in range(len(settings)):
            for j in range(i + 1, len(settings)):
                s1, s2 = settings[i], settings[j]
                overlap_rows.append({
                    "direction": direction,
                    "setting_1": s1,
                    "setting_2": s2,
                    "jaccard_overlap": jaccard(sets[s1], sets[s2]),
                    "intersection_size": len(sets[s1] & sets[s2]),
                })

    overlap = pd.DataFrame(overlap_rows).sort_values(["direction", "jaccard_overlap"], ascending=[True, False])
    overlap.to_csv(OUT_DIR / "feature_overlap_jaccard.csv", index=False)

    print("Saved to:", OUT_DIR.resolve())
    print("Files created:")
    for p in sorted(OUT_DIR.glob("*.csv")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
