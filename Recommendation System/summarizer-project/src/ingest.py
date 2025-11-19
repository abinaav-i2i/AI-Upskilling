# src/ingest.py
import os
import pandas as pd
from embedder import Embedder
from src.qdrant_utils import get_client, create_collection, upsert_points
import argparse

DATA_DEFAULT = "data/AmesHousing.csv"

def preprocess_ames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a textual 'text' field useful for embedding from AmesHousing columns.
    Picks relevant columns and builds a short natural-language description per row.
    """
    # columns we want to include in textual description if available
    cols = [
        "Neighborhood", "MS Zoning", "Bldg Type", "House Style", "Overall Qual",
        "Overall Cond", "Year Built", "Year Remod/Add", "Gr Liv Area",
        "Full Bath", "Half Bath", "Bedroom AbvGr", "Kitchen Qual", "TotRms AbvGrd",
        "Garage Type", "Garage Cars", "Garage Area", "Pool Area", "SaleType", "SaleCondition"
    ]
    present = [c for c in cols if c in df.columns]
    # build a short sentence-style description
    def make_text(row):
        parts = []
        for c in present:
            val = row.get(c)
            if pd.isna(val) or val=="":
                continue
            parts.append(f"{c}: {val}")
        # include SalePrice as an approximate price indicator if present
        if "SalePrice" in row and not pd.isna(row["SalePrice"]):
            parts.append(f"SalePrice: {int(row['SalePrice'])}")
        return ". ".join(parts)

    df = df.reset_index(drop=True)
    df["text"] = df.apply(make_text, axis=1)
    return df

def preprocess_generic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generic fallback: combine the two largest string columns.
    """
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not text_cols:
        df['text'] = df.apply(lambda row: ' '.join(str(v) for v in row.values), axis=1)
        return df
    # choose up to first 3 string columns
    chosen = text_cols[:3]
    df['text'] = df[chosen].fillna('').agg(' '.join, axis=1)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # detect AmesHousing by presence of 'SalePrice' and many known columns
    ames_indicators = {"SalePrice", "Gr Liv Area", "Neighborhood", "MS Zoning"}
    if ames_indicators.issubset(set(df.columns)):
        return preprocess_ames(df)
    else:
        return preprocess_generic(df)

def run(csv_path: str = DATA_DEFAULT, model_name: str = None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    # read csv robustly (handles common encodings)
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1")
    print(f"Loaded {len(df)} rows from {csv_path}")

    df = preprocess(df)
    texts = df['text'].astype(str).tolist()
    emb = Embedder(model_name) if model_name else Embedder()
    vectors = emb.embed_texts(texts)
    client = get_client()
    create_collection(client, vector_size=vectors.shape[1])
    ids = [int(i) + 1 for i in df.index.tolist()]

    # Prepare payload: keep a minimal set of useful fields for AmesHousing
    keep = []
    for k in ["SalePrice", "Gr Liv Area", "Neighborhood", "Year Built", "MS Zoning", "PID"]:
        if k in df.columns:
            keep.append(k)
    payloads = []
    for _, row in df.iterrows():
        p = {}
        for k in keep:
            if k in row:
                val = row[k]
                if pd.isna(val):
                    val = None
                p[k] = val
        # always store original text for summarization
        p['text'] = row['text']
        payloads.append(p)

    upsert_points(client, ids, vectors, payloads)
    print(f"Ingested {len(ids)} items into Qdrant collection 'listings'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=DATA_DEFAULT, help="Path to dataset CSV")
    parser.add_argument("--model", type=str, default=None, help="SentenceTransformer model")
    args = parser.parse_args()
    run(csv_path=args.csv, model_name=args.model)
