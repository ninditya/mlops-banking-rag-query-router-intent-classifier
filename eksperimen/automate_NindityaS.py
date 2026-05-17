"""
Automate preprocessing untuk Banking77 Intent Classification.
Mengkonversi langkah-langkah dari notebook eksperimen ke pipeline otomatis.

Usage:
    python automate_NamaKamu.py --input banking77_raw --output banking77_preprocessing
"""

import argparse
import logging
import os
import re
import pickle
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(source: str = "huggingface") -> pd.DataFrame:
    """Load Banking77 dataset dari HuggingFace atau file lokal."""
    if source == "huggingface":
        logger.info("Loading Banking77 from HuggingFace...")
        ds = load_dataset("PolyAI/banking77")
        train_df = pd.DataFrame(ds["train"])
        test_df = pd.DataFrame(ds["test"])
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        df = pd.read_csv(source)
    logger.info(f"Loaded {len(df)} samples, {df['label'].nunique()} intents")
    return df


def clean_text(text: str) -> str:
    """Text normalization: lowercase, remove special chars, strip whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\?\!\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Pipeline preprocessing lengkap:
    1. Text cleaning
    2. Label encoding
    3. TF-IDF vectorization (1-2 ngrams)
    4. Stratified train-test split
    """
    logger.info("Cleaning text...")
    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)

    logger.info("Encoding labels...")
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label"])

    logger.info("Splitting data (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"],
        df["label_encoded"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_encoded"],
    )

    logger.info("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    logger.info(f"TF-IDF shape: train={X_train_vec.shape}, test={X_test_vec.shape}")

    return {
        "X_train": X_train_vec,
        "X_test": X_test_vec,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_text": X_train,
        "X_test_text": X_test,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
    }


def save_artifacts(artifacts: dict, output_dir: str):
    """Simpan hasil preprocessing untuk dipakai di stage modelling."""
    os.makedirs(output_dir, exist_ok=True)

    # Simpan text + label sebagai CSV (gampang dibaca, version control friendly)
    pd.DataFrame({"text": artifacts["X_train_text"], "label": artifacts["y_train"]}).to_csv(
        os.path.join(output_dir, "train.csv"), index=False
    )
    pd.DataFrame({"text": artifacts["X_test_text"], "label": artifacts["y_test"]}).to_csv(
        os.path.join(output_dir, "test.csv"), index=False
    )

    # Simpan vectorizer + label encoder sebagai pickle
    with open(os.path.join(output_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(artifacts["vectorizer"], f)
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(artifacts["label_encoder"], f)

    logger.info(f"Artifacts saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="huggingface", help="Data source")
    parser.add_argument("--output", default="banking77_preprocessing", help="Output dir")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_data(args.input)
    artifacts = preprocess_data(df, test_size=args.test_size)
    save_artifacts(artifacts, args.output)
    logger.info("✅ Preprocessing complete")


if __name__ == "__main__":
    main()
