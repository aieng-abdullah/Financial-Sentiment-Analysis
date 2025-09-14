# preprocessing.py

import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer

# === Paths ===
RAW_DATA_PATH = "data/raw/financial_phrasebank.csv"
OUTPUT_DIR = "data/preprocess"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
MAX_LENGTH = 64

def clean_text(text: str) -> str:
    """Clean text by removing URLs, hashtags, mentions, and unwanted characters."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)             
    text = re.sub(r"@[\w]+|#[\w]+", "", text)              
    text = re.sub(r"[^a-zA-Z0-9$€%.,!? ]+", " ", text)     
    return re.sub(r"\s+", " ", text).strip().lower()       
    
def preprocess():
    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    # Clean text
    df["sentence"] = df["sentence"].apply(clean_text)

    # Validate numeric labels
    valid_labels = {0, 1, 2}
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset(valid_labels):
        raise ValueError(f"Invalid labels found: {unique_labels - valid_labels}")

    # Convert labels to tensor
    labels = torch.tensor(df["label"].values, dtype=torch.long)

    # Tokenize text
    tokens = tokenizer(
        df["sentence"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    # Save tokenized inputs and labels
    torch.save(tokens["input_ids"], os.path.join(OUTPUT_DIR, "tokenized_input_ids.pt"))
    torch.save(tokens["attention_mask"], os.path.join(OUTPUT_DIR, "tokenized_attention_mask.pt"))
    torch.save(labels, os.path.join(OUTPUT_DIR, "labels.pt"))

    # Save cleaned data for reference
    df.to_csv(os.path.join(OUTPUT_DIR, "cleaned.csv"), index=False)

    print(" Preprocessing complete. Files saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    preprocess()

