import os
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

def load_model_and_tokenizer(model_path: str, device: torch.device):
    """
    Load a transformer model and tokenizer from a local directory.
    If tokenizer not found locally, fall back to default BERT tokenizer.
    """
    config_path = os.path.join(model_path, "config.json")
    print(f"[INFO] Looking for config at: {os.path.abspath(config_path)}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] Missing config.json at {config_path}")

    # Load model config
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    print("[INFO] Model config loaded.")

    # Load model weights
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        local_files_only=True
    ).to(device)
    print("[INFO] Model loaded and moved to device.")

    # Try loading tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print("[INFO] Tokenizer loaded from local model path.")
    except Exception as e:
        print(f"[WARN] Tokenizer not found in {model_path}, falling back to 'bert-base-uncased'")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer


def predict_sentiment(texts, model, tokenizer, device, max_length=128):
    """
    Predict sentiment from a list of texts using the loaded model.
    """
    model.eval()
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    return predictions.cpu().tolist()


def main():
    model_path = r"C:\Users\teamp\Desktop\Financial Sentiment Analysis\models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    # Example input texts
    texts = [
        "Stocks soared after the earnings report beat expectations.",
        "The market is crashing amid recession fears.",
        "The CEO's statement had little effect on the stock price."
    ]

    # Predict
    predictions = predict_sentiment(texts, model, tokenizer, device)

    # Label mapping (as per config.json)
    label_map = {0: "positive", 1: "negative", 2: "neutral"}

    for text, pred in zip(texts, predictions):
        sentiment = label_map.get(pred, f"Label {pred}")
        print(f"[RESULT] Text: {text}\n→ Sentiment: {sentiment}\n")


if __name__ == "__main__":
    main()
