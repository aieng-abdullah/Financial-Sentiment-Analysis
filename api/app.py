import os
import torch
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# ------------------------ Logging Setup ------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#  FastAPI App Init 
app = FastAPI(title="Financial Sentiment API")


class InputTexts(BaseModel):
    texts: list[str]

#  Model Paths 
MODEL_PATH = r"C:\Users\teamp\Desktop\Financial Sentiment Analysis\output"
TOKENIZER_PATH = r"C:\Users\teamp\Desktop\Financial Sentiment Analysis\output"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


try:
    logger.info(f"Loading config from: {MODEL_PATH}")
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)

    logger.info(f"Loading model from: {MODEL_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        config=config,
        local_files_only=True
    ).to(device)

    logger.info(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load model/tokenizer: {e}", exc_info=True)
    raise RuntimeError("Startup failed due to model/tokenizer loading error.")

#  Prediction Endpoint 
@app.post("/predict")
async def predict_sentiment(data: InputTexts):
    """
    Predict sentiment labels for a list of financial news/texts.
    """
    try:
        model.eval()
        with torch.no_grad():
            # Tokenize input texts
            inputs = tokenizer(
                data.texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

        # Customize label mapping if different in your case
        label_map = {0: "positive", 1: "negative", 2: "neutral"}
        results = [{"text": text, "sentiment": label_map.get(pred, f"Label {pred}")} for text, pred in zip(data.texts, preds)]

        logger.info(f"Prediction completed for {len(data.texts)} texts.")
        return {"results": results}

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Model inference failed.")

# ------------------------ To Run ------------------------ #
# uvicorn app:app --host 0.0.0.0 --port 8000
