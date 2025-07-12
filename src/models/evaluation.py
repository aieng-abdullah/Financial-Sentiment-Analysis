import os
import torch
import logging
from transformers import AutoConfig, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_data(data_dir):
    """
    Load evaluation tensors from disk.
    Expects input_ids.pt, attention_masks.pt, labels.pt in data_dir.
    """
    for fname in ["input_ids.pt", "attention_masks.pt", "labels.pt"]:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            logger.error(f"Required data file missing: {fpath}")
            raise FileNotFoundError(f"Required data file missing: {fpath}")
    
    input_ids = torch.load(os.path.join(data_dir, "input_ids.pt"))
    attention_masks = torch.load(os.path.join(data_dir, "attention_masks.pt"))
    labels = torch.load(os.path.join(data_dir, "labels.pt"))
    logger.info(f"Loaded evaluation data from {data_dir}")
    return input_ids, attention_masks, labels

def evaluate(model, dataloader, device):
    """
    Run model inference on dataloader and compute accuracy + classification report.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    return accuracy, report

def main(model_path, data_path, save_path=None, batch_size=32):
    logger.info("Starting evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and config
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        local_files_only=True
    ).to(device)
    logger.info(f"Loaded model from {model_path}")
    
    # Load evaluation data
    input_ids, attention_masks, labels = load_data(data_path)
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    accuracy, report = evaluate(model, dataloader, device)
    
    logger.info(f"Evaluation completed on device: {device}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:\n" + report)

    # Save the model if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        logger.info(f"Saved evaluated model to {save_path}")

if __name__ == "__main__":
    MODEL_PATH = r"C:\Users\teamp\Desktop\Financial Sentiment Analysis\models"
    DATA_PATH = r"C:\Users\teamp\Desktop\Financial Sentiment Analysis\data\preprocess"
    SAVE_PATH = r"C:\Users\teamp\Desktop\Financial Sentiment Analysis\models\evaluated_model"  # Change or set to None to skip saving

    main(MODEL_PATH, DATA_PATH, save_path=SAVE_PATH)
