import torch
from transformers import AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import os
import logging

# --- Configure Logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh = logging.FileHandler('finetuning.log')
fh.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)


class Finetuner:
    def __init__(self,
                 data_dir=r"data/preprocess",
                 model_name="model.safetensors",
                 num_labels=3,
                 batch_size=16,
                 epochs=3,
                 learning_rate=2e-5,
                 device=None):
        self.DATA_DIR = data_dir
        self.MODEL_NAME = model_name
        self.NUM_LABELS = num_labels
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate
        self.DEVICE = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        logger.info(f"Using device: {self.DEVICE}")

    def load_preprocessed_tensors(self):
        logger.info(f"Loading preprocessed tensors from: {os.path.abspath(self.DATA_DIR)}...")
        try:
            input_ids = torch.load(os.path.join(self.DATA_DIR, "tokenized_input_ids.pt"))
            attention_mask = torch.load(os.path.join(self.DATA_DIR, "tokenized_attention_mask.pt"))
            labels = torch.load(os.path.join(self.DATA_DIR, "labels.pt"))
        except FileNotFoundError as e:
            logger.error(
                f"Error loading tensor file: {e}. "
                f"Ensure 'tokenized_input_ids.pt', 'tokenized_attention_mask.pt', and 'labels.pt' "
                f"are located in '{os.path.abspath(self.DATA_DIR)}'."
            )
            raise

        logger.info(f"Unique labels loaded: {torch.unique(labels)}")
        if not (labels.min() >= 0 and labels.max() < self.NUM_LABELS):
            logger.error(
                f"Labels out of expected range: {torch.unique(labels)}. Expected [0, {self.NUM_LABELS - 1}]."
            )
            raise AssertionError("Labels are out of the expected range.")
        logger.info("Preprocessed tensors loaded and labels validated successfully.")
        return input_ids, attention_mask, labels

    def create_dataloaders(self, input_ids, attention_mask, labels, train_ratio=0.8):
        logger.info("Creating DataLoaders...")
        dataset = TensorDataset(input_ids, attention_mask, labels)

        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE)
        logger.info(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
        logger.info("DataLoaders created successfully.")
        return train_loader, val_loader

    def setup_model_optimizer_scheduler(self, train_loader):
        logger.info("Loading model, optimizer, and scheduler...")
        model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=self.NUM_LABELS)
        model.to(self.DEVICE)

        optimizer = AdamW(model.parameters(), lr=self.LEARNING_RATE)

        num_training_steps = self.EPOCHS * len(train_loader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        logger.info("Model, optimizer, and scheduler initialized successfully.")
        return model, optimizer, lr_scheduler

    @staticmethod
    def compute_metrics(logits, labels):
        preds = logits.argmax(dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
        return acc, f1

    def train_and_validate(self, model, train_loader, val_loader, optimizer, lr_scheduler):
        logger.info("Starting training and validation loop...")
        for epoch in range(self.EPOCHS):
            model.train()
            total_loss = 0
            for batch_idx, (input_ids_batch, attention_mask_batch, labels_batch) in enumerate(
                    tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")):
                input_ids_batch = input_ids_batch.to(self.DEVICE)
                attention_mask_batch = attention_mask_batch.to(self.DEVICE)
                labels_batch = labels_batch.to(self.DEVICE)

                outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if (batch_idx + 1) % 100 == 0:
                    logger.debug(f"Epoch {epoch + 1}, Batch {batch_idx + 1} - Training Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} - Average Training Loss: {avg_loss:.4f}")

            model.eval()
            all_logits = []
            all_labels = []
            with torch.no_grad():
                for input_ids_batch, attention_mask_batch, labels_batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
                    input_ids_batch = input_ids_batch.to(self.DEVICE)
                    attention_mask_batch = attention_mask_batch.to(self.DEVICE)
                    labels_batch = labels_batch.to(self.DEVICE)

                    outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
                    all_logits.append(outputs.logits)
                    all_labels.append(labels_batch)

            logits = torch.cat(all_logits)
            labels_all = torch.cat(all_labels)
            val_acc, val_f1 = self.compute_metrics(logits, labels_all)

            logger.info(f"Epoch {epoch + 1} | Validation Accuracy: {val_acc:.4f} | Validation F1-Score: {val_f1:.4f}")

        logger.info("Training and validation complete.")


if __name__ == "__main__":
    finetuner = Finetuner()
    logger.info(f"Starting fine-tuning script. Current device: {finetuner.DEVICE}")
    logger.info(f"Looking for data files in: {os.path.abspath(finetuner.DATA_DIR)}")

    try:
        input_ids, attention_mask, labels = finetuner.load_preprocessed_tensors()
        train_loader, val_loader = finetuner.create_dataloaders(input_ids, attention_mask, labels)
        model, optimizer, lr_scheduler = finetuner.setup_model_optimizer_scheduler(train_loader)
        finetuner.train_and_validate(model, train_loader, val_loader, optimizer, lr_scheduler)

        # Optionally save model
        model_save_path = r"C:\Users\teamp\Desktop\Financial Sentiment Analysis\output"
        model.save_pretrained(model_save_path)
        logger.info(f"Fine-tuned model saved to {model_save_path}")

    except Exception as e:
        logger.critical(f"An unhandled error occurred during the fine-tuning process: {e}", exc_info=True)
