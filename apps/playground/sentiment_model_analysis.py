import torch
import json
from typing import Dict, List, Union
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
import contractions
import emoji
import re
from pathlib import Path
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import spacy
nlp = spacy.load('en_core_web_sm')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    # Download required NLTK resources
    nltk.download('punkt_tab')

"""
6 Granular_Sentiment in the dataset are:

joy	        5822
sadness	    4866
anger	    2244
fear	    2076
love	    1350
surprise	605
"""


class SentimentPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    def clean_text(self, text):
        """Enhanced text cleaning"""
        # Convert to lowercase
        text = text.lower()

        # Handle emojis
        text = emoji.demojize(text)

        # Expand contractions
        text = contractions.fix(text)

        # Handle special characters while keeping emoticons
        text = re.sub(r'[^\w\s:;)(><\/\\]', ' ', text)

        # Handle repeated characters (e.g., 'happppy' -> 'happy')
        text = re.sub(r'(.)\1+', r'\1\1', text)

        # Handle spacing
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def preprocess_text(self, text):
        """Complete preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)

        # Tokenize
        tokens = word_tokenize(text)

        # Lemmatize while preserving important short words
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return tokens


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.roberta = AutoModel.from_pretrained('roberta-base')
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.roberta.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop(pooled_output)
        return self.fc(output)


def train_model(model, train_loader, val_loader, device, epochs=5):
    """Modified train_model function that returns the trained model and training info"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_model_state = None
    training_info = {}

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        training_info[f'epoch_{epoch+1}'] = {
            'train_loss': total_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'classification_report': classification_report(true_labels, predictions)
        }

        print(f'Epoch {epoch + 1}:')
        print(f'Average training loss: {total_loss / len(train_loader)}')
        print(f'Validation loss: {val_loss / len(val_loader)}')
        print('\nClassification Report:')
        print(classification_report(true_labels, predictions))

    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, training_info


def save_trained_model(model, tokenizer, label_encoder, training_info, save_dir="sentiment_model"):
    """Function to save all model components"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Save model state
    torch.save(model.state_dict(), save_dir / "model_state.pt")

    # Save tokenizer
    tokenizer.save_pretrained(save_dir / "tokenizer")

    # Save label encoder classes
    pd.Series(label_encoder.classes_).to_json(
        save_dir / "label_encoder_classes.json")

    # Save config and training info
    config = {
        "model_type": "roberta-base",
        "max_length": 128,
        "num_classes": len(label_encoder.classes_),
        "version": "1.0",
        "training_info": training_info
    }

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"Model and components saved to {save_dir}")


def main():
    # Load and preprocess data
    df = pd.read_csv('sentimentanalysismulticlass.csv')
    preprocessor = SentimentPreprocessor()

    # Preprocess texts
    df['processed_text'] = df['Text'].apply(preprocessor.preprocess_text)

    # Prepare labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['Granular_Sentiment'])

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Text'].values, labels, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = SentimentDataset(
        train_texts, train_labels, preprocessor.tokenizer)
    val_dataset = SentimentDataset(
        val_texts, val_labels, preprocessor.tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentClassifier(n_classes=len(label_encoder.classes_))
    model.to(device)

    # Train model
    trained_model, training_info = train_model(
        model, train_loader, val_loader, device)

    # Save model and components
    save_trained_model(
        model=trained_model,
        tokenizer=preprocessor.tokenizer,
        label_encoder=label_encoder,
        training_info=training_info
    )

    print("Training completed and model saved successfully!")
    return trained_model, preprocessor.tokenizer, label_encoder, training_info


# if __name__ == '__main__':
#     trained_model, tokenizer, label_encoder, training_info = main()


class SentimentModelManager:
    def __init__(self, model_dir: str = "sentiment_tr_model"):
        """Initialize the model manager with a directory for model artifacts."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.config = None

    def save_model(self, model, tokenizer, label_encoder, config: Dict = None) -> None:
        """Save all model components and configuration."""
        # Save model state
        torch.save(model.state_dict(),
                   self.model_dir / "sentiment_model_6cls.pt")

        # Save tokenizer
        tokenizer.save_pretrained(self.model_dir / "tokenizer")

        # Save label encoder
        pd.Series(label_encoder.classes_).to_json(
            self.model_dir / "label_encoder_classes.json")

        # Save config
        if config is None:
            config = {
                "model_type": "roberta-base",
                "max_length": 128,
                "num_classes": len(label_encoder.classes_),
                "version": "1.0"
            }

        with open(self.model_dir / "config.json", "w") as f:
            json.dump(config, f)

        print(f"Model and components saved to {self.model_dir}")

    def load_model(self) -> None:
        """Load all model components."""
        # Load config
        with open(self.model_dir / "config.json", "r") as f:
            self.config = json.load(f)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir / "tokenizer")

        # Load label encoder
        classes = pd.read_json(
            self.model_dir / "label_encoder_classes.json",
            typ="series"
        ).values
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = classes

        # Initialize and load model
        self.model = SentimentClassifier(self.config["num_classes"])
        self.model.load_state_dict(
            torch.load(self.model_dir / "sentiment_model_6cls.pt",
                       map_location=torch.device('cpu'))
        )
        self.model.eval()

        print("Model and components loaded successfully")

    def predict(self,
                texts: Union[str, List[str]],
                batch_size: int = 16) -> Dict[str, np.ndarray]:
        """
        Run predictions on a single text or list of texts.

        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing multiple texts

        Returns:
            Dictionary containing predictions and confidence scores
        """
        if isinstance(texts, str):
            texts = [texts]

        # Prepare dataset
        dataset = SentimentDataset(
            texts=texts,
            labels=[0] * len(texts),  # Dummy labels
            tokenizer=self.tokenizer,
            max_length=self.config["max_length"]
        )

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Get predictions
        predictions = []
        confidence_scores = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)

                confidence, preds = torch.max(probs, dim=1)

                predictions.extend(preds.cpu().numpy())
                confidence_scores.extend(confidence.cpu().numpy())

        # Convert numeric predictions to labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)

        return {
            'texts': texts,
            'predictions': predicted_labels,
            'confidence': np.array(confidence_scores),
            'prediction_ids': np.array(predictions)
        }


def format_predictions(prediction_results: Dict) -> pd.DataFrame:
    """Format prediction results into a readable DataFrame."""
    return pd.DataFrame({
        'text': prediction_results['texts'],
        'predicted_sentiment': prediction_results['predictions'],
        'confidence': prediction_results['confidence'].round(3)
    })


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = SentimentModelManager()

    # Example of saving model (after training)
    """
    manager.save_model(
        model=trained_model,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        config={
            "model_type": "roberta-base",
            "max_length": 128,
            "num_classes": 6,
            "version": "1.0"
        }
    )
    """

    # Load model
    manager.load_model()

    # Example texts for prediction
    test_texts = [
        "I'm feeling really happy today!",
        "This makes me so angry and frustrated.",
        "I'm worried about the upcoming exam.",
        "The sunset was absolutely beautiful.",
        "I'm feeling quite sad and lonely."
    ]

    # Run predictions
    results = manager.predict(test_texts)

    # Format and display results
    df_results = format_predictions(results)
    print("\nPrediction Results:")
    print(df_results)

    # Example of running prediction on a single text
    single_result = manager.predict("I'm really excited about this!")
    print("\nSingle Text Prediction:")
    print(format_predictions(single_result))
