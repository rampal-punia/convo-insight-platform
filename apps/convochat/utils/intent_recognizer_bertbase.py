import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from typing import List, Dict
from torch.utils.data import Dataset


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length', max_length=128)
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


class IntentRecognizer:
    def __init__(self, model_name: str = "distilbert-base-uncased", intent_labels=None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.intent_labels = intent_labels or [
            "question", "complaint", "request", "feedback", "greeting", "other"]
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=len(self.intent_labels)).to(self.device)

    def fine_tune(self, texts: List[str], labels: List[int], epochs: int = 3):
        dataset = IntentDataset(texts, labels, self.tokenizer)
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

    def recognize_intent(self, texts: List[str]) -> List[Dict[str, float]]:
        results = []
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            intent_scores = {intent: score.item()
                             for intent, score in zip(self.intent_labels, probs[0])}
            predicted_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[predicted_intent]

            results.append({
                "text": text,
                "predicted_intent": predicted_intent,
                "confidence": confidence,
                "all_intents": intent_scores
            })

        return results


# Usage example
if __name__ == "__main__":
    recognizer = IntentRecognizer()

    # Fine-tuning data (you would need a larger dataset in practice)
    train_texts = [
        "What are your business hours?",
        "I'm very unhappy with the service I received.",
        "Can you help me reset my password?",
        "I think your product is great!",
        "Hello, how are you doing today?",
        "I'd like to make a suggestion about your product."
    ]
    train_labels = [0, 1, 2, 3, 4, 5]  # Corresponding to the intent_labels

    # Fine-tune the model
    # recognizer.fine_tune(train_texts, train_labels)

    # Test the model
    test_texts = [
        "When do you close?",
        "This product is terrible!",
        "How do I change my account settings?",
        "Your service is excellent!",
        "Hi there!",
        "I have an idea for improving your app."
    ]
    results = recognizer.recognize_intent(test_texts)
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Predicted Intent: {result['predicted_intent']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("All Intents:")
        for intent, score in result['all_intents'].items():
            print(f"  {intent}: {score:.2f}")
        print()
