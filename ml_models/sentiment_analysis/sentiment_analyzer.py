# ml_model/sentiment_analysis/sentiment_analyzer.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict


class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name).to(self.device)

    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, float]]:
        encoded_inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt")
        encoded_inputs = {k: v.to(self.device)
                          for k, v in encoded_inputs.items()}

        with torch.no_grad():
            outputs = self.model(**encoded_inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiments = []

        for prob in probabilities:
            sentiment = {
                "negative": prob[0].item(),
                "positive": prob[1].item(),
                # Range from -1 (very negative) to 1 (very positive)
                "overall": prob[1].item() - prob[0].item()
            }
            sentiments.append(sentiment)

        return sentiments


# Usage example
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    texts = [
        "I love this product! It's amazing!",
        "This is terrible, I'm very disappointed.",
        "It's okay, not great but not bad either."
    ]
    results = analyzer.analyze_sentiment(texts)
    for text, sentiment in zip(texts, results):
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print()
