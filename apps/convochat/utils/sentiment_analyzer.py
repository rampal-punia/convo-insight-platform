import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import List, Dict, Union


class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
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
            predicted_class_id = prob.argmax().item()

            sentiment = {
                "negative": prob[0].item(),
                "positive": prob[1].item(),
                # Range from -1 (very negative) to 1 (very positive)
                "overall": prob[1].item() - prob[0].item()
            }
            sentiment['category'] = self.model.config.id2label[predicted_class_id]
            sentiments.append(sentiment)

        return sentiments[0] if isinstance(texts, str) else sentiments


# Usage example
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    texts = "Yeah, that may work"
    results = analyzer.analyze_sentiment(texts)
    print(results)
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
