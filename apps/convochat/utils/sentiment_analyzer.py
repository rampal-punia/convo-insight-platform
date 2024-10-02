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

    def analyze_sentiment(self, texts: Union[str, List[str]]) -> Union[Dict[str, float], List[Dict[str, float]]]:
        if isinstance(texts, str):
            texts = [texts]
        sentiments = []
        for text in texts:
            encoded_inputs = self.tokenizer(
                text, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                probabilities = self.model(**encoded_inputs).logits

            for prob in probabilities:
                predicted_class_id = prob.argmax().item()
                sentiments.append(
                    self.model.config.id2label[predicted_class_id])

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
