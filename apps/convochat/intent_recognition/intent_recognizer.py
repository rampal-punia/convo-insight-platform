# apps/convochat/intent_recognition/intent_recognizer.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict


class IntentRecognizer:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name).to(self.device)
        self.intent_labels = ["question", "complaint",
                              "request", "feedback", "greeting", "other"]

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
    texts = [
        "What are your business hours?",
        "I'm very unhappy with the service I received.",
        "Can you help me reset my password?",
        "I think your product is great!",
        "Hello, how are you doing today?"
    ]
    results = recognizer.recognize_intent(texts)
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Predicted Intent: {result['predicted_intent']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("All Intents:")
        for intent, score in result['all_intents'].items():
            print(f"  {intent}: {score:.2f}")
        print()
