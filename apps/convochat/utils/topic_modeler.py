# apps/convochat/topic_modeling/topic_modeler.py

from bertopic import BERTopic
from typing import List, Dict
import numpy as np


class TopicModeler:
    def __init__(self):
        self.model = BERTopic(language="english",
                              calculate_probabilities=True, verbose=True)

    def fit_transform(self, documents: List[str]) -> Dict[str, any]:
        topics, probs = self.model.fit_transform(documents)
        return self._process_results(documents, topics, probs)

    def transform(self, documents: List[str]) -> Dict[str, any]:
        topics, probs = self.model.transform(documents)
        return self._process_results(documents, topics, probs)

    def _process_results(self, documents: List[str], topics: List[int], probs: np.ndarray) -> Dict[str, any]:
        results = []
        for doc, topic, prob in zip(documents, topics, probs):
            topic_info = self.model.get_topic(topic)
            topic_words = [word for word,
                           _ in topic_info] if topic_info else []
            results.append({
                "document": doc,
                "topic": topic,
                "probability": prob.max(),
                "topic_words": topic_words
            })
        return {
            "results": results,
            "topic_info": self.model.get_topic_info().to_dict('records')
        }


# Usage example
if __name__ == "__main__":
    modeler = TopicModeler()
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "The sky is blue and the grass is green.",
        "Natural language processing deals with the interaction between computers and human language."
    ]
    results = modeler.fit_transform(documents)
    for result in results['results']:
        print(f"Document: {result['document']}")
        print(f"Topic: {result['topic']}")
        print(f"Probability: {result['probability']:.2f}")
        print(f"Topic words: {', '.join(result['topic_words'])}")
        print()
