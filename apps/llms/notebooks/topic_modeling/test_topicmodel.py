import unittest
import joblib
import dill
import numpy as np
import pandas as pd
from gensim.models import LdaMulticore
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)


class TestLDATopicModeling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the trained model and necessary data
        cls.lda_model = joblib.load("topic_model_data/lda_model.joblib")
        cls.dictionary = joblib.load("topic_model_data/dictionary.joblib")
        cls.corpus = joblib.load("topic_model_data/corpus.joblib")
        cls.df = joblib.load("topic_model_data/preprocessed_df.joblib")

        with open("topic_model_data/preprocess_text_function.dill", "rb") as f:
            cls.preprocess_text = dill.load(f)

        # Load a small test set, now using the pre-tokenized texts
        cls.test_texts = cls.df['tokenized_text'].head(100).tolist()
        cls.test_processed = [cls.preprocess_text(
            text) for text in cls.test_texts]

    def test_model_type(self):
        self.assertIsInstance(self.lda_model, LdaMulticore)
        self.assertIsInstance(self.dictionary, corpora.Dictionary)
        self.assertIsInstance(self.corpus, list)

    def test_model_output_format(self):
        for doc in self.test_processed[:5]:  # Test with first 5 documents
            bow = self.dictionary.doc2bow(doc)
            topics = self.lda_model.get_document_topics(bow)
            self.assertTrue(all(isinstance(topic[0], int) and isinstance(
                topic[1], float) for topic in topics))
            self.assertTrue(all(0 <= prob <= 1 for _, prob in topics))

    def test_topic_coherence(self):
        coherence_model = CoherenceModel(
            model=self.lda_model, texts=self.test_processed, dictionary=self.dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        self.assertGreater(coherence_score, 0.0)  # Adjust threshold as needed

    def test_topic_diversity(self):
        N = 20
        topic_words = [word for topic in self.lda_model.show_topics(num_topics=-1, num_words=N, formatted=False)
                       for word, _ in topic[1]]
        diversity_score = len(set(topic_words)) / \
            (self.lda_model.num_topics * N)
        self.assertGreater(diversity_score, 0.0)  # Adjust threshold as needed

    def test_new_text_assignment(self):
        new_text = "This is a sample text about artificial intelligence and machine learning."
        processed_text = self.preprocess_text(new_text)
        bow = self.dictionary.doc2bow(processed_text)
        topics = self.lda_model.get_document_topics(bow)
        self.assertTrue(len(topics) > 0)
        self.assertIsInstance(topics[0][0], int)
        self.assertIsInstance(topics[0][1], float)
        self.assertTrue(0 <= topics[0][1] <= 1)

    def test_model_consistency(self):
        text = "This is a test text for consistency."
        processed_text = self.preprocess_text(text)
        bow = self.dictionary.doc2bow(processed_text)
        topics1 = self.lda_model.get_document_topics(bow)
        topics2 = self.lda_model.get_document_topics(bow)
        self.assertEqual(topics1, topics2)

    def test_preprocessing_function(self):
        text = "This is a TEST sentence with UPPERCASE words and punctuation!"
        processed = self.preprocess_text(text)
        self.assertTrue(all(word.islower() for word in processed))
        self.assertTrue(all(word.isalpha() for word in processed))

    def test_dictionary_filter(self):
        # Check if extreme words have been filtered out
        self.assertTrue(
            all(self.dictionary.dfs[id] >= 5 for id in self.dictionary.dfs))
        self.assertTrue(all(
            self.dictionary.dfs[id] / len(self.corpus) <= 0.5 for id in self.dictionary.dfs))

    def test_corpus_format(self):
        self.assertTrue(all(isinstance(doc, list)
                        for doc in self.corpus[:5]))  # Test with first 5 documents
        self.assertTrue(all(isinstance(word_id, int) and isinstance(count, int)
                            for doc in self.corpus[:5] for word_id, count in doc))


if __name__ == '__main__':
    unittest.main()
