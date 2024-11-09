from sentence_transformers import SentenceTransformer
from typing import List, Dict
from .models import RAGTextClassificationDocument
from django.db.models import Q
from django.db.models.expressions import RawSQL
from django.db.models import Case, When
from django.db.models.functions import Cast
from pgvector.django import CosineDistance, VectorField
from django.db import transaction


class PGVectorStoreTextClassification:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    @transaction.atomic
    def add_texts(self, texts: List[str], task: str, metadata: List[Dict] = None):
        """Add texts to vector store with their embeddings"""
        if not texts:
            return
        embeddings = self.model.encode(texts)

        documents = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            doc = RAGTextClassificationDocument(
                task=task,
                content=text,
                embedding=embedding.tolist(),
                metadata=metadata[i] if metadata else {}
            )
            documents.append(doc)

        RAGTextClassificationDocument.objects.bulk_create(documents)

    def similarity_search(self, query: str, task: str, k: int = 5):
        """Search for similar documents for a given task"""
        query_embedding = self.model.encode(query)

        # Create the cosine distance expression
        distance = CosineDistance('embedding', query_embedding.tolist())

        # Use cosine distance for similarity search
        documents = RAGTextClassificationDocument.objects.filter(
            task=task
        ).annotate(
            distance=distance
        ).order_by('distance')[:k]

        return documents

    def similarity_search_with_score(self,
                                     query: str,
                                     task: str,
                                     k: int = 5) -> List[tuple[RAGTextClassificationDocument, float]]:
        """Search for similar documents and return with similarity scores"""
        # Generate embedding for query
        query_embedding = self.model.encode(query)

        # Create the cosine distance expression
        distance = CosineDistance('embedding', query_embedding.tolist())

        # Annotate with distance scores and order by them
        documents = RAGTextClassificationDocument.objects.filter(
            task=task
        ).annotate(
            distance=distance
        ).order_by('distance')[:k]

        # Convert distance to similarity score (1 - distance)
        return [(doc, 1 - doc.distance) for doc in documents]
