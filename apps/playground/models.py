from django.db import models, connection
from pgvector.django import VectorField


class RAGTextClassificationDocument(models.Model):
    """Store documents for RAG with their embeddings"""
    class TaskType(models.TextChoices):
        SENTIMENT = 'SE', 'Sentiment Analysis'
        INTENT = 'IN', 'Intent Recognition'
        TOPIC = 'TO', 'Topic Classification'

    task = models.CharField(max_length=2, choices=TaskType.choices)
    content = models.TextField()
    metadata = models.JSONField(default=dict)
    embedding = VectorField(dimensions=384)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['task']),
            # For regular querying
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"{self.get_task_display()}: {self.content[:50]}..."

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # Create vector index if it doesn't exist
        if not hasattr(self, '_vector_index_created'):
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS content_embedding_idx 
                    ON playground_ragdocument 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                    """
                )
            self._vector_index_created = True
