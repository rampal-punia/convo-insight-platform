from django.core.management.base import BaseCommand
from django.forms.models import model_to_dict
from playground.text_classification_vector_store import PGVectorStoreTextClassification
from convochat.models import Intent, Topic, Sentiment


class Command(BaseCommand):
    help = 'Populate RAG vector store with examples'

    def handle(self, *args, **kwargs):
        vector_store = PGVectorStoreTextClassification()

        # Add sentiment examples
        self.stdout.write("Processing sentiment examples...")
        sentiments = Sentiment.objects.select_related(
            'message', 'category', 'granular_emotion').all()[:100]
        texts = []
        metadata = []

        for sentiment in sentiments:
            if sentiment.message and sentiment.message.content:
                texts.append(sentiment.message.content)
                metadata.append({
                    'sentiment_category': sentiment.category.name if sentiment.category else None,
                    'sentiment_score': float(sentiment.score) if sentiment.score else 0.0,
                    'emotion': sentiment.granular_emotion.name if sentiment.granular_emotion else None
                })

        if texts:  # Only add if we have valid texts
            vector_store.add_texts(texts, 'SE', metadata)
            self.stdout.write(self.style.SUCCESS(
                f'Added {len(texts)} sentiment examples'))

        # Add intent examples
        self.stdout.write("Processing intent examples...")
        intents = Intent.objects.all()
        texts = []
        metadata = []

        for intent in intents:
            if intent.description:  # Only add if there's a description
                texts.append(intent.description)
                metadata.append({
                    'intent_name': intent.name,
                })

        if texts:
            vector_store.add_texts(texts, 'IN', metadata)
            self.stdout.write(self.style.SUCCESS(
                f'Added {len(texts)} intent examples'))

        # Add topic examples
        self.stdout.write("Processing topic examples...")
        topics = Topic.objects.filter(is_active=True)
        texts = []
        metadata = []

        for topic in topics:
            if topic.description:  # Only add if there's a description
                texts.append(topic.description)
                metadata.append({
                    'topic_name': topic.name,
                    # Use get_field_display() for choice fields
                    'category': topic.get_category_display(),
                    'priority_weight': float(topic.priority_weight)
                })

        if texts:
            vector_store.add_texts(texts, 'TO', metadata)
            self.stdout.write(self.style.SUCCESS(
                f'Added {len(texts)} topic examples'))

        self.stdout.write(self.style.SUCCESS(
            'Successfully populated RAG store'))
