import logging

from celery import shared_task
from convochat.models import Intent, Topic, Sentiment, Conversation, Message, UserText, AIText
from analysis.models import IntentPrediction

logger = logging.getLogger(__name__)


@shared_task
def recognize_intent(message_id):
    from convochat.utils.intent_recognizer_bertbase import IntentRecognizer
    usertext = UserText.objects.get(message__id=message_id)
    intents = list(Intent.objects.all())
    logger.info("Intents in intent model database are: ", intents)
    recognizer = IntentRecognizer(intent_labels=intents)
    result = recognizer.recognize_intent([usertext.content])[0]
    logger.info("result for the intent is: ", result)
    logger.info('*'*40)
    logger.info(result['predicted_intent'])

    IntentPrediction.objects.create(
        message=usertext,
        intent=result['predicted_intent'],
        confidence_score=result['confidence'],
    )
    return result['predicted_intent']


@shared_task
def analyze_sentiment(message_id):
    from convochat.utils.sentiment_analyzer import SentimentAnalyzer
    usertext = UserText.objects.get(message__id=message_id)
    analyzer = SentimentAnalyzer()

    result = analyzer.analyze_sentiment([usertext.content])[0]

    Sentiment.objects.create(
        message=usertext,
        score=result['overall'],
        category='PO' if result['category'] == 'POSITIVE' else 'NE'
    )


@shared_task
def process_user_message(message_id, predicted_sentiment, predicted_intent):
    # Perform immediate analysis
    message = UserText.objects.get(message__id=message_id)
    # topics = generate_topic_distribution(content)

    # message.sentiment_score=sentiment_score
    message.intent = predicted_intent
    # Set the highest weighted topic as primary
    # message.primary_topic=max(topics, key=lambda x: x[1])[0]
    message.save()
