from celery import shared_task
from convochat.models import Intent, Topic, Sentiment, Conversation, Message, UserText, AIText
from convochat.utils.intent_recognizer_bertbase import IntentRecognizer
from convochat.utils.sentiment_analyzer import SentimentAnalyzer
from analysis.models import IntentPrediction


@shared_task
def recognize_intent(message_id):
    usertext = UserText.objects.get(message__id=message_id)
    intents = list(Intent.objects.all())
    print("Intents in intent model database are: ", intents)
    recognizer = IntentRecognizer(intent_labels=intents)
    result = recognizer.recognize_intent([usertext.content])
    print("result for the intent is: ", result)

    IntentPrediction.objects.create(
        message=usertext,
        intent=result['predicted_intent'],
        confidence_score=result['confidence']
    )
    return result['predicted_intent']


@shared_task
def analyze_Sentiment(message_id):
    usertext = UserText.objects.get(message__id=message_id)
    sentiments = list(Sentiment.objects.all())
    print("Sentiments in Sentimen model database are: ", sentiments)
