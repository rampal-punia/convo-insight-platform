# apps/convochat/tasks.py

from celery import shared_task
from .models import Conversation, UserMessage, AIMessage
from analysis.models import (
    LLMAgentPerformance,
    ConversationMetrics,
    Recommendation,
    TopicDistribution,
    IntentPrediction
)
from analysis.agent_performance.evaluator import AgentPerformanceEvaluator
from .models import Intent, Topic
from .utils.nlp_helper import perform_sentiment_analysis, detect_intent, extract_topics
from .sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from .intent_recognition.intent_recognizer import IntentRecognizer
from django.db import transaction
from .topic_modelling.topic_modeler import TopicModeler
from django.utils import timezone


@shared_task
def analyze_conversation_topics(conversation_id):
    conversation = Conversation.objects.get(id=conversation_id)
    modeler = TopicModeler()

    # Get all the messages in the conversation
    messages = Conversation.messages.all().order_by('created')
    documents = [message.content for message in messages]

    # Perform topic modelling
    results = modeler.fit_transform(documents)

    # Save topic distributions
    with transaction.atomic():
        TopicDistribution.objects.filter(conversation=conversation)
        for result in results['results']:
            topic, _ = Topic.objects.get_or_create(
                name=f"Topic {result['topic']}")
            TopicDistribution.objects.create(
                conversation=conversation,
                topic=topic,
                weight=result['probability']
            )

    # Update conversation with dominant topic
    dominant_topic = max(results['results'], key=lambda x: x['probability'])
    conversation.dominant_topic = Topic.objects.get(
        name=f"Topic {dominant_topic['topic']}")
    conversation.save()

    return results


@shared_task
def analyze_conversation_sentiment(conversation_id):
    conversation = Conversation.objects.get(id=conversation_id)
    analyzer = SentimentAnalyzer()

    # Get all the user messages in the conversation
    user_messages = conversation.messages.filter(is_from_user=True)
    texts = [message.content for message in user_messages]

    # Analyze sentiment
    sentiments = analyzer.analyze_sentiment(texts)

    # Calculate average sentiments
    sentiments = analyzer.analyze_sentiment(texts)

    # Calculate average sentiment
    avg_sentiment = sum(s['overall'] for s in sentiments) / \
        len(sentiments) if sentiments else 0

    # Update or create ConversationMetrices
    metrices, created = ConversationMetrics.objects.get_or_create(
        conversation=conversation
    )
    metrices.sentiment_score = avg_sentiment
    metrices.save()

    return avg_sentiment


@shared_task
def recognize_message_intent(message_id):
    message = UserMessage.objects.get(id=message_id)
    recognizer = IntentRecognizer()

    result = recognizer.recognize_intent([message.content])[0]

    intent, created = Intent.objects.get_or_create(
        name=result['predicted_intent'])

    IntentPrediction.objects.create(
        message=message,
        intent=intent,
        confidence_score=result['confidence']
    )
    return result['predicted_intent']


@shared_task
def save_user_message(conversation_id, content, is_from_user=True, in_reply_to=None):
    conversation = Conversation.objects.get(id=conversation_id)
    return UserMessage.objects.create(
        conversation=conversation,
        content=content,
        is_from_user=is_from_user,
        in_reply_to=in_reply_to
    ).id


@shared_task
def save_ai_message(conversation_id, full_response, is_from_user, in_reply_to=None):
    conversation = Conversation.objects.get(id=conversation_id)
    return AIMessage.objects.create(
        conversation=conversation,
        content=full_response,
        is_from_user=is_from_user,
        in_reply_to=in_reply_to,
    ).id


@shared_task
def save_intent_prediction(message_id, intent):
    message = UserMessage.objects.get(id=message_id)
    IntentPrediction.objects.create(
        message=message,
        intent=intent['intent'],
        confidence_score=intent['confidence']
    )


@shared_task
def save_topic_distribution(conversation_id, topics):
    conversation = Conversation.objects.get(id=conversation_id)
    for topic, weight in topics.items():
        TopicDistribution.objects.create(
            conversation=conversation,
            topic=topic,
            weight=weight
        )


@shared_task
def generate_and_save_recommendations(conversation_id, response):
    conversation = Conversation.objects.get(id=conversation_id)
    # Implement logic to generate recommendations based on the conversation and response
    # This is a placeholder and should be replaced with actual recommendation logic
    recommendations = [
        {"content": "Sample recommendation", "confidence_score": 0.8}]

    for rec in recommendations:
        Recommendation.objects.create(
            conversation=conversation,
            content=rec['content'],
            confidence_score=rec['confidence_score']
        )


@shared_task
def update_conversation_metrics(conversation_id, sentiment_score):
    conversation = Conversation.objects.get(id=conversation_id)
    metrics, created = ConversationMetrics.objects.get_or_create(
        conversation=conversation)
    # Update metrics based on the new message and sentiment
    # This is a placeholder and should be replaced with actual metric calculation logic
    metrics.overall_satisfaction_score = sentiment_score
    metrics.save()


@shared_task
def evaluate_llm_performance(conversation_id, ai_message_id):
    conversation = Conversation.objects.get(id=conversation_id)
    ai_message = AIMessage.objects.get(id=ai_message_id)
    user_message = ai_message.in_reply_to

    # Calculate response time
    response_time = ai_message.created - user_message.created

    # Here you would typically use more sophisticated methods to calculate these scores
    # For demonstration, we're using placeholder values
    accuracy_score = 0.8
    relevance_score = 0.75
    customer_satisfaction_score = 0.9
    quality_score = (accuracy_score + relevance_score +
                     customer_satisfaction_score) / 3

    LLMAgentPerformance.objects.create(
        conversation=conversation,
        response_time=response_time,
        accuracy_score=accuracy_score,
        relevance_score=relevance_score,
        customer_satisfaction_score=customer_satisfaction_score,
        quality_score=quality_score
    )

    # Trigger overall performance evaluation
    evaluate_overall_performance.delay(conversation_id)


@shared_task
def evaluate_overall_performance(conversation_id):
    evaluator = AgentPerformanceEvaluator()
    result = evaluator.evaluate_conversation(conversation_id)

    # Here you could save the result to a new model, send notifications, etc.
    print(
        f"Performance evaluation for conversation {conversation_id}: {result}")

    return result


@shared_task
def process_user_message(conversation_id, user_message_content):
    # Perform NLP tasks
    sentiment = perform_sentiment_analysis(user_message_content)
    intent = detect_intent(user_message_content)

    # Save user message
    user_message = save_user_message(
        conversation_id=conversation_id,
        content=user_message_content,
        is_from_user=True
    )

    # Trigger intent recognition
    recognize_message_intent.delay(user_message.id)

    # Save intent prediction
    save_intent_prediction.delay(user_message, intent)

    # Update conversation metrics
    update_conversation_metrics.delay(conversation_id, sentiment['score'])

    return user_message.id, sentiment, intent


@shared_task
def process_ai_response(conversation_id, ai_response_content, user_message_id):
    # Extract topics
    topics = extract_topics(ai_response_content)

    # Save AI message
    ai_message = save_ai_message(
        conversation_id=conversation_id,
        content=ai_response_content,
        is_from_user=False,
        in_reply_to=user_message_id
    )

    # Trigger topic modeling
    analyze_conversation_topics.delay(conversation_id)

    # Save topic distribution
    save_topic_distribution.delay(conversation_id, topics)

    # Generate and save recommendations
    generate_and_save_recommendations.delay(
        conversation_id, ai_response_content)

    # Evaluate LLM performance
    evaluate_llm_performance.delay(conversation_id, ai_message.id)

    return ai_message.id, topics


@shared_task
def save_conversation_title(conversation_id, title):
    conversation = Conversation.objects.get(id=conversation_id)
    conversation.title = title
    conversation.save()
