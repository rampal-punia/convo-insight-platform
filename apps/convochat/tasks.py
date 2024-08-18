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
from .utils.nlp_helper import perform_sentiment_analysis, detect_intent, extract_topics


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
    # Implement logic to evaluate LLM performance
    # This is a placeholder and should be replaced with actual evaluation logic
    LLMAgentPerformance.objects.create(
        conversation=conversation,
        response_time=ai_message.created - ai_message.in_reply_to.created,
        accuracy_score=0.9,
        relevance_score=0.85,
        customer_satisfaction_score=0.8,
        quality_score=0.87
    )


@shared_task
def process_user_message(conversation_id, user_message_content):
    # Perform NLP tasks
    sentiment = perform_sentiment_analysis(user_message_content)
    intent = detect_intent(user_message_content)

    # Save user message
    user_message_id = save_user_message(conversation_id, user_message_content)

    # Save intent prediction
    save_intent_prediction.delay(user_message_id, intent)

    # Update conversation metrics
    update_conversation_metrics.delay(conversation_id, sentiment['score'])

    return user_message_id, sentiment, intent


@shared_task
def process_ai_response(conversation_id, ai_response_content, user_message_id):
    # Extract topics
    topics = extract_topics(ai_response_content)

    # Save AI message
    ai_message_id = save_ai_message(
        conversation_id, ai_response_content, False, user_message_id)

    # Save topic distribution
    save_topic_distribution.delay(conversation_id, topics)

    # Generate and save recommendations
    generate_and_save_recommendations.delay(
        conversation_id, ai_response_content)

    # Evaluate LLM performance
    evaluate_llm_performance.delay(conversation_id, ai_message_id)

    return ai_message_id, topics


@shared_task
def save_conversation_title(conversation_id, title):
    conversation = Conversation.objects.get(id=conversation_id)
    conversation.title = title
    conversation.save()
