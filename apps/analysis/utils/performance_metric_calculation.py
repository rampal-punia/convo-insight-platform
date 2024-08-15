from django.db.models import Avg
from django.utils import timezone
from .models import Conversation, Message, UserMessage, AIMessage, LLMAgentPerformance, ConversationMetrics
from .utils import calculate_accuracy, calculate_relevance, calculate_satisfaction


def process_conversation_metrics(conversation_id):
    conversation = Conversation.objects.get(id=conversation_id)
    messages = conversation.messages.all().order_by('created')

    llm_performances = []
    response_times = []
    accuracy_scores = []
    relevance_scores = []
    satisfaction_scores = []

    for i, message in enumerate(messages):
        if isinstance(message, AIMessage):
            # Calculate response time
            if i > 0:
                response_time = message.created - messages[i-1].created
                response_times.append(response_time)

            # Calculate other metrics
            accuracy = calculate_accuracy(message)
            relevance = calculate_relevance(message)
            satisfaction = calculate_satisfaction(message)

            # Create LLMAgentPerformance instance
            performance = LLMAgentPerformance.objects.create(
                conversation=conversation,
                response_time=response_time,
                accuracy_score=accuracy,
                relevance_score=relevance,
                customer_satisfaction_score=satisfaction,
                # Simple average for quality
                quality_score=(accuracy + relevance + satisfaction) / 3,
                feedback="",  # This could be filled by a human reviewer later
                # This could be updated based on conversation analysis or user feedback
                issue_resolved=False
            )
            llm_performances.append(performance)

            # Append scores for overall metrics
            accuracy_scores.append(accuracy)
            relevance_scores.append(relevance)
            satisfaction_scores.append(satisfaction)

    # Calculate overall metrics
    avg_response_time = sum(response_times, timezone.timedelta(
    )) / len(response_times) if response_times else None
    avg_accuracy = sum(accuracy_scores) / \
        len(accuracy_scores) if accuracy_scores else None
    avg_relevance = sum(relevance_scores) / \
        len(relevance_scores) if relevance_scores else None
    overall_satisfaction = sum(
        satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else None

    # Create ConversationMetrics instance
    ConversationMetrics.objects.create(
        conversation=conversation,
        overall_satisfaction_score=overall_satisfaction,
        average_response_time=avg_response_time,
        average_accuracy_score=avg_accuracy,
        average_relevance_score=avg_relevance,
        feedback="",  # This could be filled by a human reviewer or aggregated from user feedback
        evaluated_at=timezone.now()
    )

    return llm_performances


# Usage
conversation_id = "some-uuid-here"
process_conversation_metrics(conversation_id)
