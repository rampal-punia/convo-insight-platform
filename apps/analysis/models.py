from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class LLMAgentPerformance(models.Model):
    conversation = models.ForeignKey(
        'dashboard.Conversation', on_delete=models.CASCADE, related_name='performance_evaluations')
    response_time = models.DurationField()
    accuracy_score = models.FloatField(null=True, blank=True)
    relevance_score = models.FloatField(null=True, blank=True)
    customer_satisfaction_score = models.FloatField(null=True, blank=True)
    quality_score = models.FloatField(null=True, blank=True)
    feedback = models.TextField(blank=True)
    evaluated_at = models.DateTimeField(auto_now_add=True)
    issue_resolved = models.BooleanField(default=False)

    def __str__(self):
        return f"LLM Agent Performance in conversation {self.conversation.id}"


class ConversationMetrics(models.Model):
    conversation = models.OneToOneField(
        'dashboard.Conversation', on_delete=models.CASCADE, related_name='metrics')
    overall_satisfaction_score = models.FloatField(null=True, blank=True)
    average_response_time = models.DurationField(null=True, blank=True)
    average_accuracy_score = models.FloatField(null=True, blank=True)
    average_relevance_score = models.FloatField(null=True, blank=True)
    feedback = models.TextField(blank=True)
    evaluated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Metrics for conversation {self.conversation.id}"


class Recommendation(models.Model):
    conversation = models.ForeignKey(
        'dashboard.Conversation', on_delete=models.CASCADE, related_name='recommendations')
    content = models.TextField()
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    applied = models.BooleanField(default=False)

    def __str__(self):
        return f"Recommendation for conversation {self.conversation.id}"


class TopicDistribution(models.Model):
    conversation = models.ForeignKey(
        'dashboard.Conversation', on_delete=models.CASCADE, related_name='topic_distributions')
    topic = models.ForeignKey('dashboard.Topic', on_delete=models.CASCADE)
    weight = models.FloatField()

    class Meta:
        unique_together = ('conversation', 'topic')

    def __str__(self):
        return f"Topic distribution for conversation {self.conversation.id}"


class IntentPrediction(models.Model):
    message = models.OneToOneField(
        'dashboard.UserMessage', on_delete=models.CASCADE, related_name='intent_prediction')
    intent = models.ForeignKey('dashboard.Intent', on_delete=models.CASCADE)
    confidence_score = models.FloatField()

    def __str__(self):
        return f"Intent prediction for message {self.message.id}"
