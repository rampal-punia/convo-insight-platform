# apps/convochat/models.py

import uuid

from django.db import models
from django.utils.translation import gettext_lazy as _
from config.models import CreationModificationDateBase
from django.contrib.auth import get_user_model


User = get_user_model()


class Conversation(CreationModificationDateBase):
    class Status(models.TextChoices):
        ACTIVE = 'AC', _('Active')
        ARCHIVED = 'AR', _('Archived')
        ENDED = 'EN', _('Ended')

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    title = models.CharField(
        max_length=255,
        default='Untitled Conversation'
    )
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='conversations',
    )
    status = models.CharField(
        max_length=2,
        choices=Status.choices,
        default=Status.ACTIVE
    )
    overall_sentiment = models.FloatField(null=True, blank=True)
    dominant_topic = models.ForeignKey(
        'Topic',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='conversations',
    )

    class Meta:
        ordering = ['-created']
        indexes = [
            models.Index(fields=['user', 'created']),
            models.Index(fields=['status'])
        ]

    def __str__(self) -> str:
        return f"{self.user.username} - {self.title}"


class Message(CreationModificationDateBase):
    conversation = models.ForeignKey(
        'convochat.Conversation',
        on_delete=models.CASCADE,
        related_name='messages'
    )
    content = models.TextField()
    is_from_user = models.BooleanField(default=True)
    in_reply_to = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='replies'
    )

    class Meta:
        ordering = ['created']
        indexes = [
            models.Index(fields=['conversation', 'created']),
            models.Index(fields=['is_from_user'])
        ]

    def __str__(self) -> str:
        return f"{self.id} - {self.content[:50]}"


class UserMessage(Message):
    sentiment_score = models.FloatField(
        null=True,
        blank=True
    )
    intent = models.ForeignKey(
        'Intent',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='mesages'
    )
    topic = models.ForeignKey(
        'Topic',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='messages'
    )


class AIMessage(Message):
    '''
    Use case::

        # When an agent applies a recommendation
        recommendation = Recommendation.objects.get(id=1)  # Get a specific recommendation

        ai_message = AIMessage.objects.create(
        conversation=conversation,
        content="Thank you for your patience. I understand your frustration with the delayed shipment. 
        Let me check the status for you right away.",
        is_from_user=False,
        recommendation=recommendation
        )
    '''
    recommendation = models.ForeignKey(
        'analysis.Recommendation',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='applied_messages'
    )


class Intent(models.Model):
    '''
        Examples of Intents:
        - Ask Question
        - Request Information
        - File Complaint
        - Seek Assistance
        - Make Purchase
        - Cancel Service
        - Provide Feedback
    '''
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)

    def __str__(self) -> str:
        return self.name


class Topic(models.Model):
    '''
        Examples of Topics:

        - Product Inquiries
        - Billing Issues
        - Technical Support
        - Feature Requests
        - General Feedback
        - Shipping and Delivery
        - Account Management
    '''

    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)

    def __str__(self) -> str:
        return self.name


class Sentiment(models.Model):
    class SentimentCategory(models.TextChoices):
        POSITIVE = 'PO', _('Positive')
        NEGATIVE = 'NE', _('Negative')
        NEUTRAL = 'NU', _('Neutral')

    category = models.CharField(
        max_length=2,
        choices=SentimentCategory.choices
    )
    score = models.FloatField()
    message = models.OneToOneField(
        UserMessage,
        on_delete=models.CASCADE,
        related_name='detailed_sentiment'
    )
    granular_category = models.CharField(
        max_length=50,
        blank=True,
        null=True
    )

    def __str__(self) -> str:
        return f"{self.get_category_display()} ({self.score})"
