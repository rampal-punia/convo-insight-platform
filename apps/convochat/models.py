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
    summary = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ['-created']
        indexes = [
            models.Index(fields=['user', 'created']),
            models.Index(fields=['status'])
        ]

    def __str__(self) -> str:
        return f"{self.user.username} - {self.title[:40]}"


class Message(CreationModificationDateBase):
    class ContentType(models.TextChoices):
        AUDIO = 'AU', _('Audio')
        DOCUMENT = 'DO', _('Document')
        IMAGE = 'IM', _('Image')
        TEXT = 'TE', _('Text')

    conversation = models.ForeignKey(
        'convochat.Conversation',
        on_delete=models.CASCADE,
        related_name='messages'
    )
    content_type = models.CharField(
        max_length=2,
        choices=ContentType.choices,
        default=ContentType.TEXT
    )
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
            models.Index(fields=['is_from_user', 'created'])
        ]


class UserText(Message):
    content = models.TextField()
    sentiment_score = models.FloatField(null=True, blank=True)
    intent = models.ForeignKey(
        'Intent',
        on_delete=models.SET_NULL,
        null=True,
        related_name='user_texts'
    )
    primary_topic = models.ForeignKey(
        'Topic',
        on_delete=models.SET_NULL,
        null=True,
        related_name='primary_user_texts'
    )


class AIText(Message):
    '''
    Use case::

        # When an agent applies a recommendation
        recommendation = Recommendation.objects.get(id=1)  # Get a specific recommendation

        ai_message = AIText.objects.create(
        conversation=conversation,
        content="Thank you for your patience. I understand your frustration with the delayed shipment. 
        Let me check the status for you right away.",
        is_from_user=False,
        recommendation=recommendation
        )
    '''
    content = models.TextField(null=True, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
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
        UserText,
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
