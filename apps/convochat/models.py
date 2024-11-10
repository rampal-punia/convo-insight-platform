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

    class ResolutionStatus(models.TextChoices):
        RESOLVED = 'RE', _('Resolved')
        IN_PROGRESS = 'PR', _('In Progress')
        UNRESOLVED = 'UN', _('Unresolved')

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
    # Add current_intent field
    current_intent = models.CharField(
        max_length=50,
        null=True,
        blank=True,
        help_text="Current conversation intent/context"
    )
    overall_sentiment_score = models.FloatField(null=True, blank=True)
    dominant_topic = models.ForeignKey(
        'Topic',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='conversations',
    )
    summary = models.TextField(null=True, blank=True)
    resolution_status = models.CharField(
        max_length=2,
        choices=ResolutionStatus.choices,
        default=ResolutionStatus.UNRESOLVED,
        null=True,
        blank=True
    )

    class Meta:
        ordering = ['-created']
        indexes = [
            models.Index(fields=['user', 'created']),
            models.Index(fields=['status'])
        ]

    def __str__(self) -> str:
        return f"{self.user.username} - {self.title[:40]}"

    def lasted_for(self):
        return float(self.modified - self.created)


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


class UserText(models.Model):
    message = models.OneToOneField(
        'convochat.Message',
        on_delete=models.CASCADE,
        related_name="user_text",
        null=True
    )
    content = models.TextField()
    sentiment_score = models.FloatField(null=True, blank=True)
    intent = models.ForeignKey(
        'Intent',
        on_delete=models.SET_NULL,
        null=True,
        related_name='user_text'
    )
    primary_topic = models.ForeignKey(
        'Topic',
        on_delete=models.SET_NULL,
        null=True,
        related_name='primary_user_texts'
    )


class AIText(models.Model):
    content = models.TextField(null=True, blank=True)
    message = models.OneToOneField(
        'convochat.Message',
        on_delete=models.CASCADE,
        related_name="ai_text",
        null=True
    )
    confidence_score = models.FloatField(null=True, blank=True)
    recommendation = models.ForeignKey(
        'analysis.Recommendation',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='applied_messages'
    )
    tool_calls = models.JSONField(default=list, blank=True)


class Intent(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)

    def __str__(self) -> str:
        return self.name


# Modified Topic Model (convochat_models.py)
class Topic(models.Model):
    class Category(models.TextChoices):
        PRODUCT = 'PR', _('Product-Related')
        ORDER = 'OR', _('Order Management')
        PAYMENT = 'PA', _('Payment & Account')
        EXPERIENCE = 'EX', _('Customer Experience')

    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    category = models.CharField(
        max_length=2,
        choices=Category.choices,
        default=Category.PRODUCT
    )
    # For weighting topic importance in analysis
    priority_weight = models.FloatField(
        default=1.0,
        help_text="Weight factor for topic importance in analysis"
    )
    # For tracking topic engagement
    usage_count = models.PositiveIntegerField(
        default=0,
        help_text="Number of times this topic was identified in conversations"
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this topic is currently in use"
    )
    created = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    modified = models.DateTimeField(auto_now=True, blank=True, null=True)

    class Meta:
        ordering = ['category', 'name']
        indexes = [
            models.Index(fields=['category', 'name']),
            models.Index(fields=['is_active', 'usage_count'])
        ]

    def __str__(self) -> str:
        return f"{self.get_category_display()} - {self.name}"

    def increment_usage(self):
        """Increment the usage count for this topic"""
        self.usage_count += 1
        self.save(update_fields=['usage_count', 'modified'])


class SentimentCategory(models.Model):
    """Base sentiment categories that can be used across the system"""
    name = models.CharField(
        max_length=100, unique=True)  # e.g., "Positive", "Negative", "Neutral"
    description = models.TextField(
        help_text="Description of when this sentiment category should be used"
    )
    priority_weight = models.FloatField(
        default=1.0,
        help_text="Weight factor for sentiment importance in analysis"
    )

    class Meta:
        verbose_name_plural = "Sentiment Categories"

    def __str__(self):
        return self.name


class GranularEmotion(models.Model):
    """Detailed emotional categories for nuanced sentiment analysis"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(
        help_text="Description of this emotional category and when it applies"
    )
    associated_sentiment = models.ForeignKey(
        SentimentCategory,
        on_delete=models.CASCADE,
        related_name='emotions',
        help_text="The primary sentiment category this emotion is typically associated with"
    )
    usage_count = models.PositiveIntegerField(
        default=0,
        help_text="Number of times this emotion was identified"
    )

    def __str__(self):
        return f"{self.name} ({self.associated_sentiment.name})"


class Sentiment(models.Model):
    """Individual sentiment analysis results"""
    category = models.ForeignKey(
        SentimentCategory,
        on_delete=models.PROTECT,
        related_name='sentiments'
    )
    granular_emotion = models.ForeignKey(
        GranularEmotion,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='sentiments'
    )
    score = models.FloatField(
        help_text="Sentiment intensity score between -1 (very negative) and 1 (very positive)"
    )
    confidence = models.FloatField(
        help_text="Model's confidence in this sentiment classification (0-1)",
        default=1.0
    )
    message = models.OneToOneField(
        'UserText',
        on_delete=models.CASCADE,
        related_name='sentiment_analysis'
    )

    class Meta:
        indexes = [
            models.Index(fields=['category', 'score']),
        ]

    def __str__(self):
        emotion = f" ({self.granular_emotion.name})" if self.granular_emotion else ""
        return f"{self.category.name}{emotion}: {self.score}"

    def increment_emotion_usage(self):
        """Increment the usage count for the granular emotion if present"""
        if self.granular_emotion:
            self.granular_emotion.usage_count += 1
            self.granular_emotion.save(update_fields=['usage_count'])
