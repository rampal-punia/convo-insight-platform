from django.db import models
from django.utils.translation import gettext_lazy as _
from config.models import CreationModificationDateBase
import uuid
import logging
import json
from django.core.serializers.json import DjangoJSONEncoder

logger = logging.getLogger('orders')


class ConversationSnapshot(CreationModificationDateBase):
    """Model for storing point-in-time snapshots of conversation states."""

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text=_("Unique identifier for the snapshot")
    )

    conversation = models.ForeignKey(
        'convochat.Conversation',
        on_delete=models.CASCADE,
        related_name='snapshots',
        help_text=_("The conversation this snapshot belongs to")
    )

    # Store serialized state data as JSONField
    state_data = models.JSONField(
        help_text=_("Complete serialized conversation state data"),
        default=dict,
        encoder=DjangoJSONEncoder  # Use Django's JSON encoder for better type handling
    )

    # Store conversation metrics separately for easier querying
    metrics_data = models.JSONField(
        help_text=_("Conversation metrics at time of snapshot"),
        default=dict,
        encoder=DjangoJSONEncoder
    )

    total_messages = models.PositiveIntegerField(
        default=0,
        help_text=_("Total number of messages at snapshot time")
    )

    user_messages = models.PositiveIntegerField(
        default=0,
        help_text=_("Number of user messages at snapshot time")
    )

    ai_messages = models.PositiveIntegerField(
        default=0,
        help_text=_("Number of AI messages at snapshot time")
    )

    intent_changes = models.PositiveIntegerField(
        default=0,
        help_text=_("Number of intent changes at snapshot time")
    )

    tool_uses = models.PositiveIntegerField(
        default=0,
        help_text=_("Number of tool uses at snapshot time")
    )

    avg_response_time = models.FloatField(
        null=True,
        blank=True,
        help_text=_("Average AI response time in seconds at snapshot time")
    )

    current_intent = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        help_text=_("Current conversation intent at snapshot time")
    )

    previous_intent = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        help_text=_("Previous conversation intent at snapshot time")
    )

    snapshot_type = models.CharField(
        max_length=2,
        choices=[
            ('AU', _('Automatic')),  # Regular interval snapshot
            ('MN', _('Manual')),     # Manually triggered snapshot
            ('EV', _('Event')),      # Event-triggered snapshot
            ('FN', _('Final'))       # Final state snapshot
        ],
        default='AU',
        help_text=_("Type of snapshot")
    )

    class Meta:
        ordering = ['-created']
        indexes = [
            models.Index(fields=['conversation', '-created']),
            models.Index(fields=['current_intent']),
            models.Index(fields=['snapshot_type'])
        ]
        verbose_name = _("Conversation Snapshot")
        verbose_name_plural = _("Conversation Snapshots")

    def __str__(self):
        return f"Snapshot of {self.conversation.id} at {self.created}"

    def save(self, *args, **kwargs):
        """Override save to ensure proper JSON serialization"""
        try:
            # Ensure state_data is properly serialized
            if isinstance(self.state_data, str):
                self.state_data = json.loads(self.state_data)

            # Ensure metrics_data is properly serialized
            if isinstance(self.metrics_data, str):
                self.metrics_data = json.loads(self.metrics_data)

            super().save(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error saving snapshot: {str(e)}")
            raise
