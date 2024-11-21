from django.db import models
from django.utils.translation import gettext_lazy as _
from config.models import CreationModificationDateBase
import uuid


class ConversationSnapshot(CreationModificationDateBase):
    """
    Model for storing point-in-time snapshots of conversation states.
    Used to track conversation progression and enable state restoration.
    """

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
        default=dict
    )

    # Store conversation metrics separately for easier querying
    metrics_data = models.JSONField(
        help_text=_("Conversation metrics at time of snapshot"),
        default=dict
    )

    # Additional metrics fields for quick access without JSON deserialization
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
        """
        Override save to extract and store quick-access metrics
        from metrics_data JSON field
        """
        if self.metrics_data:
            self.total_messages = self.metrics_data.get('total_messages', 0)
            self.user_messages = self.metrics_data.get('user_messages', 0)
            self.ai_messages = self.metrics_data.get('ai_messages', 0)
            self.intent_changes = self.metrics_data.get('intent_changes', 0)
            self.tool_uses = self.metrics_data.get('tool_uses', 0)
            self.avg_response_time = self.metrics_data.get('avg_response_time')

        if self.state_data:
            self.current_intent = self.state_data.get('current_intent')
            self.previous_intent = self.state_data.get('previous_intent')

        super().save(*args, **kwargs)

    @classmethod
    def create_snapshot(cls, conversation_id: str, state_data: dict, metrics_data: dict, snapshot_type: str = 'AU'):
        """
        Class method to create a new snapshot with proper data extraction.

        Args:
            conversation_id: ID of the conversation
            state_data: Complete state data dictionary
            metrics_data: Metrics data dictionary
            snapshot_type: Type of snapshot (AU/MN/EV/FN)

        Returns:
            Created ConversationSnapshot instance
        """
        return cls.objects.create(
            conversation_id=conversation_id,
            state_data=state_data,
            metrics_data=metrics_data,
            snapshot_type=snapshot_type
        )
