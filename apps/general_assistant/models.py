import uuid

from django.db import models
from django.utils.translation import gettext_lazy as _
from config.models import CreationModificationDateBase
from django.contrib.auth import get_user_model


User = get_user_model()


class GeneralConversation(CreationModificationDateBase):
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
        related_name='general_conversations',
    )
    status = models.CharField(
        max_length=2,
        choices=Status.choices,
        default=Status.ACTIVE
    )

    class Meta:
        ordering = ['-created']
        indexes = [
            models.Index(fields=['user', 'created']),
            models.Index(fields=['status'])
        ]

    def __str__(self) -> str:
        return f"{self.user.username} - {self.title[:40]}..."


class GeneralMessage(CreationModificationDateBase):
    class ContentType(models.TextChoices):
        TEXT = 'TE', _('Text')
        IMAGE = 'IM', _('Image')
        AUDIO = 'AU', _('Audio')
        VIDEO = 'VI', _('Video')
        DOCUMENT = 'DO', _('Document')

    conversation = models.ForeignKey(
        'GeneralConversation',
        on_delete=models.CASCADE,
        related_name='general_messages'
    )
    content_type = models.CharField(
        max_length=2,
        choices=ContentType.choices,
        default=ContentType.TEXT
    )
    content = models.TextField(default='No Text Data Found')
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


class AudioMessage(models.Model):
    message = models.OneToOneField(
        'GeneralMessage',
        on_delete=models.CASCADE,
        related_name='audio_content'
    )
    audio_file = models.FileField(upload_to='voice_messages/')
    transcript = models.TextField()
    duration = models.FloatField()  # in seconds
