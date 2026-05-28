"""Unit tests for the GeneralAssistant app models."""

import uuid

import pytest
from django.contrib.auth import get_user_model

from general_assistant.models import (
    GeneralConversation,
    GeneralMessage,
    AudioMessage,
    ImageMessage,
)

pytestmark = pytest.mark.django_db

User = get_user_model()


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def user():
    return User.objects.create_user(username="ga_user", password="pw12345")


@pytest.fixture
def conversation(user):
    return GeneralConversation.objects.create(user=user, title="Test Chat")


# ── GeneralConversation ─────────────────────────────────────────────────────


class TestGeneralConversation:
    def test_create_conversation(self, user):
        conv = GeneralConversation.objects.create(user=user)
        assert conv.pk is not None
        assert isinstance(conv.pk, uuid.UUID)
        assert conv.status == GeneralConversation.Status.ACTIVE
        assert conv.title == "Untitled Conversation"

    def test_custom_title(self, user):
        conv = GeneralConversation.objects.create(
            user=user, title="My Chat"
        )
        assert conv.title == "My Chat"

    def test_str_representation(self, user):
        conv = GeneralConversation.objects.create(user=user, title="A" * 50)
        assert "ga_user" in str(conv)
        # title is truncated to 40 chars in __str__
        assert len(str(conv).split("...")[0].split(" - ")[1]) <= 40

    def test_ordering(self, user):
        c1 = GeneralConversation.objects.create(user=user)
        c2 = GeneralConversation.objects.create(user=user)
        assert list(GeneralConversation.objects.all()) == [c2, c1]

    def test_status_choices(self):
        expected = {"AC", "AR", "EN"}
        actual = {c[0] for c in GeneralConversation.Status.choices}
        assert actual == expected

    def test_auto_timestamps(self, user):
        conv = GeneralConversation.objects.create(user=user)
        assert conv.created is not None
        assert conv.modified is not None

    def test_cascade_delete_user(self, user):
        conv = GeneralConversation.objects.create(user=user)
        assert GeneralConversation.objects.count() == 1
        user.delete()
        assert GeneralConversation.objects.count() == 0


# ── GeneralMessage ──────────────────────────────────────────────────────────


class TestGeneralMessage:
    def test_create_text_message(self, conversation):
        msg = GeneralMessage.objects.create(
            conversation=conversation,
            content_type=GeneralMessage.ContentType.TEXT,
            content="Hello",
            is_from_user=True,
        )
        assert msg.pk is not None
        assert msg.content == "Hello"
        assert msg.is_from_user is True
        assert msg.in_reply_to is None

    def test_default_content(self, conversation):
        msg = GeneralMessage.objects.create(conversation=conversation)
        assert msg.content == "No Text Data Found"

    def test_default_content_type(self, conversation):
        msg = GeneralMessage.objects.create(conversation=conversation)
        assert msg.content_type == GeneralMessage.ContentType.TEXT

    def test_reply_relationship(self, conversation):
        parent = GeneralMessage.objects.create(
            conversation=conversation,
            content="Question",
            is_from_user=True,
        )
        reply = GeneralMessage.objects.create(
            conversation=conversation,
            content="Answer",
            is_from_user=False,
            in_reply_to=parent,
        )
        assert reply.in_reply_to == parent
        assert parent.replies.count() == 1
        assert parent.replies.first() == reply

    def test_content_type_choices(self):
        expected = {"TE", "IM", "AU", "VI", "DO"}
        actual = {c[0] for c in GeneralMessage.ContentType.choices}
        assert actual == expected

    def test_ordering(self, conversation):
        m1 = GeneralMessage.objects.create(conversation=conversation, content="first")
        m2 = GeneralMessage.objects.create(conversation=conversation, content="second")
        assert list(
            GeneralMessage.objects.values_list("content", flat=True)
        ) == ["first", "second"]

    def test_cascade_delete_conversation(self, conversation):
        GeneralMessage.objects.create(conversation=conversation, content="bye")
        assert GeneralMessage.objects.count() == 1
        conversation.delete()
        assert GeneralMessage.objects.count() == 0


# ── AudioMessage ────────────────────────────────────────────────────────────


class TestAudioMessage:
    def test_create_audio_message(self, conversation):
        from django.core.files.base import ContentFile

        msg = GeneralMessage.objects.create(
            conversation=conversation,
            content_type=GeneralMessage.ContentType.AUDIO,
        )
        audio = AudioMessage.objects.create(
            message=msg,
            audio_file=ContentFile(b"fake-audio", name="audio_1.wav"),
            transcript="hello world",
            duration=3.5,
        )
        assert audio.pk is not None
        assert audio.transcript == "hello world"
        assert audio.duration == 3.5

    def test_one_to_one_relationship(self, conversation):
        msg = GeneralMessage.objects.create(
            conversation=conversation,
            content_type=GeneralMessage.ContentType.AUDIO,
        )
        from django.core.files.base import ContentFile
        AudioMessage.objects.create(
            message=msg,
            audio_file=ContentFile(b"fake", name="a.wav"),
            transcript="",
            duration=0.0,
        )
        assert msg.audio_content is not None

    def test_cascade_delete_message(self, conversation):
        from django.core.files.base import ContentFile

        msg = GeneralMessage.objects.create(
            conversation=conversation,
            content_type=GeneralMessage.ContentType.AUDIO,
        )
        AudioMessage.objects.create(
            message=msg,
            audio_file=ContentFile(b"fake", name="a.wav"),
            transcript="test",
            duration=1.0,
        )
        assert AudioMessage.objects.count() == 1
        msg.delete()
        assert AudioMessage.objects.count() == 0


# ── ImageMessage ────────────────────────────────────────────────────────────


class TestImageMessage:
    def test_create_image_message(self, conversation):
        from django.core.files.base import ContentFile

        msg = GeneralMessage.objects.create(
            conversation=conversation,
            content_type=GeneralMessage.ContentType.IMAGE,
        )
        img = ImageMessage.objects.create(
            message=msg,
            image=ContentFile(
                b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, name="test.png"
            ),
            width=720,
            height=405,
            description="A sunset",
        )
        assert img.pk is not None
        assert img.width == 720
        assert img.description == "A sunset"
        assert "Image Message" in str(img)

    def test_cascade_delete_message(self, conversation):
        from django.core.files.base import ContentFile

        msg = GeneralMessage.objects.create(
            conversation=conversation,
            content_type=GeneralMessage.ContentType.IMAGE,
        )
        ImageMessage.objects.create(
            message=msg,
            image=ContentFile(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, name="t.png"),
            width=100,
            height=100,
            description="test",
        )
        assert ImageMessage.objects.count() == 1
        msg.delete()
        assert ImageMessage.objects.count() == 0
