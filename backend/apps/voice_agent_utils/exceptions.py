"""Exceptions raised by voice providers."""

from __future__ import annotations


class VoiceProviderError(Exception):
    """Base class for any voice-provider failure."""


class TranscriptionError(VoiceProviderError):
    """Speech-to-text request failed."""


class SynthesisError(VoiceProviderError):
    """Text-to-speech request failed."""


class ProviderNotFoundError(VoiceProviderError):
    """Requested provider name is not registered."""
