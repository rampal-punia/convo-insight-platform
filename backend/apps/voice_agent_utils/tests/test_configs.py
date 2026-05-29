"""Tests for STTConfig / TTSConfig dataclasses."""

from voice_agent_utils import STTConfig, TTSConfig


class TestSTTConfig:
    def test_defaults(self):
        cfg = STTConfig()
        assert cfg.provider == "deepgram"
        assert cfg.model == "nova-3"
        assert cfg.language == "en"
        assert cfg.punctuate is True
        assert cfg.smart_format is True
        assert cfg.diarize is False
        assert cfg.extra == {}

    def test_override(self):
        cfg = STTConfig(
            provider="bhashini",
            model="conformer",
            language="hi",
            sample_rate=16000,
            encoding="linear16",
            diarize=True,
            extra={"vendor_specific": "x"},
        )
        assert cfg.provider == "bhashini"
        assert cfg.model == "conformer"
        assert cfg.language == "hi"
        assert cfg.sample_rate == 16000
        assert cfg.encoding == "linear16"
        assert cfg.diarize is True
        assert cfg.extra == {"vendor_specific": "x"}

    def test_is_frozen(self):
        import dataclasses

        cfg = STTConfig()
        try:
            cfg.provider = "other"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("STTConfig should be frozen")


class TestTTSConfig:
    def test_defaults(self):
        cfg = TTSConfig()
        assert cfg.provider == "deepgram"
        assert cfg.model == "aura-2-thalia-en"
        assert cfg.encoding == "mp3"
        assert cfg.extra == {}

    def test_override(self):
        cfg = TTSConfig(
            provider="cloudflare",
            model="some-voice",
            encoding="linear16",
            container="wav",
            sample_rate=24000,
            speed=1.1,
        )
        assert cfg.provider == "cloudflare"
        assert cfg.encoding == "linear16"
        assert cfg.container == "wav"
        assert cfg.sample_rate == 24000
        assert cfg.speed == 1.1
