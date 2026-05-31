"""Management command for quick STT / TTS smoke-tests.

Quick wins for interns:

    # Transcribe a WAV file (STT)
    python manage.py voice_check stt path/to/audio.wav

    # Synthesise speech from text (TTS) — saves MP3 next to cwd
    python manage.py voice_check tts "Life is awesome and Python is great."

    # TTS with custom output path
    python manage.py voice_check tts "Hello world" --out /tmp/hello.mp3

    # Use a non-default model / voice
    python manage.py voice_check stt audio.wav --model nova-2
    python manage.py voice_check tts "Hi" --model aura-2-asteria-en
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import time

from django.core.management.base import BaseCommand, CommandError

from voice_agent_utils import (
    STTConfig,
    TTSConfig,
    get_stt_provider,
    get_tts_provider,
)
from voice_agent_utils.exceptions import VoiceProviderError


class Command(BaseCommand):
    help = (
        "Smoke-test the STT or TTS service from the command line. "
        "Great for verifying Deepgram credentials and provider config."
    )

    # ------------------------------------------------------------------ #
    #  Argument definition                                                  #
    # ------------------------------------------------------------------ #

    def add_arguments(self, parser):
        sub = parser.add_subparsers(dest="mode", metavar="MODE")
        sub.required = True

        # -- stt sub-command -------------------------------------------
        stt_p = sub.add_parser(
            "stt",
            help="Transcribe a WAV (or any audio) file → print transcript.",
        )
        stt_p.add_argument(
            "audio_file",
            metavar="AUDIO_FILE",
            help="Path to the audio file to transcribe.",
        )
        stt_p.add_argument(
            "--model",
            default="nova-3",
            metavar="MODEL",
            help="Deepgram STT model (default: nova-3, aura-2-asteria-en).",
        )
        stt_p.add_argument(
            "--language",
            default="en",
            metavar="LANG",
            help="BCP-47 language code (default: en).",
        )
        stt_p.add_argument(
            "--no-smart-format",
            dest="smart_format",
            action="store_false",
            default=True,
            help="Disable smart formatting.",
        )
        stt_p.add_argument(
            "--diarize",
            action="store_true",
            default=False,
            help="Enable speaker diarization.",
        )

        # -- tts sub-command -------------------------------------------
        tts_p = sub.add_parser(
            "tts",
            help="Synthesise text → save audio file.",
        )
        tts_p.add_argument(
            "text",
            metavar="TEXT",
            help="The text to synthesise. Quote multi-word strings.",
        )
        tts_p.add_argument(
            "--out",
            dest="output_file",
            default=None,
            metavar="OUTPUT_FILE",
            help=(
                "Path for the output audio file. "
                "Default: voice_check_output_<timestamp>.mp3 in the current directory."
            ),
        )
        tts_p.add_argument(
            "--model",
            default="aura-2-thalia-en",
            metavar="MODEL",
            help="Deepgram TTS voice model (default: aura-2-thalia-en).",
        )
        tts_p.add_argument(
            "--encoding",
            default="mp3",
            choices=["mp3", "linear16", "flac", "mulaw", "alaw", "opus", "aac"],
            help="Audio encoding / format (default: mp3).",
        )

    # ------------------------------------------------------------------ #
    #  Entry point                                                          #
    # ------------------------------------------------------------------ #

    def handle(self, *args, **options):
        mode = options["mode"]
        if mode == "stt":
            asyncio.run(self._run_stt(options))
        elif mode == "tts":
            asyncio.run(self._run_tts(options))
        else:
            raise CommandError(f"Unknown mode: {mode!r}")

    # ------------------------------------------------------------------ #
    #  STT                                                                  #
    # ------------------------------------------------------------------ #

    async def _run_stt(self, options: dict) -> None:
        audio_path = pathlib.Path(options["audio_file"])
        if not audio_path.exists():
            raise CommandError(f"File not found: {audio_path}")
        if not audio_path.is_file():
            raise CommandError(f"Not a file: {audio_path}")

        cfg = STTConfig(
            provider="deepgram",
            model=options["model"],
            language=options["language"],
            smart_format=options["smart_format"],
            diarize=options["diarize"],
        )

        self.stdout.write(
            self.style.HTTP_INFO(
                f"\n[STT] Transcribing {audio_path.name} "
                f"(model={cfg.model}, lang={cfg.language}) …"
            )
        )

        provider = get_stt_provider(cfg)
        t0 = time.perf_counter()
        try:
            transcript = await provider.transcribe_file(str(audio_path))
        except VoiceProviderError as exc:
            raise CommandError(f"STT failed: {exc}") from exc

        elapsed = time.perf_counter() - t0
        size_kb = audio_path.stat().st_size / 1024

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Transcript:"))
        self.stdout.write(f"  {transcript or '(empty)'}")
        self.stdout.write("")
        self.stdout.write(f"  [{size_kb:.1f} KB audio → {elapsed:.2f}s]")

    # ------------------------------------------------------------------ #
    #  TTS                                                                  #
    # ------------------------------------------------------------------ #

    async def _run_tts(self, options: dict) -> None:
        text: str = options["text"].strip()
        if not text:
            raise CommandError("TEXT must not be empty.")

        encoding: str = options["encoding"]
        out_path: pathlib.Path = (
            pathlib.Path(options["output_file"])
            if options["output_file"]
            else pathlib.Path.cwd()
            / f"voice_check_output_{int(time.time())}.{encoding}"
        )

        cfg = TTSConfig(
            provider="deepgram",
            model=options["model"],
            encoding=encoding,
        )

        self.stdout.write(
            self.style.HTTP_INFO(
                f"\n[TTS] Synthesising {len(text)} chars "
                f"(model={cfg.model}, encoding={cfg.encoding}) …"
            )
        )

        provider = get_tts_provider(cfg)
        t0 = time.perf_counter()
        try:
            audio_bytes = await provider.synthesize(text)
        except VoiceProviderError as exc:
            raise CommandError(f"TTS failed: {exc}") from exc

        elapsed = time.perf_counter() - t0

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio_bytes)

        size_kb = len(audio_bytes) / 1024
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(f"Audio saved → {out_path}"))
        self.stdout.write(f"  {size_kb:.1f} KB | {elapsed:.2f}s")
        self.stdout.write(f'  Text: "{text[:80]}{"…" if len(text) > 80 else ""}"')
        self.stdout.write("")
