from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import APIConnectOptions, NOT_GIVEN, NotGivenOr

logger = logging.getLogger("agent.local_stt")

DEFAULT_VOSK_MODEL_PATH = ".models/vosk-model-small-en-us-0.15"
DEFAULT_VOSK_SAMPLE_RATE = 16000
DEFAULT_VOSK_LANGUAGE = "en"
DEFAULT_VOSK_GRAMMAR_TERMS = (
    "Sense HAT",
    "sensehat",
    "Raspberry Pi",
    "temperature",
    "humidity",
    "pressure",
    "air pressure",
    "barometric pressure",
    "compass",
    "gyroscope",
    "accelerometer",
    "joystick",
    "LED matrix",
    "light",
    "brightness",
    "color",
    "colour",
    "[unk]",
)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(
            "Invalid integer env var; using default",
            extra={"env": name, "value": value, "default": default},
        )
        return default


def _env_terms(name: str, default: Sequence[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    terms = [term.strip() for term in raw.split(",") if term.strip()]
    if "[unk]" not in terms:
        terms.append("[unk]")
    return terms or list(default)


def configured_vosk_model_path() -> Path:
    return Path(os.getenv("VOSK_MODEL_PATH", DEFAULT_VOSK_MODEL_PATH)).expanduser()


def configured_vosk_sample_rate() -> int:
    return _env_int("VOSK_SAMPLE_RATE", DEFAULT_VOSK_SAMPLE_RATE)


def configured_vosk_language() -> str:
    return os.getenv("VOSK_LANGUAGE", DEFAULT_VOSK_LANGUAGE)


def configured_vosk_grammar_terms() -> list[str]:
    return _env_terms("VOSK_GRAMMAR_TERMS", DEFAULT_VOSK_GRAMMAR_TERMS)


def load_vosk_model(model_path: str | Path | None = None) -> Any:
    resolved = Path(model_path or configured_vosk_model_path())
    if not resolved.exists():
        raise RuntimeError(
            f"Vosk model path does not exist: {resolved}. Download vosk-model-small-en-us-0.15 and set VOSK_MODEL_PATH."
        )
    try:
        from vosk import Model
    except ImportError as exc:
        raise RuntimeError(
            "The vosk package is not installed. Install it with `uv add vosk==0.3.45` or `pip install vosk==0.3.45`."
        ) from exc

    started_at = time.perf_counter()
    model = Model(str(resolved))
    logger.info(
        "Local Vosk model loaded",
        extra={
            "model_path": str(resolved),
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
        },
    )
    return model


class VoskSTT(stt.STT):
    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        model: Any | None = None,
        sample_rate: int | None = None,
        language: str | None = None,
        grammar_terms: Sequence[str] | None = None,
        model_factory: Callable[[str], Any] | None = None,
        recognizer_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                diarization=False,
                offline_recognize=True,
            )
        )
        self._model_path = Path(model_path or configured_vosk_model_path())
        self._model = model
        self._sample_rate = sample_rate or configured_vosk_sample_rate()
        self._language = language or configured_vosk_language()
        self._grammar_terms = list(grammar_terms or configured_vosk_grammar_terms())
        self._model_factory = model_factory
        self._recognizer_factory = recognizer_factory

    @property
    def model(self) -> str:
        return self._model_path.name

    @property
    def provider(self) -> str:
        return "local_vosk"

    def prewarm(self) -> None:
        self._get_model()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        started_at = time.perf_counter()
        frame = utils.merge_frames(buffer)
        pcm = _frame_to_pcm16_mono(frame, sample_rate=self._sample_rate)
        recognizer = self._new_recognizer()
        recognizer.AcceptWaveform(pcm)
        result = _parse_vosk_json(getattr(recognizer, "FinalResult", recognizer.Result)())
        text = result.get("text", "").strip()
        confidence = _confidence(result)
        logger.info(
            "Local Vosk recognition completed",
            extra={
                "chars": len(text),
                "confidence": round(confidence, 3),
                "input_sample_rate": frame.sample_rate,
                "input_channels": frame.num_channels,
                "target_sample_rate": self._sample_rate,
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
        )
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=language if language is not NOT_GIVEN else self._language,
                    text=text,
                    confidence=confidence,
                )
            ],
        )

    def _get_model(self) -> Any:
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(str(self._model_path))
            else:
                self._model = load_vosk_model(self._model_path)
        return self._model

    def _new_recognizer(self) -> Any:
        if self._recognizer_factory is None:
            from vosk import KaldiRecognizer

            factory = KaldiRecognizer
        else:
            factory = self._recognizer_factory

        model = self._get_model()
        if self._grammar_terms:
            return factory(model, self._sample_rate, json.dumps(self._grammar_terms))
        return factory(model, self._sample_rate)


def _frame_to_pcm16_mono(frame: rtc.AudioFrame, *, sample_rate: int) -> bytes:
    frames: list[rtc.AudioFrame]
    if frame.sample_rate != sample_rate:
        resampler = rtc.AudioResampler(
            frame.sample_rate,
            sample_rate,
            quality=rtc.AudioResamplerQuality.HIGH,
        )
        frames = list(resampler.push(frame))
        frames.extend(resampler.flush())
    else:
        frames = [frame]

    chunks: list[bytes] = []
    for item in frames:
        samples = np.frombuffer(bytes(item.data), dtype=np.int16)
        if item.num_channels > 1:
            samples = samples.reshape(-1, item.num_channels).mean(axis=1).astype(np.int16)
        chunks.append(samples.tobytes())
    return b"".join(chunks)


def _parse_vosk_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Vosk returned invalid JSON", extra={"raw": raw[:120]})
        return {"text": ""}
    if not isinstance(parsed, dict):
        return {"text": ""}
    return parsed


def _confidence(result: dict[str, Any]) -> float:
    words = result.get("result")
    if not isinstance(words, list) or not words:
        return 0.0
    confidences = [
        float(word["conf"])
        for word in words
        if isinstance(word, dict) and isinstance(word.get("conf"), int | float)
    ]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)
