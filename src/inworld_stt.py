from __future__ import annotations

import base64
import io
import logging
import os
import time
import wave
from typing import Any
from urllib.parse import urljoin

import aiohttp
from livekit.agents import stt, utils
from livekit.agents._exceptions import APIConnectionError, APIStatusError, APITimeoutError
from livekit.agents.types import APIConnectOptions, NOT_GIVEN, NotGivenOr

from local_stt import _frame_to_pcm16_mono

logger = logging.getLogger("agent.inworld_stt")

DEFAULT_INWORLD_STT_BASE_URL = "https://api.inworld.ai/"
DEFAULT_INWORLD_STT_MODEL = "inworld/inworld-stt-1"
DEFAULT_INWORLD_STT_LANGUAGE = "en-US"
DEFAULT_INWORLD_STT_SAMPLE_RATE = 16000
DEFAULT_INWORLD_STT_AUDIO_ENCODING = "AUTO_DETECT"


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


def configured_inworld_stt_model() -> str:
    return os.getenv("INWORLD_STT_MODEL", DEFAULT_INWORLD_STT_MODEL)


def configured_inworld_stt_language() -> str:
    return os.getenv("INWORLD_STT_LANGUAGE", DEFAULT_INWORLD_STT_LANGUAGE)


def configured_inworld_stt_sample_rate() -> int:
    return _env_int("INWORLD_STT_SAMPLE_RATE", DEFAULT_INWORLD_STT_SAMPLE_RATE)


def configured_inworld_stt_base_url() -> str:
    return os.getenv("INWORLD_STT_BASE_URL", DEFAULT_INWORLD_STT_BASE_URL)


def configured_inworld_stt_audio_encoding() -> str:
    return os.getenv("INWORLD_STT_AUDIO_ENCODING", DEFAULT_INWORLD_STT_AUDIO_ENCODING)


class InworldSTT(stt.STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        language: str | None = None,
        sample_rate: int | None = None,
        audio_encoding: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                diarization=False,
                offline_recognize=True,
            )
        )
        self._api_key = api_key or os.getenv("INWORLD_API_KEY")
        if not self._api_key:
            raise ValueError(
                "INWORLD_API_KEY is required when STT_PROVIDER=inworld"
            )
        self._model = model or configured_inworld_stt_model()
        self._language = language or configured_inworld_stt_language()
        self._sample_rate = sample_rate or configured_inworld_stt_sample_rate()
        self._audio_encoding = audio_encoding or configured_inworld_stt_audio_encoding()
        self._base_url = base_url or configured_inworld_stt_base_url()
        self._session = http_session

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "inworld"

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
        wav = _pcm16_mono_to_wav(pcm, sample_rate=self._sample_rate)
        request_language = language if language is not NOT_GIVEN else self._language
        payload = self._build_payload(wav, language=str(request_language))
        timeout = aiohttp.ClientTimeout(total=conn_options.timeout)
        try:
            async with self._ensure_session().post(
                urljoin(self._base_url, "/stt/v1/transcribe"),
                headers={
                    "Authorization": f"Basic {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            ) as response:
                if not response.ok:
                    body = await response.text()
                    raise APIStatusError(
                        message=body,
                        status_code=response.status,
                        request_id=None,
                        body=body,
                    )
                data = await response.json()
        except TimeoutError as exc:
            raise APITimeoutError("Inworld STT request timed out") from exc
        except APIStatusError:
            raise
        except Exception as exc:
            raise APIConnectionError("Inworld STT request failed") from exc

        text = _extract_transcript(data)
        confidence = _extract_confidence(data)
        logger.info(
            "Inworld STT recognition completed",
            extra={
                "model": self._model,
                "chars": len(text),
                "confidence": round(confidence, 3),
                "input_sample_rate": frame.sample_rate,
                "input_channels": frame.num_channels,
                "target_sample_rate": self._sample_rate,
                "audio_encoding": self._audio_encoding,
                "audio_container": "wav",
                "audio_bytes": len(wav),
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
        )
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=str(request_language),
                    text=text,
                    confidence=confidence,
                )
            ],
        )

    def _build_payload(self, wav: bytes, *, language: str) -> dict[str, Any]:
        return {
            "transcribeConfig": {
                "modelId": self._model,
                "audioEncoding": self._audio_encoding,
                "language": language,
                "sampleRateHertz": self._sample_rate,
                "numberOfChannels": 1,
            },
            "audioData": {
                "content": base64.b64encode(wav).decode("ascii"),
            },
        }

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session


def _extract_transcript(data: dict[str, Any]) -> str:
    transcription = data.get("transcription")
    if isinstance(transcription, dict):
        transcript = transcription.get("transcript") or transcription.get("text")
        if isinstance(transcript, str):
            return transcript.strip()
    result = data.get("result")
    if isinstance(result, dict):
        return _extract_transcript(result)
    return ""


def _extract_confidence(data: dict[str, Any]) -> float:
    transcription = data.get("transcription")
    if not isinstance(transcription, dict):
        result = data.get("result")
        if isinstance(result, dict):
            return _extract_confidence(result)
        return 0.0

    for key in ("confidence", "transcriptConfidence"):
        value = transcription.get(key)
        if isinstance(value, int | float):
            return float(value)

    timestamps = transcription.get("wordTimestamps") or transcription.get("word_timestamps")
    if not isinstance(timestamps, list) or not timestamps:
        return 0.0
    confidences = [
        float(item["confidence"])
        for item in timestamps
        if isinstance(item, dict) and isinstance(item.get("confidence"), int | float)
    ]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def _pcm16_mono_to_wav(pcm: bytes, *, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()
