from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    RunContext,
    ToolError,
    cli,
    function_tool,
    inference,
    llm,
    room_io,
    stt,
)
from livekit.plugins import ai_coustics, noise_cancellation, silero
from livekit.plugins import inworld as livekit_inworld
from livekit.plugins import openai as livekit_openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from inworld_stt import InworldSTT
from local_stt import VoskSTT, configured_vosk_model_path, load_vosk_model
from memory import (
    ContextController,
    DisabledContextController,
    call_default_llm_node,
    has_memory_note,
    latest_user_message,
    records_for_tool,
)

logger = logging.getLogger("agent")
data_flow_logger = logging.getLogger("agent.data_flow")

load_dotenv(".env.local")

STT_PROVIDER_LOCAL_VOSK = "local_vosk"
STT_PROVIDER_LIVEKIT_INFERENCE = "livekit_inference"
STT_PROVIDER_INWORLD = "inworld"
TTS_PROVIDER_LIVEKIT_INFERENCE = "livekit_inference"
TTS_PROVIDER_INWORLD = "inworld"
AUDIO_ENHANCEMENT_PROVIDER_NONE = "none"
AUDIO_ENHANCEMENT_PROVIDER_KRISP = "krisp"
AUDIO_ENHANCEMENT_PROVIDER_AI_COUSTICS = "ai_coustics"
LLM_PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
LLM_PROVIDER_LIVEKIT_INFERENCE = "livekit_inference"
DEFAULT_LIVEKIT_INFERENCE_LLM_MODEL = "openai/gpt-5.3-chat-latest"


Pixel = tuple[int, int, int]

OFF: Pixel = (0, 0, 0)
DIM: Pixel = (8, 8, 8)
GREEN: Pixel = (0, 120, 35)
BLUE: Pixel = (0, 40, 140)
CYAN: Pixel = (0, 130, 130)
YELLOW: Pixel = (150, 115, 0)
PURPLE: Pixel = (90, 0, 120)
TEAL: Pixel = (0, 100, 85)
WHITE: Pixel = (120, 120, 120)
RED: Pixel = (150, 0, 0)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _round_sensor_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {key: _round_sensor_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_sensor_value(item) for item in value]
    return value


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


class DataFlowLogger:
    def __init__(
        self,
        *,
        path: str | Path | None = None,
        enabled: bool | None = None,
        max_text_chars: int | None = None,
    ) -> None:
        self._enabled = (
            _env_bool("DATA_FLOW_LOG_ENABLED", True) if enabled is None else enabled
        )
        self._path = Path(path or os.getenv("DATA_FLOW_LOG_PATH", ".data/data_flow.jsonl"))
        self._max_text_chars = max_text_chars or _env_int(
            "DATA_FLOW_LOG_MAX_TEXT_CHARS", 12000
        )

    @property
    def path(self) -> Path:
        return self._path

    def write(self, event: str, **fields: Any) -> None:
        if not self._enabled:
            return
        record = {
            "timestamp": _utc_now_iso(),
            "event": event,
            **self._normalize(fields),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            data_flow_logger.info(
                "Data-flow event logged",
                extra={"event": event, "path": str(self._path)},
            )
        except Exception as exc:
            logger.warning(
                "Data-flow log write failed",
                extra={
                    "event": event,
                    "path": str(self._path),
                    "error_type": type(exc).__name__,
                },
            )

    def _normalize(self, value: Any) -> Any:
        if isinstance(value, str):
            return _truncate_text(value, self._max_text_chars)
        if isinstance(value, dict):
            return {str(key): self._normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize(item) for item in value]
        return value


@dataclass(frozen=True)
class TranscriptCorrectionResult:
    original: str
    deterministic: str
    corrected: str
    channel: str
    llm_model: str | None
    elapsed_ms: float


def configured_stt_provider() -> str:
    return os.getenv("STT_PROVIDER", STT_PROVIDER_LOCAL_VOSK).strip().lower()


def configured_tts_provider() -> str:
    return os.getenv("TTS_PROVIDER", TTS_PROVIDER_LIVEKIT_INFERENCE).strip().lower()


def configured_audio_enhancement_provider() -> str:
    return os.getenv(
        "AUDIO_ENHANCEMENT_PROVIDER", AUDIO_ENHANCEMENT_PROVIDER_NONE
    ).strip().lower()


def turn_detection_enabled() -> bool:
    return _env_bool("TURN_DETECTION_ENABLED", False)


def build_session_stt(proc_userdata: dict[str, Any]) -> Any:
    provider = configured_stt_provider()
    if provider == STT_PROVIDER_INWORLD:
        logger.info(
            "Using Inworld STT",
            extra={
                "provider": provider,
                "model": os.getenv("INWORLD_STT_MODEL", "inworld/inworld-stt-1"),
            },
        )
        return InworldSTT()
    if provider == STT_PROVIDER_LIVEKIT_INFERENCE:
        logger.warning("Using external STT provider", extra={"provider": provider})
        return inference.STT(model="deepgram/nova-3", language="multi")
    if provider != STT_PROVIDER_LOCAL_VOSK:
        raise RuntimeError(
            f"Unsupported STT_PROVIDER={provider!r}. Use inworld, local_vosk, or livekit_inference."
        )
    model = proc_userdata.get("vosk_model")
    if model is None:
        model = load_vosk_model(configured_vosk_model_path())
        proc_userdata["vosk_model"] = model
    logger.info("Using local Vosk STT", extra={"provider": provider})
    return VoskSTT(model_path=configured_vosk_model_path(), model=model)


def _optional_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float env var; ignoring", extra={"env": name, "value": value})
        return None


def build_session_tts() -> Any:
    provider = configured_tts_provider()
    if provider == TTS_PROVIDER_INWORLD:
        model = _required_env_value("INWORLD_TTS_MODEL")
        voice = _required_env_value("INWORLD_TTS_VOICE")
        kwargs: dict[str, Any] = {}
        temperature = _optional_float_env("INWORLD_TTS_TEMPERATURE")
        speaking_rate = _optional_float_env("INWORLD_TTS_SPEAKING_RATE")
        text_normalization = _optional_env_value("INWORLD_TTS_TEXT_NORMALIZATION")
        if temperature is not None:
            kwargs["temperature"] = temperature
        if speaking_rate is not None:
            kwargs["speaking_rate"] = speaking_rate
        if text_normalization:
            kwargs["text_normalization"] = text_normalization
        logger.info(
            "Using Inworld TTS",
            extra={"provider": provider, "model": model, "voice": voice},
        )
        return livekit_inworld.TTS(model=model, voice=voice, **kwargs)
    if provider == TTS_PROVIDER_LIVEKIT_INFERENCE:
        model = _configured_tts_model_for_log()
        voice = os.getenv("TTS_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
        logger.info(
            "Using LiveKit Inference TTS",
            extra={"provider": provider, "model": model, "voice": voice},
        )
        return inference.TTS(model=model, voice=voice)
    raise RuntimeError(
        f"Unsupported TTS_PROVIDER={provider!r}. Use inworld or livekit_inference."
    )


def build_turn_detection() -> MultilingualModel | None:
    if not turn_detection_enabled():
        logger.info("Turn detection disabled", extra={"enabled": False})
        return None
    logger.info("Turn detection enabled", extra={"enabled": True})
    return MultilingualModel()


def build_audio_input_options() -> room_io.AudioInputOptions:
    provider = configured_audio_enhancement_provider()
    logger.info("Audio enhancement selected", extra={"provider": provider})
    if provider == AUDIO_ENHANCEMENT_PROVIDER_NONE:
        return room_io.AudioInputOptions(noise_cancellation=None)
    if provider == AUDIO_ENHANCEMENT_PROVIDER_KRISP:
        return room_io.AudioInputOptions(
            noise_cancellation=lambda params: (
                noise_cancellation.BVCTelephony()
                if params.participant.kind
                == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC()
            )
        )
    if provider == AUDIO_ENHANCEMENT_PROVIDER_AI_COUSTICS:
        return room_io.AudioInputOptions(
            noise_cancellation=ai_coustics.audio_enhancement(
                model=ai_coustics.EnhancerModel.QUAIL_VF_L
            )
        )
    raise RuntimeError(
        "Unsupported AUDIO_ENHANCEMENT_PROVIDER="
        f"{provider!r}. Use none, krisp, or ai_coustics."
    )


def configured_llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", LLM_PROVIDER_OPENAI_COMPATIBLE).strip().lower()


def _required_env_value(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    joined = " or ".join(names)
    raise RuntimeError(f"Missing required environment variable: {joined}")


def _optional_env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _safe_url_for_log(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return url


def _configured_llm_model_for_log() -> str:
    provider = configured_llm_provider()
    if provider == LLM_PROVIDER_LIVEKIT_INFERENCE:
        return os.getenv("LLM_MODEL", DEFAULT_LIVEKIT_INFERENCE_LLM_MODEL)
    return os.getenv("LLM_MODEL", "unknown")


def _configured_tts_model_for_log() -> str:
    if configured_tts_provider() == TTS_PROVIDER_INWORLD:
        return os.getenv("INWORLD_TTS_MODEL", "inworld-tts-1.5-max")
    return os.getenv("TTS_MODEL", "cartesia/sonic-3")


def _chat_context_for_log(chat_ctx: llm.ChatContext) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, item in enumerate(chat_ctx.items):
        item_type = getattr(item, "type", type(item).__name__)
        record: dict[str, Any] = {"index": index, "type": item_type}
        role = getattr(item, "role", None)
        if role is not None:
            record["role"] = role
        text_content = getattr(item, "text_content", None)
        if text_content:
            record["text"] = text_content
        if item_type == "function_call":
            record.update(
                {
                    "name": getattr(item, "name", None),
                    "arguments": getattr(item, "arguments", None),
                    "call_id": getattr(item, "call_id", None),
                }
            )
        elif item_type == "function_call_output":
            record.update(
                {
                    "name": getattr(item, "name", None),
                    "output": getattr(item, "output", None),
                    "call_id": getattr(item, "call_id", None),
                }
            )
        items.append(record)
    return items


def _tools_for_log(tools: list[llm.Tool]) -> list[dict[str, Any]]:
    return [{"id": tool.id} for tool in tools]


def _chunk_text_for_log(chunk: llm.ChatChunk | str | Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.content:
        return chunk.delta.content
    return ""


def _chunk_tool_calls_for_log(chunk: llm.ChatChunk | str | Any) -> list[dict[str, Any]]:
    if not isinstance(chunk, llm.ChatChunk) or not chunk.delta:
        return []
    return [
        {
            "name": tool_call.name,
            "arguments": tool_call.arguments,
            "call_id": tool_call.call_id,
        }
        for tool_call in chunk.delta.tool_calls
    ]


def build_session_llm() -> llm.LLM:
    provider = configured_llm_provider()
    if provider == LLM_PROVIDER_LIVEKIT_INFERENCE:
        model = os.getenv("LLM_MODEL", DEFAULT_LIVEKIT_INFERENCE_LLM_MODEL)
        logger.warning(
            "Using LiveKit Inference LLM",
            extra={"provider": provider, "model": model},
        )
        return inference.LLM(model=model)
    if provider != LLM_PROVIDER_OPENAI_COMPATIBLE:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER={provider!r}. Use openai_compatible or livekit_inference."
        )

    model = _required_env_value("LLM_MODEL")
    api_key = _required_env_value("LLM_API_KEY", "OPENROUTER_API_KEY")
    base_url = _required_env_value("LLM_BASE_URL", "OPENROUTER_BASE_URL")
    temperature = _optional_env_value("LLM_TEMPERATURE")
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        try:
            kwargs["temperature"] = float(temperature)
        except ValueError:
            logger.warning(
                "Invalid LLM_TEMPERATURE; ignoring",
                extra={"value": temperature},
            )

    logger.info(
        "Using OpenAI-compatible LLM",
        extra={
            "provider": provider,
            "model": model,
            "base_url": _safe_url_for_log(base_url),
        },
    )
    return livekit_openai.LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def _solid(color: Pixel) -> list[Pixel]:
    return [color] * 64


def _pattern(rows: list[str], colors: dict[str, Pixel]) -> list[Pixel]:
    return [colors.get(char, OFF) for row in rows for char in row]


STATUS_PATTERNS: dict[str, list[Pixel]] = {
    "ready": _pattern(
        [
            "GGGGGGGG",
            "G......G",
            "G......G",
            "G......G",
            "G......G",
            "G......G",
            "G......G",
            "GGGGGGGG",
        ],
        {"G": GREEN, ".": DIM},
    ),
    "in_room": _solid(BLUE),
    "stt": _pattern(
        [
            "..C..C..",
            "..C..C..",
            ".CC.CC..",
            ".CC.CC..",
            "CCCCCCC.",
            "CCCCCCC.",
            ".CC.CC..",
            "..C..C..",
        ],
        {"C": CYAN},
    ),
    "llm": _pattern(
        [
            "Y.Y.Y.Y.",
            ".Y.Y.Y.Y",
            "Y.Y.Y.Y.",
            ".Y.Y.Y.Y",
            "Y.Y.Y.Y.",
            ".Y.Y.Y.Y",
            "Y.Y.Y.Y.",
            ".Y.Y.Y.Y",
        ],
        {"Y": YELLOW},
    ),
    "tts": _pattern(
        [
            "........",
            "PPPPPPPP",
            "........",
            ".PPPPPP.",
            ".PPPPPP.",
            "........",
            "PPPPPPPP",
            "........",
        ],
        {"P": PURPLE},
    ),
    "memory_insert": _pattern(
        [
            "...G....",
            "...G....",
            ".GGGGG..",
            "...G....",
            "...G....",
            ".GGGGG..",
            ".G...G..",
            ".GGGGG..",
        ],
        {"G": GREEN},
    ),
    "memory_retrieve": _pattern(
        [
            "..TTT...",
            ".T...T..",
            ".T...T..",
            "..TTT...",
            "...T....",
            "....T...",
            ".....T..",
            "......T.",
        ],
        {"T": TEAL},
    ),
    "sensehat_tool": _pattern(
        [
            "W......W",
            "........",
            "...W....",
            "..WWW...",
            "...W....",
            "........",
            "........",
            "W......W",
        ],
        {"W": WHITE},
    ),
    "error": _pattern(
        [
            "R......R",
            ".R....R.",
            "..R..R..",
            "...RR...",
            "...RR...",
            "..R..R..",
            ".R....R.",
            "R......R",
        ],
        {"R": RED},
    ),
}


class StatusDisplay:
    def __init__(self, sense_hat: Any | None = None, *, enabled: bool | None = None) -> None:
        self._sense_hat = sense_hat
        self._enabled = _env_bool("SENSEHAT_STATUS_DISPLAY_ENABLED", True) if enabled is None else enabled
        self._state_stack: list[str] = []
        self._current_state: str | None = None

    @property
    def current_state(self) -> str | None:
        return self._current_state

    @property
    def sense_hat(self) -> Any:
        if self._sense_hat is None:
            from sense_hat import SenseHat

            self._sense_hat = SenseHat()
        return self._sense_hat

    def set_state(self, state: str) -> None:
        if not self._enabled:
            logger.debug("Sense HAT status display disabled", extra={"state": state})
            self._current_state = state
            return

        pattern = STATUS_PATTERNS.get(state)
        if pattern is None:
            logger.warning("Unknown Sense HAT status state", extra={"state": state})
            return

        try:
            sense = self.sense_hat
            if hasattr(sense, "low_light"):
                sense.low_light = True
            sense.set_pixels([list(pixel) for pixel in pattern])
            self._current_state = state
            logger.info("Sense HAT status displayed", extra={"state": state})
        except Exception as exc:
            self._current_state = state
            logger.warning(
                "Sense HAT status display failed",
                extra={"state": state, "error_type": type(exc).__name__},
            )

    def push_state(self, state: str) -> None:
        if self._current_state:
            self._state_stack.append(self._current_state)
        self.set_state(state)

    def pop_state(self) -> None:
        previous = self._state_stack.pop() if self._state_stack else "in_room"
        self.set_state(previous)

    @contextlib.contextmanager
    def showing(self, state: str):
        self.push_state(state)
        try:
            yield
        except Exception:
            self.set_state("error")
            raise
        finally:
            self.pop_state()


DOMAIN_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:sunset|sense\s+had|sense\s+head|sense\s+hot|cents\s+hat|sensor\s+hat)\b", re.IGNORECASE), "Sense HAT"),
    (re.compile(r"\bsensehat\b", re.IGNORECASE), "Sense HAT"),
    (re.compile(r"\braspberry\s+pie\b", re.IGNORECASE), "Raspberry Pi"),
    (re.compile(r"\bair\s+pressure\b", re.IGNORECASE), "air pressure"),
    (re.compile(r"\bbarometric\s+pressure\b", re.IGNORECASE), "barometric pressure"),
    (re.compile(r"\bhue\s+midday\b", re.IGNORECASE), "humidity"),
    (re.compile(r"\bhumid(?: ity)?\b", re.IGNORECASE), "humidity"),
    (re.compile(r"\bexcelerometer\b", re.IGNORECASE), "accelerometer"),
    (re.compile(r"\baccelerator\b", re.IGNORECASE), "accelerometer"),
    (re.compile(r"\bgyro\s+scope\b", re.IGNORECASE), "gyroscope"),
    (re.compile(r"\bjoy\s+stick\b", re.IGNORECASE), "joystick"),
    (re.compile(r"\bled\s+metrics\b", re.IGNORECASE), "LED matrix"),
)

STT_CORRECTION_TRIGGER = re.compile(
    r"\b(sunset|sense|sensor|cents|raspberry|pie|pressure|temperature|humidity|humid|compass|gyro|acceler|joystick|matrix|led|light|brightness|colour|color)\b",
    re.IGNORECASE,
)


class TranscriptCorrector:
    def __init__(
        self,
        *,
        llm_enabled: bool | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._llm_enabled = _env_bool("STT_CORRECTION_LLM_ENABLED", True) if llm_enabled is None else llm_enabled
        self._model = model or os.getenv("STT_CORRECTION_MODEL", "openai/gpt-4.1-mini")
        self._timeout = timeout or float(os.getenv("STT_CORRECTION_TIMEOUT_SECONDS", "2.0"))

    async def correct(self, text: str) -> str:
        result = await self.correct_with_metadata(text)
        return result.corrected

    async def correct_with_metadata(self, text: str) -> TranscriptCorrectionResult:
        started_at = time.perf_counter()
        deterministic = self.correct_deterministic(text)
        channel = "none" if deterministic == text else "deterministic"
        corrected = deterministic

        if self._llm_enabled and STT_CORRECTION_TRIGGER.search(text):
            llm_result = await self._correct_with_llm(deterministic)
            if llm_result:
                corrected = llm_result
                channel = "llm"

        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info(
            "STT transcript correction completed",
            extra={
                "changed": corrected != text,
                "channel": channel,
                "llm_model": self._model if channel == "llm" else None,
                "elapsed_ms": elapsed_ms,
                "original": text,
                "corrected": corrected,
            },
        )
        return TranscriptCorrectionResult(
            original=text,
            deterministic=deterministic,
            corrected=corrected,
            channel=channel,
            llm_model=self._model if channel == "llm" else None,
            elapsed_ms=elapsed_ms,
        )

    def correct_deterministic(self, text: str) -> str:
        corrected = text
        for pattern, replacement in DOMAIN_REPLACEMENTS:
            corrected = pattern.sub(replacement, corrected)
        return corrected

    async def _correct_with_llm(self, text: str) -> str | None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.debug("STT LLM correction skipped: OPENROUTER_API_KEY is not set")
            return None

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                timeout=self._timeout,
            )
            prompt = (
                "Correct obvious speech-to-text mistakes for a Raspberry Pi Sense HAT voice agent. "
                "Preserve the user's meaning, do not add facts, and return only the corrected transcript.\n\n"
                f"Transcript: {text}"
            )
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=80,
                ),
                timeout=self._timeout + 0.5,
            )
            corrected = (response.choices[0].message.content or "").strip()
            return corrected or None
        except Exception as exc:
            logger.warning(
                "STT LLM correction failed; using deterministic correction",
                extra={"model": self._model, "error_type": type(exc).__name__},
            )
            return None


class SenseHatReader:
    def __init__(self, sense_hat: Any | None = None) -> None:
        self._sense_hat = sense_hat

    @property
    def sense_hat(self) -> Any:
        if self._sense_hat is None:
            try:
                from sense_hat import SenseHat
            except ImportError as exc:
                raise RuntimeError(
                    "The Sense HAT Python package is not available. Install it on the Raspberry Pi with `sudo apt install sense-hat`, then run the agent where the package can be imported."
                ) from exc

            self._sense_hat = SenseHat()

        return self._sense_hat

    def environment(self) -> dict[str, Any]:
        sense = self.sense_hat
        return _round_sensor_value(
            {
                "temperature_c": sense.get_temperature(),
                "temperature_from_humidity_c": sense.get_temperature_from_humidity(),
                "temperature_from_pressure_c": sense.get_temperature_from_pressure(),
                "humidity_percent": sense.get_humidity(),
                "pressure_millibars": sense.get_pressure(),
            }
        )

    def motion(self) -> dict[str, Any]:
        sense = self.sense_hat
        return _round_sensor_value(
            {
                "orientation_degrees": sense.get_orientation_degrees(),
                "orientation_radians": sense.get_orientation_radians(),
                "compass_degrees": sense.get_compass(),
                "compass_raw_microteslas": sense.get_compass_raw(),
                "gyroscope_degrees": sense.get_gyroscope(),
                "gyroscope_raw_radians_per_second": sense.get_gyroscope_raw(),
                "accelerometer_degrees": sense.get_accelerometer(),
                "accelerometer_raw_g": sense.get_accelerometer_raw(),
            }
        )

    def joystick(self) -> dict[str, Any]:
        events = self.sense_hat.stick.get_events()
        return {
            "events": [
                {
                    "timestamp": event.timestamp,
                    "direction": event.direction,
                    "action": event.action,
                }
                for event in events
            ]
        }

    def light(self) -> dict[str, Any]:
        sense = self.sense_hat
        colour_sensor = getattr(sense, "colour", None) or getattr(sense, "color", None)
        if colour_sensor is None:
            return {
                "available": False,
                "reason": "This Sense HAT does not expose a colour and brightness sensor through the Python API.",
            }

        red, green, blue, clear = colour_sensor.colour
        colour_raw = getattr(colour_sensor, "colour_raw", None)
        rgb_raw = getattr(colour_sensor, "rgb", None)
        return {
            "available": True,
            "red": red,
            "green": green,
            "blue": blue,
            "clear_brightness": clear,
            "brightness_raw": getattr(colour_sensor, "brightness", None),
            "colour_raw": list(colour_raw) if colour_raw is not None else None,
            "rgb_raw": list(rgb_raw) if rgb_raw is not None else None,
            "gain": getattr(colour_sensor, "gain", None),
            "integration_cycles": getattr(colour_sensor, "integration_cycles", None),
            "integration_time_seconds": getattr(
                colour_sensor, "integration_time", None
            ),
        }

    def display(self) -> dict[str, Any]:
        sense = self.sense_hat
        return {
            "rotation_degrees": sense.rotation,
            "low_light": sense.low_light,
            "gamma": list(sense.gamma),
            "pixels": sense.get_pixels(),
        }

    def all(self) -> dict[str, Any]:
        return {
            "environment": self.environment(),
            "motion": self.motion(),
            "joystick": self.joystick(),
            "light": self.light(),
            "display": self.display(),
        }


class Assistant(Agent):
    def __init__(
        self,
        sense_hat_reader: SenseHatReader | None = None,
        context_controller: Any | None = None,
        status_display: StatusDisplay | None = None,
        transcript_corrector: TranscriptCorrector | None = None,
        data_flow: DataFlowLogger | None = None,
    ) -> None:
        self._sense_hat_reader = sense_hat_reader or SenseHatReader()
        self._context_controller = context_controller or DisabledContextController()
        self._status_display = status_display or StatusDisplay(enabled=False)
        self._transcript_corrector = transcript_corrector or TranscriptCorrector()
        self._data_flow = data_flow or DataFlowLogger()
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            When the user asks about the Raspberry Pi Sense HAT, sensor readings, local temperature, humidity, pressure, orientation, movement, compass direction, light, brightness, colour, joystick input, or LED display state, use the Sense HAT tools before answering.
            Do not answer questions about current sensor values from memory or general knowledge. Current Sense HAT readings require a Sense HAT tool call.
            When the user asks you to remember something, or asks what you remember, use the memory tools and answer plainly.
            Sensor tool values are live readings from the device. State the units clearly and keep the answer short enough for voice.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )
        logger.info(
            "Assistant initialized",
            extra={"tool_ids": ",".join(tool.id for tool in self.tools)},
        )

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        text = new_message.text_content
        logger.info(
            "User turn completed",
            extra={"chars": len(text or ""), "memory_enabled": bool(text)},
        )
        await self._context_controller.observe_user_turn(
            text, turn_id=getattr(new_message, "id", None)
        )
        build_memory_note = getattr(self._context_controller, "build_memory_note", None)
        if text and build_memory_note and not has_memory_note(turn_ctx):
            with self._status_display.showing("memory_retrieve"):
                note = await build_memory_note(text)
                if note:
                    turn_ctx.add_message(role="assistant", content=note)
                    logger.info("Injected memory context in user-turn hook")

    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncIterable[stt.SpeechEvent | str]:
        logger.info("STT node starting")
        self._status_display.push_state("stt")
        try:
            async for event in Agent.default.stt_node(self, audio, model_settings):
                yield await self._correct_stt_event(event)
        finally:
            self._status_display.pop_state()
            logger.info("STT node finished")

    async def _correct_stt_event(
        self, event: stt.SpeechEvent | str
    ) -> stt.SpeechEvent | str:
        if isinstance(event, str):
            result = await self._transcript_corrector.correct_with_metadata(event)
            self._data_flow.write(
                "stt_correction",
                original_stt=result.original,
                stt_correction_channel=result.channel,
                deterministic_stt=result.deterministic,
                corrected_stt=result.corrected,
                correction_llm_model=result.llm_model,
                elapsed_ms=result.elapsed_ms,
            )
            return result.corrected
        if event.type != stt.SpeechEventType.FINAL_TRANSCRIPT or not event.alternatives:
            return event

        original = event.alternatives[0].text
        result = await self._transcript_corrector.correct_with_metadata(original)
        self._data_flow.write(
            "stt_correction",
            original_stt=result.original,
            stt_correction_channel=result.channel,
            deterministic_stt=result.deterministic,
            corrected_stt=result.corrected,
            correction_llm_model=result.llm_model,
            elapsed_ms=result.elapsed_ms,
        )
        corrected = result.corrected
        if corrected != original:
            event.alternatives[0].text = corrected
            logger.info(
                "Final STT transcript corrected",
                extra={
                    "original": original,
                    "corrected": corrected,
                    "channel": result.channel,
                },
            )
        return event

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: Any,
    ) -> AsyncIterable[llm.ChatChunk | str | Any]:
        with self._status_display.showing("llm"):
            user_message = latest_user_message(chat_ctx)
            latest_text = user_message.text_content if user_message is not None else None
            await self._context_controller.observe_user_turn(
                latest_text, turn_id=getattr(user_message, "id", None)
            )
            logger.info(
                "LLM node starting",
                extra={
                    "tools": ",".join(tool.id for tool in tools),
                    "latest_user_chars": len(latest_text or ""),
                    "context_items": len(chat_ctx.items),
                },
            )
            controlled_ctx = await self._context_controller.prepare_llm_context(
                chat_ctx,
                latest_user_text=latest_text,
            )
            llm_model = _configured_llm_model_for_log()
            self._data_flow.write(
                "llm_input",
                llm_provider=configured_llm_provider(),
                llm_model=llm_model,
                tools=_tools_for_log(tools),
                context_items=_chat_context_for_log(controlled_ctx),
            )
            stream = await call_default_llm_node(self, controlled_ctx, tools, model_settings)
            output_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            started_at = time.perf_counter()
            async for chunk in stream:
                output_parts.append(_chunk_text_for_log(chunk))
                tool_calls.extend(_chunk_tool_calls_for_log(chunk))
                yield chunk
            self._data_flow.write(
                "llm_output",
                llm_provider=configured_llm_provider(),
                llm_model=llm_model,
                text="".join(output_parts).strip(),
                tool_calls=tool_calls,
                elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable[rtc.AudioFrame]:
        logger.info("TTS node starting")
        self._status_display.push_state("tts")
        tts_parts: list[str] = []

        async def logged_text() -> AsyncIterable[str]:
            async for chunk in text:
                chunk_text = _coerce_text(chunk)
                tts_parts.append(chunk_text)
                yield chunk

        try:
            started_at = time.perf_counter()
            async for frame in Agent.default.tts_node(self, logged_text(), model_settings):
                yield frame
            self._data_flow.write(
                "tts_input",
                tts_provider=configured_tts_provider(),
                tts_model=_configured_tts_model_for_log(),
                text="".join(tts_parts).strip(),
                elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
        finally:
            self._status_display.pop_state()
            logger.info("TTS node finished")

    def _read_sense_hat(self, category: str) -> dict[str, Any]:
        try:
            with self._status_display.showing("sensehat_tool"):
                logger.info("Sense HAT tool read started", extra={"category": category})
                reader_method = getattr(self._sense_hat_reader, category)
                result = reader_method()
                logger.info(
                    "Sense HAT tool read completed",
                    extra={
                        "category": category,
                        "keys": ",".join(result.keys()) if isinstance(result, dict) else "",
                    },
                )
                return result
        except Exception as exc:
            logger.exception("Unable to read Sense HAT %s data", category)
            raise ToolError(
                "I could not read the Sense HAT right now. Check that the Sense HAT is attached, enabled, and that the `sense_hat` Python package is available to this process."
            ) from exc

    @function_tool()
    async def get_sensehat_environment(self, context: RunContext) -> dict[str, Any]:
        """Read the Raspberry Pi Sense HAT environmental sensors.

        Use this when the user asks for local temperature, humidity, air pressure, barometric pressure, or general room conditions from the Sense HAT. Returns Celsius, relative humidity percent, and millibars.
        """

        return self._read_sense_hat("environment")

    @function_tool()
    async def get_sensehat_motion(self, context: RunContext) -> dict[str, Any]:
        """Read the Raspberry Pi Sense HAT IMU and compass sensors.

        Use this when the user asks for orientation, pitch, roll, yaw, compass heading, north direction, gyroscope, accelerometer, magnetometer, movement, tilt, or rotation from the Sense HAT.
        """

        return self._read_sense_hat("motion")

    @function_tool()
    async def get_sensehat_joystick(self, context: RunContext) -> dict[str, Any]:
        """Read pending Raspberry Pi Sense HAT joystick events.

        Use this when the user asks whether the Sense HAT joystick has been pressed, held, released, or moved up, down, left, right, or middle. Returns events since the last joystick read.
        """

        return self._read_sense_hat("joystick")

    @function_tool()
    async def get_sensehat_light(self, context: RunContext) -> dict[str, Any]:
        """Read the Raspberry Pi Sense HAT v2 colour and brightness sensor.

        Use this when the user asks for colour, color, red, green, blue, clear light, brightness, ambient light, light level, or raw colour sensor values from the Sense HAT. Returns unavailable if the attached Sense HAT does not have this v2 sensor.
        """

        return self._read_sense_hat("light")

    @function_tool()
    async def get_sensehat_display(self, context: RunContext) -> dict[str, Any]:
        """Read the Raspberry Pi Sense HAT LED matrix display state.

        Use this when the user asks what is currently shown on the LED matrix, the display rotation, low light mode, gamma table, or pixel colors.
        """

        return self._read_sense_hat("display")

    @function_tool()
    async def get_sensehat_snapshot(self, context: RunContext) -> dict[str, Any]:
        """Read all available Raspberry Pi Sense HAT information in one snapshot.

        Use this when the user asks for everything the Sense HAT can provide or asks for a full status report. Includes environmental readings, IMU and compass readings, joystick events, colour and brightness readings when available, and LED matrix state.
        """

        return self._read_sense_hat("all")

    @function_tool()
    async def search_memory(self, context: RunContext, query: str) -> list[dict[str, Any]]:
        """Search the agent's long-run memory.

        Use this when the user asks what you remember about a topic, asks you to recall a saved preference, or asks whether something is stored in memory.

        Args:
            query: A short phrase describing what to look for in memory.
        """

        logger.info("Memory search tool called", extra={"query_chars": len(query)})
        records = await self._context_controller.search(query)
        return records_for_tool(records)

    @function_tool()
    async def list_memories(
        self, context: RunContext, limit: int = 10
    ) -> list[dict[str, Any]]:
        """List the most important long-run memories currently stored.

        Use this when the user asks what you remember, asks to inspect memory, or asks for a memory list.

        Args:
            limit: Maximum number of memories to return. Use 10 unless the user requests a different count.
        """

        logger.info("Memory list tool called", extra={"limit": limit})
        records = self._context_controller.list_memories(limit=limit)
        return records_for_tool(records)

    @function_tool()
    async def forget_memory(self, context: RunContext, query: str) -> dict[str, Any]:
        """Forget the best matching long-run memory.

        Use this when the user asks you to forget a specific remembered fact or preference.

        Args:
            query: A short phrase describing the memory to forget.
        """

        logger.warning("Memory forget tool called", extra={"query_chars": len(query)})
        record = await self._context_controller.forget_memory(query)
        if record is None:
            return {"deleted": False, "message": "No matching memory was found."}
        return {"deleted": True, "memory": records_for_tool([record])[0]}

    @function_tool()
    async def forget_all_memories(
        self, context: RunContext, confirm: bool
    ) -> dict[str, Any]:
        """Forget all long-run memories after explicit confirmation.

        Use this only when the user clearly asks you to forget or delete all memories. Require confirmation before setting confirm to true.

        Args:
            confirm: True only after the user explicitly confirms deleting all memories.
        """

        logger.warning("Forget all memories tool called", extra={"confirm": confirm})
        if not confirm:
            return {
                "deleted": 0,
                "message": "Confirmation is required before deleting all memories.",
            }
        deleted = self._context_controller.forget_all_memories()
        return {"deleted": deleted}


server = AgentServer(num_idle_processes=1)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    if configured_stt_provider() == STT_PROVIDER_LOCAL_VOSK:
        proc.userdata["vosk_model"] = load_vosk_model(configured_vosk_model_path())
    status_display = StatusDisplay()
    status_display.set_state("ready")
    proc.userdata["status_display"] = status_display


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    turn_detection = build_turn_detection()
    session_kwargs: dict[str, Any] = {}
    if turn_detection is not None:
        session_kwargs["turn_detection"] = turn_detection

    # Set up a voice AI pipeline using the configured STT, LLM, and TTS providers.
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=build_session_stt(ctx.proc.userdata),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=build_session_llm(),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=build_session_tts(),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
        **session_kwargs,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    status_display = ctx.proc.userdata.get("status_display") or StatusDisplay()
    status_display.set_state("ready")
    context_controller = ContextController(status_reporter=status_display)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(
            context_controller=context_controller,
            status_display=status_display,
        ),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=build_audio_input_options(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()
    status_display.set_state("in_room")


if __name__ == "__main__":
    cli.run_app(server)
