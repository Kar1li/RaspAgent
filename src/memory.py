from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
import uuid
from collections.abc import AsyncIterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from livekit.agents import Agent, ModelSettings, llm

logger = logging.getLogger("agent.memory")

DEFAULT_EMBEDDING_MODELS = (
    "openai/text-embedding-3-small",
    "qwen/qwen3-embedding-8b",
)
STATUS_NOTE_PREFIX = "Relevant current status for this turn."
MEMORY_NOTE_PREFIX = STATUS_NOTE_PREFIX
DEFAULT_QUIET_HOURS_START = "22:00"
DEFAULT_QUIET_HOURS_END = "08:00"
KNOWN_REMINDER_TYPES = ("drink_water", "nap", "check_in")

SENSITIVE_PATTERN = re.compile(
    r"\b(password|passcode|api[-_ ]?key|secret|token|credential|ssn|social security|credit card|bank account|medical record|diagnosis|private key)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MemoryCandidate:
    text: str
    kind: str
    importance: float
    source_text: str


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    text: str
    kind: str
    importance: float
    source_text: str
    created_at: float
    updated_at: float
    embedding_model: str | None = None
    embedding_dimension: int | None = None


@dataclass(frozen=True)
class EmbeddingResult:
    model: str
    vector: list[float]
    used_fallback: bool


@dataclass
class ReminderPolicy:
    reminder_type: str
    enabled: bool
    interval_minutes: int | None = None
    window_start: str | None = None
    window_end: str | None = None
    max_per_day: int = 0
    updated_at: float = 0.0


@dataclass
class UserProfile:
    preferred_name: str | None = None
    timezone: str = "UTC"
    quiet_hours_start: str | None = DEFAULT_QUIET_HOURS_START
    quiet_hours_end: str | None = DEFAULT_QUIET_HOURS_END
    updated_at: float = 0.0


@dataclass
class RuntimeStatus:
    current_activity: str | None = None
    activity_expires_at: float | None = None
    busy_state: str | None = None
    busy_expires_at: float | None = None
    summary_entries: list[str] = field(default_factory=list)
    last_user_interaction_at: float | None = None
    last_assistant_speech_at: float | None = None
    last_water_reminder_at: float | None = None
    last_check_in_at: float | None = None
    last_nap_suggestion_at: float | None = None
    water_cooldown_until: float | None = None
    check_in_cooldown_until: float | None = None
    nap_cooldown_until: float | None = None
    posture_monitoring_active: bool = False
    posture_workflow_active: bool = False
    posture_session_id: str | None = None
    latest_posture_label: str | None = None
    latest_posture_severity: str | None = None
    latest_posture_reason_codes: list[str] = field(default_factory=list)
    latest_posture_metrics: dict[str, float] = field(default_factory=dict)
    latest_posture_prompt_key: str | None = None
    last_posture_event_at: float | None = None
    posture_issue_cooldowns: dict[str, float] = field(default_factory=dict)
    posture_preview_enabled: bool = False
    posture_preview_active: bool = False
    last_posture_callback_at: float | None = None
    last_posture_callback_event: str | None = None
    posture_reminder_cooldown_seconds: float = 60.0
    updated_at: float = 0.0

    @property
    def recent_summary(self) -> str:
        return " ".join(entry.strip() for entry in self.summary_entries if entry.strip()).strip()


@dataclass
class DailyStatus:
    day: str
    water_reminders_sent: int = 0
    water_acknowledged: int = 0
    water_snoozed: int = 0
    water_dismissed: int = 0
    nap_suggestions_sent: int = 0
    check_ins_sent: int = 0
    last_event_at: float | None = None
    updated_at: float = 0.0


@dataclass(frozen=True)
class ProactiveAction:
    reminder_type: str
    mode: str
    text: str | None = None
    instructions: str | None = None


@dataclass
class StructuredUpdateResult:
    profile_changed: bool = False
    status_changed: bool = False
    policies_changed: bool = False
    explicit_memories: list[MemoryRecord] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.profile_changed or self.status_changed or self.policies_changed


class Embedder(Protocol):
    async def embed(self, text: str) -> EmbeddingResult: ...


class StatusReporter(Protocol):
    def push_state(self, state: str) -> None: ...
    def pop_state(self) -> None: ...


class EmbeddingUnavailableError(RuntimeError):
    pass


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


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(
            "Invalid float env var; using default",
            extra={"env": name, "value": value, "default": default},
        )
        return default


def _configured_embedding_models() -> list[str]:
    raw = os.getenv("MEMORY_EMBEDDING_MODELS")
    if not raw:
        return list(DEFAULT_EMBEDDING_MODELS)
    models = [model.strip() for model in raw.split(",") if model.strip()]
    return models or list(DEFAULT_EMBEDDING_MODELS)


def _safe_snippet(text: str, *, limit: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _table_suffix(model: str, dimension: int) -> str:
    digest = hashlib.sha256(f"{model}:{dimension}".encode()).hexdigest()[:16]
    return f"{digest}_{dimension}"


def _fts_query(text: str) -> str:
    terms = re.findall(r"[A-Za-z0-9_]+", text.lower())
    if not terms:
        return '""'
    return " OR ".join(f'"{term}"' for term in terms[:12])


def _zoneinfo(name: str | None) -> ZoneInfo:
    if not name or name.upper() == "UTC":
        return UTC  # type: ignore[return-value]
    try:
        return ZoneInfo(name)
    except Exception:
        logger.warning("Invalid timezone; using UTC", extra={"timezone": name})
        return UTC  # type: ignore[return-value]


def _now_in_timezone(name: str | None, *, now: datetime | None = None) -> datetime:
    base = now or datetime.now(UTC)
    if base.tzinfo is None:
        base = base.replace(tzinfo=UTC)
    return base.astimezone(_zoneinfo(name))


def _iso_day(name: str | None, *, now: datetime | None = None) -> str:
    return _now_in_timezone(name, now=now).date().isoformat()


def _parse_hhmm(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", value.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _format_hhmm(hour: int, minute: int = 0) -> str:
    return f"{hour:02d}:{minute:02d}"


def _parse_time_phrase(value: str) -> str | None:
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", value.strip(), re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    meridiem = (match.group(3) or "").lower()
    if meridiem:
        if hour == 12:
            hour = 0
        if meridiem == "pm":
            hour += 12
    hour = max(0, min(hour, 23))
    minute = max(0, min(minute, 59))
    return _format_hhmm(hour, minute)


def _time_in_window(now_local: datetime, start: str | None, end: str | None) -> bool:
    parsed_start = _parse_hhmm(start)
    parsed_end = _parse_hhmm(end)
    if parsed_start is None or parsed_end is None:
        return True
    current = now_local.hour * 60 + now_local.minute
    start_minutes = parsed_start[0] * 60 + parsed_start[1]
    end_minutes = parsed_end[0] * 60 + parsed_end[1]
    if start_minutes == end_minutes:
        return True
    if start_minutes < end_minutes:
        return start_minutes <= current < end_minutes
    return current >= start_minutes or current < end_minutes


def _in_quiet_hours(profile: UserProfile, *, now: datetime | None = None) -> bool:
    now_local = _now_in_timezone(profile.timezone, now=now)
    start = profile.quiet_hours_start
    end = profile.quiet_hours_end
    if not start or not end:
        return False
    return _time_in_window(now_local, start, end)


def _normalize_summary_entries(entries: list[str], *, max_entries: int, max_chars: int) -> list[str]:
    kept: list[str] = []
    total = 0
    for entry in entries:
        snippet = _safe_snippet(entry, limit=140)
        if not snippet:
            continue
        if kept and kept[-1] == snippet:
            continue
        if total + len(snippet) > max_chars and kept:
            break
        kept.append(snippet)
        total += len(snippet) + 1
        if len(kept) >= max_entries:
            break
    return kept


def _normalize_optional_scalar(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.casefold() in {"none", "null"}:
            return None
        return stripped
    return value


def _normalize_reminder_type(value: Any) -> str:
    normalized = _normalize_optional_scalar(value)
    if normalized is None:
        raise ValueError("reminder_type is required")
    key = str(normalized).strip().casefold().replace("-", " ").replace("_", " ")
    aliases = {
        "drink water": "drink_water",
        "water": "drink_water",
        "drinkwater": "drink_water",
        "drink_water": "drink_water",
        "nap": "nap",
        "check in": "check_in",
        "checkin": "check_in",
        "check_in": "check_in",
    }
    resolved = aliases.get(key)
    if resolved not in KNOWN_REMINDER_TYPES:
        raise ValueError(
            "reminder_type must be one of drink_water, nap, or check_in"
        )
    return resolved


def _normalize_optional_int(
    value: Any,
    *,
    field_name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    normalized = _normalize_optional_scalar(value)
    if normalized is None:
        return None
    if isinstance(normalized, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(normalized, int):
        amount = normalized
    elif isinstance(normalized, str) and re.fullmatch(r"-?\d+", normalized):
        amount = int(normalized)
    else:
        raise ValueError(f"{field_name} must be an integer")
    if minimum is not None and amount < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    if maximum is not None and amount > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}")
    return amount


def _normalize_optional_bool(value: Any, *, field_name: str) -> bool | None:
    normalized = _normalize_optional_scalar(value)
    if normalized is None:
        return None
    if isinstance(normalized, bool):
        return normalized
    if isinstance(normalized, str):
        lowered = normalized.casefold()
        if lowered in {"true", "yes", "on", "1"}:
            return True
        if lowered in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"{field_name} must be true or false")


def _normalize_optional_time(value: Any, *, field_name: str) -> str | None:
    normalized = _normalize_optional_scalar(value)
    if normalized is None:
        return None
    parsed = _parse_time_phrase(str(normalized))
    if parsed is None:
        raise ValueError(f"{field_name} must be in HH:MM or am/pm format")
    return parsed


def _normalize_optional_timezone(value: Any, *, field_name: str) -> str | None:
    normalized = _normalize_optional_scalar(value)
    if normalized is None:
        return None
    candidate = str(normalized)
    if candidate.upper() == "UTC":
        return "UTC"
    try:
        ZoneInfo(candidate)
    except Exception as exc:
        raise ValueError(f"{field_name} must be a valid IANA timezone like UTC or Asia/Shanghai") from exc
    return candidate


def _normalize_optional_name(value: Any, *, field_name: str) -> str | None:
    normalized = _normalize_optional_scalar(value)
    if normalized is None:
        return None
    candidate = " ".join(str(normalized).split()).strip(" ,.")
    if not candidate:
        return None
    if len(candidate) > 80:
        raise ValueError(f"{field_name} must be 80 characters or fewer")
    return candidate


def _load_json_list(value: Any) -> list[str]:
    try:
        loaded = json.loads(value or "[]")
    except Exception:
        return []
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded if str(item).strip()]


def _load_json_float_dict(value: Any) -> dict[str, float]:
    try:
        loaded = json.loads(value or "{}")
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in loaded.items():
        try:
            result[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return result


def _normalize_posture_issue_key(
    *,
    reason_codes: list[str],
    prompt_key: str | None,
    posture_label: str | None,
) -> str:
    normalized_codes = {str(code).strip().replace("_", " ").lower() for code in reason_codes}
    if "forward head" in normalized_codes or "neck" in normalized_codes:
        return "neck_alignment"
    if "rounded back" in normalized_codes or "trunk lean" in normalized_codes or "back" in normalized_codes:
        return "back_alignment"
    if (prompt_key or "").strip() == "remind_adjust_camera" or posture_label == "insufficient_data":
        return "camera_adjustment"
    return "general_posture"


def _posture_guidance_text(issue_key: str) -> str:
    mapping = {
        "neck_alignment": "Bring your head back a little and straighten your neck.",
        "back_alignment": "Sit taller and straighten your back.",
        "camera_adjustment": "I cannot read your posture clearly. Please adjust the camera angle or lighting.",
        "general_posture": "Please sit properly and adjust your posture.",
    }
    return mapping.get(issue_key, mapping["general_posture"])


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _reminder_counter_field(reminder_type: str, outcome: str) -> str | None:
    if reminder_type == "drink_water":
        return {
            "sent": "water_reminders_sent",
            "acknowledged": "water_acknowledged",
            "snoozed": "water_snoozed",
            "dismissed": "water_dismissed",
        }.get(outcome)
    if reminder_type == "nap" and outcome == "sent":
        return "nap_suggestions_sent"
    if reminder_type == "check_in" and outcome == "sent":
        return "check_ins_sent"
    return None


class OpenRouterEmbedder:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        models: list[str] | None = None,
        timeout: float = 20.0,
        client_factory: Any | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self._base_url = base_url or os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self._models = models or _configured_embedding_models()
        self._timeout = timeout
        self._client_factory = client_factory

    async def embed(self, text: str) -> EmbeddingResult:
        if not self._api_key:
            logger.warning("Memory embeddings unavailable: OPENROUTER_API_KEY is not set")
            raise EmbeddingUnavailableError("OPENROUTER_API_KEY is not set")

        if self._client_factory is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                logger.error("Memory embeddings unavailable: openai package is missing")
                raise EmbeddingUnavailableError("openai package is missing") from exc

            client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        else:
            client = self._client_factory(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        errors: list[str] = []

        for index, model in enumerate(self._models):
            used_fallback = index > 0
            for attempt in range(2):
                started_at = time.perf_counter()
                logger.info(
                    "Embedding request started",
                    extra={
                        "model": model,
                        "attempt": attempt + 1,
                        "used_fallback": used_fallback,
                        "chars": len(text),
                    },
                )
                try:
                    response = await client.embeddings.create(model=model, input=text)
                    vector = [float(value) for value in response.data[0].embedding]
                    logger.info(
                        "Embedding request completed",
                        extra={
                            "model": model,
                            "dimension": len(vector),
                            "used_fallback": used_fallback,
                            "elapsed_ms": round(
                                (time.perf_counter() - started_at) * 1000, 2
                            ),
                        },
                    )
                    return EmbeddingResult(
                        model=model,
                        vector=vector,
                        used_fallback=used_fallback,
                    )
                except Exception as exc:
                    errors.append(f"{model}: {type(exc).__name__}: {exc}")
                    logger.warning(
                        "Embedding request failed",
                        extra={
                            "model": model,
                            "attempt": attempt + 1,
                            "used_fallback": used_fallback,
                            "elapsed_ms": round(
                                (time.perf_counter() - started_at) * 1000, 2
                            ),
                            "error_type": type(exc).__name__,
                        },
                    )
                    if attempt == 0:
                        await asyncio.sleep(0.25)

            if index + 1 < len(self._models):
                logger.warning(
                    "Embedding model failed; trying fallback",
                    extra={"failed_model": model, "fallback_model": self._models[index + 1]},
                )

        logger.error("All embedding models failed", extra={"models": self._models})
        raise EmbeddingUnavailableError("; ".join(errors))


class MemoryExtractor:
    def __init__(self, *, min_importance: float) -> None:
        self._min_importance = min_importance

    async def extract(self, user_text: str) -> list[MemoryCandidate]:
        text = " ".join(user_text.split())
        if not text:
            logger.debug("Memory extraction skipped: empty user text")
            return []

        logger.info("Memory extraction started", extra={"chars": len(text)})
        candidates: list[MemoryCandidate] = []
        lower = text.lower()

        if SENSITIVE_PATTERN.search(text):
            logger.warning(
                "Memory extraction rejected sensitive user text",
                extra={"snippet": _safe_snippet(text)},
            )
            return []

        remember_match = re.search(r"\bremember(?: that)?\s+(.+)", text, re.IGNORECASE)
        if remember_match:
            candidates.append(
                MemoryCandidate(
                    text=remember_match.group(1).strip().rstrip("."),
                    kind="explicit_fact",
                    importance=0.95,
                    source_text=text,
                )
            )

        name_match = re.search(
            r"\b(?:my name is|call me)\s+([A-Za-z][A-Za-z0-9 _'-]{0,60})",
            text,
            re.IGNORECASE,
        )
        if name_match:
            name = name_match.group(1).strip().rstrip(".")
            candidates.append(
                MemoryCandidate(
                    text=f"The user's preferred name is {name}.",
                    kind="user_identity",
                    importance=0.9,
                    source_text=text,
                )
            )

        preference_match = re.search(
            r"\b(?:i prefer|i like|i don't like|i do not like|i want you to|please always|please don't|please do not)\b(.+)",
            text,
            re.IGNORECASE,
        )
        if preference_match:
            candidates.append(
                MemoryCandidate(
                    text=f"The user preference is: {text.rstrip('.')}.",
                    kind="preference",
                    importance=0.82,
                    source_text=text,
                )
            )

        if any(
            term in lower
            for term in (
                "this project",
                "the project",
                "raspberry pi",
                "sense hat",
                "sensehat",
            )
        ) and any(
            verb in lower
            for verb in ("is ", "uses ", "runs ", "needs ", "requires ", "should ")
        ):
            candidates.append(
                MemoryCandidate(
                    text=f"Project context: {text.rstrip('.')}.",
                    kind="project_fact",
                    importance=0.72,
                    source_text=text,
                )
            )

        accepted = [
            candidate
            for candidate in candidates
            if candidate.text and candidate.importance >= self._min_importance
        ]
        logger.info(
            "Memory extraction completed",
            extra={
                "candidates": len(candidates),
                "accepted": len(accepted),
                "rejected": len(candidates) - len(accepted),
            },
        )
        return accepted


class SQLiteVectorMemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._vector_available = False
        self._load_sqlite_vec()
        self._init_schema()
        self._ensure_defaults()

    @property
    def vector_available(self) -> bool:
        return self._vector_available

    def _load_sqlite_vec(self) -> None:
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._vector_available = True
            logger.info("sqlite-vec loaded", extra={"db_path": str(self._db_path)})
        except Exception as exc:
            self._vector_available = False
            logger.warning(
                "sqlite-vec unavailable; vector memory will use FTS fallback",
                extra={"db_path": str(self._db_path), "error_type": type(exc).__name__},
            )

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column in columns:
            return
        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

    def _init_schema(self) -> None:
        logger.info("Initializing memory database", extra={"db_path": str(self._db_path)})
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                kind TEXT NOT NULL,
                importance REAL NOT NULL,
                source_text TEXT NOT NULL,
                embedding_model TEXT,
                embedding_dimension INTEGER,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                deleted_at REAL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                memory_id UNINDEXED,
                text,
                source_text
            );

            CREATE TABLE IF NOT EXISTS agent_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                preferred_name TEXT,
                timezone TEXT NOT NULL DEFAULT 'UTC',
                quiet_hours_start TEXT,
                quiet_hours_end TEXT,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_status (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                current_activity TEXT,
                activity_expires_at REAL,
                busy_state TEXT,
                busy_expires_at REAL,
                summary_entries_json TEXT NOT NULL DEFAULT '[]',
                last_user_interaction_at REAL,
                last_assistant_speech_at REAL,
                last_water_reminder_at REAL,
                last_check_in_at REAL,
                last_nap_suggestion_at REAL,
                water_cooldown_until REAL,
                check_in_cooldown_until REAL,
                nap_cooldown_until REAL,
                posture_monitoring_active INTEGER NOT NULL DEFAULT 0,
                posture_workflow_active INTEGER NOT NULL DEFAULT 0,
                posture_session_id TEXT,
                latest_posture_label TEXT,
                latest_posture_severity TEXT,
                latest_posture_reason_codes_json TEXT NOT NULL DEFAULT '[]',
                latest_posture_metrics_json TEXT NOT NULL DEFAULT '{}',
                latest_posture_prompt_key TEXT,
                last_posture_event_at REAL,
                posture_issue_cooldowns_json TEXT NOT NULL DEFAULT '{}',
                posture_preview_enabled INTEGER NOT NULL DEFAULT 0,
                posture_preview_active INTEGER NOT NULL DEFAULT 0,
                last_posture_callback_at REAL,
                last_posture_callback_event TEXT,
                posture_reminder_cooldown_seconds REAL NOT NULL DEFAULT 60.0,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reminder_policies (
                reminder_type TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL,
                interval_minutes INTEGER,
                window_start TEXT,
                window_end TEXT,
                max_per_day INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS daily_status_log (
                day TEXT PRIMARY KEY,
                water_reminders_sent INTEGER NOT NULL DEFAULT 0,
                water_acknowledged INTEGER NOT NULL DEFAULT 0,
                water_snoozed INTEGER NOT NULL DEFAULT 0,
                water_dismissed INTEGER NOT NULL DEFAULT 0,
                nap_suggestions_sent INTEGER NOT NULL DEFAULT 0,
                check_ins_sent INTEGER NOT NULL DEFAULT 0,
                last_event_at REAL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS proactive_event_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );
            """
        )
        self._ensure_column(
            "agent_status",
            "summary_entries_json",
            "summary_entries_json TEXT NOT NULL DEFAULT '[]'",
        )
        self._ensure_column(
            "agent_status",
            "last_assistant_speech_at",
            "last_assistant_speech_at REAL",
        )
        self._ensure_column(
            "agent_status",
            "posture_monitoring_active",
            "posture_monitoring_active INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            "agent_status",
            "posture_workflow_active",
            "posture_workflow_active INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            "agent_status",
            "posture_session_id",
            "posture_session_id TEXT",
        )
        self._ensure_column(
            "agent_status",
            "latest_posture_label",
            "latest_posture_label TEXT",
        )
        self._ensure_column(
            "agent_status",
            "latest_posture_severity",
            "latest_posture_severity TEXT",
        )
        self._ensure_column(
            "agent_status",
            "latest_posture_reason_codes_json",
            "latest_posture_reason_codes_json TEXT NOT NULL DEFAULT '[]'",
        )
        self._ensure_column(
            "agent_status",
            "latest_posture_metrics_json",
            "latest_posture_metrics_json TEXT NOT NULL DEFAULT '{}'",
        )
        self._ensure_column(
            "agent_status",
            "latest_posture_prompt_key",
            "latest_posture_prompt_key TEXT",
        )
        self._ensure_column(
            "agent_status",
            "last_posture_event_at",
            "last_posture_event_at REAL",
        )
        self._ensure_column(
            "agent_status",
            "posture_issue_cooldowns_json",
            "posture_issue_cooldowns_json TEXT NOT NULL DEFAULT '{}'",
        )
        self._ensure_column(
            "agent_status",
            "posture_preview_enabled",
            "posture_preview_enabled INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            "agent_status",
            "posture_preview_active",
            "posture_preview_active INTEGER NOT NULL DEFAULT 0",
        )
        self._ensure_column(
            "agent_status",
            "last_posture_callback_at",
            "last_posture_callback_at REAL",
        )
        self._ensure_column(
            "agent_status",
            "last_posture_callback_event",
            "last_posture_callback_event TEXT",
        )
        self._ensure_column(
            "agent_status",
            "posture_reminder_cooldown_seconds",
            "posture_reminder_cooldown_seconds REAL NOT NULL DEFAULT 60.0",
        )
        self._conn.commit()
        logger.info("Memory database initialized", extra={"db_path": str(self._db_path)})

    def _ensure_defaults(self) -> None:
        now = time.time()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO agent_profile(
                id, preferred_name, timezone, quiet_hours_start, quiet_hours_end, updated_at
            ) VALUES (1, NULL, 'UTC', ?, ?, ?)
            """,
            (DEFAULT_QUIET_HOURS_START, DEFAULT_QUIET_HOURS_END, now),
        )
        self._conn.execute(
            """
            INSERT OR IGNORE INTO agent_status(
                id, summary_entries_json, updated_at
            ) VALUES (1, '[]', ?)
            """,
            (now,),
        )
        for policy in default_reminder_policies().values():
            self._conn.execute(
                """
                INSERT OR IGNORE INTO reminder_policies(
                    reminder_type, enabled, interval_minutes, window_start, window_end,
                    max_per_day, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    policy.reminder_type,
                    int(policy.enabled),
                    policy.interval_minutes,
                    policy.window_start,
                    policy.window_end,
                    policy.max_per_day,
                    now,
                ),
            )
        self._conn.commit()

    def _ensure_vector_table(self, model: str, dimension: int) -> str | None:
        if not self._vector_available:
            return None
        table_name = f"memory_vec_{_table_suffix(model, dimension)}"
        try:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table_name} USING vec0(embedding float[{dimension}])"
            )
            self._conn.commit()
            return table_name
        except Exception as exc:
            logger.warning(
                "Vector table unavailable; falling back to FTS",
                extra={
                    "table": table_name,
                    "model": model,
                    "dimension": dimension,
                    "error_type": type(exc).__name__,
                },
            )
            return None

    def add_memory(
        self,
        candidate: MemoryCandidate,
        *,
        embedding: EmbeddingResult | None,
    ) -> MemoryRecord | None:
        normalized = candidate.text.strip().casefold()
        if not normalized:
            logger.debug("Skipping empty memory candidate")
            return None

        existing = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE normalized_text = ? AND deleted_at IS NULL
            LIMIT 1
            """,
            (normalized,),
        ).fetchone()
        now = time.time()
        if existing:
            self._conn.execute(
                """
                UPDATE memories
                SET importance = max(importance, ?), updated_at = ?
                WHERE id = ?
                """,
                (candidate.importance, now, existing["id"]),
            )
            self._conn.commit()
            logger.info(
                "Duplicate memory merged",
                extra={"memory_id": existing["uid"], "kind": existing["kind"]},
            )
            refreshed = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (existing["id"],)
            ).fetchone()
            return self._row_to_record(refreshed or existing)

        dimension = len(embedding.vector) if embedding else None
        cursor = self._conn.execute(
            """
            INSERT INTO memories (
                uid, text, normalized_text, kind, importance, source_text,
                embedding_model, embedding_dimension, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                candidate.text,
                normalized,
                candidate.kind,
                candidate.importance,
                candidate.source_text,
                embedding.model if embedding else None,
                dimension,
                now,
                now,
            ),
        )
        memory_pk = int(cursor.lastrowid)
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_pk,)
        ).fetchone()
        self._conn.execute(
            "INSERT INTO memories_fts(memory_id, text, source_text) VALUES (?, ?, ?)",
            (memory_pk, candidate.text, candidate.source_text),
        )

        if embedding and dimension:
            table_name = self._ensure_vector_table(embedding.model, dimension)
            if table_name:
                try:
                    import sqlite_vec

                    self._conn.execute(
                        f"INSERT OR REPLACE INTO {table_name}(rowid, embedding) VALUES (?, ?)",
                        (memory_pk, sqlite_vec.serialize_float32(embedding.vector)),
                    )
                    logger.debug(
                        "Memory vector stored",
                        extra={
                            "memory_id": row["uid"],
                            "model": embedding.model,
                            "dimension": dimension,
                            "table": table_name,
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to store memory vector; FTS still available",
                        extra={
                            "memory_id": row["uid"],
                            "model": embedding.model,
                            "error_type": type(exc).__name__,
                        },
                    )

        self._conn.commit()
        logger.info(
            "Memory inserted",
            extra={
                "memory_id": row["uid"],
                "kind": candidate.kind,
                "importance": candidate.importance,
                "embedding_model": embedding.model if embedding else None,
                "embedding_dimension": dimension,
            },
        )
        logger.debug("Memory text inserted", extra={"snippet": _safe_snippet(candidate.text)})
        return self._row_to_record(row)

    def search(
        self,
        query: str,
        *,
        embedding: EmbeddingResult | None,
        limit: int,
    ) -> list[MemoryRecord]:
        started_at = time.perf_counter()
        records: list[MemoryRecord] = []
        seen: set[str] = set()

        if embedding:
            records.extend(self._vector_search(query, embedding=embedding, limit=limit))
            seen.update(record.id for record in records)

        if len(records) < limit:
            for record in self._fts_search(query, limit=limit):
                if record.id not in seen:
                    records.append(record)
                    seen.add(record.id)
                if len(records) >= limit:
                    break

        logger.info(
            "Memory search completed",
            extra={
                "query_chars": len(query),
                "results": len(records),
                "vector_used": bool(embedding),
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
        )
        return records[:limit]

    def _vector_search(
        self,
        query: str,
        *,
        embedding: EmbeddingResult,
        limit: int,
    ) -> list[MemoryRecord]:
        dimension = len(embedding.vector)
        table_name = self._ensure_vector_table(embedding.model, dimension)
        if not table_name:
            return []
        try:
            import sqlite_vec

            rows = self._conn.execute(
                f"""
                SELECT m.*
                FROM (
                    SELECT rowid, distance
                    FROM {table_name}
                    WHERE embedding MATCH ?
                    ORDER BY distance
                    LIMIT ?
                ) v
                JOIN memories m ON m.id = v.rowid
                WHERE m.deleted_at IS NULL
                ORDER BY v.distance
                """,
                (sqlite_vec.serialize_float32(embedding.vector), limit),
            ).fetchall()
            logger.debug(
                "Vector memory search completed",
                extra={
                    "model": embedding.model,
                    "dimension": dimension,
                    "results": len(rows),
                    "query": _safe_snippet(query),
                },
            )
            return [self._row_to_record(row) for row in rows]
        except Exception as exc:
            logger.warning(
                "Vector memory search failed; using FTS fallback",
                extra={
                    "model": embedding.model,
                    "dimension": dimension,
                    "table": table_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return []

    def _fts_search(self, query: str, *, limit: int) -> list[MemoryRecord]:
        fts_query = _fts_query(query)
        rows: list[sqlite3.Row] = []
        try:
            rows = self._conn.execute(
                """
                SELECT m.*
                FROM memories_fts f
                JOIN memories m ON m.id = f.memory_id
                WHERE memories_fts MATCH ? AND m.deleted_at IS NULL
                ORDER BY bm25(memories_fts)
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        except Exception as exc:
            logger.warning(
                "FTS memory search failed; using LIKE fallback",
                extra={"error_type": type(exc).__name__},
            )

        if not rows:
            terms = re.findall(r"[A-Za-z0-9_]+", query.lower())
            scored: list[tuple[int, sqlite3.Row]] = []
            for row in self._conn.execute(
                "SELECT * FROM memories WHERE deleted_at IS NULL"
            ).fetchall():
                haystack = f"{row['text']} {row['source_text']}".lower()
                score = sum(1 for term in terms if term in haystack)
                if score:
                    scored.append((score, row))
            scored.sort(key=lambda item: (-item[0], -float(item[1]["importance"])))
            rows = [row for _, row in scored[:limit]]

        logger.debug("FTS memory search completed", extra={"results": len(rows)})
        return [self._row_to_record(row) for row in rows]

    def list_memories(self, *, limit: int) -> list[MemoryRecord]:
        rows = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE deleted_at IS NULL
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
            """,
            (max(1, min(limit, 50)),),
        ).fetchall()
        logger.info("Memory list completed", extra={"results": len(rows)})
        return [self._row_to_record(row) for row in rows]

    def delete_memory(self, memory_id: str) -> bool:
        now = time.time()
        cursor = self._conn.execute(
            "UPDATE memories SET deleted_at = ?, updated_at = ? WHERE uid = ? AND deleted_at IS NULL",
            (now, now, memory_id),
        )
        self._conn.commit()
        deleted = cursor.rowcount > 0
        logger.info("Memory delete requested", extra={"memory_id": memory_id, "deleted": deleted})
        return deleted

    def delete_all(self) -> int:
        now = time.time()
        cursor = self._conn.execute(
            "UPDATE memories SET deleted_at = ?, updated_at = ? WHERE deleted_at IS NULL",
            (now, now),
        )
        self._conn.commit()
        logger.warning("All memories soft-deleted", extra={"deleted": cursor.rowcount})
        return int(cursor.rowcount)

    def load_profile(self) -> UserProfile:
        row = self._conn.execute("SELECT * FROM agent_profile WHERE id = 1").fetchone()
        if row is None:
            self._ensure_defaults()
            row = self._conn.execute("SELECT * FROM agent_profile WHERE id = 1").fetchone()
        return UserProfile(
            preferred_name=row["preferred_name"],
            timezone=row["timezone"] or "UTC",
            quiet_hours_start=row["quiet_hours_start"],
            quiet_hours_end=row["quiet_hours_end"],
            updated_at=float(row["updated_at"]),
        )

    def save_profile(self, profile: UserProfile) -> None:
        profile.updated_at = time.time()
        self._conn.execute(
            """
            UPDATE agent_profile
            SET preferred_name = ?, timezone = ?, quiet_hours_start = ?, quiet_hours_end = ?, updated_at = ?
            WHERE id = 1
            """,
            (
                profile.preferred_name,
                profile.timezone,
                profile.quiet_hours_start,
                profile.quiet_hours_end,
                profile.updated_at,
            ),
        )
        self._conn.commit()

    def load_status(self) -> RuntimeStatus:
        row = self._conn.execute("SELECT * FROM agent_status WHERE id = 1").fetchone()
        if row is None:
            self._ensure_defaults()
            row = self._conn.execute("SELECT * FROM agent_status WHERE id = 1").fetchone()
        entries: list[str] = []
        try:
            raw_entries = row["summary_entries_json"] or "[]"
            loaded = json.loads(raw_entries)
            if isinstance(loaded, list):
                entries = [str(item) for item in loaded if str(item).strip()]
        except Exception:
            logger.warning("Invalid stored summary entries; resetting")
        return RuntimeStatus(
            current_activity=row["current_activity"],
            activity_expires_at=row["activity_expires_at"],
            busy_state=row["busy_state"],
            busy_expires_at=row["busy_expires_at"],
            summary_entries=entries,
            last_user_interaction_at=row["last_user_interaction_at"],
            last_assistant_speech_at=row["last_assistant_speech_at"],
            last_water_reminder_at=row["last_water_reminder_at"],
            last_check_in_at=row["last_check_in_at"],
            last_nap_suggestion_at=row["last_nap_suggestion_at"],
            water_cooldown_until=row["water_cooldown_until"],
            check_in_cooldown_until=row["check_in_cooldown_until"],
            nap_cooldown_until=row["nap_cooldown_until"],
            posture_monitoring_active=bool(row["posture_monitoring_active"]),
            posture_workflow_active=bool(row["posture_workflow_active"]),
            posture_session_id=row["posture_session_id"],
            latest_posture_label=row["latest_posture_label"],
            latest_posture_severity=row["latest_posture_severity"],
            latest_posture_reason_codes=_load_json_list(row["latest_posture_reason_codes_json"]),
            latest_posture_metrics=_load_json_float_dict(row["latest_posture_metrics_json"]),
            latest_posture_prompt_key=row["latest_posture_prompt_key"],
            last_posture_event_at=row["last_posture_event_at"],
            posture_issue_cooldowns=_load_json_float_dict(row["posture_issue_cooldowns_json"]),
            posture_preview_enabled=bool(row["posture_preview_enabled"]),
            posture_preview_active=bool(row["posture_preview_active"]),
            last_posture_callback_at=row["last_posture_callback_at"],
            last_posture_callback_event=row["last_posture_callback_event"],
            posture_reminder_cooldown_seconds=float(
                row["posture_reminder_cooldown_seconds"] or 60.0
            ),
            updated_at=float(row["updated_at"]),
        )

    def save_status(self, status: RuntimeStatus) -> None:
        status.updated_at = time.time()
        self._conn.execute(
            """
            UPDATE agent_status
            SET current_activity = ?, activity_expires_at = ?, busy_state = ?, busy_expires_at = ?,
                summary_entries_json = ?, last_user_interaction_at = ?, last_assistant_speech_at = ?,
                last_water_reminder_at = ?, last_check_in_at = ?, last_nap_suggestion_at = ?, water_cooldown_until = ?,
                check_in_cooldown_until = ?, nap_cooldown_until = ?, posture_monitoring_active = ?,
                posture_workflow_active = ?, posture_session_id = ?, latest_posture_label = ?, latest_posture_severity = ?,
                latest_posture_reason_codes_json = ?, latest_posture_metrics_json = ?, latest_posture_prompt_key = ?,
                last_posture_event_at = ?, posture_issue_cooldowns_json = ?, posture_preview_enabled = ?,
                posture_preview_active = ?, last_posture_callback_at = ?, last_posture_callback_event = ?,
                posture_reminder_cooldown_seconds = ?, updated_at = ?
            WHERE id = 1
            """,
            (
                status.current_activity,
                status.activity_expires_at,
                status.busy_state,
                status.busy_expires_at,
                json.dumps(status.summary_entries, ensure_ascii=False),
                status.last_user_interaction_at,
                status.last_assistant_speech_at,
                status.last_water_reminder_at,
                status.last_check_in_at,
                status.last_nap_suggestion_at,
                status.water_cooldown_until,
                status.check_in_cooldown_until,
                status.nap_cooldown_until,
                int(status.posture_monitoring_active),
                int(status.posture_workflow_active),
                status.posture_session_id,
                status.latest_posture_label,
                status.latest_posture_severity,
                json.dumps(status.latest_posture_reason_codes, ensure_ascii=False),
                json.dumps(status.latest_posture_metrics, ensure_ascii=False),
                status.latest_posture_prompt_key,
                status.last_posture_event_at,
                json.dumps(status.posture_issue_cooldowns, ensure_ascii=False),
                int(status.posture_preview_enabled),
                int(status.posture_preview_active),
                status.last_posture_callback_at,
                status.last_posture_callback_event,
                status.posture_reminder_cooldown_seconds,
                status.updated_at,
            ),
        )
        self._conn.commit()

    def load_reminder_policies(self) -> dict[str, ReminderPolicy]:
        rows = self._conn.execute(
            "SELECT * FROM reminder_policies ORDER BY reminder_type"
        ).fetchall()
        result: dict[str, ReminderPolicy] = {}
        for row in rows:
            result[row["reminder_type"]] = ReminderPolicy(
                reminder_type=row["reminder_type"],
                enabled=bool(row["enabled"]),
                interval_minutes=row["interval_minutes"],
                window_start=row["window_start"],
                window_end=row["window_end"],
                max_per_day=int(row["max_per_day"]),
                updated_at=float(row["updated_at"]),
            )
        defaults = default_reminder_policies()
        for reminder_type, policy in defaults.items():
            result.setdefault(reminder_type, policy)
        return result

    def save_reminder_policy(self, policy: ReminderPolicy) -> None:
        policy.updated_at = time.time()
        self._conn.execute(
            """
            INSERT INTO reminder_policies(
                reminder_type, enabled, interval_minutes, window_start, window_end,
                max_per_day, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(reminder_type) DO UPDATE SET
                enabled = excluded.enabled,
                interval_minutes = excluded.interval_minutes,
                window_start = excluded.window_start,
                window_end = excluded.window_end,
                max_per_day = excluded.max_per_day,
                updated_at = excluded.updated_at
            """,
            (
                policy.reminder_type,
                int(policy.enabled),
                policy.interval_minutes,
                policy.window_start,
                policy.window_end,
                policy.max_per_day,
                policy.updated_at,
            ),
        )
        self._conn.commit()

    def ensure_daily_row(self, day: str) -> DailyStatus:
        now = time.time()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO daily_status_log(
                day, updated_at
            ) VALUES (?, ?)
            """,
            (day, now),
        )
        self._conn.commit()
        return self.get_daily_status(day) or DailyStatus(day=day, updated_at=now)

    def get_daily_status(self, day: str) -> DailyStatus | None:
        row = self._conn.execute(
            "SELECT * FROM daily_status_log WHERE day = ?",
            (day,),
        ).fetchone()
        if row is None:
            return None
        return DailyStatus(
            day=row["day"],
            water_reminders_sent=int(row["water_reminders_sent"]),
            water_acknowledged=int(row["water_acknowledged"]),
            water_snoozed=int(row["water_snoozed"]),
            water_dismissed=int(row["water_dismissed"]),
            nap_suggestions_sent=int(row["nap_suggestions_sent"]),
            check_ins_sent=int(row["check_ins_sent"]),
            last_event_at=row["last_event_at"],
            updated_at=float(row["updated_at"]),
        )

    def get_recent_daily_status(self, *, limit: int) -> list[DailyStatus]:
        rows = self._conn.execute(
            """
            SELECT * FROM daily_status_log
            ORDER BY day DESC
            LIMIT ?
            """,
            (max(1, min(limit, 30)),),
        ).fetchall()
        return [
            DailyStatus(
                day=row["day"],
                water_reminders_sent=int(row["water_reminders_sent"]),
                water_acknowledged=int(row["water_acknowledged"]),
                water_snoozed=int(row["water_snoozed"]),
                water_dismissed=int(row["water_dismissed"]),
                nap_suggestions_sent=int(row["nap_suggestions_sent"]),
                check_ins_sent=int(row["check_ins_sent"]),
                last_event_at=row["last_event_at"],
                updated_at=float(row["updated_at"]),
            )
            for row in rows
        ]

    def increment_daily_counter(self, day: str, field_name: str, *, delta: int = 1) -> None:
        if field_name not in {
            "water_reminders_sent",
            "water_acknowledged",
            "water_snoozed",
            "water_dismissed",
            "nap_suggestions_sent",
            "check_ins_sent",
        }:
            raise ValueError(f"Unsupported daily counter {field_name!r}")
        self.ensure_daily_row(day)
        now = time.time()
        self._conn.execute(
            f"""
            UPDATE daily_status_log
            SET {field_name} = {field_name} + ?,
                last_event_at = ?,
                updated_at = ?
            WHERE day = ?
            """,
            (delta, now, now, day),
        )
        self._conn.commit()

    def append_proactive_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO proactive_event_log(event_type, payload_json, created_at)
            VALUES (?, ?, ?)
            """,
            (event_type, json.dumps(payload, ensure_ascii=False), time.time()),
        )
        self._conn.commit()

    def delete_daily_status(self, day: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM daily_status_log WHERE day = ?",
            (day,),
        )
        self._conn.commit()
        deleted = cursor.rowcount > 0
        logger.info("Daily status delete requested", extra={"day": day, "deleted": deleted})
        return deleted

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["uid"],
            text=row["text"],
            kind=row["kind"],
            importance=float(row["importance"]),
            source_text=row["source_text"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            embedding_model=row["embedding_model"],
            embedding_dimension=row["embedding_dimension"],
        )


def default_reminder_policies() -> dict[str, ReminderPolicy]:
    return {
        "drink_water": ReminderPolicy(
            reminder_type="drink_water",
            enabled=True,
            interval_minutes=60,
            window_start="09:00",
            window_end="21:00",
            max_per_day=24,
        ),
        "nap": ReminderPolicy(
            reminder_type="nap",
            enabled=True,
            interval_minutes=None,
            window_start="13:00",
            window_end="16:00",
            max_per_day=1,
        ),
        "check_in": ReminderPolicy(
            reminder_type="check_in",
            enabled=True,
            interval_minutes=180,
            window_start="09:00",
            window_end="21:00",
            max_per_day=6,
        ),
    }


class ContextController:
    def __init__(
        self,
        *,
        store: SQLiteVectorMemoryStore | None = None,
        embedder: Embedder | None = None,
        extractor: MemoryExtractor | None = None,
        db_path: str | Path | None = None,
        short_window_messages: int | None = None,
        top_k: int | None = None,
        min_importance: float | None = None,
        status_reporter: StatusReporter | None = None,
    ) -> None:
        self.short_window_messages = short_window_messages or _env_int(
            "MEMORY_SHORT_WINDOW_MESSAGES", 8
        )
        self.top_k = top_k or _env_int("MEMORY_TOP_K", 5)
        self.min_importance = min_importance or _env_float("MEMORY_MIN_IMPORTANCE", 0.65)
        memory_db_path = db_path or os.getenv(
            "MEMORY_DB_PATH", ".data/agent_memory.sqlite3"
        )
        self.store = store or SQLiteVectorMemoryStore(memory_db_path)
        self.embedder = embedder or OpenRouterEmbedder()
        self.extractor = extractor or MemoryExtractor(min_importance=self.min_importance)
        self.status_reporter = status_reporter
        self._observed_turn_ids: set[str] = set()
        self.profile = self.store.load_profile()
        self.status = self.store.load_status()
        self.policies = self.store.load_reminder_policies()
        self._recent_activity_window_seconds = (
            _env_int("PROACTIVE_RECENT_ACTIVITY_WINDOW_MINUTES", 240) * 60
        )
        self._post_user_grace_seconds = _env_int(
            "PROACTIVE_POST_USER_GRACE_SECONDS", 90
        )
        self._post_assistant_grace_seconds = _env_int(
            "PROACTIVE_POST_ASSISTANT_GRACE_SECONDS", 60
        )
        self._summary_max_entries = _env_int("STATUS_SUMMARY_MAX_ENTRIES", 6)
        self._summary_max_chars = _env_int("STATUS_SUMMARY_MAX_CHARS", 420)
        self._posture_issue_cooldown_seconds = _env_float(
            "POSTURE_ISSUE_REMINDER_COOLDOWN_SECONDS", 60.0
        )
        self._posture_event_stale_seconds = _env_float(
            "POSTURE_EVENT_STALE_SECONDS", 30.0
        )
        self._data_flow_logger = logging.getLogger("agent.data_flow")
        if self.status.posture_reminder_cooldown_seconds <= 0:
            self.status.posture_reminder_cooldown_seconds = (
                self._posture_issue_cooldown_seconds
            )
            self.store.save_status(self.status)
        logger.info(
            "Context controller initialized",
            extra={
                "short_window_messages": self.short_window_messages,
                "top_k": self.top_k,
                "min_importance": self.min_importance,
                "db_path": str(memory_db_path),
                "post_user_grace_seconds": self._post_user_grace_seconds,
                "post_assistant_grace_seconds": self._post_assistant_grace_seconds,
                "posture_issue_cooldown_seconds": self._posture_issue_cooldown_seconds,
                "stored_posture_reminder_cooldown_seconds": self.status.posture_reminder_cooldown_seconds,
                "posture_event_stale_seconds": self._posture_event_stale_seconds,
            },
        )

    async def observe_user_turn(
        self, text: str | None, *, turn_id: str | None = None
    ) -> list[MemoryRecord]:
        if not text:
            logger.debug("Skipping memory observation: no user text")
            return []

        observation_key = turn_id or hashlib.sha256(text.strip().encode()).hexdigest()
        if observation_key:
            if observation_key in self._observed_turn_ids:
                logger.debug("Skipping memory observation: turn already observed")
                return []
            self._observed_turn_ids.add(observation_key)

        started_at = time.perf_counter()
        logger.info(
            "Memory observation started",
            extra={"chars": len(text), "turn_id": turn_id, "snippet": _safe_snippet(text)},
        )
        records: list[MemoryRecord] = []
        try:
            candidates = await self.extractor.extract(text)
            for candidate in candidates:
                self._push_status("memory_insert")
                try:
                    embedding = await self._embed_or_none(candidate.text)
                    record = self.store.add_memory(candidate, embedding=embedding)
                    if record:
                            records.append(record)
                finally:
                    self._pop_status()

            structured_result = self._observe_structured_status(text)
            self._emit_data_flow(
                "status_observation",
                user_text=text,
                structured_changed=structured_result.changed,
                profile_changed=structured_result.profile_changed,
                status_changed=structured_result.status_changed,
                policies_changed=structured_result.policies_changed,
                explicit_memories=records_for_tool(structured_result.explicit_memories),
                profile=self._profile_for_log(),
                runtime=self._runtime_for_log(),
                policies=self._policies_for_log(),
            )
            logger.info(
                "Memory observation completed",
                extra={
                    "stored": len(records),
                    "structured_changed": structured_result.changed,
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )
            return records
        except Exception as exc:
            logger.error(
                "Memory observation failed",
                extra={"error_type": type(exc).__name__},
                exc_info=True,
            )
            return records

    def _observe_structured_status(self, text: str) -> StructuredUpdateResult:
        result = StructuredUpdateResult()
        normalized = " ".join(text.split())
        lowered = normalized.lower()
        now = time.time()
        self._expire_temporary_state(now=now)

        self.status.last_user_interaction_at = now
        result.status_changed = True

        profile_name = self._extract_preferred_name(normalized)
        if profile_name and profile_name != self.profile.preferred_name:
            self.profile.preferred_name = profile_name
            result.profile_changed = True

        timezone_name = self._extract_timezone(normalized)
        if timezone_name and timezone_name != self.profile.timezone:
            self.profile.timezone = timezone_name
            result.profile_changed = True

        after_time = self._extract_single_time(lowered, r"\b(?:don't|do not) remind me after ([^.,]+)")
        if after_time and after_time != self.profile.quiet_hours_start:
            self.profile.quiet_hours_start = after_time
            if not self.profile.quiet_hours_end:
                self.profile.quiet_hours_end = DEFAULT_QUIET_HOURS_END
            result.profile_changed = True

        before_time = self._extract_single_time(lowered, r"\b(?:don't|do not) remind me before ([^.,]+)")
        if before_time and before_time != self.profile.quiet_hours_end:
            self.profile.quiet_hours_end = before_time
            if not self.profile.quiet_hours_start:
                self.profile.quiet_hours_start = DEFAULT_QUIET_HOURS_START
            result.profile_changed = True

        if "don't remind me to drink water" in lowered or "do not remind me to drink water" in lowered:
            result.policies_changed |= self._set_policy_enabled("drink_water", False)
        elif "remind me to drink water" in lowered or "drink water reminder" in lowered:
            result.policies_changed |= self._set_policy_enabled("drink_water", True)

        if "don't suggest a nap" in lowered or "do not suggest a nap" in lowered:
            result.policies_changed |= self._set_policy_enabled("nap", False)
        elif "suggest a nap" in lowered or "nap reminder" in lowered:
            result.policies_changed |= self._set_policy_enabled("nap", True)

        cadence = self._extract_drink_water_interval(normalized)
        if cadence is not None:
            result.policies_changed |= self._set_policy_interval("drink_water", cadence)

        activity, duration_minutes = self._extract_activity(normalized)
        if activity:
            self._apply_activity(activity, duration_minutes=duration_minutes)
            result.status_changed = True

        self._update_summary(normalized)
        result.status_changed = True

        if result.profile_changed:
            self.store.save_profile(self.profile)
        if result.status_changed:
            self.store.save_status(self.status)
        if result.policies_changed:
            for policy in self.policies.values():
                self.store.save_reminder_policy(policy)
        logger.info(
            "Structured status observed",
            extra={
                "profile_changed": result.profile_changed,
                "status_changed": result.status_changed,
                "policies_changed": result.policies_changed,
                "preferred_name": self.profile.preferred_name,
                "timezone": self.profile.timezone,
                "current_activity": self.status.current_activity,
                "busy_state": self.status.busy_state,
                "summary_entries": len(self.status.summary_entries),
            },
        )
        return result

    def _extract_preferred_name(self, text: str) -> str | None:
        match = re.search(
            r"\b(?:my name is|call me)\s+([A-Za-z][A-Za-z0-9 _'-]{0,60})",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        raw = match.group(1).strip().rstrip(".")
        cleaned = re.split(r"\b(?:and|but|please|also)\b", raw, maxsplit=1, flags=re.IGNORECASE)[0]
        return cleaned.strip().rstrip(",.") or None

    def _extract_timezone(self, text: str) -> str | None:
        match = re.search(
            r"\b(?:my timezone is|i am in timezone)\s+([A-Za-z0-9_/\-+]+)",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        candidate = match.group(1).strip()
        try:
            ZoneInfo(candidate)
        except Exception:
            logger.warning("Ignoring invalid timezone from user turn", extra={"timezone": candidate})
            return None
        return candidate

    def _extract_single_time(self, text: str, pattern: str) -> str | None:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        return _parse_time_phrase(match.group(1))

    def _extract_drink_water_interval(self, text: str) -> int | None:
        match = re.search(
            r"\bremind me to drink water every (\d+)\s*(minute|minutes|hour|hours)\b",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        amount = int(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("hour"):
            amount *= 60
        return max(1, min(amount, 24 * 60))

    def _extract_activity(self, text: str) -> tuple[str | None, int | None]:
        patterns = (
            (r"\b(?:i am|i'm)\s+(busy|working|coding|resting|out|napping)\b", None),
            (r"\b(?:i am|i'm)\s+taking a nap\b", "napping"),
        )
        activity: str | None = None
        for pattern, fixed in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                activity = fixed or match.group(1).lower()
                break
        if not activity:
            return None, None
        duration_match = re.search(r"\bfor (\d+)\s*(minute|minutes|hour|hours)\b", text, re.IGNORECASE)
        if not duration_match:
            return activity, None
        amount = int(duration_match.group(1))
        unit = duration_match.group(2).lower()
        if unit.startswith("hour"):
            amount *= 60
        return activity, max(5, min(amount, 24 * 60))

    def _set_policy_enabled(self, reminder_type: str, enabled: bool) -> bool:
        policy = self.policies[reminder_type]
        if policy.enabled == enabled:
            return False
        policy.enabled = enabled
        self.store.save_reminder_policy(policy)
        return True

    def _set_policy_interval(self, reminder_type: str, interval_minutes: int) -> bool:
        policy = self.policies[reminder_type]
        if policy.interval_minutes == interval_minutes:
            return False
        policy.interval_minutes = interval_minutes
        self.store.save_reminder_policy(policy)
        return True

    def _apply_activity(self, activity: str, *, duration_minutes: int | None = None) -> None:
        now = time.time()
        duration_seconds = (duration_minutes or _env_int("TEMPORARY_ACTIVITY_DURATION_MINUTES", 120)) * 60
        self.status.current_activity = activity
        self.status.activity_expires_at = now + duration_seconds
        if activity in {"busy", "working", "coding"}:
            self.status.busy_state = activity
            self.status.busy_expires_at = now + duration_seconds
        else:
            self.status.busy_state = None
            self.status.busy_expires_at = None

    def _update_summary(self, text: str) -> None:
        if not text:
            return
        candidate = _safe_snippet(text, limit=120)
        entries = [candidate, *self.status.summary_entries]
        self.status.summary_entries = _normalize_summary_entries(
            entries,
            max_entries=self._summary_max_entries,
            max_chars=self._summary_max_chars,
        )
        logger.info(
            "Conversation summary updated",
            extra={
                "entries": len(self.status.summary_entries),
                "chars": len(self.status.recent_summary),
                "latest_entry": candidate,
            },
        )

    def _expire_temporary_state(self, *, now: float | None = None) -> bool:
        now_ts = now or time.time()
        changed = False
        if self.status.activity_expires_at and self.status.activity_expires_at <= now_ts:
            self.status.current_activity = None
            self.status.activity_expires_at = None
            changed = True
        if self.status.busy_expires_at and self.status.busy_expires_at <= now_ts:
            self.status.busy_state = None
            self.status.busy_expires_at = None
            changed = True
        if changed:
            self.store.save_status(self.status)
            logger.info(
                "Temporary status expired",
                extra={
                    "current_activity": self.status.current_activity,
                    "busy_state": self.status.busy_state,
                },
            )
        return changed

    def build_status_note(self) -> str:
        self._expire_temporary_state()
        lines = [
            f"{STATUS_NOTE_PREFIX} Use only if helpful. Do not mention storage unless asked. Do not claim live sensor readings without a Sense HAT tool call.",
        ]
        if self.profile.preferred_name:
            lines.append(f"- User preferred name: {self.profile.preferred_name}.")
        if self.profile.quiet_hours_start and self.profile.quiet_hours_end:
            lines.append(
                f"- Quiet hours: {self.profile.quiet_hours_start} to {self.profile.quiet_hours_end} in timezone {self.profile.timezone}."
            )
        if self.status.current_activity:
            lines.append(f"- Current temporary activity: {self.status.current_activity}.")
        if self.status.posture_monitoring_active:
            lines.append("- Posture monitoring is active.")
            lines.append(
                f"- Posture coaching cadence: every {round(self.status.posture_reminder_cooldown_seconds / 60.0, 2)} minutes."
            )
            lines.append(
                f"- Posture preview: enabled={self.status.posture_preview_enabled}, active={self.status.posture_preview_active}."
            )
            if self.status.latest_posture_label:
                posture_bits = [self.status.latest_posture_label]
                if self.status.latest_posture_reason_codes:
                    posture_bits.append(", ".join(self.status.latest_posture_reason_codes))
                lines.append(f"- Latest posture status: {' | '.join(posture_bits)}.")
            if self.status.last_posture_callback_event:
                lines.append(
                    f"- Last posture callback: {self.status.last_posture_callback_event} at {self.status.last_posture_callback_at}."
                )

        active_policies = []
        for reminder_type in KNOWN_REMINDER_TYPES:
            policy = self.policies.get(reminder_type)
            if not policy or not policy.enabled:
                continue
            label = reminder_type.replace("_", " ")
            if policy.interval_minutes:
                active_policies.append(f"{label} every {policy.interval_minutes} minutes")
            elif policy.window_start and policy.window_end:
                active_policies.append(f"{label} during {policy.window_start}-{policy.window_end}")
            else:
                active_policies.append(label)
        if active_policies:
            lines.append(f"- Active proactive schedule: {', '.join(active_policies)}.")
        if self.status.recent_summary:
            lines.append(f"- Recent conversation summary: {self.status.recent_summary}")
        note = "\n".join(lines)
        logger.info(
            "Status note built",
            extra={
                "chars": len(note),
                "preferred_name": self.profile.preferred_name,
                "current_activity": self.status.current_activity,
                "summary_chars": len(self.status.recent_summary),
                "posture_preview_active": self.status.posture_preview_active,
                "last_posture_callback_event": self.status.last_posture_callback_event,
            },
        )
        self._emit_data_flow(
            "status_note",
            note=note,
            profile=self._profile_for_log(),
            runtime=self._runtime_for_log(),
            policies=self._policies_for_log(),
        )
        return note

    async def search(self, query: str, *, limit: int | None = None) -> list[MemoryRecord]:
        self._push_status("memory_retrieve")
        try:
            embedding = await self._embed_or_none(query)
            return self.store.search(
                query,
                embedding=embedding,
                limit=limit or self.top_k,
            )
        finally:
            self._pop_status()

    def list_memories(self, *, limit: int = 10) -> list[MemoryRecord]:
        return self.store.list_memories(limit=limit)

    async def forget_memory(self, query: str) -> MemoryRecord | None:
        records = await self.search(query, limit=1)
        if not records:
            logger.info("Forget memory found no match", extra={"query_chars": len(query)})
            return None
        record = records[0]
        self.store.delete_memory(record.id)
        return record

    def forget_all_memories(self) -> int:
        return self.store.delete_all()

    async def prepare_llm_context(
        self,
        chat_ctx: llm.ChatContext,
        *,
        latest_user_text: str | None,
    ) -> llm.ChatContext:
        started_at = time.perf_counter()
        controlled = chat_ctx.copy(
            exclude_config_update=True,
        )
        before_count = len(controlled.items)
        before_types = _context_shape(controlled)
        controlled.truncate(max_items=self.short_window_messages)
        note = self.build_status_note()
        latest_message = latest_user_message(controlled)
        created_at = (
            max(0.0, latest_message.created_at - 0.001)
            if latest_message is not None
            else time.time()
        )
        controlled.add_message(role="assistant", content=note, created_at=created_at)
        logger.info(
            "LLM context prepared",
            extra={
                "items_before": before_count,
                "items_after": len(controlled.items),
                "status_note": True,
                "status_note_chars": len(note),
                "shape_before": before_types,
                "shape_after": _context_shape(controlled),
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
        )
        return controlled

    async def _embed_or_none(self, text: str) -> EmbeddingResult | None:
        try:
            return await self.embedder.embed(text)
        except EmbeddingUnavailableError as exc:
            logger.warning(
                "Embedding unavailable; using non-vector memory path",
                extra={"reason": str(exc)},
            )
            return None
        except Exception as exc:
            logger.error(
                "Unexpected embedding failure; using non-vector memory path",
                extra={"error_type": type(exc).__name__},
                exc_info=True,
            )
            return None

    def _push_status(self, state: str) -> None:
        if not self.status_reporter:
            return
        try:
            self.status_reporter.push_state(state)
        except Exception as exc:
            logger.warning(
                "Status reporter push failed",
                extra={"state": state, "error_type": type(exc).__name__},
            )

    def _pop_status(self) -> None:
        if not self.status_reporter:
            return
        try:
            self.status_reporter.pop_state()
        except Exception as exc:
            logger.warning(
                "Status reporter pop failed",
                extra={"error_type": type(exc).__name__},
            )

    def status_summary(self, *, now: datetime | None = None) -> dict[str, Any]:
        day = _iso_day(self.profile.timezone, now=now)
        today = self.store.ensure_daily_row(day)
        self._expire_temporary_state(now=time.time())
        summary = {
            "profile": {
                "preferred_name": self.profile.preferred_name,
                "timezone": self.profile.timezone,
                "quiet_hours_start": self.profile.quiet_hours_start,
                "quiet_hours_end": self.profile.quiet_hours_end,
            },
            "runtime": {
                "current_activity": self.status.current_activity,
                "activity_expires_at": self.status.activity_expires_at,
                "busy_state": self.status.busy_state,
                "busy_expires_at": self.status.busy_expires_at,
                "recent_summary": self.status.recent_summary,
                "last_user_interaction_at": self.status.last_user_interaction_at,
                "last_assistant_speech_at": self.status.last_assistant_speech_at,
                "last_water_reminder_at": self.status.last_water_reminder_at,
                "last_check_in_at": self.status.last_check_in_at,
                "last_nap_suggestion_at": self.status.last_nap_suggestion_at,
                "water_cooldown_until": self.status.water_cooldown_until,
                "check_in_cooldown_until": self.status.check_in_cooldown_until,
                "nap_cooldown_until": self.status.nap_cooldown_until,
                "posture_monitoring_active": self.status.posture_monitoring_active,
                "posture_workflow_active": self.status.posture_workflow_active,
                "posture_session_id": self.status.posture_session_id,
                "latest_posture_label": self.status.latest_posture_label,
                "latest_posture_severity": self.status.latest_posture_severity,
                "latest_posture_reason_codes": list(self.status.latest_posture_reason_codes),
                "latest_posture_metrics": dict(self.status.latest_posture_metrics),
                "latest_posture_prompt_key": self.status.latest_posture_prompt_key,
                "last_posture_event_at": self.status.last_posture_event_at,
                "posture_preview_enabled": self.status.posture_preview_enabled,
                "posture_preview_active": self.status.posture_preview_active,
                "last_posture_callback_at": self.status.last_posture_callback_at,
                "last_posture_callback_event": self.status.last_posture_callback_event,
                "posture_reminder_cooldown_seconds": self.status.posture_reminder_cooldown_seconds,
                "posture_issue_cooldowns": dict(self.status.posture_issue_cooldowns),
            },
            "policies": {
                key: {
                    "enabled": policy.enabled,
                    "interval_minutes": policy.interval_minutes,
                    "window_start": policy.window_start,
                    "window_end": policy.window_end,
                    "max_per_day": policy.max_per_day,
                }
                for key, policy in self.policies.items()
            },
            "today": asdict(today),
        }
        logger.info(
            "Status summary prepared",
            extra={
                "day": day,
                "preferred_name": self.profile.preferred_name,
                "current_activity": self.status.current_activity,
                "recent_summary_chars": len(self.status.recent_summary),
                "posture_preview_active": self.status.posture_preview_active,
                "last_posture_callback_event": self.status.last_posture_callback_event,
            },
        )
        self._emit_data_flow("status_summary", summary=summary)
        return summary

    def update_reminder_settings(
        self,
        reminder_type: str,
        *,
        enabled: bool | None = None,
        interval_minutes: int | None = None,
        snooze_minutes: int | None = None,
        quiet_hours_start: str | None = None,
        quiet_hours_end: str | None = None,
    ) -> dict[str, Any]:
        raw_args = {
            "reminder_type": reminder_type,
            "enabled": enabled,
            "interval_minutes": interval_minutes,
            "snooze_minutes": snooze_minutes,
            "quiet_hours_start": quiet_hours_start,
            "quiet_hours_end": quiet_hours_end,
        }
        logger.info("Reminder settings update requested", extra={"raw_args": _json_safe(raw_args)})

        reminder_key = _normalize_reminder_type(reminder_type)
        normalized_enabled = _normalize_optional_bool(enabled, field_name="enabled")
        normalized_interval = _normalize_optional_int(
            interval_minutes,
            field_name="interval_minutes",
            minimum=1,
            maximum=24 * 60,
        )
        normalized_snooze = _normalize_optional_int(
            snooze_minutes,
            field_name="snooze_minutes",
            minimum=1,
            maximum=24 * 60,
        )
        normalized_quiet_start = _normalize_optional_time(
            quiet_hours_start,
            field_name="quiet_hours_start",
        )
        normalized_quiet_end = _normalize_optional_time(
            quiet_hours_end,
            field_name="quiet_hours_end",
        )
        if (
            normalized_enabled is None
            and normalized_interval is None
            and normalized_snooze is None
            and normalized_quiet_start is None
            and normalized_quiet_end is None
        ):
            raise ValueError("No reminder setting change was provided")

        normalized_args = {
            "reminder_type": reminder_key,
            "enabled": normalized_enabled,
            "interval_minutes": normalized_interval,
            "snooze_minutes": normalized_snooze,
            "quiet_hours_start": normalized_quiet_start,
            "quiet_hours_end": normalized_quiet_end,
        }
        logger.info(
            "Reminder settings arguments normalized",
            extra={"normalized_args": _json_safe(normalized_args)},
        )
        policy = self.policies[reminder_key]
        if normalized_enabled is not None:
            policy.enabled = normalized_enabled
        if normalized_interval is not None:
            policy.interval_minutes = normalized_interval
        if normalized_quiet_start is not None:
            self.profile.quiet_hours_start = normalized_quiet_start
        if normalized_quiet_end is not None:
            self.profile.quiet_hours_end = normalized_quiet_end
        if normalized_quiet_start is not None or normalized_quiet_end is not None:
            self.store.save_profile(self.profile)
        self.store.save_reminder_policy(policy)
        if normalized_snooze:
            until = time.time() + normalized_snooze * 60
            if reminder_key == "drink_water":
                self.status.water_cooldown_until = until
            elif reminder_key == "nap":
                self.status.nap_cooldown_until = until
            elif reminder_key == "check_in":
                self.status.check_in_cooldown_until = until
            self.store.save_status(self.status)
            self.record_proactive_outcome(reminder_key, "snoozed")
        summary = self.status_summary()
        logger.info(
            "Reminder settings updated",
            extra={
                "reminder_type": reminder_key,
                "enabled": policy.enabled,
                "interval_minutes": policy.interval_minutes,
                "quiet_hours_start": self.profile.quiet_hours_start,
                "quiet_hours_end": self.profile.quiet_hours_end,
                "snooze_minutes": normalized_snooze,
            },
        )
        self._emit_data_flow(
            "status_settings_update",
            reminder_type=reminder_key,
            enabled=policy.enabled,
            interval_minutes=policy.interval_minutes,
            quiet_hours_start=self.profile.quiet_hours_start,
            quiet_hours_end=self.profile.quiet_hours_end,
            snooze_minutes=normalized_snooze,
            summary=summary,
        )
        return summary

    def set_water_reminder_interval(self, interval_minutes: int | str) -> dict[str, Any]:
        normalized_interval = _normalize_optional_int(
            interval_minutes,
            field_name="interval_minutes",
            minimum=1,
            maximum=24 * 60,
        )
        if normalized_interval is None:
            raise ValueError("interval_minutes is required")
        return self.update_reminder_settings(
            "drink_water",
            enabled=True,
            interval_minutes=normalized_interval,
        )

    def set_check_in_interval(self, interval_minutes: int | str) -> dict[str, Any]:
        normalized_interval = _normalize_optional_int(
            interval_minutes,
            field_name="interval_minutes",
            minimum=1,
            maximum=24 * 60,
        )
        if normalized_interval is None:
            raise ValueError("interval_minutes is required")
        return self.update_reminder_settings(
            "check_in",
            enabled=True,
            interval_minutes=normalized_interval,
        )

    def set_posture_coaching_interval(self, interval_minutes: int | str) -> dict[str, Any]:
        normalized_interval = _normalize_optional_int(
            interval_minutes,
            field_name="interval_minutes",
            minimum=1,
            maximum=24 * 60,
        )
        if normalized_interval is None:
            raise ValueError("interval_minutes is required")
        self.status.posture_reminder_cooldown_seconds = float(normalized_interval * 60)
        self.store.save_status(self.status)
        summary = self.status_summary()
        logger.info(
            "Posture coaching cadence updated",
            extra={
                "interval_minutes": normalized_interval,
                "posture_reminder_cooldown_seconds": self.status.posture_reminder_cooldown_seconds,
            },
        )
        self._emit_data_flow(
            "status_posture_cadence_update",
            interval_minutes=normalized_interval,
            posture_reminder_cooldown_seconds=self.status.posture_reminder_cooldown_seconds,
            summary=summary,
        )
        return summary

    def set_reminder_enabled(self, reminder_type: str, enabled: bool | str) -> dict[str, Any]:
        normalized_enabled = _normalize_optional_bool(enabled, field_name="enabled")
        if normalized_enabled is None:
            raise ValueError("enabled is required")
        return self.update_reminder_settings(
            reminder_type,
            enabled=normalized_enabled,
        )

    def get_reminder_policy(self, reminder_type: str) -> dict[str, Any]:
        reminder_key = _normalize_reminder_type(reminder_type)
        policy = self.policies[reminder_key]
        result = {
            "reminder_type": reminder_key,
            "enabled": policy.enabled,
            "interval_minutes": policy.interval_minutes,
            "window_start": policy.window_start,
            "window_end": policy.window_end,
            "max_per_day": policy.max_per_day,
            "updated_at": policy.updated_at,
        }
        logger.info("Reminder policy prepared", extra={"reminder_type": reminder_key})
        self._emit_data_flow("reminder_policy", reminder_type=reminder_key, policy=result)
        return result

    def snooze_reminder(self, reminder_type: str, minutes: int | str) -> dict[str, Any]:
        normalized_minutes = _normalize_optional_int(
            minutes,
            field_name="minutes",
            minimum=1,
            maximum=24 * 60,
        )
        if normalized_minutes is None:
            raise ValueError("minutes is required")
        return self.update_reminder_settings(
            reminder_type,
            snooze_minutes=normalized_minutes,
        )

    def clear_reminder_snooze(self, reminder_type: str) -> dict[str, Any]:
        reminder_key = _normalize_reminder_type(reminder_type)
        if reminder_key == "drink_water":
            self.status.water_cooldown_until = None
        elif reminder_key == "nap":
            self.status.nap_cooldown_until = None
        elif reminder_key == "check_in":
            self.status.check_in_cooldown_until = None
        self.store.save_status(self.status)
        summary = self.status_summary()
        logger.info("Reminder snooze cleared", extra={"reminder_type": reminder_key})
        self._emit_data_flow(
            "status_snooze_cleared",
            reminder_type=reminder_key,
            summary=summary,
        )
        return summary

    def reset_reminder_policy(self, reminder_type: str) -> dict[str, Any]:
        reminder_key = _normalize_reminder_type(reminder_type)
        defaults = default_reminder_policies()[reminder_key]
        self.policies[reminder_key] = ReminderPolicy(
            reminder_type=defaults.reminder_type,
            enabled=defaults.enabled,
            interval_minutes=defaults.interval_minutes,
            window_start=defaults.window_start,
            window_end=defaults.window_end,
            max_per_day=defaults.max_per_day,
        )
        self.store.save_reminder_policy(self.policies[reminder_key])
        if reminder_key == "drink_water":
            self.status.water_cooldown_until = None
            self.status.last_water_reminder_at = None
        elif reminder_key == "nap":
            self.status.nap_cooldown_until = None
            self.status.last_nap_suggestion_at = None
        elif reminder_key == "check_in":
            self.status.check_in_cooldown_until = None
            self.status.last_check_in_at = None
        self.store.save_status(self.status)
        summary = self.status_summary()
        logger.info("Reminder policy reset", extra={"reminder_type": reminder_key})
        self._emit_data_flow(
            "status_policy_reset",
            reminder_type=reminder_key,
            summary=summary,
        )
        return summary

    def set_quiet_hours(self, start: str, end: str) -> dict[str, Any]:
        normalized_start = _normalize_optional_time(start, field_name="start")
        normalized_end = _normalize_optional_time(end, field_name="end")
        if normalized_start is None or normalized_end is None:
            raise ValueError("start and end are required")
        self.profile.quiet_hours_start = normalized_start
        self.profile.quiet_hours_end = normalized_end
        self.store.save_profile(self.profile)
        summary = self.status_summary()
        logger.info(
            "Quiet hours updated",
            extra={
                "quiet_hours_start": self.profile.quiet_hours_start,
                "quiet_hours_end": self.profile.quiet_hours_end,
            },
        )
        self._emit_data_flow(
            "status_quiet_hours_update",
            quiet_hours_start=self.profile.quiet_hours_start,
            quiet_hours_end=self.profile.quiet_hours_end,
            summary=summary,
        )
        return summary

    def reset_quiet_hours(self) -> dict[str, Any]:
        self.profile.quiet_hours_start = DEFAULT_QUIET_HOURS_START
        self.profile.quiet_hours_end = DEFAULT_QUIET_HOURS_END
        self.store.save_profile(self.profile)
        summary = self.status_summary()
        logger.info(
            "Quiet hours reset",
            extra={
                "quiet_hours_start": self.profile.quiet_hours_start,
                "quiet_hours_end": self.profile.quiet_hours_end,
            },
        )
        self._emit_data_flow(
            "status_quiet_hours_reset",
            quiet_hours_start=self.profile.quiet_hours_start,
            quiet_hours_end=self.profile.quiet_hours_end,
            summary=summary,
        )
        return summary

    def update_user_profile(
        self,
        *,
        preferred_name: str | None = None,
        timezone: str | None = None,
    ) -> dict[str, Any]:
        normalized_name = _normalize_optional_name(preferred_name, field_name="preferred_name")
        normalized_timezone = _normalize_optional_timezone(timezone, field_name="timezone")
        if normalized_name is None and normalized_timezone is None:
            raise ValueError("Provide preferred_name or timezone to update the profile")
        if normalized_name is not None:
            self.profile.preferred_name = normalized_name
        if normalized_timezone is not None:
            self.profile.timezone = normalized_timezone
        self.store.save_profile(self.profile)
        summary = self.status_summary()
        logger.info(
            "User profile updated",
            extra={
                "preferred_name": self.profile.preferred_name,
                "timezone": self.profile.timezone,
            },
        )
        self._emit_data_flow(
            "status_profile_update",
            preferred_name=self.profile.preferred_name,
            timezone=self.profile.timezone,
            summary=summary,
        )
        return summary

    def clear_preferred_name(self) -> dict[str, Any]:
        self.profile.preferred_name = None
        self.store.save_profile(self.profile)
        summary = self.status_summary()
        logger.info("Preferred name cleared")
        self._emit_data_flow("status_profile_name_cleared", summary=summary)
        return summary

    def set_current_activity(self, activity: str, *, duration_minutes: int | None = None) -> dict[str, Any]:
        self._apply_activity(activity.strip().lower(), duration_minutes=duration_minutes)
        self.store.save_status(self.status)
        summary = self.status_summary()
        logger.info(
            "Current activity updated",
            extra={
                "activity": self.status.current_activity,
                "activity_expires_at": self.status.activity_expires_at,
                "busy_state": self.status.busy_state,
            },
        )
        self._emit_data_flow(
            "status_activity_update",
            activity=self.status.current_activity,
            activity_expires_at=self.status.activity_expires_at,
            busy_state=self.status.busy_state,
            summary=summary,
        )
        return summary

    def clear_temporary_status(self) -> dict[str, Any]:
        self.status.current_activity = None
        self.status.activity_expires_at = None
        self.status.busy_state = None
        self.status.busy_expires_at = None
        self.store.save_status(self.status)
        summary = self.status_summary()
        logger.info("Temporary status cleared")
        self._emit_data_flow("status_activity_cleared", summary=summary)
        return summary

    def clear_recent_summary(self) -> dict[str, Any]:
        self.status.summary_entries = []
        self.store.save_status(self.status)
        summary = self.status_summary()
        logger.info("Recent conversation summary cleared")
        self._emit_data_flow("status_summary_cleared", summary=summary)
        return summary

    def start_posture_monitoring(
        self,
        session_id: str,
        *,
        preview_enabled: bool = False,
        preview_active: bool = False,
    ) -> dict[str, Any]:
        if not session_id.strip():
            raise ValueError("session_id is required")
        self.status.posture_monitoring_active = True
        self.status.posture_workflow_active = True
        self.status.posture_session_id = session_id.strip()
        self.status.latest_posture_label = None
        self.status.latest_posture_severity = None
        self.status.latest_posture_reason_codes = []
        self.status.latest_posture_metrics = {}
        self.status.latest_posture_prompt_key = None
        self.status.last_posture_event_at = None
        self.status.posture_issue_cooldowns = {}
        self.status.posture_preview_enabled = bool(preview_enabled)
        self.status.posture_preview_active = bool(preview_active)
        self.status.last_posture_callback_at = None
        self.status.last_posture_callback_event = None
        self.store.save_status(self.status)
        summary = self.status_summary()
        logger.info(
            "Posture monitoring started",
            extra={
                "posture_session_id": self.status.posture_session_id,
                "posture_preview_enabled": self.status.posture_preview_enabled,
                "posture_preview_active": self.status.posture_preview_active,
            },
        )
        self._emit_data_flow(
            "posture_monitoring_started",
            posture_session_id=self.status.posture_session_id,
            posture_preview_enabled=self.status.posture_preview_enabled,
            posture_preview_active=self.status.posture_preview_active,
            summary=summary,
        )
        return summary

    def stop_posture_monitoring(self) -> dict[str, Any]:
        self.status.posture_monitoring_active = False
        self.status.posture_workflow_active = False
        self.status.posture_session_id = None
        self.status.latest_posture_label = None
        self.status.latest_posture_severity = None
        self.status.latest_posture_reason_codes = []
        self.status.latest_posture_metrics = {}
        self.status.latest_posture_prompt_key = None
        self.status.last_posture_event_at = None
        self.status.posture_issue_cooldowns = {}
        self.status.posture_preview_enabled = False
        self.status.posture_preview_active = False
        self.status.last_posture_callback_at = None
        self.status.last_posture_callback_event = None
        self.store.save_status(self.status)
        summary = self.status_summary()
        logger.info("Posture monitoring stopped")
        self._emit_data_flow("posture_monitoring_stopped", summary=summary)
        return summary

    def posture_monitoring_status(self) -> dict[str, Any]:
        runtime = self.status_summary()["runtime"]
        result = {
            "active": runtime["posture_monitoring_active"],
            "workflow_active": runtime["posture_workflow_active"],
            "session_id": runtime["posture_session_id"],
            "latest_posture_label": runtime["latest_posture_label"],
            "latest_posture_severity": runtime["latest_posture_severity"],
            "latest_posture_reason_codes": runtime["latest_posture_reason_codes"],
            "latest_posture_metrics": runtime["latest_posture_metrics"],
            "latest_posture_prompt_key": runtime["latest_posture_prompt_key"],
            "last_posture_event_at": runtime["last_posture_event_at"],
            "preview_enabled": runtime["posture_preview_enabled"],
            "preview_active": runtime["posture_preview_active"],
            "last_callback_at": runtime["last_posture_callback_at"],
            "last_callback_event": runtime["last_posture_callback_event"],
            "coaching_interval_seconds": runtime["posture_reminder_cooldown_seconds"],
            "coaching_interval_minutes": round(
                float(runtime["posture_reminder_cooldown_seconds"]) / 60.0,
                2,
            ),
        }
        logger.info(
            "Posture monitoring status prepared",
            extra={"active": result["active"], "session_id": result["session_id"]},
        )
        self._emit_data_flow("posture_monitoring_status", status=result)
        return result

    def ingest_posture_event(
        self,
        *,
        session_id: str,
        event_name: str,
        timestamp: str | None = None,
        severity: str | None = None,
        posture_label: str | None = None,
        reason_codes: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
        prompt_key: str | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        normalized_session_id = session_id.strip()
        if not self.status.posture_monitoring_active or self.status.posture_session_id != normalized_session_id:
            logger.info(
                "Ignoring posture event for inactive or mismatched session",
                extra={
                    "posture_session_id": self.status.posture_session_id,
                    "incoming_session_id": normalized_session_id,
                    "event_name": event_name,
                },
            )
            return {"accepted": False, "reason": "session_mismatch"}
        normalized_event = event_name.strip().lower()
        normalized_reasons = [
            str(code).strip().replace("_", " ").lower()
            for code in (reason_codes or [])
            if str(code).strip()
        ]
        normalized_metrics = {
            str(key): float(value)
            for key, value in (metrics or {}).items()
            if isinstance(value, (int, float))
        }
        callback_received_at = time.time()
        self.status.last_posture_event_at = callback_received_at
        self.status.last_posture_callback_at = callback_received_at
        self.status.last_posture_callback_event = normalized_event
        if normalized_event == "posture.normal":
            self.status.latest_posture_label = "standard"
            self.status.latest_posture_severity = "normal"
            self.status.latest_posture_reason_codes = []
            self.status.latest_posture_metrics = normalized_metrics
            self.status.latest_posture_prompt_key = prompt_key
        elif normalized_event == "camera.error":
            self.status.latest_posture_label = "insufficient_data"
            self.status.latest_posture_severity = severity or "warning"
            self.status.latest_posture_reason_codes = normalized_reasons or ["camera error"]
            self.status.latest_posture_metrics = normalized_metrics
            self.status.latest_posture_prompt_key = prompt_key or "remind_adjust_camera"
        elif normalized_event == "session.stopped":
            self.status.posture_monitoring_active = False
            self.status.posture_workflow_active = False
            self.status.posture_preview_active = False
        else:
            self.status.latest_posture_label = posture_label or "needs_adjustment"
            self.status.latest_posture_severity = severity or "mild"
            self.status.latest_posture_reason_codes = normalized_reasons
            self.status.latest_posture_metrics = normalized_metrics
            self.status.latest_posture_prompt_key = prompt_key
        self.store.save_status(self.status)
        result = {
            "accepted": True,
            "session_id": normalized_session_id,
            "event_name": normalized_event,
            "latest_posture_label": self.status.latest_posture_label,
            "latest_posture_severity": self.status.latest_posture_severity,
            "latest_posture_reason_codes": list(self.status.latest_posture_reason_codes),
            "latest_posture_metrics": dict(self.status.latest_posture_metrics),
            "message": message,
            "timestamp": timestamp,
        }
        logger.info(
            "Posture event ingested",
            extra={
                "event_name": normalized_event,
                "posture_session_id": normalized_session_id,
                "latest_posture_label": self.status.latest_posture_label,
                "latest_posture_severity": self.status.latest_posture_severity,
                "last_posture_callback_at": self.status.last_posture_callback_at,
                "last_posture_callback_event": self.status.last_posture_callback_event,
            },
        )
        self._emit_data_flow("posture_runtime_updated", result=result, runtime=self._runtime_for_log())
        return result

    def daily_status(self, day: str) -> dict[str, Any]:
        resolved = self._resolve_day(day)
        status = self.store.get_daily_status(resolved) or DailyStatus(day=resolved)
        result = asdict(status)
        logger.info("Daily status prepared", extra={"day": resolved})
        self._emit_data_flow("daily_status", day=resolved, status=result)
        return result

    def recent_status_history(self, *, days: int) -> list[dict[str, Any]]:
        history = [asdict(item) for item in self.store.get_recent_daily_status(limit=days)]
        logger.info("Recent status history prepared", extra={"days": days, "rows": len(history)})
        self._emit_data_flow("recent_status_history", days=days, history=history)
        return history

    def delete_daily_status(self, day: str) -> dict[str, Any]:
        resolved = self._resolve_day(day)
        deleted = self.store.delete_daily_status(resolved)
        result = {"day": resolved, "deleted": deleted}
        logger.info("Daily status deleted", extra=result)
        self._emit_data_flow("daily_status_deleted", **result)
        return result

    def _resolve_day(self, day: str) -> str:
        lowered = day.strip().lower()
        now_local = _now_in_timezone(self.profile.timezone)
        if lowered == "today":
            return now_local.date().isoformat()
        if lowered == "yesterday":
            return (now_local.date() - timedelta(days=1)).isoformat()
        datetime.fromisoformat(day)
        return day

    def is_recently_active(self, *, now: float | None = None) -> bool:
        last = self.status.last_user_interaction_at
        if last is None:
            return False
        return (now or time.time()) - last <= self._recent_activity_window_seconds

    def record_assistant_speech(self, *, text: str | None = None, now: float | None = None) -> None:
        if not (text or "").strip():
            return
        now_ts = now or time.time()
        self.status.last_assistant_speech_at = now_ts
        self.store.save_status(self.status)
        logger.info(
            "Assistant speech recorded",
            extra={
                "chars": len((text or "").strip()),
                "last_assistant_speech_at": self.status.last_assistant_speech_at,
            },
        )
        self._emit_data_flow(
            "assistant_speech_recorded",
            chars=len((text or "").strip()),
            runtime=self._runtime_for_log(),
        )

    def proactive_gate_reason(self, *, now: float | None = None) -> str | None:
        now_ts = now or time.time()
        if self.status.last_user_interaction_at is not None:
            elapsed = now_ts - self.status.last_user_interaction_at
            if elapsed < self._post_user_grace_seconds:
                return "post_user_grace"
        if self.status.last_assistant_speech_at is not None:
            elapsed = now_ts - self.status.last_assistant_speech_at
            if elapsed < self._post_assistant_grace_seconds:
                return "post_assistant_grace"
        return None

    def _next_posture_action(self, *, now_ts: float) -> ProactiveAction | None:
        if not self.status.posture_monitoring_active:
            self._emit_data_flow(
                "posture_reminder_decision",
                decision="skip",
                reason="monitoring_inactive",
                runtime=self._runtime_for_log(),
            )
            return None
        if self.status.last_posture_event_at is None:
            self._emit_data_flow(
                "posture_reminder_decision",
                decision="skip",
                reason="no_posture_event",
                runtime=self._runtime_for_log(),
            )
            return None
        if now_ts - self.status.last_posture_event_at > self._posture_event_stale_seconds:
            self._emit_data_flow(
                "posture_reminder_decision",
                decision="skip",
                reason="stale_event",
                event_age_seconds=round(now_ts - self.status.last_posture_event_at, 2),
                stale_after_seconds=self._posture_event_stale_seconds,
                runtime=self._runtime_for_log(),
            )
            return None
        posture_label = self.status.latest_posture_label
        if posture_label in {None, "standard"}:
            self._emit_data_flow(
                "posture_reminder_decision",
                decision="skip",
                reason="posture_standard",
                posture_label=posture_label,
                runtime=self._runtime_for_log(),
            )
            return None
        issue_key = _normalize_posture_issue_key(
            reason_codes=self.status.latest_posture_reason_codes,
            prompt_key=self.status.latest_posture_prompt_key,
            posture_label=posture_label,
        )
        cooldown_until = self.status.posture_issue_cooldowns.get(issue_key)
        if cooldown_until is not None and now_ts < cooldown_until:
            logger.info(
                "Posture reminder skipped due to cooldown",
                extra={"issue_key": issue_key, "cooldown_until": cooldown_until},
            )
            self._emit_data_flow(
                "posture_reminder_decision",
                decision="skip",
                reason="cooldown_active",
                issue_key=issue_key,
                cooldown_until=cooldown_until,
                runtime=self._runtime_for_log(),
            )
            return None
        text = _posture_guidance_text(issue_key)
        logger.info(
            "Posture reminder due",
            extra={"issue_key": issue_key, "text": text, "posture_label": posture_label},
        )
        self._emit_data_flow(
            "posture_reminder_decision",
            decision="speak",
            issue_key=issue_key,
            posture_label=posture_label,
            latest_posture_reason_codes=list(self.status.latest_posture_reason_codes),
            latest_posture_prompt_key=self.status.latest_posture_prompt_key,
            text=text,
        )
        return ProactiveAction(
            reminder_type=f"posture:{issue_key}",
            mode="say",
            text=text,
        )

    def next_proactive_action(self, *, now: datetime | None = None) -> ProactiveAction | None:
        now_dt = now or datetime.now(UTC)
        now_ts = now_dt.timestamp()
        self._expire_temporary_state(now=now_ts)
        if _in_quiet_hours(self.profile, now=now_dt):
            logger.info("Proactive action skipped", extra={"reason": "quiet_hours"})
            return None
        if not self.is_recently_active(now=now_ts):
            logger.info("Proactive action skipped", extra={"reason": "inactive_user"})
            return None
        posture_action = self._next_posture_action(now_ts=now_ts)
        if posture_action is not None:
            return posture_action
        now_local = _now_in_timezone(self.profile.timezone, now=now_dt)
        day = now_local.date().isoformat()
        today = self.store.ensure_daily_row(day)

        if self._is_due("drink_water", now_local=now_local, now_ts=now_ts):
            action = ProactiveAction(
                reminder_type="drink_water",
                mode="say",
                text="Quick reminder to drink some water.",
            )
            logger.info("Proactive action due", extra={"reminder_type": action.reminder_type, "mode": action.mode})
            return action

        if self.status.current_activity not in {"napping", "out"} and self._is_due(
            "nap",
            now_local=now_local,
            now_ts=now_ts,
            today=today,
        ):
            action = ProactiveAction(
                reminder_type="nap",
                mode="say",
                text="It is afternoon. If you are tired, this could be a good time for a short nap.",
            )
            logger.info("Proactive action due", extra={"reminder_type": action.reminder_type, "mode": action.mode})
            return action

        if self.status.busy_state is None and self._is_due(
            "check_in",
            now_local=now_local,
            now_ts=now_ts,
            today=today,
        ):
            action = ProactiveAction(
                reminder_type="check_in",
                mode="generate_reply",
                instructions=(
                    "Offer one short proactive check-in. Ask what the user is doing right now "
                    "or whether they want help with their current task. Keep it brief and natural."
                ),
            )
            logger.info("Proactive action due", extra={"reminder_type": action.reminder_type, "mode": action.mode})
            return action
        logger.info("Proactive action skipped", extra={"reason": "no_due_action"})
        return None

    def _is_due(
        self,
        reminder_type: str,
        *,
        now_local: datetime,
        now_ts: float,
        today: DailyStatus | None = None,
    ) -> bool:
        policy = self.policies.get(reminder_type)
        if not policy or not policy.enabled:
            return False
        if not _time_in_window(now_local, policy.window_start, policy.window_end):
            return False
        if reminder_type == "drink_water":
            if self.status.water_cooldown_until and now_ts < self.status.water_cooldown_until:
                logger.debug("Reminder not due due to cooldown", extra={"reminder_type": reminder_type})
                return False
            last = self.status.last_water_reminder_at
            return last is None or (
                policy.interval_minutes is not None
                and now_ts - last >= policy.interval_minutes * 60
            )
        if reminder_type == "nap":
            if self.status.nap_cooldown_until and now_ts < self.status.nap_cooldown_until:
                logger.debug("Reminder not due due to cooldown", extra={"reminder_type": reminder_type})
                return False
            current_day = today or self.store.ensure_daily_row(now_local.date().isoformat())
            return current_day.nap_suggestions_sent < max(1, policy.max_per_day)
        if reminder_type == "check_in":
            if self.status.check_in_cooldown_until and now_ts < self.status.check_in_cooldown_until:
                logger.debug("Reminder not due due to cooldown", extra={"reminder_type": reminder_type})
                return False
            last = self.status.last_check_in_at
            return last is None or (
                policy.interval_minutes is not None
                and now_ts - last >= policy.interval_minutes * 60
            )
        return False

    def record_proactive_outcome(
        self,
        reminder_type: str,
        outcome: str,
        *,
        now: datetime | None = None,
    ) -> None:
        now_dt = now or datetime.now(UTC)
        now_ts = now_dt.timestamp()
        day = _iso_day(self.profile.timezone, now=now_dt)
        if reminder_type == "drink_water":
            if outcome == "sent":
                self.status.last_water_reminder_at = now_ts
            elif outcome == "snoozed":
                self.status.water_cooldown_until = now_ts + 30 * 60
        elif reminder_type == "nap":
            if outcome == "sent":
                self.status.last_nap_suggestion_at = now_ts
            elif outcome == "snoozed":
                self.status.nap_cooldown_until = now_ts + 60 * 60
        elif reminder_type == "check_in":
            if outcome == "sent":
                self.status.last_check_in_at = now_ts
            elif outcome == "snoozed":
                self.status.check_in_cooldown_until = now_ts + 60 * 60
        self.store.save_status(self.status)
        counter_field = _reminder_counter_field(reminder_type, outcome)
        if counter_field:
            self.store.increment_daily_counter(day, counter_field)
        self.store.append_proactive_event(
            f"{reminder_type}.{outcome}",
            {"reminder_type": reminder_type, "outcome": outcome, "day": day},
        )
        logger.info(
            "Proactive outcome recorded",
            extra={
                "reminder_type": reminder_type,
                "outcome": outcome,
                "day": day,
                "last_water_reminder_at": self.status.last_water_reminder_at,
                "last_nap_suggestion_at": self.status.last_nap_suggestion_at,
                "last_check_in_at": self.status.last_check_in_at,
            },
        )
        self._emit_data_flow(
            "proactive_outcome",
            reminder_type=reminder_type,
            outcome=outcome,
            day=day,
            runtime=self._runtime_for_log(),
        )

    def record_posture_reminder_sent(self, reminder_type: str, *, now: datetime | None = None) -> None:
        now_dt = now or datetime.now(UTC)
        now_ts = now_dt.timestamp()
        issue_key = reminder_type.split(":", 1)[1] if ":" in reminder_type else reminder_type
        self.status.posture_issue_cooldowns[issue_key] = (
            now_ts + self.status.posture_reminder_cooldown_seconds
        )
        self.store.save_status(self.status)
        logger.info(
            "Posture reminder recorded",
            extra={
                "issue_key": issue_key,
                "cooldown_until": self.status.posture_issue_cooldowns[issue_key],
                "posture_reminder_cooldown_seconds": self.status.posture_reminder_cooldown_seconds,
            },
        )
        self._emit_data_flow(
            "posture_reminder_spoken",
            issue_key=issue_key,
            cooldown_until=self.status.posture_issue_cooldowns[issue_key],
            runtime=self._runtime_for_log(),
        )

    def _profile_for_log(self) -> dict[str, Any]:
        return {
            "preferred_name": self.profile.preferred_name,
            "timezone": self.profile.timezone,
            "quiet_hours_start": self.profile.quiet_hours_start,
            "quiet_hours_end": self.profile.quiet_hours_end,
        }

    def _runtime_for_log(self) -> dict[str, Any]:
        return {
            "current_activity": self.status.current_activity,
            "activity_expires_at": self.status.activity_expires_at,
            "busy_state": self.status.busy_state,
            "busy_expires_at": self.status.busy_expires_at,
            "recent_summary": self.status.recent_summary,
            "last_user_interaction_at": self.status.last_user_interaction_at,
            "last_assistant_speech_at": self.status.last_assistant_speech_at,
            "last_water_reminder_at": self.status.last_water_reminder_at,
            "last_check_in_at": self.status.last_check_in_at,
            "last_nap_suggestion_at": self.status.last_nap_suggestion_at,
            "water_cooldown_until": self.status.water_cooldown_until,
            "check_in_cooldown_until": self.status.check_in_cooldown_until,
            "nap_cooldown_until": self.status.nap_cooldown_until,
            "posture_monitoring_active": self.status.posture_monitoring_active,
            "posture_workflow_active": self.status.posture_workflow_active,
            "posture_session_id": self.status.posture_session_id,
            "latest_posture_label": self.status.latest_posture_label,
            "latest_posture_severity": self.status.latest_posture_severity,
            "latest_posture_reason_codes": list(self.status.latest_posture_reason_codes),
            "latest_posture_metrics": dict(self.status.latest_posture_metrics),
            "latest_posture_prompt_key": self.status.latest_posture_prompt_key,
            "last_posture_event_at": self.status.last_posture_event_at,
            "posture_preview_enabled": self.status.posture_preview_enabled,
            "posture_preview_active": self.status.posture_preview_active,
            "last_posture_callback_at": self.status.last_posture_callback_at,
            "last_posture_callback_event": self.status.last_posture_callback_event,
            "posture_reminder_cooldown_seconds": self.status.posture_reminder_cooldown_seconds,
            "posture_issue_cooldowns": dict(self.status.posture_issue_cooldowns),
        }

    def _policies_for_log(self) -> dict[str, Any]:
        return {
            key: {
                "enabled": policy.enabled,
                "interval_minutes": policy.interval_minutes,
                "window_start": policy.window_start,
                "window_end": policy.window_end,
                "max_per_day": policy.max_per_day,
            }
            for key, policy in self.policies.items()
        }

    def _emit_data_flow(self, event: str, **fields: Any) -> None:
        try:
            self._data_flow_logger.info(
                "Structured memory event",
                extra={"event": event, **_json_safe(fields)},
            )
        except Exception:
            logger.debug("Structured memory data-flow emit failed", exc_info=True)


class DisabledContextController:
    async def observe_user_turn(
        self, text: str | None, *, turn_id: str | None = None
    ) -> list[MemoryRecord]:
        logger.debug("Memory disabled: observe_user_turn ignored")
        return []

    async def prepare_llm_context(
        self,
        chat_ctx: llm.ChatContext,
        *,
        latest_user_text: str | None,
    ) -> llm.ChatContext:
        controlled = chat_ctx.copy(exclude_config_update=True)
        controlled.truncate(max_items=_env_int("MEMORY_SHORT_WINDOW_MESSAGES", 8))
        return controlled

    async def search(self, query: str, *, limit: int | None = None) -> list[MemoryRecord]:
        return []

    def list_memories(self, *, limit: int = 10) -> list[MemoryRecord]:
        return []

    async def forget_memory(self, query: str) -> MemoryRecord | None:
        return None

    def forget_all_memories(self) -> int:
        return 0

    def status_summary(self, *, now: datetime | None = None) -> dict[str, Any]:
        return {}

    def update_reminder_settings(self, reminder_type: str, **kwargs: Any) -> dict[str, Any]:
        return {}

    def set_water_reminder_interval(self, interval_minutes: int | str) -> dict[str, Any]:
        return {}

    def set_check_in_interval(self, interval_minutes: int | str) -> dict[str, Any]:
        return {}

    def set_posture_coaching_interval(self, interval_minutes: int | str) -> dict[str, Any]:
        return {}

    def set_reminder_enabled(self, reminder_type: str, enabled: bool | str) -> dict[str, Any]:
        return {}

    def get_reminder_policy(self, reminder_type: str) -> dict[str, Any]:
        return {}

    def snooze_reminder(self, reminder_type: str, minutes: int | str) -> dict[str, Any]:
        return {}

    def clear_reminder_snooze(self, reminder_type: str) -> dict[str, Any]:
        return {}

    def reset_reminder_policy(self, reminder_type: str) -> dict[str, Any]:
        return {}

    def set_quiet_hours(self, start: str, end: str) -> dict[str, Any]:
        return {}

    def reset_quiet_hours(self) -> dict[str, Any]:
        return {}

    def update_user_profile(
        self,
        *,
        preferred_name: str | None = None,
        timezone: str | None = None,
    ) -> dict[str, Any]:
        return {}

    def clear_preferred_name(self) -> dict[str, Any]:
        return {}

    def set_current_activity(self, activity: str, *, duration_minutes: int | None = None) -> dict[str, Any]:
        return {}

    def clear_temporary_status(self) -> dict[str, Any]:
        return {}

    def clear_recent_summary(self) -> dict[str, Any]:
        return {}

    def start_posture_monitoring(
        self,
        session_id: str,
        *,
        preview_enabled: bool = False,
        preview_active: bool = False,
    ) -> dict[str, Any]:
        return {}

    def stop_posture_monitoring(self) -> dict[str, Any]:
        return {}

    def posture_monitoring_status(self) -> dict[str, Any]:
        return {}

    def ingest_posture_event(self, **kwargs: Any) -> dict[str, Any]:
        return {}

    def daily_status(self, day: str) -> dict[str, Any]:
        return {}

    def recent_status_history(self, *, days: int) -> list[dict[str, Any]]:
        return []

    def delete_daily_status(self, day: str) -> dict[str, Any]:
        return {}

    def next_proactive_action(self, *, now: datetime | None = None) -> ProactiveAction | None:
        return None

    def record_proactive_outcome(
        self,
        reminder_type: str,
        outcome: str,
        *,
        now: datetime | None = None,
    ) -> None:
        return None

    def record_posture_reminder_sent(self, reminder_type: str, *, now: datetime | None = None) -> None:
        return None

    def is_recently_active(self, *, now: float | None = None) -> bool:
        return False

    def record_assistant_speech(self, *, text: str | None = None, now: float | None = None) -> None:
        return None

    def proactive_gate_reason(self, *, now: float | None = None) -> str | None:
        return None


def latest_user_message(chat_ctx: llm.ChatContext) -> llm.ChatMessage | None:
    for item in reversed(chat_ctx.items):
        if item.type == "message" and item.role == "user":
            return item
    return None


def latest_user_text(chat_ctx: llm.ChatContext) -> str | None:
    message = latest_user_message(chat_ctx)
    return message.text_content if message is not None else None


def has_memory_note(chat_ctx: llm.ChatContext) -> bool:
    for item in chat_ctx.items:
        if item.type == "message" and (item.text_content or "").startswith(
            STATUS_NOTE_PREFIX
        ):
            return True
    return False


def _context_shape(chat_ctx: llm.ChatContext) -> str:
    labels: list[str] = []
    for item in chat_ctx.items:
        if item.type == "message":
            labels.append(f"message:{item.role}")
        else:
            labels.append(item.type)
    return ",".join(labels)


async def call_default_llm_node(
    agent: Agent,
    chat_ctx: llm.ChatContext,
    tools: list[llm.Tool],
    model_settings: ModelSettings,
) -> AsyncIterable[llm.ChatChunk | str | Any]:
    result = Agent.default.llm_node(agent, chat_ctx, tools, model_settings)
    if hasattr(result, "__await__"):
        result = await result
    if isinstance(result, (str, llm.ChatChunk)) or result is None:

        async def single_item() -> AsyncIterable[llm.ChatChunk | str | Any]:
            if result is not None:
                yield result

        return single_item()
    return result


def records_for_tool(records: list[MemoryRecord]) -> list[dict[str, Any]]:
    return [
        {
            "id": record.id,
            "text": record.text,
            "kind": record.kind,
            "importance": round(record.importance, 2),
            "embedding_model": record.embedding_model,
        }
        for record in records
    ]


def cosine_like_embedding(text: str, *, dimensions: int = 8) -> list[float]:
    buckets = [0.0] * dimensions
    for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        for index in range(dimensions):
            buckets[index] += digest[index % len(digest)] / 255.0
    norm = math.sqrt(sum(value * value for value in buckets)) or 1.0
    return [value / norm for value in buckets]
