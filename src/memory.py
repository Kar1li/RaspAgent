from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import re
import sqlite3
import time
import uuid
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from livekit.agents import Agent, ModelSettings, llm

logger = logging.getLogger("agent.memory")


DEFAULT_EMBEDDING_MODELS = (
    "openai/text-embedding-3-small",
    "qwen/qwen3-embedding-8b",
)

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


MEMORY_NOTE_PREFIX = "Relevant long-run memory for this turn."


def _table_suffix(model: str, dimension: int) -> str:
    digest = hashlib.sha256(f"{model}:{dimension}".encode()).hexdigest()[:16]
    return f"{digest}_{dimension}"


def _fts_query(text: str) -> str:
    terms = re.findall(r"[A-Za-z0-9_]+", text.lower())
    if not terms:
        return '""'
    return " OR ".join(f'"{term}"' for term in terms[:12])


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
            """
        )
        self._conn.commit()
        logger.info("Memory database initialized", extra={"db_path": str(self._db_path)})

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
            return self._row_to_record(existing)

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
        logger.info(
            "Context controller initialized",
            extra={
                "short_window_messages": self.short_window_messages,
                "top_k": self.top_k,
                "min_importance": self.min_importance,
                "db_path": str(memory_db_path),
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
            logger.info(
                "Memory observation completed",
                extra={
                    "stored": len(records),
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

    async def build_memory_note(self, query: str | None) -> str | None:
        if not query:
            return None
        records = await self.search(query, limit=self.top_k)
        if not records:
            logger.debug("No memories selected for prompt injection")
            return None
        lines = [f"- {record.text}" for record in records[: self.top_k]]
        note = (
            f"{MEMORY_NOTE_PREFIX} Use only if helpful and do not mention memory storage unless asked. Memories are not live sensor readings.\n"
            + "\n".join(lines)
        )
        logger.info(
            "Memory prompt note built",
            extra={"memories": len(records), "chars": len(note)},
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
        note = None
        if latest_user_text and not has_memory_note(controlled):
            note = await self.build_memory_note(latest_user_text)
            latest_message = latest_user_message(controlled)
            created_at = (
                max(0.0, latest_message.created_at - 0.001)
                if latest_message is not None
                else time.time()
            )
            if note:
                controlled.add_message(
                    role="assistant",
                    content=note,
                    created_at=created_at,
                )
        logger.info(
            "LLM context prepared",
            extra={
                "items_before": before_count,
                "items_after": len(controlled.items),
                "memory_note": bool(note),
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
        except Exception as exc:
            logger.error(
                "Unexpected embedding failure; using non-vector memory path",
                extra={"error_type": type(exc).__name__},
                exc_info=True,
            )
            return None


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
            MEMORY_NOTE_PREFIX
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
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = digest[0] % dimensions
        buckets[index] += 1.0
    magnitude = math.sqrt(sum(value * value for value in buckets)) or 1.0
    return [value / magnitude for value in buckets]
