from types import SimpleNamespace
from datetime import datetime, timezone

import pytest
from livekit.agents import llm

from memory import (
    ContextController,
    EmbeddingResult,
    EmbeddingUnavailableError,
    STATUS_NOTE_PREFIX,
    MemoryCandidate,
    MemoryExtractor,
    OpenRouterEmbedder,
    SQLiteVectorMemoryStore,
    cosine_like_embedding,
)


class FakeEmbedder:
    def __init__(self, *, model: str = "openai/text-embedding-3-small") -> None:
        self.model = model
        self.calls: list[str] = []

    async def embed(self, text: str) -> EmbeddingResult:
        self.calls.append(text)
        return EmbeddingResult(
            model=self.model,
            vector=cosine_like_embedding(text),
            used_fallback=self.model != "openai/text-embedding-3-small",
        )


class FailingEmbedder:
    async def embed(self, text: str) -> EmbeddingResult:
        raise EmbeddingUnavailableError("no embeddings")


class FakeStatusReporter:
    def __init__(self) -> None:
        self.events: list[str] = []

    def push_state(self, state: str) -> None:
        self.events.append(f"push:{state}")

    def pop_state(self) -> None:
        self.events.append("pop")


@pytest.mark.asyncio
async def test_memory_extractor_accepts_explicit_facts_and_rejects_sensitive_data():
    extractor = MemoryExtractor(min_importance=0.65)

    facts = await extractor.extract("Remember that my name is Karl.")
    sensitive = await extractor.extract("Remember that my password is player11.")

    assert any("Karl" in candidate.text for candidate in facts)
    assert sensitive == []


def test_sqlite_vector_store_inserts_searches_and_deletes(tmp_path):
    store = SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3")
    candidate = MemoryCandidate(
        text="The user's preferred name is Karl.",
        kind="user_identity",
        importance=0.9,
        source_text="My name is Karl.",
    )
    embedding = EmbeddingResult(
        model="openai/text-embedding-3-small",
        vector=cosine_like_embedding(candidate.text),
        used_fallback=False,
    )

    record = store.add_memory(candidate, embedding=embedding)
    results = store.search(
        "What is my name?",
        embedding=EmbeddingResult(
            model="openai/text-embedding-3-small",
            vector=cosine_like_embedding("What is my name Karl?"),
            used_fallback=False,
        ),
        limit=5,
    )

    assert record is not None
    assert results
    assert results[0].id == record.id
    assert store.delete_memory(record.id)
    assert store.search("Karl", embedding=None, limit=5) == []


def test_sqlite_vector_store_searches_qwen_vector_table(tmp_path):
    store = SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3")
    candidate = MemoryCandidate(
        text="The user's preferred name is Karl.",
        kind="user_identity",
        importance=0.9,
        source_text="My name is Karl.",
    )
    embedding = EmbeddingResult(
        model="qwen/qwen3-embedding-8b",
        vector=cosine_like_embedding(candidate.text, dimensions=4096),
        used_fallback=True,
    )

    record = store.add_memory(candidate, embedding=embedding)
    results = store.search(
        "What is my name?",
        embedding=EmbeddingResult(
            model="qwen/qwen3-embedding-8b",
            vector=cosine_like_embedding("What is my name Karl?", dimensions=4096),
            used_fallback=True,
        ),
        limit=5,
    )

    assert record is not None
    assert results
    assert results[0].id == record.id


@pytest.mark.asyncio
async def test_context_controller_observes_memory_and_injects_prompt_note(tmp_path):
    embedder = FakeEmbedder()
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=embedder,
        short_window_messages=8,
        top_k=5,
        min_importance=0.65,
    )

    records = await controller.observe_user_turn("Remember that my name is Karl.")
    chat_ctx = llm.ChatContext()
    for index in range(6):
        chat_ctx.add_message(role="user", content=f"user turn {index}")
        chat_ctx.add_message(role="assistant", content=f"assistant turn {index}")
    chat_ctx.add_message(role="user", content="What is my name?")

    controlled_ctx = await controller.prepare_llm_context(
        chat_ctx,
        latest_user_text="What is my name?",
    )
    assistant_notes = [
        item.text_content
        for item in controlled_ctx.items
        if item.type == "message" and item.role == "assistant"
        and (item.text_content or "").startswith(STATUS_NOTE_PREFIX)
    ]

    assert records
    assert len(embedder.calls) == 2
    assert len(controlled_ctx.items) <= 9
    assert any(note and "Karl" in note for note in assistant_notes)
    assert any(note and "Recent conversation summary" in note for note in assistant_notes)
    assert controlled_ctx.items[-1].role == "user"


@pytest.mark.asyncio
async def test_context_controller_preserves_function_call_items(tmp_path):
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=FailingEmbedder(),
        short_window_messages=8,
        top_k=5,
        min_importance=0.65,
    )
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="What is the temperature?")
    chat_ctx.items.append(
        llm.FunctionCall(
            call_id="call-1",
            name="get_sensehat_environment",
            arguments="{}",
        )
    )
    chat_ctx.items.append(
        llm.FunctionCallOutput(
            call_id="call-1",
            name="get_sensehat_environment",
            output='{"temperature_c": 22.5}',
            is_error=False,
        )
    )

    controlled_ctx = await controller.prepare_llm_context(
        chat_ctx,
        latest_user_text="What is the temperature?",
    )

    assert any(item.type == "function_call" for item in controlled_ctx.items)
    assert any(item.type == "function_call_output" for item in controlled_ctx.items)


@pytest.mark.asyncio
async def test_context_controller_uses_fts_when_embeddings_fail(tmp_path):
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=FailingEmbedder(),
        short_window_messages=8,
        top_k=5,
        min_importance=0.65,
    )

    await controller.observe_user_turn("Remember that my favorite sensor is humidity.")
    results = await controller.search("humidity", limit=5)

    assert results
    assert "humidity" in results[0].text.lower()


@pytest.mark.asyncio
async def test_context_controller_tracks_structured_status_and_daily_history(tmp_path):
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=FailingEmbedder(),
    )

    await controller.observe_user_turn("My name is Karl and remind me to drink water every 45 minutes.")
    controller.record_proactive_outcome("drink_water", "sent")

    summary = controller.status_summary()
    today = summary["today"]["day"]

    assert summary["profile"]["preferred_name"] == "Karl"
    assert summary["policies"]["drink_water"]["interval_minutes"] == 45
    assert controller.daily_status(today)["water_reminders_sent"] == 1


@pytest.mark.asyncio
async def test_context_controller_normalizes_reminder_settings_aliases(tmp_path):
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=FailingEmbedder(),
    )
    original_start = controller.profile.quiet_hours_start
    original_end = controller.profile.quiet_hours_end

    summary = controller.update_reminder_settings(
        "drink water",
        enabled="true",
        interval_minutes="1",
        snooze_minutes="",
        quiet_hours_start="None",
        quiet_hours_end="null",
    )

    assert summary["policies"]["drink_water"]["enabled"] is True
    assert summary["policies"]["drink_water"]["interval_minutes"] == 1
    assert summary["profile"]["quiet_hours_start"] == original_start
    assert summary["profile"]["quiet_hours_end"] == original_end
    assert summary["runtime"]["water_cooldown_until"] is None


@pytest.mark.asyncio
async def test_set_water_reminder_interval_enables_policy_without_snoozing(tmp_path):
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=FailingEmbedder(),
    )
    controller.policies["drink_water"].enabled = False
    controller.store.save_reminder_policy(controller.policies["drink_water"])

    summary = controller.set_water_reminder_interval(1)

    assert summary["policies"]["drink_water"]["enabled"] is True
    assert summary["policies"]["drink_water"]["interval_minutes"] == 1
    assert summary["runtime"]["water_cooldown_until"] is None


@pytest.mark.asyncio
async def test_set_check_in_interval_enables_policy_without_snoozing(tmp_path):
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=FailingEmbedder(),
    )
    controller.policies["check_in"].enabled = False
    controller.store.save_reminder_policy(controller.policies["check_in"])

    summary = controller.set_check_in_interval(15)

    assert summary["policies"]["check_in"]["enabled"] is True
    assert summary["policies"]["check_in"]["interval_minutes"] == 15
    assert summary["runtime"]["check_in_cooldown_until"] is None


@pytest.mark.asyncio
async def test_update_user_profile_and_clear_preferred_name(tmp_path):
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=FailingEmbedder(),
    )

    summary = controller.update_user_profile(
        preferred_name="Karl",
        timezone="UTC",
    )

    assert summary["profile"]["preferred_name"] == "Karl"
    assert summary["profile"]["timezone"] == "UTC"

    cleared = controller.clear_preferred_name()

    assert cleared["profile"]["preferred_name"] is None
    assert cleared["profile"]["timezone"] == "UTC"


@pytest.mark.asyncio
async def test_reset_reminder_policy_and_clear_snooze(tmp_path):
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=FailingEmbedder(),
    )
    controller.update_reminder_settings(
        "drink_water",
        enabled=False,
        interval_minutes=7,
        snooze_minutes=5,
    )

    cleared = controller.clear_reminder_snooze("drink_water")
    assert cleared["runtime"]["water_cooldown_until"] is None

    reset = controller.reset_reminder_policy("drink_water")
    assert reset["policies"]["drink_water"]["enabled"] is True
    assert reset["policies"]["drink_water"]["interval_minutes"] == 60
    assert reset["runtime"]["last_water_reminder_at"] is None


@pytest.mark.asyncio
async def test_clear_recent_summary_and_delete_daily_status(tmp_path):
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=FailingEmbedder(),
    )
    await controller.observe_user_turn("My name is Karl and I am working.")
    controller.record_proactive_outcome("drink_water", "sent")
    day = controller.status_summary()["today"]["day"]

    cleared = controller.clear_recent_summary()
    assert cleared["runtime"]["recent_summary"] == ""

    deleted = controller.delete_daily_status(day)
    assert deleted == {"day": day, "deleted": True}
    assert controller.daily_status(day)["water_reminders_sent"] == 0


@pytest.mark.asyncio
async def test_context_controller_next_proactive_action_respects_quiet_hours(tmp_path):
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=FailingEmbedder(),
    )
    await controller.observe_user_turn("My name is Karl.")
    controller.profile.timezone = "UTC"
    controller.profile.quiet_hours_start = "22:00"
    controller.profile.quiet_hours_end = "08:00"
    controller.store.save_profile(controller.profile)
    controller.status.last_user_interaction_at = datetime(2026, 4, 22, 1, 0, tzinfo=timezone.utc).timestamp()
    controller.status.last_water_reminder_at = None
    controller.store.save_status(controller.status)

    action = controller.next_proactive_action(
        now=datetime(2026, 4, 22, 2, 0, tzinfo=timezone.utc)
    )

    assert action is None


@pytest.mark.asyncio
async def test_context_controller_proactive_gate_respects_post_user_grace(tmp_path):
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=FailingEmbedder(),
    )
    controller.status.last_user_interaction_at = 100.0
    controller.store.save_status(controller.status)

    reason = controller.proactive_gate_reason(now=150.0)

    assert reason == "post_user_grace"


@pytest.mark.asyncio
async def test_context_controller_proactive_gate_respects_post_assistant_grace(tmp_path):
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=FailingEmbedder(),
    )
    controller.status.last_user_interaction_at = 0.0
    controller.status.last_assistant_speech_at = 100.0
    controller.store.save_status(controller.status)

    reason = controller.proactive_gate_reason(now=150.0)

    assert reason == "post_assistant_grace"


def test_context_controller_recent_status_history_reads_multiple_days(tmp_path):
    store = SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3")
    controller = ContextController(store=store, embedder=FailingEmbedder())
    store.increment_daily_counter("2026-04-20", "water_reminders_sent")
    store.increment_daily_counter("2026-04-21", "check_ins_sent")

    history = controller.recent_status_history(days=2)

    assert history[0]["day"] == "2026-04-21"
    assert history[1]["day"] == "2026-04-20"


@pytest.mark.asyncio
async def test_context_controller_reports_memory_status(tmp_path):
    status = FakeStatusReporter()
    controller = ContextController(
        store=SQLiteVectorMemoryStore(tmp_path / "memory.sqlite3"),
        embedder=FailingEmbedder(),
        short_window_messages=8,
        top_k=5,
        min_importance=0.65,
        status_reporter=status,
    )

    await controller.observe_user_turn("Remember that my favorite sensor is pressure.")
    await controller.search("pressure", limit=5)

    assert "push:memory_insert" in status.events
    assert "push:memory_retrieve" in status.events
    assert status.events.count("pop") >= 2


@pytest.mark.asyncio
async def test_openrouter_embedder_falls_back_to_qwen_model():
    class FakeEmbeddings:
        def __init__(self) -> None:
            self.models: list[str] = []

        async def create(self, **kwargs):
            model = kwargs["model"]
            self.models.append(model)
            if model == "openai/text-embedding-3-small":
                raise RuntimeError("primary failed")
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])]
            )

    fake_embeddings = FakeEmbeddings()

    def factory(**kwargs):
        return SimpleNamespace(embeddings=fake_embeddings)

    embedder = OpenRouterEmbedder(
        api_key="test-key",
        models=["openai/text-embedding-3-small", "qwen/qwen3-embedding-8b"],
        client_factory=factory,
    )

    result = await embedder.embed("remember my preferred name is Karl")

    assert result.model == "qwen/qwen3-embedding-8b"
    assert result.used_fallback is True
    assert fake_embeddings.models == [
        "openai/text-embedding-3-small",
        "openai/text-embedding-3-small",
        "qwen/qwen3-embedding-8b",
    ]
