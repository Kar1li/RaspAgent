from __future__ import annotations

import json
from pathlib import Path

import pytest
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from local_stt import VoskSTT, _frame_to_pcm16_mono


class FakeModel:
    pass


class FakeRecognizer:
    calls: list[dict[str, object]] = []
    result_text = "sense hat temperature"

    def __init__(self, model, sample_rate, grammar=None) -> None:
        self.model = model
        self.sample_rate = sample_rate
        self.grammar = grammar
        self.audio = b""
        self.calls.append(
            {
                "model": model,
                "sample_rate": sample_rate,
                "grammar": grammar,
            }
        )

    def AcceptWaveform(self, audio: bytes) -> bool:
        self.audio += audio
        return True

    def Result(self) -> str:
        return self.FinalResult()

    def FinalResult(self) -> str:
        return json.dumps(
            {
                "text": self.result_text,
                "result": [{"conf": 0.7}, {"conf": 0.9}],
            }
        )


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=(b"\x01\x00\x02\x00") * 80,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )


@pytest.mark.asyncio
async def test_vosk_stt_returns_final_transcript() -> None:
    FakeRecognizer.calls = []
    local_stt = VoskSTT(
        model_path=Path(".models/test"),
        model=FakeModel(),
        sample_rate=16000,
        language="en",
        grammar_terms=["Sense HAT", "[unk]"],
        recognizer_factory=FakeRecognizer,
    )

    event = await local_stt._recognize_impl(
        _frame(),
        language="en",
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
    )

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "sense hat temperature"
    assert event.alternatives[0].confidence == pytest.approx(0.8)
    assert FakeRecognizer.calls[0]["sample_rate"] == 16000
    assert json.loads(FakeRecognizer.calls[0]["grammar"]) == ["Sense HAT", "[unk]"]


@pytest.mark.asyncio
async def test_vosk_stt_uses_model_factory_once() -> None:
    calls: list[str] = []

    def model_factory(path: str) -> FakeModel:
        calls.append(path)
        return FakeModel()

    local_stt = VoskSTT(
        model_path=Path(".models/test"),
        sample_rate=16000,
        model_factory=model_factory,
        recognizer_factory=FakeRecognizer,
    )

    local_stt.prewarm()
    local_stt.prewarm()

    assert calls == [".models\\test"] or calls == [".models/test"]


def test_frame_to_pcm16_mono_downmixes_stereo() -> None:
    frame = rtc.AudioFrame(
        data=b"\x02\x00\x06\x00\x04\x00\x08\x00",
        sample_rate=16000,
        num_channels=2,
        samples_per_channel=2,
    )

    pcm = _frame_to_pcm16_mono(frame, sample_rate=16000)

    assert pcm == b"\x04\x00\x06\x00"
