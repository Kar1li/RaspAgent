from __future__ import annotations

import base64
import io
import wave

import pytest
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from inworld_stt import InworldSTT, _pcm16_mono_to_wav


class FakeResponse:
    ok = True
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def json(self):
        return {
            "transcription": {
                "transcript": "sense hat temperature",
                "word_timestamps": [
                    {"word": "sense", "confidence": 0.8},
                    {"word": "hat", "confidence": 0.9},
                ],
            }
        }

    async def text(self):
        return ""


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeResponse()


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=(b"\x01\x00\x02\x00") * 80,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )


@pytest.mark.asyncio
async def test_inworld_stt_posts_transcribe_request() -> None:
    session = FakeSession()
    inworld_stt = InworldSTT(
        api_key="test-key",
        model="inworld/inworld-stt-test",
        language="en-US",
        sample_rate=16000,
        base_url="https://api.inworld.ai/",
        http_session=session,
    )

    event = await inworld_stt._recognize_impl(
        _frame(),
        language="en-US",
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
    )

    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "sense hat temperature"
    assert event.alternatives[0].confidence == pytest.approx(0.85)
    call = session.calls[0]
    assert call["url"] == "https://api.inworld.ai/stt/v1/transcribe"
    assert call["headers"]["Authorization"] == "Basic test-key"
    payload = call["json"]
    assert payload["transcribeConfig"] == {
        "modelId": "inworld/inworld-stt-test",
        "audioEncoding": "AUTO_DETECT",
        "language": "en-US",
        "sampleRateHertz": 16000,
        "numberOfChannels": 1,
    }
    audio = base64.b64decode(payload["audioData"]["content"])
    assert audio.startswith(b"RIFF")
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000


def test_pcm16_mono_to_wav_wraps_audio() -> None:
    audio = _pcm16_mono_to_wav(b"\x01\x00\x02\x00", sample_rate=16000)

    assert audio.startswith(b"RIFF")
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.readframes(2) == b"\x01\x00\x02\x00"


def test_inworld_stt_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("INWORLD_API_KEY", raising=False)

    with pytest.raises(ValueError, match="INWORLD_API_KEY"):
        InworldSTT()
