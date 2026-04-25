import pytest

from posture_integration import PostureEventReceiver


class _FakeContextController:
    def __init__(self, result=None) -> None:
        self.result = result or {"accepted": True}
        self.calls: list[dict[str, object]] = []

    def ingest_posture_event(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def test_posture_event_receiver_rejects_invalid_auth() -> None:
    controller = _FakeContextController()
    receiver = PostureEventReceiver(
        expected_auth="secret",
        context_controller=controller,
    )

    status, body = receiver.process(
        headers={"x-posture-auth": "wrong"},
        payload={"session_id": "abc", "event_name": "posture.warning"},
    )

    assert status == 401
    assert body["reason"] == "invalid_auth"
    assert controller.calls == []


def test_posture_event_receiver_rejects_invalid_payload() -> None:
    controller = _FakeContextController()
    receiver = PostureEventReceiver(
        expected_auth="secret",
        context_controller=controller,
    )

    status, body = receiver.process(
        headers={"x-posture-auth": "secret"},
        payload={"event_name": "posture.warning"},
    )

    assert status == 400
    assert body["reason"] == "invalid_payload"
    assert controller.calls == []


def test_posture_event_receiver_returns_accepted_context_result() -> None:
    controller = _FakeContextController(
        result={"accepted": True, "session_id": "abc", "event_name": "posture.warning"}
    )
    receiver = PostureEventReceiver(
        expected_auth="secret",
        context_controller=controller,
    )

    status, body = receiver.process(
        headers={"x-posture-auth": "secret"},
        payload={
            "session_id": "abc",
            "event_name": "posture.warning",
            "reason_codes": ["forward head"],
            "metrics": {"head_offset_px": 12.0},
        },
    )

    assert status == 200
    assert body["accepted"] is True
    assert controller.calls[0]["reason_codes"] == ["forward head"]
    assert controller.calls[0]["metrics"] == {"head_offset_px": 12.0}


def test_posture_event_receiver_returns_202_for_session_mismatch() -> None:
    controller = _FakeContextController(result={"accepted": False, "reason": "session_mismatch"})
    receiver = PostureEventReceiver(
        expected_auth="secret",
        context_controller=controller,
    )

    status, body = receiver.process(
        headers={"x-posture-auth": "secret"},
        payload={"session_id": "old", "event_name": "posture.warning"},
    )

    assert status == 202
    assert body["reason"] == "session_mismatch"
