from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger("agent.posture")


class PostureControlError(RuntimeError):
    pass


@dataclass(frozen=True)
class PostureControlClient:
    base_url: str
    timeout_seconds: float = 5.0

    def start_session(
        self,
        *,
        callback_url: str,
        callback_auth: str,
        duration_sec: int,
        preview_enabled: bool = False,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/sessions/start",
            {
                "callback_url": callback_url,
                "callback_auth": callback_auth,
                "duration_sec": duration_sec,
                "preview_enabled": preview_enabled,
            },
        )

    def stop_session(self, *, session_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if session_id:
            payload["session_id"] = session_id
        return self._request("POST", "/sessions/stop", payload)

    def current_session(self) -> dict[str, Any]:
        return self._request("GET", "/sessions/current", None)

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", None)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        body = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise PostureControlError(raw or f"HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise PostureControlError(str(exc.reason)) from exc


class PostureEventReceiver:
    def __init__(
        self,
        *,
        expected_auth: str,
        context_controller: Any,
        data_flow: Any | None = None,
    ) -> None:
        self._expected_auth = expected_auth
        self._context_controller = context_controller
        self._data_flow = data_flow

    def process(self, *, headers: dict[str, str], payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        provided_auth = headers.get("x-posture-auth", "")
        logger.info(
            "Posture callback received",
            extra={
                "event_name": payload.get("event_name"),
                "session_id": payload.get("session_id"),
                "auth_valid": provided_auth == self._expected_auth,
            },
        )
        if provided_auth != self._expected_auth:
            logger.warning("Rejected posture event due to invalid auth")
            self._write_data_flow("posture_callback_rejected", reason="invalid_auth", payload=payload)
            self._write_data_flow("posture_event_ignored", reason="invalid_auth", payload=payload)
            return 401, {"accepted": False, "reason": "invalid_auth"}

        session_id = str(payload.get("session_id", "")).strip()
        event_name = str(payload.get("event_name", "")).strip()
        if not session_id or not event_name:
            logger.warning("Rejected posture event due to invalid payload")
            self._write_data_flow("posture_callback_rejected", reason="invalid_payload", payload=payload)
            self._write_data_flow("posture_event_ignored", reason="invalid_payload", payload=payload)
            return 400, {"accepted": False, "reason": "invalid_payload"}

        normalized_payload = {
            "session_id": session_id,
            "event_name": event_name,
            "timestamp": payload.get("timestamp"),
            "severity": payload.get("severity"),
            "posture_label": payload.get("posture_label"),
            "reason_codes": list(payload.get("reason_codes", [])),
            "metrics": dict(payload.get("metrics", {})),
            "prompt_key": payload.get("prompt_key"),
            "message": payload.get("message"),
        }
        self._write_data_flow("posture_event_received", payload=normalized_payload)
        result = self._context_controller.ingest_posture_event(
            session_id=session_id,
            event_name=event_name,
            timestamp=payload.get("timestamp"),
            severity=payload.get("severity"),
            posture_label=payload.get("posture_label"),
            reason_codes=list(payload.get("reason_codes", [])),
            metrics=dict(payload.get("metrics", {})),
            prompt_key=payload.get("prompt_key"),
            message=payload.get("message"),
        )
        if not result.get("accepted", False):
            logger.info(
                "Posture callback ignored",
                extra={
                    "event_name": event_name,
                    "session_id": session_id,
                    "reason": result.get("reason", "ignored"),
                },
            )
            self._write_data_flow("posture_callback_rejected", reason=result.get("reason", "ignored"), payload=payload)
            self._write_data_flow("posture_event_ignored", reason=result.get("reason", "ignored"), payload=payload)
            return 202, result
        logger.info(
            "Posture callback accepted",
            extra={
                "event_name": event_name,
                "session_id": session_id,
                "latest_posture_label": result.get("latest_posture_label"),
            },
        )
        self._write_data_flow("posture_callback_accepted", result=result)
        return 200, result

    def _write_data_flow(self, event: str, **fields: Any) -> None:
        if self._data_flow is None:
            return
        try:
            self._data_flow.write(event, **fields)
        except Exception:
            logger.debug("Failed to write posture data-flow event", exc_info=True)


class PostureIntakeServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        receiver: PostureEventReceiver,
    ) -> None:
        self._host = host
        self._port = port
        self._receiver = receiver
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._server is None:
            return self._port
        return int(self._server.server_address[1])

    def callback_url(self) -> str:
        return f"http://{self._host}:{self.port}/internal/posture/events"

    def start(self) -> None:
        if self._server is not None:
            return
        receiver = self._receiver

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/internal/posture/health":
                    self.send_error(404)
                    return
                self._json_response(200, {"ok": True})

            def do_POST(self) -> None:
                if self.path != "/internal/posture/events":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._json_response(400, {"accepted": False, "reason": "invalid_json"})
                    return
                status, body = receiver.process(
                    headers={str(key).lower(): value for key, value in self.headers.items()},
                    payload=payload,
                )
                self._json_response(status, body)

            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("Posture intake server: " + format, *args)

            def _json_response(self, status: int, body: dict[str, Any]) -> None:
                encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="posture-intake-server",
            daemon=True,
        )
        self._thread.start()
        logger.info("Posture intake server started", extra={"host": self._host, "port": self.port})

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Posture intake server stopped", extra={"host": self._host, "port": self.port})
        self._server = None
        self._thread = None
