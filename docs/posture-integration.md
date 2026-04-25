# Posture Detection Integration

This project supports a local posture-monitoring loop between the LiveKit voice agent and a separate posture detection service running on the Raspberry Pi.

## Design

- the voice agent owns policy, speech, reminder gating, and user-facing tools
- the posture detection project owns camera capture, posture analysis, and realtime posture events
- both services run on the same Raspberry Pi
- all posture HTTP traffic is local-only on `127.0.0.1`

## Control Flow

1. the user asks the agent for posture help while working
2. the agent calls `start_posture_monitoring`
3. the agent sends `POST /sessions/start` to the posture service
4. the posture service starts a monitoring session and remembers the callback URL and auth token
5. the posture service pushes posture events to `POST /internal/posture/events` on the agent
6. the agent stores the latest posture state in structured runtime status
7. the proactive scheduler turns posture warnings into short spoken coaching with per-issue cooldowns

## Agent Interfaces

### Tools

- `start_posture_monitoring()`
- `stop_posture_monitoring()`
- `get_posture_monitoring_status()`

### Local HTTP intake

- `POST /internal/posture/events`
- `GET /internal/posture/health`

Required header:

- `X-Posture-Auth`

## Posture Service Interfaces

- `POST /sessions/start`
- `POST /sessions/stop`
- `GET /sessions/current`
- `GET /health`

## Runtime Fields

The structured runtime status stores:

- `posture_monitoring_active`
- `posture_workflow_active`
- `posture_session_id`
- `latest_posture_label`
- `latest_posture_severity`
- `latest_posture_reason_codes`
- `latest_posture_metrics`
- `latest_posture_prompt_key`
- `last_posture_event_at`
- `posture_issue_cooldowns`

## Reminder Mapping

- neck or forward-head issues -> neck coaching
- rounded-back or trunk-lean issues -> back coaching
- insufficient-data or camera issues -> camera adjustment prompt

## Environment

Agent-side settings:

- `POSTURE_SERVICE_URL`
- `POSTURE_AGENT_INTAKE_HOST`
- `POSTURE_AGENT_INTAKE_PORT`
- `POSTURE_SHARED_SECRET`
- `POSTURE_DEFAULT_DURATION_SECONDS`
- `POSTURE_SERVICE_TIMEOUT_SECONDS`
- `POSTURE_ISSUE_REMINDER_COOLDOWN_SECONDS`
- `POSTURE_EVENT_STALE_SECONDS`
