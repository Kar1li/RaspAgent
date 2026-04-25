# Posture Integration Developer Review

This note is for developers reviewing the current posture-monitoring implementation and picking up follow-up work on either the LiveKit agent side or the Raspberry Pi posture service side.

It documents:

- what was added on the agent side
- what was added on the posture service side
- how the two processes interact now
- the main failures we hit during development
- how those failures were diagnosed and fixed

## Current Architecture

The posture feature is now split into two runtime channels:

1. the posture detection service owns camera access, posture analysis, session state, and callback delivery
2. the voice agent owns user-facing tools, reminder policy, speech, gating, and structured posture memory
3. a separate preview helper process renders the posture preview window on the Pi desktop

The important design decision is that the posture service remains the only owner of the camera. The preview helper does not open the camera itself. Instead, it renders frames published by the posture service over localhost.

## Agent Side

Main files:

- [src/agent.py](D:/livekit_project/my-agent/src/agent.py)
- [src/posture_integration.py](D:/livekit_project/my-agent/src/posture_integration.py)
- [src/memory.py](D:/livekit_project/my-agent/src/memory.py)

### What the agent now does

- exposes posture tools:
  - `start_posture_monitoring`
  - `stop_posture_monitoring`
  - `get_posture_monitoring_status`
  - `set_posture_coaching_interval`
- starts a local posture intake server inside the active agent job
- accepts local posture callbacks on `/internal/posture/events`
- stores posture runtime state in the structured status system
- uses proactive reminder logic to speak short posture coaching messages
- merges live posture-service runtime status into `get_posture_monitoring_status`

### Code walkthrough

#### `Assistant.start_posture_monitoring()`

This sends `POST /sessions/start` to the posture service with:

- callback URL
- callback auth token
- session duration
- preview enabled flag

Then it stores the returned session state into the structured runtime status.

#### `Assistant.get_posture_monitoring_status()`

This used to return only the agent’s locally stored posture runtime. That caused misleading answers when the posture service had frames or posture events but the agent had not yet accepted a callback.

It now merges:

- agent-side structured runtime state
- posture service `GET /sessions/current`

That gives the user and developers one tool response that includes:

- whether the session is active
- whether preview was requested
- whether the preview helper is running
- frame counts and timestamps
- callback health
- latest emitted posture event

#### `PostureEventReceiver.process()`

This handles callbacks from the posture service.

It now logs and distinguishes:

- callback received
- rejected because auth is invalid
- rejected because payload is invalid
- ignored because the session does not match
- accepted and stored

It also writes posture callback acceptance and rejection events into the data-flow log.

#### `ContextController.ingest_posture_event()`

This is where the structured runtime state is updated after a callback is accepted.

It stores:

- current posture label and severity
- reason codes
- metrics
- last posture callback time
- last posture callback event
- posture issue cooldowns

That data is later used by the proactive scheduler and by the status tools.

## Posture Service Side

Main files on the Pi:

- [/home/karl/posture_detection/services/http_service.py](/home/karl/posture_detection/services/http_service.py)
- [/home/karl/posture_detection/services/session_controller.py](/home/karl/posture_detection/services/session_controller.py)
- [/home/karl/posture_detection/preview_helper.py](/home/karl/posture_detection/preview_helper.py)
- [/home/karl/posture_detection/service_main.py](/home/karl/posture_detection/service_main.py)

### What the posture service now does

- defaults to `RealCameraAdapter` in runtime mode
- owns camera open, warmup, frame reads, and posture analysis
- exposes richer session health through `GET /sessions/current`
- publishes the latest annotated preview frame as a local JPEG feed
- launches and stops a separate preview helper process
- tracks preview-helper heartbeat state
- probes callback reachability
- logs callback attempts, success, and failure

### Code walkthrough

#### `PostureServiceRunner`

This is now the main integration point.

It owns:

- current `SessionController`
- callback configuration
- preview helper process state
- callback health state
- live posture session health state

The important additions are:

- `preview_requested`
- `preview_process_running`
- `preview_last_heartbeat_at`
- `callback_target_healthy`
- `last_callback_attempt_at`
- `last_callback_success_at`
- `last_callback_error`

#### `current_session()`

This is the main runtime truth endpoint now. It returns:

- camera adapter and backend
- stream name
- frame count
- last frame timestamp
- last analysis timestamp
- last emitted posture event
- preview helper state
- callback health
- session progress markers

Those fields are what the agent now merges into the tool response.

#### `start_session()`

This now does more than just start the monitoring state machine.

It also:

- resets callback and preview health state
- performs a callback health probe
- launches the preview helper when preview is requested

#### `preview_helper.py`

This is the new desktop rendering process.

It:

- runs outside the headless posture service logic
- polls `GET /sessions/current`
- fetches annotated frames from `GET /preview/frame.jpg`
- displays them with OpenCV in a desktop window
- posts heartbeats to `POST /preview/heartbeat`
- exits when the session changes, preview is no longer requested, or monitoring stops

This split was introduced because the direct `systemd` service window path was not reliable in the Pi desktop environment.

#### `SessionController`

The session controller still owns the camera/frame loop and posture analysis.

Important additions:

- explicit runtime markers:
  - `camera_opened`
  - `first_frame_received`
  - `first_frame_analyzed`
  - `first_posture_event_emitted`
- tracked frame counters and timestamps
- latest emitted posture event
- latest preview frame JPEG snapshot

The frame loop now updates a JPEG preview snapshot that the helper process can render without trying to open the camera a second time.

## Problems Encountered and Fixes

### 1. The service was using `InMemoryCameraAdapter`

Problem:

- posture sessions started successfully
- preview thread started
- but `read_frame()` always returned no frame
- no preview image appeared
- no posture analysis happened
- no `posture.warning` or `posture.normal` callbacks were emitted

Root cause:

- the controller still defaulted to `InMemoryCameraAdapter()` when no adapter was injected

Fix:

- changed the runtime service path to default to `RealCameraAdapter`
- left fake adapters available only for tests and explicit injection

### 2. “Session started” did not mean “frames are flowing”

Problem:

- the posture service could report a running session
- but there was no reliable runtime proof that frames were actually being read

Fix:

- added live health fields to `GET /sessions/current`
- added:
  - `frames_read_count`
  - `last_frame_at`
  - `last_analyzed_at`
  - `last_emitted_posture_event`
  - first-frame and first-analysis markers

This made it possible to tell whether the camera path was healthy without guessing from the absence of logs.

### 3. The window did not appear reliably when owned by the system service

Problem:

- logs showed the preview thread started
- even logs later showed frame rendering
- but the user still did not consistently see a desktop window

Likely cause:

- the OpenCV window was being created from a long-running service context
- desktop session ownership and window visibility were fragile in that path

Fix:

- moved preview rendering into a separate helper process
- kept camera ownership inside the posture service
- exposed the latest annotated frame over localhost
- made the helper render that feed and send heartbeats back

This separated “camera and analysis are working” from “desktop window is visible”.

### 4. Callback failure was mixed together with camera failure

Problem:

- sometimes the posture service was healthy and producing posture events
- but the agent still said no data had arrived

Root cause:

- callback delivery could fail independently of camera and analysis
- early status reporting did not clearly separate those failures

Fix:

- added callback health probe and timestamps
- added callback attempt/success/failure logging
- added agent-side callback acceptance/rejection logging
- merged callback health into `get_posture_monitoring_status`

After that, it became clear whether the problem was:

- no frames
- no posture event emitted
- callback target unreachable
- callback rejected by the agent

### 5. The agent’s posture status response could be stale

Problem:

- the agent could say it was waiting for the first reading
- while the posture service had already started producing frames

Fix:

- `get_posture_monitoring_status()` now merges service-side runtime status
- the tool response now reflects live service state, not only agent memory

## Validation Done

### Agent-side tests

Updated tests cover:

- tool exposure
- posture start/stop flow
- merged posture status output
- callback acceptance/rejection logging path

### Posture service tests

Updated tests cover:

- runtime default adapter path
- callback flow without `session.started`
- `GET /sessions/current` health fields
- preview helper launch state
- preview heartbeat state

### Pi smoke validation

Verified on the Pi:

- both services restart cleanly
- posture service reports `RealCameraAdapter`
- frame counts increase during a live monitoring session
- posture events are emitted
- preview helper launches and heartbeats

One important caveat remains:

- from SSH, we can verify the preview helper process and heartbeat
- but actual human-visible confirmation of the desktop window still depends on observing the Pi desktop directly

## Recommended Follow-up Work

### High-value follow-up

- add a small developer script to inspect posture-service health and callback health in one command
- add a watchdog event when callback target is unhealthy for too long during a room session
- add rate limiting to preview-frame polling if CPU usage becomes an issue

### If the desktop window still disappears intermittently

The next place to inspect is the preview helper path, not the camera path:

- preview helper process logs
- heartbeat cadence
- OpenCV window creation in the user desktop session
- whether the helper should move from polling JPEG to MJPEG for smoother rendering

### If the agent still says no posture data in a real room

The next place to inspect is callback reachability:

- `callback_target_healthy`
- `last_callback_attempt_at`
- `last_callback_success_at`
- agent `posture_callback_accepted` and `posture_callback_rejected` data-flow events

## Files Added or Changed in This Round

Agent repo:

- [src/agent.py](D:/livekit_project/my-agent/src/agent.py)
- [src/posture_integration.py](D:/livekit_project/my-agent/src/posture_integration.py)
- [tests/test_agent.py](D:/livekit_project/my-agent/tests/test_agent.py)

Posture project on Pi:

- [/home/karl/posture_detection/services/http_service.py](/home/karl/posture_detection/services/http_service.py)
- [/home/karl/posture_detection/services/session_controller.py](/home/karl/posture_detection/services/session_controller.py)
- [/home/karl/posture_detection/service_main.py](/home/karl/posture_detection/service_main.py)
- [/home/karl/posture_detection/preview_helper.py](/home/karl/posture_detection/preview_helper.py)
- [/home/karl/posture_detection/tests/test_http_service.py](/home/karl/posture_detection/tests/test_http_service.py)
- [/etc/systemd/system/posture-detection.service](/etc/systemd/system/posture-detection.service)
