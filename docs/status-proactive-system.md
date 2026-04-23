# Structured Status and Proactive System

This document describes the structured status and proactive memory system implemented for the Raspberry Pi Sense HAT agent, and leaves practical notes for a human tester.

## What Was Implemented

The agent no longer depends on vector retrieval for normal proactive behavior. Instead, it uses a structured SQLite-backed status system plus a sliding recent-conversation window.

Implemented areas:

- structured profile storage
- structured runtime status storage
- structured reminder-policy storage
- daily proactive history storage
- compact status-note injection into the LLM context
- proactive scheduler with rate limits and grace-period gating
- narrow function tools for reminder and status CRUD
- detailed run-log and data-flow logging for status and reminder operations

## Persistence Model

The structured data lives in the same SQLite database used by the agent memory layer.

Tables:

- `agent_profile`
  - `preferred_name`
  - `timezone`
  - `quiet_hours_start`
  - `quiet_hours_end`
- `agent_status`
  - `current_activity`
  - `activity_expires_at`
  - `busy_state`
  - `busy_expires_at`
  - `summary_entries_json`
  - `last_user_interaction_at`
  - `last_assistant_speech_at`
  - reminder timestamps and cooldown timestamps
- `reminder_policies`
  - `drink_water`
  - `nap`
  - `check_in`
- `daily_status_log`
  - daily counters such as water reminders sent, water snoozed, nap suggestions sent, check-ins sent
- `proactive_event_log`
  - append-only event log for reminder outcomes

Persistence is event-driven. The agent writes status immediately when meaningful changes happen. There is no blind periodic persistence loop.

## Context Control

The LLM context path uses:

- the existing sliding message window
- one injected assistant-side status note

The status note can include:

- preferred name
- quiet hours and timezone
- current temporary activity
- active proactive schedules
- recent conversation summary

Vector memory remains available for explicit memory tools, but it is not the normal source for proactive prompt augmentation.

## Proactive Behavior

The scheduler currently supports:

- water reminders
- afternoon nap suggestions
- proactive check-ins

It only speaks when all gating rules pass.

Current gates:

- remote participant exists
- no current speech in progress
- not inside STT/LLM/TTS processing
- outside quiet hours
- user has been active recently
- reminder cooldown has expired
- post-user grace period has expired
- post-assistant grace period has expired

Current env defaults:

- `PROACTIVE_TICK_SECONDS=60`
- `PROACTIVE_RECENT_ACTIVITY_WINDOW_MINUTES=240`
- `PROACTIVE_POST_USER_GRACE_SECONDS=90`
- `PROACTIVE_POST_ASSISTANT_GRACE_SECONDS=60`

## Status and Reminder CRUD Surface

### Read

- `get_status_summary`
- `get_reminder_policy`
- `get_daily_status`
- `get_recent_status_history`

### Create / Update

- `update_user_profile`
- `set_water_reminder_interval`
- `set_check_in_interval`
- `set_reminder_enabled`
- `set_quiet_hours`
- `set_current_activity`
- `snooze_reminder`

### Delete / Reset / Clear

- `clear_preferred_name`
- `clear_temporary_status`
- `clear_recent_summary`
- `clear_reminder_snooze`
- `reset_reminder_policy`
- `reset_quiet_hours`
- `delete_daily_status`

## Narrow Tooling Design

Reminder-setting tools were intentionally split into narrow operations because the earlier broad tool caused repeated model mistakes such as:

- non-canonical reminder types like `drink water`
- stringified booleans and integers
- invented unrelated arguments such as snooze and quiet-hour changes

The current design uses:

- dedicated interval setters for water and check-in cadence changes
- explicit enable/disable
- explicit snooze
- explicit quiet-hours updates

The internal controller still has a broader helper, but the agent prompt and exposed tool set are designed to keep the model on narrow operations.

## Logging

Two logging channels are relevant.

### Run log

Primary service log:

```bash
sudo journalctl -u livekit-sensehat-agent.service -f
```

Useful structured events include:

- LLM node start and completion
- TTS node start and completion
- proactive scheduler tick decisions
- structured tool validation failures
- reminder and status update completion

### Data-flow log

Structured end-to-end flow:

```bash
tail -f /home/karl/RaspAgent/.data/data_flow.jsonl
```

Useful events include:

- `llm_input`
- `llm_output`
- `tts_input`
- `tool_call_received`
- `tool_arguments_normalized`
- `tool_validation_failed`
- `tool_completed`
- `tool_completed_error`
- `status_observation`
- `status_note`
- `status_summary`
- `proactive_tick`
- `proactive_action`
- `proactive_execution`
- `proactive_outcome`

## Human Tester Notes

### Before Testing

Confirm the service is running:

```bash
sudo systemctl is-active livekit-sensehat-agent.service
```

Expected result:

- `active`

Keep both logs open while testing:

```bash
sudo journalctl -u livekit-sensehat-agent.service -f
tail -f /home/karl/RaspAgent/.data/data_flow.jsonl
```

### Test 1: Water interval setter

Say:

- `remind me to drink every one minute`

Expected behavior:

- the agent should use `set_water_reminder_interval`
- the agent should not call snooze or quiet-hours tools
- the resulting water interval should be `1`
- water reminder should be enabled

Expected log clues:

- `tool_call_received` with `tool_name=set_water_reminder_interval`
- `tool_completed`

### Test 2: Check-in interval setter

Say:

- `check in with me every 15 minutes`

Expected behavior:

- the agent should use `set_check_in_interval`
- the check-in policy interval should become `15`
- check-in reminders should be enabled

### Test 3: Specific reminder read

Say:

- `what is my check in reminder setting`

Expected behavior:

- the agent should use `get_reminder_policy`
- the spoken answer should mention the check-in cadence and enabled state

### Test 4: Snooze and unsnooze

Say:

- `snooze the water reminder for 10 minutes`
- then `clear the water snooze`

Expected behavior:

- first call uses `snooze_reminder`
- second call uses `clear_reminder_snooze`
- `water_cooldown_until` should become non-null, then null

### Test 5: Quiet hours reset

Say:

- `set quiet hours from 11 pm to 7 am`
- then `reset quiet hours`

Expected behavior:

- first call uses `set_quiet_hours`
- second call uses `reset_quiet_hours`
- final values should return to defaults

Current defaults:

- start `22:00`
- end `08:00`

### Test 6: Profile CRUD

Say:

- `call me Karl`
- `my timezone is UTC`
- `forget my preferred name`

Expected behavior:

- `update_user_profile` should store the name and timezone
- `clear_preferred_name` should remove only the name
- timezone should remain stored

### Test 7: Temporary status and summary clearing

Say:

- `I am working for two hours`
- `clear my temporary status`
- `clear the recent summary`

Expected behavior:

- activity should appear, then be cleared
- the rolling recent summary should become empty

### Test 8: Daily history deletion

First trigger at least one proactive outcome or reminder-setting event during the day.

Then say:

- `delete today's reminder history`

Expected behavior:

- the agent should use `delete_daily_status`
- a later `get_daily_status today` should show a zeroed empty row unless new events are added afterward

### Test 9: Grace-period gating

Sequence:

1. greet the agent
2. wait less than 90 seconds
3. do not speak

Expected behavior:

- proactive reminders should be suppressed during the post-user grace window

Also verify:

1. after the agent speaks
2. wait less than 60 seconds

Expected behavior:

- proactive reminders should be suppressed during the post-assistant grace window

Expected log clues:

- `proactive_tick` with `reason=post_user_grace` or `reason=post_assistant_grace`

## Known Caveats

- The agent still uses a structured status note plus short conversation window, so if the window is extremely small and the summary is cleared, the agent may become intentionally less context-rich until the user speaks again.
- Daily-history deletion removes the stored row for that day. If new proactive events happen later the same day, a new row will be created.
- `UTC` is always accepted as a timezone. Other timezone names depend on valid IANA timezone identifiers.
- The old broad reminder-setting tool should no longer be used in normal conversations. If it ever appears again in logs, treat that as a regression.

## Recommended Regression Checks After Any Future Change

- Sense HAT sensor tools still trigger correctly
- water cadence change uses `set_water_reminder_interval`
- check-in cadence change uses `set_check_in_interval`
- clearing snooze only clears cooldown and does not reset the policy
- resetting a reminder policy restores its default cadence and enabled state
- proactive reminders do not fire inside grace periods
- data-flow logs still include tool lifecycle events
