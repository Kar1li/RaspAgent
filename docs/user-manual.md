# Sense HAT Voice Agent User Manual

This is a practical guide for using the Raspberry Pi Sense HAT voice agent as an end user.

## What The Agent Can Do

The agent can:

- answer normal spoken questions
- read live Sense HAT data on request
- remember useful long-run facts when you ask it to
- keep short-run conversation context
- proactively remind you about things like drinking water
- suggest a short nap in the afternoon
- check in with you from time to time
- let you inspect, change, clear, and reset reminder and status settings

## Important Behavior To Know

### Live sensor readings are not guessed

If you ask for current temperature, humidity, pressure, orientation, joystick input, light, or LED display state, the agent should call a Sense HAT tool and read the device live.

### Memory has two parts

- short-run context:
  recent conversation summary and temporary status
- long-run memory:
  explicit facts you asked the agent to remember

### Proactive reminders are rate-limited

The agent should avoid interrupting active conversation. It waits for grace periods after user speech and assistant speech before speaking proactively.

## Everyday Examples

### General questions

Say things like:

- `what can you help me with`
- `what did I ask you a minute ago`
- `what do you remember about me`

### Sense HAT readings

Say things like:

- `what is the temperature right now`
- `read the humidity`
- `what is the air pressure`
- `what direction am I facing`
- `show me everything the Sense HAT can provide`
- `has the joystick been pressed`
- `what is on the LED display`

### Save and recall memory

Say things like:

- `remember that my name is Karl`
- `remember that I prefer short answers`
- `what do you remember about me`
- `forget my preferred name`
- `forget that I prefer short answers`

## Proactive Reminders

The agent currently supports:

- drink water reminders
- afternoon nap suggestions
- proactive check-ins

### Change reminder timing

Say things like:

- `remind me to drink every 30 minutes`
- `check in with me every 15 minutes`

### Turn reminders on or off

Say things like:

- `turn off water reminders`
- `enable check in reminders`
- `disable nap suggestions`

### Snooze a reminder

Say things like:

- `snooze the water reminder for 10 minutes`
- `delay check ins for 30 minutes`

### Remove a snooze

Say things like:

- `clear the water reminder snooze`
- `unsnooze check ins`

### Reset to defaults

Say things like:

- `reset the water reminder to default`
- `reset check in reminders`
- `reset quiet hours`

## Profile And Status Controls

### Preferred name and timezone

Say things like:

- `call me Karl`
- `set my timezone to UTC`
- `set my timezone to Asia Shanghai`
- `forget my preferred name`

### Temporary activity

This affects proactive behavior for a while.

Say things like:

- `I am working`
- `I am busy for 90 minutes`
- `I am taking a nap`
- `clear my current activity`

### Quiet hours

Say things like:

- `do not remind me after 10 pm`
- `do not remind me before 8 am`
- `set quiet hours from 22:00 to 08:00`
- `reset quiet hours`

## Inspecting Stored Status

Say things like:

- `what is my current reminder setup`
- `what is the drink water reminder setting`
- `what is my current status`
- `what happened today`
- `show recent reminder history`

## Clearing Stored Status

Say things like:

- `clear the recent summary`
- `delete today's reminder history`
- `delete yesterday's reminder history`
- `clear my current activity`

## Sense HAT LED Status Meanings

The LED matrix can show program state. Depending on the current code version, you may see patterns for:

- ready but not in a room
- connected and idle in a room
- speech to text
- language model processing
- text to speech
- memory insert
- memory retrieval
- proactive speaking
- Sense HAT tool activity

If the LEDs stop updating entirely while the service is active, check the service log.

## Good Phrasing Tips

For the most reliable behavior:

- ask one thing at a time
- use direct phrases for reminders and settings
- mention `Sense HAT` explicitly for sensor questions
- when changing a schedule, say the interval clearly

Examples:

- `Sense HAT temperature`
- `remind me to drink every 20 minutes`
- `check in with me every 2 hours`
- `set quiet hours from 11 pm to 7 am`

## Known Limits

- current sensor values depend on a successful live tool call
- if speech recognition mishears a device term, the correction layer may still miss it sometimes
- proactive reminders only happen when the session is active and the gating rules allow it
- the agent can clear structured status history, but long-run memory facts are managed through separate memory tools

## Troubleshooting

### The agent did not read the sensor

Ask again more directly:

- `use the Sense HAT and tell me the temperature`

Then check the service log for a Sense HAT tool call.

### The agent did not change a reminder correctly

Use a narrow request:

- `remind me to drink every 10 minutes`
- `check in with me every 30 minutes`
- `turn off water reminders`

### The agent is speaking too often

Try:

- `turn off check in reminders`
- `snooze water reminders for 60 minutes`
- `set quiet hours from 22:00 to 08:00`

### The agent forgot temporary context

That may be expected if the recent summary or temporary activity was cleared or expired. Ask:

- `what is my current status`

### The agent forgot a long-run fact

Ask:

- `what do you remember about me`

If needed, store it again with:

- `remember that ...`

## Logs For Advanced Users

If you need to inspect behavior:

Run log:

```bash
sudo journalctl -u livekit-sensehat-agent.service -f
```

Data-flow log:

```bash
tail -f /home/karl/RaspAgent/.data/data_flow.jsonl
```

The data-flow log is the best place to see:

- STT text
- STT correction
- LLM input and output
- tool calls
- reminder updates
- proactive scheduling decisions
