# Raspberry Pi Deployment Guide

This guide is for running this agent on a Raspberry Pi 4 with a Sense HAT.

The intended deployment shape is:

- run the agent as a `systemd` service on the Pi
- join the LiveKit room from another device for testing
- avoid `console` mode on the Pi
- avoid building a React frontend on the Pi itself

For this project, that is the stable path. Building a frontend on the Pi wastes RAM and CPU for no benefit to the agent process.

## 1. Hardware and OS

Recommended baseline:

- Raspberry Pi 4
- Raspberry Pi OS 64-bit
- Sense HAT attached to the 40-pin header
- network access to LiveKit and your model providers

Update the Pi first:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

## 2. Enable the Sense HAT stack

Install the distro packages that provide the Sense HAT Python module and its IMU dependencies:

```bash
sudo apt update
sudo apt install -y \
  git \
  curl \
  build-essential \
  python3 \
  python3-venv \
  python3-dev \
  python3-sense-hat \
  sense-hat \
  python3-rtimulib \
  librtimulib-dev \
  i2c-tools
```

Enable I2C and the Sense HAT overlay:

```bash
sudo raspi-config nonint do_i2c 0
```

Then confirm `/boot/firmware/config.txt` contains:

```ini
dtparam=i2c_arm=on
dtoverlay=rpi-sense
```

If `dtoverlay=rpi-sense` is missing, add it manually and reboot:

```bash
sudo reboot
```

## 3. Install `uv`

This repo uses `uv` for environment management and execution.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

If you want `uv` available in non-interactive shells and `systemd`, make sure `$HOME/.local/bin` is on your `PATH`, or call it by absolute path as shown later in the service file.

## 4. Clone the project

```bash
cd ~
git clone <your-repo-url> RaspAgent
cd ~/RaspAgent
```

## 5. Create the virtualenv correctly

This is the critical part for Sense HAT support.

Do **not** rely on a default virtualenv if you want the `sense_hat` module from the Raspberry Pi OS packages. The working setup for this project uses:

- distro package: `python3-sense-hat`
- virtualenv with system site packages enabled

Create the environment like this:

```bash
cd ~/RaspAgent
rm -rf .venv
uv venv --python /usr/bin/python3 --system-site-packages .venv
uv sync
```

Verify that the venv can see the system-installed Sense HAT module:

```bash
cat .venv/pyvenv.cfg
.venv/bin/python -c "import sense_hat; print(sense_hat.__file__)"
```

Expected results:

- `.venv/pyvenv.cfg` contains `include-system-site-packages = true`
- `sense_hat` resolves from `/usr/lib/python3/dist-packages/sense_hat/...`

If you see `ModuleNotFoundError: No module named 'sense_hat'`, the usual cause is that `.venv` was created without `--system-site-packages`. Recreate the venv and verify again.

## 6. Configure the environment

Start from the example file:

```bash
cp .env.example .env.local
```

Fill in the values required for your deployment. At minimum:

```env
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

LLM_PROVIDER=openai_compatible
LLM_MODEL=
LLM_API_KEY=
LLM_BASE_URL=
```

The current Pi-oriented defaults in this project are already tuned to avoid the previous startup failures:

```env
STT_PROVIDER=local_vosk
TTS_PROVIDER=livekit_inference
AUDIO_ENHANCEMENT_PROVIDER=none
TURN_DETECTION_ENABLED=false
SENSEHAT_STATUS_DISPLAY_ENABLED=true
DATA_FLOW_LOG_ENABLED=true
```

Notes:

- `TURN_DETECTION_ENABLED=false` avoids the extra turn-detector model startup path, which is unnecessary on a constrained Pi unless you explicitly want it.
- `AUDIO_ENHANCEMENT_PROVIDER=none` avoids optional enhancement plugins that were previously causing service startup noise on the Pi.
- If you switch to `STT_PROVIDER=inworld` or `TTS_PROVIDER=inworld`, also fill the matching `INWORLD_*` variables.
- If you keep `STT_PROVIDER=local_vosk`, make sure `VOSK_MODEL_PATH` points to a valid local model directory.

## 7. Download model assets

Run the one-time model download step:

```bash
cd ~/RaspAgent
uv run python src/agent.py download-files
```

For the default Pi configuration, also make sure the Vosk model exists at the path from `.env.local`, for example:

```env
VOSK_MODEL_PATH=.models/vosk-model-small-en-us-0.15
```

If the directory is missing, download and unpack that Vosk model under `.models/`.

## 8. Verify locally before enabling the service

Check the Sense HAT import from the project venv:

```bash
cd ~/RaspAgent
.venv/bin/python -c "from sense_hat import SenseHat; print('sense_hat ok')"
```

Check basic sensor access:

```bash
cd ~/RaspAgent
.venv/bin/python - <<'PY'
from sense_hat import SenseHat
sense = SenseHat()
print("temperature:", sense.get_temperature())
print("humidity:", sense.get_humidity())
print("pressure:", sense.get_pressure())
PY
```

Then start the agent manually once:

```bash
cd ~/RaspAgent
uv run python src/agent.py start --log-level info
```

Join the room from another device and test there. That avoids `console` mode and avoids building a frontend on the Pi.

## 9. Run the agent as a service

Create `/etc/systemd/system/livekit-sensehat-agent.service`:

```ini
[Unit]
Description=LiveKit Sense HAT Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=karl
Group=karl
WorkingDirectory=/home/karl/RaspAgent
ExecStart=/home/karl/.local/bin/uv run python src/agent.py start --log-level info
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Adjust `User`, `Group`, `WorkingDirectory`, and `ExecStart` if your checkout lives somewhere else.

Enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable livekit-sensehat-agent.service
sudo systemctl restart livekit-sensehat-agent.service
sudo systemctl status livekit-sensehat-agent.service
```

## 10. Logs and troubleshooting

Service logs:

```bash
sudo journalctl -u livekit-sensehat-agent.service -f
```

Structured data-flow logs written by the app:

```bash
tail -f ~/RaspAgent/.data/data_flow.jsonl
```

Useful checks:

```bash
cd ~/RaspAgent
.venv/bin/python -V
.venv/bin/python -c "import sense_hat; print(sense_hat.__file__)"
grep '^include-system-site-packages' .venv/pyvenv.cfg
```

### `ModuleNotFoundError: No module named 'sense_hat'`

This was the main deployment pitfall on this project.

Fix it in this order:

1. install the OS packages:

```bash
sudo apt install -y python3-sense-hat sense-hat python3-rtimulib librtimulib-dev i2c-tools
```

2. recreate the virtualenv with system packages exposed:

```bash
cd ~/RaspAgent
rm -rf .venv
uv venv --python /usr/bin/python3 --system-site-packages .venv
uv sync
```

3. verify:

```bash
.venv/bin/python -c "import sense_hat; print(sense_hat.__file__)"
```

If that import does not resolve into `/usr/lib/python3/dist-packages/`, the venv is still wrong.

### Service starts but sensor tools fail

Check:

- the Sense HAT is physically attached correctly
- `dtoverlay=rpi-sense` is present
- I2C is enabled
- the service is running the project venv
- `.venv/bin/python` can import `sense_hat`

### The Pi struggles in `console` mode or while building a frontend

Do not use the Pi as the frontend build machine.

Use this split instead:

- Pi: run only the agent service
- laptop/desktop/phone: join the LiveKit room as the client
- if you need a web frontend, build it on another machine and point it at the same LiveKit deployment

That keeps the Pi focused on STT, tools, TTS, and sensor access.
