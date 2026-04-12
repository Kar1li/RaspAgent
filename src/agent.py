import logging
from typing import Any

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    ToolError,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


def _round_sensor_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {key: _round_sensor_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_sensor_value(item) for item in value]
    return value


class SenseHatReader:
    def __init__(self, sense_hat: Any | None = None) -> None:
        self._sense_hat = sense_hat

    @property
    def sense_hat(self) -> Any:
        if self._sense_hat is None:
            try:
                from sense_hat import SenseHat
            except ImportError as exc:
                raise RuntimeError(
                    "The Sense HAT Python package is not available. Install it on the Raspberry Pi with `sudo apt install sense-hat`, then run the agent where the package can be imported."
                ) from exc

            self._sense_hat = SenseHat()

        return self._sense_hat

    def environment(self) -> dict[str, Any]:
        sense = self.sense_hat
        return _round_sensor_value(
            {
                "temperature_c": sense.get_temperature(),
                "temperature_from_humidity_c": sense.get_temperature_from_humidity(),
                "temperature_from_pressure_c": sense.get_temperature_from_pressure(),
                "humidity_percent": sense.get_humidity(),
                "pressure_millibars": sense.get_pressure(),
            }
        )

    def motion(self) -> dict[str, Any]:
        sense = self.sense_hat
        return _round_sensor_value(
            {
                "orientation_degrees": sense.get_orientation_degrees(),
                "orientation_radians": sense.get_orientation_radians(),
                "compass_degrees": sense.get_compass(),
                "compass_raw_microteslas": sense.get_compass_raw(),
                "gyroscope_degrees": sense.get_gyroscope(),
                "gyroscope_raw_radians_per_second": sense.get_gyroscope_raw(),
                "accelerometer_degrees": sense.get_accelerometer(),
                "accelerometer_raw_g": sense.get_accelerometer_raw(),
            }
        )

    def joystick(self) -> dict[str, Any]:
        events = self.sense_hat.stick.get_events()
        return {
            "events": [
                {
                    "timestamp": event.timestamp,
                    "direction": event.direction,
                    "action": event.action,
                }
                for event in events
            ]
        }

    def light(self) -> dict[str, Any]:
        sense = self.sense_hat
        colour_sensor = getattr(sense, "colour", None) or getattr(sense, "color", None)
        if colour_sensor is None:
            return {
                "available": False,
                "reason": "This Sense HAT does not expose a colour and brightness sensor through the Python API.",
            }

        red, green, blue, clear = colour_sensor.colour
        colour_raw = getattr(colour_sensor, "colour_raw", None)
        rgb_raw = getattr(colour_sensor, "rgb", None)
        return {
            "available": True,
            "red": red,
            "green": green,
            "blue": blue,
            "clear_brightness": clear,
            "brightness_raw": getattr(colour_sensor, "brightness", None),
            "colour_raw": list(colour_raw) if colour_raw is not None else None,
            "rgb_raw": list(rgb_raw) if rgb_raw is not None else None,
            "gain": getattr(colour_sensor, "gain", None),
            "integration_cycles": getattr(colour_sensor, "integration_cycles", None),
            "integration_time_seconds": getattr(
                colour_sensor, "integration_time", None
            ),
        }

    def display(self) -> dict[str, Any]:
        sense = self.sense_hat
        return {
            "rotation_degrees": sense.rotation,
            "low_light": sense.low_light,
            "gamma": list(sense.gamma),
            "pixels": sense.get_pixels(),
        }

    def all(self) -> dict[str, Any]:
        return {
            "environment": self.environment(),
            "motion": self.motion(),
            "joystick": self.joystick(),
            "light": self.light(),
            "display": self.display(),
        }


class Assistant(Agent):
    def __init__(self, sense_hat_reader: SenseHatReader | None = None) -> None:
        self._sense_hat_reader = sense_hat_reader or SenseHatReader()
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            When the user asks about the Raspberry Pi Sense HAT, sensor readings, local temperature, humidity, pressure, orientation, movement, compass direction, light, brightness, colour, joystick input, or LED display state, use the Sense HAT tools before answering.
            Sensor tool values are live readings from the device. State the units clearly and keep the answer short enough for voice.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    def _read_sense_hat(self, category: str) -> dict[str, Any]:
        try:
            reader_method = getattr(self._sense_hat_reader, category)
            return reader_method()
        except Exception as exc:
            logger.exception("Unable to read Sense HAT %s data", category)
            raise ToolError(
                "I could not read the Sense HAT right now. Check that the Sense HAT is attached, enabled, and that the `sense_hat` Python package is available to this process."
            ) from exc

    @function_tool()
    async def get_sensehat_environment(self, context: RunContext) -> dict[str, Any]:
        """Read the Raspberry Pi Sense HAT environmental sensors.

        Use this when the user asks for local temperature, humidity, air pressure, barometric pressure, or general room conditions from the Sense HAT. Returns Celsius, relative humidity percent, and millibars.
        """

        return self._read_sense_hat("environment")

    @function_tool()
    async def get_sensehat_motion(self, context: RunContext) -> dict[str, Any]:
        """Read the Raspberry Pi Sense HAT IMU and compass sensors.

        Use this when the user asks for orientation, pitch, roll, yaw, compass heading, north direction, gyroscope, accelerometer, magnetometer, movement, tilt, or rotation from the Sense HAT.
        """

        return self._read_sense_hat("motion")

    @function_tool()
    async def get_sensehat_joystick(self, context: RunContext) -> dict[str, Any]:
        """Read pending Raspberry Pi Sense HAT joystick events.

        Use this when the user asks whether the Sense HAT joystick has been pressed, held, released, or moved up, down, left, right, or middle. Returns events since the last joystick read.
        """

        return self._read_sense_hat("joystick")

    @function_tool()
    async def get_sensehat_light(self, context: RunContext) -> dict[str, Any]:
        """Read the Raspberry Pi Sense HAT v2 colour and brightness sensor.

        Use this when the user asks for colour, color, red, green, blue, clear light, brightness, ambient light, light level, or raw colour sensor values from the Sense HAT. Returns unavailable if the attached Sense HAT does not have this v2 sensor.
        """

        return self._read_sense_hat("light")

    @function_tool()
    async def get_sensehat_display(self, context: RunContext) -> dict[str, Any]:
        """Read the Raspberry Pi Sense HAT LED matrix display state.

        Use this when the user asks what is currently shown on the LED matrix, the display rotation, low light mode, gamma table, or pixel colors.
        """

        return self._read_sense_hat("display")

    @function_tool()
    async def get_sensehat_snapshot(self, context: RunContext) -> dict[str, Any]:
        """Read all available Raspberry Pi Sense HAT information in one snapshot.

        Use this when the user asks for everything the Sense HAT can provide or asks for a full status report. Includes environmental readings, IMU and compass readings, joystick events, colour and brightness readings when available, and LED matrix state.
        """

        return self._read_sense_hat("all")


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="openai/gpt-5.3-chat-latest"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else ai_coustics.audio_enhancement(
                        model=ai_coustics.EnhancerModel.QUAIL_VF_L
                    )
                ),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
