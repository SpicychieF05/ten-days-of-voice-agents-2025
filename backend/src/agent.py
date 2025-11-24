import logging
import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


@dataclass
class WellnessState:
    """Represents a short daily wellness check-in state."""
    mood: Optional[str] = None
    energy: Optional[str] = None
    stress: Optional[str] = None
    goals: List[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Return True when required fields (mood, energy, goals) exist."""
        return bool(self.mood and self.energy and self.goals)

    def missing_fields(self) -> List[str]:
        """Return a list of missing required fields."""
        missing = []
        if not self.mood:
            missing.append("mood")
        if not self.energy:
            missing.append("energy")
        if not self.goals:
            missing.append("goals")
        return missing

    def to_dict(self) -> dict:
        return asdict(self)


class Assistant(Agent):
    def __init__(self) -> None:
        # Initialize wellness state
        self.wellness_state = WellnessState()

        # Set up wellness log file path
        self.wellness_file = os.path.join(
            os.path.dirname(__file__),
            "wellness_log.json",
        )

        # Load past entries so we can softly reference them in the persona
        self.past_entries: List[dict] = self._load_past_entries()
        last_ref = "No previous check-ins found." if not self.past_entries else (
            f"Last check-in on {self.past_entries[-1].get('timestamp', 'unknown date')}: mood '{self.past_entries[-1].get('mood','')}', energy '{self.past_entries[-1].get('energy','')}'."
        )

        # Persona instructions for a supportive wellness companion
        instructions = f"""You are a warm, calm, and encouraging daily wellness companion. You are NOT a clinician and must not provide medical advice or diagnoses.

When you begin a session:
- Greet the user in a gentle, human tone.
- Softly reference past check-ins when available (one short sentence). Example: '{last_ref}'
- Conduct a short daily check-in by asking one question at a time:
  1) How are you feeling today? (mood — free text or simple scale)
  2) What's your energy like right now? (low/medium/high or brief text)
  3) Any stress or things on your mind you'd like to note? (optional)
  4) What are 1–3 practical goals you want to accomplish today? (comma-separated or listed)

After collecting responses, offer a small, grounded, non-medical reflection focused on simple actions (e.g., small steps, breathing, breaking tasks down). Keep language simple and encouraging.

Close the session by summarizing the mood, energy, and goals, ask for confirmation to save, and then save the check-in to a JSON file when requested.

Rules:
- Do not offer medical diagnoses or definitive medical advice.
- Avoid clinical language; be supportive and practical.
- Keep responses concise and friendly.
"""

        super().__init__(
            instructions=instructions,
            tools=self._build_tools(),
        )

    def _load_past_entries(self) -> List[dict]:
        """Load past wellness entries from the JSON file if available."""
        if os.path.exists(self.wellness_file):
            try:
                with open(self.wellness_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception:
                logger.exception("Failed to read wellness log; starting fresh")
        return []
    
    def _build_tools(self):
        """Build and return wellness check-in function tools."""

        @function_tool
        async def set_mood(context: RunContext, mood: str):
            """Set the user's mood (free text or simple scale)."""
            logger.info(f"Setting mood: {mood}")
            self.wellness_state.mood = mood.strip()
            return "Thanks — noted your mood. How's your energy right now?"

        @function_tool
        async def set_energy(context: RunContext, energy: str):
            """Set the user's energy level (low/medium/high or free text)."""
            logger.info(f"Setting energy: {energy}")
            self.wellness_state.energy = energy.strip()
            return "Got it. Would you like to note any stress or things on your mind? (optional)"

        @function_tool
        async def set_stress(context: RunContext, stress: str):
            """Set optional stress/context notes."""
            logger.info(f"Setting stress: {stress}")
            self.wellness_state.stress = stress.strip()
            return "Thanks for sharing that. What are 1–3 practical goals for today?"

        @function_tool
        async def set_goals(context: RunContext, goals: str):
            """Set 1–3 goals (comma-separated or single-line)."""
            logger.info(f"Setting goals: {goals}")
            items = [g.strip() for g in goals.split(",") if g.strip()]
            # Limit to 3 goals
            self.wellness_state.goals = items[:3]
            return f"Great — I have {len(self.wellness_state.goals)} goal(s). Would you like a short reflection before we save?"

        @function_tool
        async def save_checkin(context: RunContext):
            """Validate, save the check-in to JSON, and return a recap string."""
            logger.info("Attempting to save check-in")

            if not self.wellness_state.is_complete():
                missing = self.wellness_state.missing_fields()
                return f"I can't save yet. Missing: {', '.join(missing)}"

            entry = self.wellness_state.to_dict()
            entry["timestamp"] = datetime.now().isoformat()

            # Create a brief assistant summary/reflection (non-medical, actionable)
            mood = entry.get("mood", "").strip()
            energy = entry.get("energy", "").strip()
            goals = entry.get("goals", [])
            goals_str = ", ".join(goals)
            reflection = (
                f"You said you're feeling '{mood}' with energy '{energy}'. "
                f"Today you plan: {goals_str}. A small suggestion: pick one goal to start with, set a 10–20 minute window, and take a short break after finishing."
            )
            entry["assistant_summary"] = reflection

            # Read existing log
            records = []
            if os.path.exists(self.wellness_file):
                try:
                    with open(self.wellness_file, 'r') as f:
                        records = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    records = []

            records.append(entry)

            # Write back to file (ensure folder exists)
            try:
                with open(self.wellness_file, 'w') as f:
                    json.dump(records, f, indent=2)
            except Exception as e:
                logger.exception("Failed to write wellness log")
                return "I tried to save your check-in but something went wrong. Please try again later."

            # Reset in-memory state
            self.wellness_state = WellnessState()

            logger.info(f"Check-in saved: {entry}")

            recap = (
                f"Check-in saved. {reflection} I'll keep this on file and we can reference past entries next time."
            )
            return recap

        return [
            set_mood,
            set_energy,
            set_stress,
            set_goals,
            save_checkin,
        ]


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Load VAD model (in case inference executor is disabled)
    if "vad" not in ctx.proc.userdata:
        ctx.proc.userdata["vad"] = silero.VAD.load()

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        # turn_detection=MultilingualModel(),  # Disabled due to Windows multiprocessing issues
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

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

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
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint, 
        prewarm_fnc=prewarm,
        agent_name="wellness-companion-day3"
    ))
