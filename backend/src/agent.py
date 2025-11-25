import logging
import json
import os
from typing import List, Optional, Sequence

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
    RoomInputOptions,
    FunctionTool
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Voice Constants
VOICE_LEARN = "en-US-matthew"
VOICE_QUIZ = "en-US-alicia"
VOICE_TEACH_BACK = "en-US-ken"

class Assistant(Agent):
    def __init__(self, session: AgentSession) -> None:
        self._session = session
        self.content_file = os.path.join(
            os.path.dirname(__file__),
            "shared-data",
            "day4_tutor_content.json",
        )
        self.content = self._load_content()
        
        # Initial State
        self.current_mode = "learn"
        self.current_concept_id = "variables"
        self.current_voice = VOICE_LEARN

        # Build initial instructions
        instructions = self._build_instructions()

        super().__init__(
            instructions=instructions,
            tools=self._build_tools(),
        )
        
        # Set initial voice
        self._update_voice()

    def _load_content(self) -> List[dict]:
        """Load content from the JSON file."""
        if os.path.exists(self.content_file):
            try:
                with open(self.content_file, 'r') as f:
                    return json.load(f)
            except Exception:
                logger.exception("Failed to read content file")
        return []

    def _build_instructions(self) -> str:
        """Construct the system instructions."""
        return f"""You are an Active Recall Tutor designed to help users learn concepts effectively.
You have three learning modes:
1. **Learn Mode**: You explain the concept clearly using the summary.
2. **Quiz Mode**: You ask a sample question to test the user's knowledge.
3. **Teach-Back Mode**: You ask the user to explain the concept back to you and provide feedback.

Current State:
- Mode: {self.current_mode}
- Concept ID: {self.current_concept_id}

Content Data:
{json.dumps(self.content, indent=2)}

Rules:
- When the user asks to switch modes (e.g., "quiz me", "teach me"), call `set_learning_mode`.
- When the user asks to switch concepts (e.g., "let's do loops"), call `set_concept`.
- Always adapt your response to the current mode.
- In **Learn Mode**: Read the 'summary' for the current concept.
- In **Quiz Mode**: Ask the 'sample_question' for the current concept.
- In **Teach-Back Mode**: Ask the user to explain the concept. If they explain, give qualitative feedback.
- If the user greets you, ask them which concept they want to learn or which mode they prefer.
"""

    def _update_voice(self):
        """Update the TTS voice based on the current mode."""
        if self.current_mode == "learn":
            # Matthew
            self.current_voice = VOICE_LEARN
        elif self.current_mode == "quiz":
            # Alicia
            self.current_voice = VOICE_QUIZ
        elif self.current_mode == "teach_back":
            # Ken
            self.current_voice = VOICE_TEACH_BACK
        
        # Update the session TTS if available
        if hasattr(self._session, 'tts') and self._session.tts is not None:
            try:
                self._session.tts.voice = self.current_voice  # type: ignore
            except AttributeError:
                pass  # Voice may not be settable on this TTS instance
        
        logger.info(f"Updated voice to {self.current_voice} for mode {self.current_mode}")

    def _build_tools(self) -> list:
        @function_tool
        async def set_learning_mode(context: RunContext, mode: str):
            """Switch the learning mode. Valid modes: 'learn', 'quiz', 'teach_back'."""
            mode = mode.lower().strip()
            if mode not in ["learn", "quiz", "teach_back"]:
                return "Invalid mode. Please choose learn, quiz, or teach_back."
            
            self.current_mode = mode
            self._update_voice()
            
            # Note: We don't strictly need to update self.instructions dynamically for the LLM to know,
            # as long as the tool output confirms the switch, the LLM context will have it.
            return f"Switched to {mode} mode. Voice updated to {self.current_voice}."

        @function_tool
        async def set_concept(context: RunContext, concept_id: str):
            """Switch the current concept. Valid IDs: 'variables', 'loops'."""
            concept_id = concept_id.lower().strip()
            valid_ids = [c['id'] for c in self.content]
            
            # Simple fuzzy match or direct match
            if concept_id not in valid_ids:
                # Try to find by title
                found = False
                for c in self.content:
                    if c['title'].lower() == concept_id:
                        concept_id = c['id']
                        found = True
                        break
                if not found:
                    return f"Invalid concept. Available: {', '.join(valid_ids)}"
            
            self.current_concept_id = concept_id
            return f"Switched concept to {concept_id}."

        @function_tool
        async def load_content_file(context: RunContext):
            """Reload the content JSON file."""
            self.content = self._load_content()
            return "Content file reloaded."

        return [set_learning_mode, set_concept, load_content_file]


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Load VAD model
    if "vad" not in ctx.proc.userdata:
        ctx.proc.userdata["vad"] = silero.VAD.load()

    # Initialize Session
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice=VOICE_LEARN, # Default to Learn mode voice
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Initialize Assistant with session
    assistant = Assistant(session)

    # Start the session
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),  # type: ignore
        ),
    )

    # Join the room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint, 
        prewarm_fnc=prewarm,
        agent_name="day4-active-recall-tutor"
    ))
