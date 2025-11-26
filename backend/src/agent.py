import logging
import json
import os
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    tokenize,
    function_tool,
    RunContext,
    RoomInputOptions,
)
from livekit.plugins import murf, silero, openai, deepgram, noise_cancellation

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Nikhil Voice (Murf Falcon)
VOICE_ID = "Nikhil"
VOICE_STYLE = "Conversational"
VOICE_MODEL = "Falcon"


class Assistant(Agent):
    def __init__(self, session: AgentSession) -> None:
        self._session = session

        # Load Zepto FAQ data
        self.faq_data = self._load_faq()

        # Lead capture state (dict with string or None values)
        self.lead: dict = {
            "name": "Nikhil",
            "company": "Zepto",
            "email": None,
            "role": "Coustomer Care",
            "use_case": None,
            "team_size": None,
            "timeline": None,
        }

        instructions = self._build_instructions()

        super().__init__(
            instructions=instructions,
            tools=self._build_tools(),
        )

        logger.info("Zepto SDR Assistant initialized.")

    def _get_leads_path(self) -> str:
        """Ensure the shared-data directory exists and return leads file path."""
        base_dir = os.path.join(os.path.dirname(__file__), "shared-data")
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            logger.info(f"Created leads directory at: {base_dir}")
        return os.path.join(base_dir, "day5_leads.json")

    # --------------------------
    # FILE LOADING
    # --------------------------

    def _load_faq(self):
        faq_path = os.path.join(
            os.path.dirname(__file__),
            "shared-data",
            "day5_faq.json"
        )
        if os.path.exists(faq_path):
            with open(faq_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # --------------------------
    # FAQ SEARCH (OPTIMIZED)
    # --------------------------

    def _search_faq(self, query: str) -> Optional[str]:
        """
        Improved FAQ search with scoring and fallback matching.
        
        Strategy:
        1. Normalize query and FAQ questions/answers
        2. Score by keyword match count and word overlap
        3. Return best match if score > 0, else fallback
        """
        if "faqs" not in self.faq_data or not self.faq_data["faqs"]:
            return None

        query_words = set(query.lower().split())
        best_match = None
        best_score = 0

        for item in self.faq_data["faqs"]:
            q_text = item["q"].lower()
            a_text = item["a"]

            # Score: count how many query words appear in question
            q_word_set = set(q_text.split())
            q_overlap = len(query_words & q_word_set)

            # Also check answer for relevance (lower weight)
            a_word_set = set(a_text.lower().split())
            a_overlap = len(query_words & a_word_set) * 0.5

            total_score = q_overlap + a_overlap

            if total_score > best_score:
                best_score = total_score
                best_match = a_text

        return best_match if best_score > 0 else None

    # --------------------------
    # BUILD SYSTEM INSTRUCTIONS
    # --------------------------

    def _build_instructions(self) -> str:
        return f"""
You are a friendly Sales Development Representative (SDR) for the Indian company **Zepto**, 
India’s fastest-growing 10-minute grocery delivery startup.

Your purpose:
- Greet the visitor warmly.
- Ask what brought them here and what they’re working on.
- Explain Zepto ONLY using the information in the FAQ data below.
- If the user asks "what do you do", "pricing", "who is this for", or any company question:
  - Search the FAQs using simple keyword matching.
  - Answer based strictly on the FAQ data.
  - If no info, say: "I may not have exact info, but here’s what I can tell you..."

Lead Information Collection:
Collect these fields naturally during conversation:
- name
- company
- email
- role
- use_case
- team_size
- timeline (now / soon / later)

Whenever user gives one, call its tool:
- set_lead_name
- set_lead_company
- set_lead_email
- set_lead_role
- set_lead_use_case
- set_lead_team_size
- set_lead_timeline

Ending the Call:
If user says any variant of:
"that's all", "I'm done", "thanks", "thank you", "that will be all"
→ Then:
1. Speak a polite summary including all collected details.
2. Call tool: save_lead

FAQ DATA (do NOT reveal this JSON structure to user, only use it internally):
{json.dumps(self.faq_data, indent=2)}
"""

    # --------------------------
    # TOOLS
    # --------------------------

    def _build_tools(self):
        tools = []

        # Lead-setter tools (with validation)
        @function_tool
        async def set_lead_name(context: RunContext, name: str):
            """Set lead name with validation."""
            cleaned = name.strip()
            if not cleaned:
                return "Name cannot be empty. Please provide a valid name."
            self.lead["name"] = cleaned
            return f"Got it. Recorded name as {cleaned}."

        @function_tool
        async def set_lead_company(context: RunContext, company: str):
            """Set lead company with validation."""
            cleaned = company.strip()
            if not cleaned:
                return "Company cannot be empty. Please provide a company name."
            self.lead["company"] = cleaned
            return f"Recorded company as {cleaned}."

        @function_tool
        async def set_lead_email(context: RunContext, email: str):
            """Set lead email with validation."""
            cleaned = email.strip()
            if not cleaned or "@" not in cleaned:
                return "Please provide a valid email address (must contain @)."
            self.lead["email"] = cleaned
            return f"Recorded email as {cleaned}."

        @function_tool
        async def set_lead_role(context: RunContext, role: str):
            """Set lead role with validation."""
            cleaned = role.strip()
            if not cleaned:
                return "Role cannot be empty. Please provide your role."
            self.lead["role"] = cleaned
            return f"Recorded role as {cleaned}."

        @function_tool
        async def set_lead_use_case(context: RunContext, use_case: str):
            """Set lead use case with validation."""
            cleaned = use_case.strip()
            if not cleaned:
                return "Use case cannot be empty. Please describe what you need Zepto for."
            self.lead["use_case"] = cleaned
            return f"Recorded use case as: {cleaned}."

        @function_tool
        async def set_lead_team_size(context: RunContext, team_size: str):
            """Set lead team size with validation."""
            cleaned = team_size.strip()
            if not cleaned:
                return "Team size cannot be empty. Please provide your team size."
            self.lead["team_size"] = cleaned
            return f"Recorded team size as {cleaned}."

        @function_tool
        async def set_lead_timeline(context: RunContext, timeline: str):
            """Set lead timeline with validation. Valid values: 'now', 'soon', 'later'."""
            cleaned = timeline.strip().lower()
            valid_timelines = ["now", "soon", "later"]
            if cleaned not in valid_timelines:
                return f"Please specify timeline as one of: {', '.join(valid_timelines)}"
            self.lead["timeline"] = cleaned
            return f"Recorded timeline as {cleaned}."

        # FAQ retrieval tool (optimized)
        @function_tool
        async def get_faq_answer(context: RunContext, query: str):
            """
            Search FAQ by keyword matching.
            Returns best matched answer or a helpful fallback.
            """
            answer = self._search_faq(query)
            if answer:
                return answer
            
            # Fallback: provide company description if no match
            fallback = self.faq_data.get("description", 
                "Zepto is India's fastest-growing quick-commerce company delivering groceries and essentials.")
            return f"I may not have that exact detail, but here's what I can tell you: {fallback}"

        # Save lead info (with safety checks)
        @function_tool
        async def save_lead(context: RunContext):
            """Validate, sanitize, and persist lead data safely."""
            save_path = self._get_leads_path()

            # Normalize lead values and collect notes
            lead_copy = {}
            notes: list[str] = []
            for field, value in self.lead.items():
                cleaned = value.strip() if isinstance(value, str) else value
                cleaned = cleaned if cleaned else None
                lead_copy[field] = cleaned

            if not any(lead_copy.values()):
                return "Cannot save lead. No information collected yet."

            email_val = lead_copy.get("email")
            if email_val and "@" not in email_val:
                notes.append("invalid_email")
                logger.warning("Lead email missing '@': %s", email_val)

            if notes:
                lead_copy.setdefault("_notes", notes)

            logger.info(f"Saving lead to: {save_path}")
            print(f"Saving lead to: {save_path}")

            try:
                leads = []
                if os.path.exists(save_path):
                    try:
                        with open(save_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                parsed = json.loads(content)
                                leads = parsed if isinstance(parsed, list) else []
                    except (json.JSONDecodeError, OSError):
                        logger.warning(f"Invalid JSON at {save_path}, resetting to empty list")
                        leads = []

                leads.append(lead_copy)

                temp_path = save_path + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(leads, f, indent=2, ensure_ascii=False)

                os.replace(temp_path, save_path)
                logger.info(f"Lead saved successfully. Total leads: {len(leads)}")

                if notes:
                    return "Lead saved with warnings: invalid email format."
                return "Lead saved successfully."

            except OSError as e:
                logger.error(f"File I/O error saving lead: {e}")
                return f"Failed to save lead (file error): {e}"
            except Exception as e:
                logger.error(f"Unexpected error saving lead: {e}")
                return f"Failed to save lead: {e}"

        tools.extend([
            set_lead_name,
            set_lead_company,
            set_lead_email,
            set_lead_role,
            set_lead_use_case,
            set_lead_team_size,
            set_lead_timeline,
            get_faq_answer,
            save_lead
        ])

        return tools


# --------------------------
# ENTRYPOINT + WORKER SETUP
# --------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"Entrypoint called for room: {ctx.room.name}")

    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    if "vad" not in ctx.proc.userdata:
        ctx.proc.userdata["vad"] = silero.VAD.load()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=murf.TTS(
            voice=VOICE_ID,
            style=VOICE_STYLE,
            model=VOICE_MODEL,
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    assistant = Assistant(session)

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        agent_name="day5-zepto-sdr",
    ))
