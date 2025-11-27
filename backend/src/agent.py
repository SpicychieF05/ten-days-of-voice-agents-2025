"""
Day 6: Punjab National Bank Fraud Alert Voice Agent
==================================================
This agent handles incoming fraud alert calls for Punjab National Bank.
It verifies suspicious transactions with account holders and updates fraud case status.

Features:
- Browser-based voice interface support
- LiveKit Telephony support (+1 518 600 7326)
- Secure identity verification
- Transaction fraud confirmation workflow
- JSON-based case persistence
"""

import logging
import json
import os
from typing import Optional, Dict, Any

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
from livekit.plugins import murf, silero, deepgram, noise_cancellation, google

logger = logging.getLogger("pnb-fraud-agent")

# Load environment variables from backend/.env.local
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.local")
load_dotenv(env_path)

# Murf Falcon TTS Configuration
VOICE_ID = "Nikhil"
VOICE_STYLE = "Conversational"
VOICE_MODEL = "Falcon"

# Telephony Configuration
TELEPHONY_PHONE_NUMBER = "+1 518 600 7326"
TELEPHONY_TRUNK_ID = "PN_PPN_xeEuDKznYvKd"
DISPATCH_RULE = "pnb-fraud-detection-dispatch"
AGENT_NAME = "pnb-fraud-agent"


class PNBFraudAgent(Agent):
    """
    Punjab National Bank Fraud Detection Voice Agent
    
    Handles fraud verification calls with the following workflow:
    1. Introduction as Maya Roy from PNB
    2. Ask for customer name
    3. Load fraud case from database
    4. Verify identity with security question
    5. Present suspicious transaction details
    6. Ask if customer recognizes the transaction
    7. Update case status based on response
    8. Provide summary and end call
    """

    def __init__(self, session: AgentSession) -> None:
        self._session = session

        # Load fraud cases database
        self.fraud_db = self._load_fraud_database()

        # Agent state
        self.current_case: Optional[Dict[str, Any]] = None
        self.verification_passed: bool = False
        self.user_name: Optional[str] = None

        # Build system instructions
        instructions = self._build_instructions()

        super().__init__(
            instructions=instructions,
            tools=self._build_tools(),
        )

        logger.info("Punjab National Bank Fraud Detection Agent initialized.")

    # ===========================================
    # DATABASE OPERATIONS
    # ===========================================

    def _get_fraud_cases_path(self) -> str:
        """Get path to fraud cases database file."""
        return os.path.join(
            os.path.dirname(__file__),
            "shared-data",
            "day6_fraud_cases.json"
        )

    def _load_fraud_database(self) -> Dict[str, Any]:
        """Load fraud cases database from JSON file."""
        fraud_path = self._get_fraud_cases_path()
        try:
            if os.path.exists(fraud_path):
                with open(fraud_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                logger.warning(f"Fraud cases file not found: {fraud_path}")
                return {"bankInfo": {}, "callerIdentity": {}, "fraudCases": []}
        except Exception as e:
            logger.error(f"Error loading fraud database: {e}")
            return {"bankInfo": {}, "callerIdentity": {}, "fraudCases": []}

    def _save_fraud_database(self) -> bool:
        """Persist fraud database to JSON file safely."""
        fraud_path = self._get_fraud_cases_path()
        try:
            temp_path = fraud_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.fraud_db, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, fraud_path)
            logger.info(f"Fraud database saved successfully to: {fraud_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving fraud database: {e}")
            return False

    # ===========================================
    # SYSTEM INSTRUCTIONS
    # ===========================================

    def _build_instructions(self) -> str:
        """Build the agent's system prompt with persona and workflow."""
        
        bank_name = self.fraud_db.get("bankInfo", {}).get("bankName", "Punjab National Bank")
        branch = self.fraud_db.get("bankInfo", {}).get("branch", "Salt Lake Branch, Kolkata")
        rep_name = self.fraud_db.get("callerIdentity", {}).get("representativeName", "Maya Roy")
        rep_role = self.fraud_db.get("callerIdentity", {}).get("representativeRole", "Fraud Detection Officer")

        return f"""
You are {rep_name}, a {rep_role} at {bank_name}, {branch}.

Your role is to help customers verify suspicious transactions on their accounts.

IMPORTANT SECURITY RULES:
- NEVER ask for PIN, CVV, full card number, OTP, or passwords
- NEVER request sensitive banking credentials
- ONLY use the security question from the fraud case database
- Be calm, professional, and reassuring at all times

START THE CONVERSATION IMMEDIATELY:
As soon as the call connects, greet the customer by saying:
"Hello, this is {rep_name} calling from {bank_name}, {branch}. We detected a suspicious transaction on your account and wanted to verify it with you. May I have your full name please?"

CALL FLOW:
1. INTRODUCTION (Start immediately with greeting above)

2. IDENTITY COLLECTION
   - After customer provides name, call tool: load_fraud_case(userName)
   - If case not found, politely say you'll need to verify details and end call

3. SECURITY VERIFICATION
   - Say: "For security purposes, I need to verify your identity with a quick question."
   - Ask the security question from the fraud case
   - When user answers, call tool: verify_answer(answer)
   - If verification fails, apologize and end call politely

4. TRANSACTION PRESENTATION
   - After successful verification, present the suspicious transaction details:
     * Transaction amount
     * Merchant name
     * Transaction time
     * Transaction method
     * Risk context (explain why it was flagged)
   
5. CONFIRMATION
   - Ask: "Did you authorize this transaction?"
   - If user says YES/recognizes it → call tool: mark_safe()
   - If user says NO/doesn't recognize it → call tool: mark_fraud()

6. SUMMARY & CLOSURE
   - Provide a brief summary of the action taken
   - Thank the customer for their time
   - If fraud confirmed: "We have blocked this transaction and will investigate further. You will not be charged."
   - If safe confirmed: "Thank you for confirming. The transaction will proceed normally."
   - Say goodbye professionally

TONE:
- Calm and reassuring (customers may be worried)
- Professional and courteous
- Clear and concise
- Empathetic to customer concerns

ENDING THE CALL:
If the user says any variant of "that's all", "thank you", "goodbye", "I'm done":
- Call tool: save_case() to persist any updates
- Provide final summary
- End with: "Thank you for your time. Have a great day."
"""

    # ===========================================
    # TOOLS
    # ===========================================

    def _build_tools(self):
        """Build all fraud detection tools."""
        tools = []

        @function_tool
        async def load_fraud_case(context: RunContext, userName: str) -> str:
            """
            Load fraud case for the specified customer name.
            Returns case details or error message.
            """
            cleaned_name = userName.strip()
            if not cleaned_name:
                return "Customer name cannot be empty. Please provide a valid name."

            self.user_name = cleaned_name

            # Search for case in database
            fraud_cases = self.fraud_db.get("fraudCases", [])
            for case in fraud_cases:
                if case.get("userName", "").lower() == cleaned_name.lower():
                    self.current_case = case
                    logger.info(f"Loaded fraud case for: {cleaned_name}")
                    
                    # Return structured case info for agent
                    return f"""
Fraud case loaded successfully for {cleaned_name}.

Account: {case.get('maskedAccount', 'N/A')}
Card: {case.get('maskedCard', 'N/A')}

Suspicious Transaction:
- Amount: {case.get('transactionAmount', 'N/A')}
- Merchant: {case.get('merchant', 'N/A')}
- Time: {case.get('transactionTime', 'N/A')}
- Method: {case.get('transactionMethod', 'N/A')}
- Location: {case.get('location', 'N/A')}

Risk Context: {case.get('spokenRiskContext', 'Transaction flagged by security systems.')}

Security Question: {case.get('securityQuestion', 'N/A')}

Now proceed to verify the customer's identity with the security question.
"""

            # Case not found
            logger.warning(f"No fraud case found for: {cleaned_name}")
            return f"I apologize, but I cannot find a pending fraud alert for the name '{cleaned_name}'. Please verify the name or contact our main helpline for assistance."

        @function_tool
        async def verify_answer(context: RunContext, answer: str) -> str:
            """
            Verify customer's answer to security question.
            Returns verification result.
            """
            if not self.current_case:
                return "Error: No fraud case loaded. Please provide your name first."

            cleaned_answer = answer.strip()
            correct_answer = self.current_case.get("securityAnswer", "").strip()

            # Case-insensitive comparison
            if cleaned_answer.lower() == correct_answer.lower():
                self.verification_passed = True
                logger.info(f"Identity verification passed for: {self.user_name}")
                return "Identity verification successful. Now proceed to present the suspicious transaction details and ask if the customer recognizes it."
            else:
                self.verification_passed = False
                logger.warning(f"Identity verification failed for: {self.user_name}")
                return "I'm sorry, but that answer doesn't match our records. For your security, I cannot proceed with this call. Please contact our branch directly with a valid ID. Thank you."

        @function_tool
        async def mark_safe(context: RunContext) -> str:
            """
            Mark the transaction as safe (customer authorized it).
            Updates case status to 'confirmed_safe'.
            """
            if not self.current_case:
                return "Error: No fraud case loaded."

            if not self.verification_passed:
                return "Error: Identity verification not completed."

            self.current_case["status"] = "confirmed_safe"
            self.current_case["outcomeNote"] = f"Customer {self.user_name} confirmed the transaction as legitimate."
            
            logger.info(f"Transaction marked as SAFE for: {self.user_name}")
            
            return f"Transaction confirmed as safe. The charge of {self.current_case.get('transactionAmount', 'N/A')} to {self.current_case.get('merchant', 'N/A')} will proceed normally. Thank the customer and prepare to end the call."

        @function_tool
        async def mark_fraud(context: RunContext) -> str:
            """
            Mark the transaction as fraudulent (customer did not authorize it).
            Updates case status to 'confirmed_fraud'.
            """
            if not self.current_case:
                return "Error: No fraud case loaded."

            if not self.verification_passed:
                return "Error: Identity verification not completed."

            self.current_case["status"] = "confirmed_fraud"
            self.current_case["outcomeNote"] = f"Customer {self.user_name} confirmed this is fraudulent. Transaction blocked."
            
            logger.info(f"Transaction marked as FRAUD for: {self.user_name}")
            
            return f"Transaction confirmed as fraudulent. We have blocked the charge of {self.current_case.get('transactionAmount', 'N/A')} to {self.current_case.get('merchant', 'N/A')}. The customer will not be charged. Our fraud investigation team will follow up. Thank the customer and prepare to end the call."

        @function_tool
        async def mark_verification_failed(context: RunContext) -> str:
            """
            Mark case when verification fails.
            Updates case status to 'verification_failed'.
            """
            if not self.current_case:
                return "Error: No fraud case loaded."

            self.current_case["status"] = "verification_failed"
            self.current_case["outcomeNote"] = f"Identity verification failed for {self.user_name}. Call terminated."
            
            logger.info(f"Verification FAILED for: {self.user_name}")
            
            return "Verification failed. Case marked. End the call politely and securely."

        @function_tool
        async def save_case(context: RunContext) -> str:
            """
            Save current case updates to database.
            Persists all changes to JSON file.
            """
            if not self.current_case:
                return "No case to save."

            # Update the case in the database
            fraud_cases = self.fraud_db.get("fraudCases", [])
            for i, case in enumerate(fraud_cases):
                if case.get("userName") == self.current_case.get("userName"):
                    fraud_cases[i] = self.current_case
                    break

            # Save to file
            if self._save_fraud_database():
                logger.info(f"Case saved successfully for: {self.user_name}")
                return "Case information saved successfully to database."
            else:
                logger.error("Failed to save case to database")
                return "Warning: Failed to save case to database. Please check system logs."

        tools.extend([
            load_fraud_case,
            verify_answer,
            mark_safe,
            mark_fraud,
            mark_verification_failed,
            save_case
        ])

        return tools


# ===========================================
# LIVEKIT AGENT ENTRYPOINT
# ===========================================

def prewarm(proc: JobProcess):
    """Prewarm function to preload VAD model."""
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD model preloaded")


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for LiveKit agent.
    Handles both browser-based and telephony-based connections.
    """
    logger.info(f"🔔 PNB Fraud Agent joining room: {ctx.room.name}")

    # Load VAD if not prewarmed
    if "vad" not in ctx.proc.userdata:
        ctx.proc.userdata["vad"] = silero.VAD.load()

    # Connect to room
    await ctx.connect()
    logger.info(f"✅ Connected to room: {ctx.room.name}")

    # Create assistant instance first to get greeting
    fraud_db_path = os.path.join(os.path.dirname(__file__), "shared-data", "day6_fraud_cases.json")
    try:
        with open(fraud_db_path, "r", encoding="utf-8") as f:
            fraud_db = json.load(f)
    except:
        fraud_db = {"callerIdentity": {"representativeName": "Maya Roy"}, "bankInfo": {"bankName": "Punjab National Bank", "branch": "Salt Lake Branch, Kolkata"}}
    
    rep_name = fraud_db.get("callerIdentity", {}).get("representativeName", "Maya Roy")
    bank_name = fraud_db.get("bankInfo", {}).get("bankName", "Punjab National Bank")
    branch = fraud_db.get("bankInfo", {}).get("branch", "Salt Lake Branch, Kolkata")
    
    greeting = f"Hello, this is {rep_name} calling from {bank_name}, {branch}. We detected a suspicious transaction on your account and wanted to verify it with you. May I have your full name please?"

    # Create AgentSession with Murf Falcon TTS, Deepgram STT, Gemini LLM
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.0-flash-exp"),
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

    # Create assistant instance
    assistant = PNBFraudAgent(session)

    # Start agent session
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    
    logger.info(f"✅ Agent session started")

    # Send initial greeting using the session's say method
    await session.say(greeting, allow_interruptions=True)
    
    logger.info(f"🎙️ Initial greeting sent, agent ready for fraud detection calls")


# ===========================================
# MAIN WORKER
# ===========================================

if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        agent_name=AGENT_NAME,
    ))
