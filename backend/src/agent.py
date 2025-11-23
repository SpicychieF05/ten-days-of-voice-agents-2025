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
class OrderState:
    """Represents a coffee order state for Falcon Brew Coffee."""
    drinkType: Optional[str] = None
    size: Optional[str] = None
    milk: Optional[str] = None
    extras: List[str] = field(default_factory=list)
    name: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if all required fields are filled (extras are optional)."""
        return all([
            self.drinkType is not None,
            self.size is not None,
            self.milk is not None,
            self.name is not None
        ])
    
    def missing_fields(self) -> List[str]:
        """Return list of missing required fields."""
        missing = []
        if self.drinkType is None:
            missing.append("drink type")
        if self.size is None:
            missing.append("size")
        if self.milk is None:
            missing.append("milk preference")
        if self.name is None:
            missing.append("customer name")
        return missing
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class Assistant(Agent):
    def __init__(self) -> None:
        # Initialize order state
        self.order_state = OrderState()
        
        # Set up orders file path
        self.orders_file = os.path.join(
            os.path.dirname(__file__), 
            "orders.json"
        )
        
        # Initialize the agent with barista persona and tools
        super().__init__(
            instructions="""You are a friendly and professional barista at Falcon Brew Coffee, a premium coffee shop known for exceptional service and quality beverages.

Your role:
- Greet customers warmly when they arrive
- Help them place their coffee order by asking clarifying questions one at a time
- Be patient, polite, and conversational like a real barista
- Use the provided tools to record order details AS SOON AS the customer mentions them
- Keep responses natural and concise, without emojis, asterisks, or complex formatting

Order collection process:
1. When a customer mentions a drink, immediately use set_drink_type tool
2. After drink is set, ask about size and use set_size tool when mentioned
3. After size is set, ask about milk and use set_milk tool when mentioned
4. After milk is set, ask about extras and use set_extras tool if mentioned
5. After extras (or if none), ask for name and use set_customer_name tool
6. Once all details are collected, use save_order to finalize the order

Required information:
- Drink type (latte, cappuccino, espresso, americano, mocha, flat white, etc.)
- Size (small, medium, large)
- Milk preference (whole milk, oat milk, almond milk, soy milk, skim milk, no milk)
- Customer name
- Optional extras (vanilla syrup, caramel, whipped cream, extra shot, chocolate drizzle, etc.)

Always be helpful and make customers feel welcome at Falcon Brew Coffee!""",
            tools=self._build_tools(),
        )
    
    def _build_tools(self):
        """Build and return the list of function tools for the barista."""
        
        @function_tool
        async def set_drink_type(context: RunContext, drink: str):
            """Record the type of drink the customer wants to order.
            
            Use this tool immediately when the customer mentions what drink they want.
            
            Args:
                drink: The type of drink (e.g., latte, cappuccino, espresso, americano, mocha, flat white, macchiato)
            """
            logger.info(f"Setting drink type: {drink}")
            self.order_state.drinkType = drink.strip().lower()
            return f"Got it, one {drink}. What size would you like?"
        
        @function_tool
        async def set_size(context: RunContext, size: str):
            """Record the size of the drink.
            
            Use this tool immediately when the customer mentions the size.
            
            Args:
                size: The size of the drink (small, medium, large, or short, tall, grande, venti)
            """
            logger.info(f"Setting size: {size}")
            # Normalize size variations
            size_normalized = size.strip().lower()
            if size_normalized in ["short", "tall", "small"]:
                size_normalized = "small"
            elif size_normalized in ["grande", "medium"]:
                size_normalized = "medium"
            elif size_normalized in ["venti", "large"]:
                size_normalized = "large"
            
            self.order_state.size = size_normalized
            return f"Perfect, {size_normalized} it is. What kind of milk would you like?"
        
        @function_tool
        async def set_milk(context: RunContext, milk_type: str):
            """Record the customer's milk preference.
            
            Use this tool immediately when the customer mentions their milk preference.
            
            Args:
                milk_type: The type of milk (whole milk, oat milk, almond milk, soy milk, skim milk, 2%, no milk)
            """
            logger.info(f"Setting milk type: {milk_type}")
            self.order_state.milk = milk_type.strip().lower()
            return f"Great choice, {milk_type}. Would you like any extras like vanilla syrup, caramel, whipped cream, or an extra shot?"
        
        @function_tool
        async def set_extras(context: RunContext, extras: str):
            """Record any extras or additions to the drink.
            
            Use this tool when the customer mentions extras, add-ons, or special requests.
            Can be called multiple times for different extras.
            
            Args:
                extras: The extras to add (e.g., vanilla syrup, caramel, whipped cream, extra shot, chocolate drizzle)
            """
            logger.info(f"Adding extras: {extras}")
            # Parse comma-separated extras or single extra
            extra_items = [e.strip().lower() for e in extras.split(",")]
            self.order_state.extras.extend(extra_items)
            
            if self.order_state.name:
                return f"Added {extras} to your order. Let me finalize that for you."
            else:
                return f"Added {extras}. May I have your name for the order?"
        
        @function_tool
        async def set_customer_name(context: RunContext, name: str):
            """Record the customer's name for the order.
            
            Use this tool when the customer provides their name.
            
            Args:
                name: The customer's name
            """
            logger.info(f"Setting customer name: {name}")
            self.order_state.name = name.strip()
            
            # Check if order is complete
            if self.order_state.is_complete():
                return f"Thank you {name}. Let me confirm your order and get that started for you."
            else:
                missing = self.order_state.missing_fields()
                return f"Thank you {name}. I still need to know your {missing[0]}."
        
        @function_tool
        async def save_order(context: RunContext):
            """Save the completed order to the orders file.
            
            Use this tool ONLY when all required order information has been collected:
            - drink type
            - size
            - milk preference
            - customer name
            
            This will write the order to orders.json and reset the order state.
            """
            logger.info("Attempting to save order")
            
            # Check if order is complete
            if not self.order_state.is_complete():
                missing = self.order_state.missing_fields()
                return f"I cannot save the order yet. I still need: {', '.join(missing)}"
            
            # Add timestamp
            order_dict = self.order_state.to_dict()
            order_dict["timestamp"] = datetime.now().isoformat()
            
            # Read existing orders or create new list
            orders = []
            if os.path.exists(self.orders_file):
                try:
                    with open(self.orders_file, 'r') as f:
                        orders = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    orders = []
            
            # Append new order
            orders.append(order_dict)
            
            # Write to file
            with open(self.orders_file, 'w') as f:
                json.dump(orders, f, indent=2)
            
            logger.info(f"Order saved successfully: {order_dict}")
            
            # Create summary
            extras_str = ", ".join(self.order_state.extras) if self.order_state.extras else "none"
            summary = f"""Order confirmed for {self.order_state.name}:
- {self.order_state.size} {self.order_state.drinkType}
- {self.order_state.milk}
- Extras: {extras_str}

Your order will be ready shortly. Thank you for choosing Falcon Brew Coffee!"""
            
            # Reset order state for next customer
            self.order_state = OrderState()
            
            return summary
        
        return [
            set_drink_type,
            set_size,
            set_milk,
            set_extras,
            set_customer_name,
            save_order,
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
        agent_name="coffee-barista"
    ))
