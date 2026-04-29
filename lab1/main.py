import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool, ModelSettings, StopAtTools

load_dotenv()
set_tracing_disabled(disabled=True)

# Create client for Gemini
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Select model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

@function_tool
def get_order_status(order_id: int) -> str:
    """
    Returns the order status given an order ID.
    Args: order_id (int) - Order ID of the customer's order
    Returns: string - Status message
    """
    if order_id in (100, 101):
        return "Delivered"
    elif order_id in (200, 201):
        return "In Transit"
    elif order_id in (300, 301):
        return "Processing"
    else:
        return "Order not found"

agent = Agent(
    name="HelloAgent",
    instructions="You are a helpful assistant. Use the get_order_status tool when asked about orders.",
    model=model,
    tools=[get_order_status], 
    tool_use_behavior=StopAtTools(stop_at_tool_names=["get_order_status"]),
    # model_settings=ModelSettings(tool_choice = 'required'),
)

# Get user input
user_message = input("Enter your message: ")

# Run agent
result = Runner.run_sync(agent, user_message)

print("Agent Output:", result.final_output)