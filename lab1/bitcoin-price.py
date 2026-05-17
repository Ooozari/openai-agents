import os
import asyncio
from dotenv import load_dotenv, find_dotenv

from agents import (
    Agent, 
    Runner, 
    OpenAIChatCompletionsModel, 
    AsyncOpenAI, 
    set_tracing_disabled
)

try:
    from agents import MCPServerSse
except ImportError:
    from agents.mcp import MCPServerSse

load_dotenv(find_dotenv())
set_tracing_disabled(disabled=True)

external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash-lite",
    openai_client=external_client
)

async def main():
    async with MCPServerSse(
        params={
            "url": "https://mcp.api.coingecko.com/sse",
            "timeout": 30,
            "sse_read_timeout": 60
        },
        name="coingecko",
        client_session_timeout_seconds=30
    ) as mcp_server:

        print("Connected to MCP server. Fetching tools...")
        server_tools = await mcp_server.list_tools()
        print(f"Available tools: {[t.name for t in server_tools]}")

        crypto_agent = Agent(
            name="CryptoTracker",
            instructions="You are a crypto assistant. Use the provided MCP tools to answer questions.",
            model=llm_model,
            mcp_servers=[mcp_server],
        )

        print("\nFetching Bitcoin price via MCP...")
        result = await Runner.run(crypto_agent, "What's the price of Bitcoin?")
        print(f"\nFinal Output: {result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())