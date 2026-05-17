import os
import pdfplumber
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, SQLiteSession

load_dotenv()
set_tracing_disabled(disabled=True)

# ── Gemini client (free) ──────────────────────────────────────────────────────
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client
)

# ── Read PDF ──────────────────────────────────────────────────────────────────
PDF_PATH = "document.pdf"

def load_pdf_text(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

pdf_text = load_pdf_text(PDF_PATH)

# ── Agent with PDF injected directly into instructions ────────────────────────
agent = Agent(
    name="PDFReaderTool",
    instructions=f"""You are an AI agent that answers questions ONLY from the document below.
Never use your own knowledge. Only answer from the document. Answer in one sentence.

DOCUMENT:
{pdf_text}
""",
    model=model,
)

# ── SQLite session ────────────────────────────────────────────────────────────
session = SQLiteSession("first_session")

# ── Chat loop ─────────────────────────────────────────────────────────────────
while True:
    question = input("You: ")
    result = Runner.run_sync(agent, question, session=session)
    print("Agent: ", result.final_output)