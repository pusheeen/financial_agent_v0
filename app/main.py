# main.py

"""
Main FastAPI application file, adapted to use the Google ADK agent system.
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from starlette.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

# --- ADK Imports ---
from typing import Optional, Dict, Any
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
# Add this import
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)
# --- Import your new ADK root agent ---
from app.agents.agents import root_agent # Make sure the path is correct
from dotenv import load_dotenv  # <-- ADD THIS
load_dotenv()

# --- Configuration ---
APP_DIR = Path(__file__).resolve().parent

# --- ADK Agent Runner Setup ---
class AgentCaller:
    """A simple wrapper class for interacting with an ADK agent."""
    def __init__(self, agent: Agent, runner: Runner, user_id: str, session_id: str):
        self.agent = agent
        self.runner = runner
        self.user_id = user_id
        self.session_id = session_id

    async def call(self, user_message: str, include_reasoning: bool = False) -> dict:
        content = types.Content(role='user', parts=[types.Part(text=user_message)])

        final_response = {
            'answer': "Agent did not produce a final response.",
            'status': 'error'
        }

        reasoning_steps = []
        tools_used = set()

        async for event in self.runner.run_async(user_id=self.user_id, session_id=self.session_id, new_message=content):
            if include_reasoning and event.author != self.agent.name and not event.is_final_response():
                 # Capture tool calls and observations as reasoning steps
                 if event.content and event.content.parts and hasattr(event.content.parts[0], 'tool_code'):
                     tool_call = event.content.parts[0].tool_code
                     reasoning_steps.append({
                         'tool': tool_call.name,
                         'input': str(tool_call.args),
                         'output': "Pending..."
                     })
                     tools_used.add(tool_call.name)
                 elif event.content and event.content.parts and hasattr(event.content.parts[0], 'tool_result'):
                     if reasoning_steps:
                        reasoning_steps[-1]['output'] = str(event.content.parts[0].tool_result.result)

            if event.is_final_response() and event.content and event.content.parts:
                final_response['answer'] = event.content.parts[0].text
                final_response['status'] = 'success'
                break

        if include_reasoning:
            final_response['reasoning_steps'] = reasoning_steps
            final_response['tools_used'] = list(tools_used)

        return final_response

async def make_agent_caller(agent: Agent) -> AgentCaller:
    """Factory function to create an AgentCaller instance."""
    app_name = agent.name + "_app"
    user_id = agent.name + "_user"
    session_id = agent.name + "_session_01"

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
    return AgentCaller(agent, runner, user_id, session_id)


# --- FastAPI Application ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup, initialize the ADK AgentCaller
    print("Initializing ADK Agent...")
    app.state.agent_caller = await make_agent_caller(root_agent)
    print("Agent is ready.")
    yield
    # On shutdown (not essential for this example, but good practice)
    print("Shutting down.")


app = FastAPI(
    title="Clinical Assistant API (ADK Version)",
    description="An API for interacting with the Clinical AI Assistant, powered by Google ADK.",
    version="2.0.0",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    """Defines the structure of the request body for the /chat endpoint."""
    question: str
    include_reasoning: bool = False

@app.get("/")
async def get_index():
    """Serves the main index.html file."""
    template_path = APP_DIR / "templates" / "index.html"
    return FileResponse(template_path)

@app.post("/chat")
async def chat(request: QueryRequest, http_request: Request):
    """
    Receives a question, processes it through the ADK agent, and returns the response.
    """
    # Basic validation
    if not request.question or len(request.question.strip()) < 5:
        return {
            'answer': "Invalid query: Question seems too short to be meaningful",
            'status': 'validation_error'
        }

    # Access the agent_caller initialized at startup
    agent_caller = http_request.app.state.agent_caller

    # Process the query using the ADK agent
    response = await agent_caller.call(
        user_message=request.question,
        include_reasoning=request.include_reasoning
    )
    return response