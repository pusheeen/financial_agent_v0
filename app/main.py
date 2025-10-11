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
import os
import stripe
import databases
import sqlalchemy
from passlib.context import CryptContext
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from fastapi import Form, Depends, HTTPException

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)
# --- Import your new ADK root agent and utilities ---
from dotenv import load_dotenv  # <-- ADD THIS
from app.agents.agents import root_agent, search_latest_news
from app.scoring import compute_company_scores, ScoreComputationError
load_dotenv()

# --- Configuration ---
APP_DIR = Path(__file__).resolve().parent
# Stripe configuration
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
stripe.api_key = STRIPE_SECRET_KEY or None
DOMAIN = os.getenv("DOMAIN", "http://localhost:8000")
PRICE_ID = os.getenv("STRIPE_PRICE_ID")
# Flag whether Stripe is configured
HAS_STRIPE = bool(STRIPE_SECRET_KEY and PRICE_ID)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./users.db")
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

users = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("username", sqlalchemy.String, unique=True, index=True),
    sqlalchemy.Column("email", sqlalchemy.String, unique=True, index=True),
    sqlalchemy.Column("hashed_password", sqlalchemy.String),
    sqlalchemy.Column("stripe_customer_id", sqlalchemy.String, nullable=True),
    sqlalchemy.Column("stripe_subscription_id", sqlalchemy.String, nullable=True),
    sqlalchemy.Column("subscription_status", sqlalchemy.String, nullable=True),
)

engine = sqlalchemy.create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
metadata.create_all(engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

async def get_user_by_email(email: str):
    query = users.select().where(users.c.email == email)
    return await database.fetch_one(query)

async def get_user_by_id(user_id: int):
    query = users.select().where(users.c.id == user_id)
    return await database.fetch_one(query)

async def authenticate_user(email: str, password: str):
    user = await get_user_by_email(email)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

async def get_current_user(request: Request):
    user_id = request.session.get("user")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

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

# Add session middleware and template rendering
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY", "changeme"))
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

class QueryRequest(BaseModel):
    """Defines the structure of the request body for the /chat endpoint."""
    question: str
    include_reasoning: bool = False

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serves the main index.html file."""
    user = None
    user_id = request.session.get("user")
    if user_id:
        user = await get_user_by_id(user_id)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


@app.get("/api/scores/{ticker}")
async def get_scores(ticker: str, news_only: bool = False, query: Optional[str] = None):
    """Return the latest computed scorecard for a company or fetch news snippets for a query."""
    loop = asyncio.get_event_loop()
    if news_only:
        try:
            search_query = query or ticker
            data = await loop.run_in_executor(None, search_latest_news, search_query)
            results = data.get("results", []) if isinstance(data, dict) else []
            return {"status": "success", "data": {"latest_news": results}}
        except Exception as exc:  # pragma: no cover - network failures
            return {"status": "error", "message": f"Failed to fetch news: {exc}"}
    try:
        data = await loop.run_in_executor(None, compute_company_scores, ticker.upper())
        return {"status": "success", "data": data}
    except ScoreComputationError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:  # pragma: no cover - unexpected errors bubbled to client
        return {"status": "error", "message": f"Unexpected error: {exc}"}

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
  
# User authentication and subscription routes

# Register
@app.get("/register", response_class=HTMLResponse)
async def register_get(request: Request):
    # Simple registration form
    html_content = """
<html><body>
<h2>Register</h2>
<form method='post'>
  Username: <input name='username'/><br/>
  Email: <input type='email' name='email'/><br/>
  Password: <input type='password' name='password'/><br/>
  <button type='submit'>Register</button>
</form>
</body></html>
"""
    return HTMLResponse(html_content)

@app.post("/register")
async def register_post(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    existing_user = await get_user_by_email(email)
    if existing_user:
        return HTMLResponse("Email already registered", status_code=400)
    hashed_password = get_password_hash(password)
    await database.execute(users.insert().values(username=username, email=email, hashed_password=hashed_password))
    return RedirectResponse(url="/login", status_code=302)

# Login
@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    # Simple login form
    html_content = """
<html><body>
<h2>Login</h2>
<form method='post'>
  Email: <input type='email' name='email'/><br/>
  Password: <input type='password' name='password'/><br/>
  <button type='submit'>Login</button>
</form>
</body></html>
"""
    return HTMLResponse(html_content)

@app.post("/login")
async def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    user = await authenticate_user(email, password)
    if not user:
        return HTMLResponse("Invalid credentials", status_code=401)
    request.session["user"] = user["id"]
    return RedirectResponse(url="/account", status_code=302)

# Logout
@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/", status_code=302)

# Account page
@app.get("/account", response_class=HTMLResponse)
async def account_page(request: Request, current_user=Depends(get_current_user)):
    # Display account and subscription status
    subscribed = current_user.get("subscription_status") == "active"
    html = "<html><body>"
    html += f"<h2>Account for {current_user.get('username')}</h2>"
    html += f"<p>Email: {current_user.get('email')}</p>"
    html += "<h3>Subscription</h3>"
    if subscribed:
        html += "<p>Subscribed: $10/month (Active)</p>"
    else:
        if HAS_STRIPE:
            html += "<p>Monthly subscription: $10</p>"
            html += "<form method='post' action='/create-checkout-session'><button type='submit'>Subscribe</button></form>"
        else:
            html += "<p>Subscription feature is not yet configured.</p>"
    html += "<p><a href='/logout'>Logout</a> | <a href='/'>Home</a></p>"
    html += "</body></html>"
    return HTMLResponse(html)

# Create Stripe checkout session
@app.post("/create-checkout-session")
async def create_checkout_session(request: Request, current_user=Depends(get_current_user)):
    # Ensure Stripe integration is available
    if not HAS_STRIPE:
        raise HTTPException(status_code=503, detail="Stripe subscription is not configured")
    user = await get_user_by_id(current_user["id"])
    # Create Stripe customer if not exists
    if not user["stripe_customer_id"]:
        customer = stripe.Customer.create(email=user["email"])
        update_query = users.update().where(users.c.id == user["id"]).values(stripe_customer_id=customer["id"])
        await database.execute(update_query)
        stripe_customer_id = customer["id"]
    else:
        stripe_customer_id = user["stripe_customer_id"]
    try:
        session = stripe.checkout.Session.create(
            customer=stripe_customer_id,
            payment_method_types=["card"],
            line_items=[{"price": PRICE_ID, "quantity": 1}],
            mode="subscription",
            success_url=DOMAIN + "/subscription_success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=DOMAIN + "/account",
        )
        return RedirectResponse(session.url, status_code=303)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Subscription success
@app.get("/subscription_success", response_class=HTMLResponse)
async def subscription_success(request: Request, session_id: str, current_user=Depends(get_current_user)):
    # Ensure Stripe integration is available
    if not HAS_STRIPE:
        raise HTTPException(status_code=503, detail="Stripe subscription is not configured")
    stripe_session = stripe.checkout.Session.retrieve(session_id)
    subscription_id = stripe_session.get("subscription")
    # Update subscription status
    await database.execute(users.update().where(users.c.id == current_user["id"]).values(
        stripe_subscription_id=subscription_id,
        subscription_status="active"
    ))
    html = "<html><body>"
    html += f"<h2>Subscription Successful!</h2><p>Thank you, {current_user.get('username')}.</p>"
    html += "<p>Your subscription is active. $10/month.</p>"
    html += "<p><a href='/account'>Go to Account</a> | <a href='/'>Home</a></p>"
    html += "</body></html>"
    return HTMLResponse(html)
