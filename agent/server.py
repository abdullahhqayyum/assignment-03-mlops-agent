"""Local conversational API. Bind to loopback; confirmations are separate human requests."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from agent.config import Settings
from agent.orchestrator import Agent

app = FastAPI(title="MLOps Agent", version="1.0.0")
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = Agent(Settings.from_env())
    return _agent


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=12000)


class ConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    confirmation_id: str = Field(min_length=32, max_length=32, pattern="^[a-f0-9]+$")
    approved: bool


@app.post("/chat")
def chat(request: ChatRequest):
    return get_agent().run(request.message)


@app.post("/confirm")
def confirm(request: ConfirmationRequest):
    try:
        return get_agent().registry.confirm(request.confirmation_id, request.approved)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
