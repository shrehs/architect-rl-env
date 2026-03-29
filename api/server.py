from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from env.environment import ArchitectEnv
from env.models import Action, Observation
from env.tasks import TASKS


class StepRequest(BaseModel):
    action: Optional[Action] = None
    user_reply: Optional[str] = None


app = FastAPI(title="ArchitectRL OpenEnv API", version="1.0.0")
_env = ArchitectEnv(task_id="easy")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {"tasks": TASKS}


@app.api_route("/reset", methods=["GET", "POST"], response_model=Observation)
def reset(request: Request = None) -> Observation:
    global _env
    _env = ArchitectEnv(task_id="easy")
    return _env.reset()


@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    if req.action is not None:
        action = req.action
    elif req.user_reply is not None:
        action = Action(user_reply=req.user_reply)
    else:
        raise HTTPException(status_code=422, detail="Provide either action or user_reply")

    try:
        observation, reward, done, info = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "observation": observation.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    return _env.state()
