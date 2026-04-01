from typing import Dict, List

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(min_length=1)
    content: str = Field(min_length=1)


class Observation(BaseModel):
    last_assistant_message: str
    constraints_collected: Dict[str, str]
    missing_constraints: List[str]
    mode: str
    step_count: int = Field(ge=0)


class Action(BaseModel):
    type: str = Field(min_length=1)


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    reason: str
