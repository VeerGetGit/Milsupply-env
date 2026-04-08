"""
milsupply-env — Military Logistics & Supply Chain OpenEnv Server
=================================================================
Implements the full OpenEnv interface:
  POST /reset   → returns initial observation
  POST /step    → takes action, returns observation + reward + done + info
  GET  /state   → returns current env state
  GET  /health  → health check
"""

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tasks.priority_classify import PriorityClassifyTask
from tasks.shortage_detect import ShortageDetectTask
from tasks.optimize_allocation import OptimizeAllocationTask

app = FastAPI(
    title="milsupply-env",
    description="Military Logistics & Supply Chain OpenEnv Environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Registry of tasks
# ---------------------------------------------------------------------------

TASKS = {
    "priority-classify": PriorityClassifyTask,
    "shortage-detect":   ShortageDetectTask,
    "optimize-allocation": OptimizeAllocationTask,
}

# Active session (single-user env — stateful server)
_active_task_name: str = "priority-classify"
_active_task: Any = PriorityClassifyTask()
_last_observation: Dict[str, Any] = {}
_episode_done: bool = False


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "priority-classify"


class StepRequest(BaseModel):
    task: str
    payload: Dict[str, Any]


class ObservationResponse(BaseModel):
    observation: Dict[str, Any]
    done: bool = False
    reward: float = 0.0
    info: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env": "milsupply-env", "version": "1.0.0"}


@app.post("/reset", response_model=ObservationResponse)
def reset(req: ResetRequest = ResetRequest()):
    global _active_task_name, _active_task, _last_observation, _episode_done

    task_name = req.task if req.task else "priority-classify"
    if task_name not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Choose from: {list(TASKS.keys())}"
        )

    _active_task_name = task_name
    _active_task = TASKS[task_name]()
    _last_observation = _active_task.reset()
    _episode_done = False

    return ObservationResponse(
        observation=_last_observation,
        done=False,
        reward=0.0,
        info={"task": task_name, "message": "Episode reset successfully"},
    )


@app.post("/step", response_model=ObservationResponse)
def step(req: StepRequest):
    global _last_observation, _episode_done

    if _episode_done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset to start a new episode."
        )

    if req.task != _active_task_name:
        raise HTTPException(
            status_code=400,
            detail=f"Active task is '{_active_task_name}', but action is for '{req.task}'. Call /reset first."
        )

    obs, reward, done, info = _active_task.step(req.payload)
    _last_observation = obs
    _episode_done = done

    return ObservationResponse(
        observation=obs,
        reward=round(reward, 4),
        done=done,
        info=info,
    )


@app.get("/state")
def state():
    return {
        "active_task": _active_task_name,
        "episode_done": _episode_done,
        "task_state": _active_task.state() if _active_task else {},
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "priority-classify",
                "difficulty": "easy",
                "description": "Classify supply requests by urgency (critical/high/routine)",
            },
            {
                "name": "shortage-detect",
                "difficulty": "medium",
                "description": "Identify critically short supply items given inventory + pending requests",
            },
            {
                "name": "optimize-allocation",
                "difficulty": "hard",
                "description": "Allocate limited supplies across units to maximize operational readiness",
            },
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
