"""
Typed Pydantic models for milsupply-env.
Defines Action, Observation, and State models per OpenEnv spec.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action models (what the agent sends)
# ---------------------------------------------------------------------------

class MilSupplyAction(BaseModel):
    """Unified action model for all three tasks."""
    task: str = Field(..., description="Task name: priority-classify | shortage-detect | optimize-allocation")
    # priority-classify
    classifications: Optional[Dict[str, str]] = Field(
        None, description="Map of request_id -> priority: 'critical' | 'high' | 'routine'"
    )
    # shortage-detect
    shortage_items: Optional[List[str]] = Field(
        None, description="List of item names identified as critically short"
    )
    # optimize-allocation
    allocations: Optional[List[Dict[str, Any]]] = Field(
        None, description="List of {unit, item, quantity_allocated} dicts"
    )


# ---------------------------------------------------------------------------
# Observation model (what the env returns)
# ---------------------------------------------------------------------------

class MilSupplyObservation(BaseModel):
    """Unified observation model for all three tasks."""
    task: str
    context: str
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}
    # task-specific fields
    supply_requests: Optional[List[Dict[str, Any]]] = None
    inventory: Optional[List[Dict[str, Any]]] = None
    pending_requests: Optional[List[Dict[str, Any]]] = None
    available_stock: Optional[Dict[str, int]] = None
    units: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------

class MilSupplyState(BaseModel):
    """Environment state."""
    active_task: str = "priority-classify"
    episode_done: bool = False
    step_count: int = 0
    last_reward: float = 0.0