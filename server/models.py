"""
Typed Pydantic models for milsupply-env (Military Logistics & Supply Chain).
Defines Action, Observation, and Reward models per OpenEnv spec.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class SupplyRequest(BaseModel):
    request_id: str
    unit: str
    item: str
    quantity_requested: int
    urgency_stated: str  # stated by the requesting unit (may be wrong)
    location: str
    mission_criticality: str  # "combat", "support", "training"


class InventoryItem(BaseModel):
    item: str
    quantity_available: int
    reorder_threshold: int
    days_until_resupply: int


class UnitStatus(BaseModel):
    unit: str
    location: str
    personnel: int
    current_readiness_pct: float  # 0–100
    critical_items_needed: List[str]


# ---------------------------------------------------------------------------
# Action models (what the agent sends)
# ---------------------------------------------------------------------------

class PriorityClassifyAction(BaseModel):
    """Task: priority-classify — agent assigns urgency to each request."""
    classifications: Dict[str, str] = Field(
        ...,
        description="Map of request_id -> priority: 'critical' | 'high' | 'routine'"
    )


class ShortageDetectAction(BaseModel):
    """Task: shortage-detect — agent lists items it believes are critically short."""
    shortage_items: List[str] = Field(
        ...,
        description="List of item names the agent identifies as critically short"
    )
    reasoning: Optional[str] = Field(
        None,
        description="Optional brief explanation of shortage reasoning"
    )


class OptimizeAllocationAction(BaseModel):
    """Task: optimize-allocation — agent allocates quantities across units."""
    allocations: List[Dict[str, Any]] = Field(
        ...,
        description="List of {unit, item, quantity_allocated} dicts"
    )
    justification: Optional[str] = Field(
        None,
        description="Optional justification for allocation decisions"
    )


# Union action type used by the API
class AgentAction(BaseModel):
    task: str = Field(..., description="Task name: priority-classify | shortage-detect | optimize-allocation")
    payload: Dict[str, Any] = Field(..., description="Task-specific action payload")


# ---------------------------------------------------------------------------
# Observation models (what the env returns)
# ---------------------------------------------------------------------------

class PriorityClassifyObservation(BaseModel):
    task: str = "priority-classify"
    supply_requests: List[SupplyRequest]
    context: str  # situational briefing


class ShortageDetectObservation(BaseModel):
    task: str = "shortage-detect"
    inventory: List[InventoryItem]
    pending_requests: List[SupplyRequest]
    context: str


class OptimizeAllocationObservation(BaseModel):
    task: str = "optimize-allocation"
    available_stock: Dict[str, int]   # item -> qty available
    units: List[UnitStatus]
    pending_requests: List[SupplyRequest]
    context: str


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Dict[str, Any]
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = {}
