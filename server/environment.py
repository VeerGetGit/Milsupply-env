"""
milsupply-env — Military Logistics & Supply Chain
===================================================
Uses OpenEnv Environment base class with graders for all 3 tasks.
"""

import random
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from openenv_core.env_server import Environment
from models import MilSupplyAction, MilSupplyObservation, MilSupplyState


# ===========================================================================
# SCENARIO DATA
# ===========================================================================

PRIORITY_SCENARIOS = [
    {
        "context": "FOB Alpha is conducting active combat operations. Multiple units have submitted supply requests simultaneously.",
        "requests": [
            {"request_id": "REQ-001", "unit": "1st Infantry", "item": "Blood plasma", "quantity_requested": 20, "urgency_stated": "urgent", "location": "Grid 4421", "mission_criticality": "combat", "_ground_truth": "critical"},
            {"request_id": "REQ-002", "unit": "Support Battalion", "item": "Printer paper", "quantity_requested": 500, "urgency_stated": "urgent", "location": "Base Camp", "mission_criticality": "support", "_ground_truth": "routine"},
            {"request_id": "REQ-003", "unit": "2nd Armored", "item": "Tank ammunition (120mm)", "quantity_requested": 40, "urgency_stated": "routine", "location": "Grid 4419", "mission_criticality": "combat", "_ground_truth": "critical"},
            {"request_id": "REQ-004", "unit": "Medical Corps", "item": "Surgical gloves", "quantity_requested": 200, "urgency_stated": "high", "location": "Field Hospital", "mission_criticality": "support", "_ground_truth": "high"},
            {"request_id": "REQ-005", "unit": "Signal Corps", "item": "Radio batteries", "quantity_requested": 50, "urgency_stated": "high", "location": "Comms Hub", "mission_criticality": "combat", "_ground_truth": "high"},
        ],
    },
    {
        "context": "FOB Bravo has received 6 simultaneous supply requests during a training exercise that escalated into a real emergency.",
        "requests": [
            {"request_id": "REQ-101", "unit": "Recon Platoon", "item": "Night vision goggles", "quantity_requested": 8, "urgency_stated": "routine", "location": "Forward OP", "mission_criticality": "combat", "_ground_truth": "critical"},
            {"request_id": "REQ-102", "unit": "HQ Staff", "item": "Coffee and rations", "quantity_requested": 100, "urgency_stated": "urgent", "location": "HQ Tent", "mission_criticality": "support", "_ground_truth": "routine"},
            {"request_id": "REQ-103", "unit": "Combat Engineers", "item": "C4 explosive charges", "quantity_requested": 15, "urgency_stated": "high", "location": "Grid 5530", "mission_criticality": "combat", "_ground_truth": "critical"},
            {"request_id": "REQ-104", "unit": "Logistics", "item": "Fuel (diesel)", "quantity_requested": 2000, "urgency_stated": "high", "location": "Motor Pool", "mission_criticality": "support", "_ground_truth": "high"},
            {"request_id": "REQ-105", "unit": "Medical", "item": "Morphine auto-injectors", "quantity_requested": 30, "urgency_stated": "urgent", "location": "Aid Station", "mission_criticality": "combat", "_ground_truth": "critical"},
            {"request_id": "REQ-106", "unit": "Admin", "item": "Laptop chargers", "quantity_requested": 5, "urgency_stated": "high", "location": "Admin Tent", "mission_criticality": "support", "_ground_truth": "routine"},
        ],
    },
]

SHORTAGE_SCENARIOS = [
    {
        "context": "Theater logistics command has provided current inventory levels for FOB Delta. Identify critically short items.",
        "inventory": [
            {"item": "5.56mm ammunition", "quantity_available": 500, "reorder_threshold": 2000, "days_until_resupply": 7},
            {"item": "MRE rations", "quantity_available": 3000, "reorder_threshold": 1000, "days_until_resupply": 2},
            {"item": "Medical bandages", "quantity_available": 80, "reorder_threshold": 200, "days_until_resupply": 5},
            {"item": "Diesel fuel (liters)", "quantity_available": 5000, "reorder_threshold": 3000, "days_until_resupply": 1},
            {"item": "Night vision batteries", "quantity_available": 20, "reorder_threshold": 100, "days_until_resupply": 6},
            {"item": "Morphine auto-injectors", "quantity_available": 10, "reorder_threshold": 50, "days_until_resupply": 8},
        ],
        "pending_requests": [
            {"request_id": "R-201", "unit": "1st Infantry", "item": "5.56mm ammunition", "quantity_requested": 1000, "urgency_stated": "critical", "location": "Grid 1122", "mission_criticality": "combat"},
            {"request_id": "R-202", "unit": "Support Base", "item": "MRE rations", "quantity_requested": 500, "urgency_stated": "routine", "location": "Base", "mission_criticality": "support"},
            {"request_id": "R-203", "unit": "Medical Corps", "item": "Medical bandages", "quantity_requested": 300, "urgency_stated": "high", "location": "Aid Stn", "mission_criticality": "combat"},
            {"request_id": "R-204", "unit": "Recon Team", "item": "Night vision batteries", "quantity_requested": 60, "urgency_stated": "high", "location": "Grid 3344", "mission_criticality": "combat"},
            {"request_id": "R-206", "unit": "Combat Medics", "item": "Morphine auto-injectors", "quantity_requested": 40, "urgency_stated": "critical", "location": "Grid 1122", "mission_criticality": "combat"},
        ],
        "_ground_truth_shortages": ["5.56mm ammunition", "Medical bandages", "Night vision batteries", "Morphine auto-injectors"],
    },
    {
        "context": "FOB Eagle is supporting a rapid advance. Inventory drawn down heavily. Flag items at critical shortage.",
        "inventory": [
            {"item": "Tank rounds (120mm)", "quantity_available": 15, "reorder_threshold": 80, "days_until_resupply": 5},
            {"item": "Engineer tape", "quantity_available": 200, "reorder_threshold": 50, "days_until_resupply": 2},
            {"item": "IV saline bags", "quantity_available": 30, "reorder_threshold": 100, "days_until_resupply": 9},
            {"item": "Smoke grenades", "quantity_available": 10, "reorder_threshold": 50, "days_until_resupply": 6},
            {"item": "GPS handheld units", "quantity_available": 2, "reorder_threshold": 10, "days_until_resupply": 12},
        ],
        "pending_requests": [
            {"request_id": "R-301", "unit": "3rd Armor", "item": "Tank rounds (120mm)", "quantity_requested": 60, "urgency_stated": "critical", "location": "Grid 7788", "mission_criticality": "combat"},
            {"request_id": "R-303", "unit": "Field Hospital", "item": "IV saline bags", "quantity_requested": 80, "urgency_stated": "critical", "location": "Aid Stn", "mission_criticality": "combat"},
            {"request_id": "R-305", "unit": "Infantry Co.", "item": "Smoke grenades", "quantity_requested": 40, "urgency_stated": "high", "location": "Grid 7790", "mission_criticality": "combat"},
            {"request_id": "R-306", "unit": "Recon", "item": "GPS handheld units", "quantity_requested": 8, "urgency_stated": "critical", "location": "Grid 8800", "mission_criticality": "combat"},
        ],
        "_ground_truth_shortages": ["Tank rounds (120mm)", "IV saline bags", "Smoke grenades", "GPS handheld units"],
    },
]

ALLOCATION_SCENARIOS = [
    {
        "context": "Theater command has released emergency resupply stocks. Allocate items across 4 active units to maximize readiness.",
        "available_stock": {"5.56mm ammunition": 800, "Medical bandages": 150, "Radio batteries": 80, "MRE rations": 500},
        "units": [
            {"unit": "Alpha Company", "location": "Grid 1100", "personnel": 120, "current_readiness_pct": 40.0, "critical_items_needed": ["5.56mm ammunition", "MRE rations"], "_needed_qty": {"5.56mm ammunition": 400, "MRE rations": 200}},
            {"unit": "Bravo Medical", "location": "Aid Station", "personnel": 30, "current_readiness_pct": 55.0, "critical_items_needed": ["Medical bandages", "MRE rations"], "_needed_qty": {"Medical bandages": 100, "MRE rations": 80}},
            {"unit": "Charlie Comms", "location": "Comms Hub", "personnel": 20, "current_readiness_pct": 60.0, "critical_items_needed": ["Radio batteries"], "_needed_qty": {"Radio batteries": 60}},
            {"unit": "Delta Support", "location": "Base Camp", "personnel": 50, "current_readiness_pct": 70.0, "critical_items_needed": ["MRE rations"], "_needed_qty": {"MRE rations": 150}},
        ],
        "pending_requests": [
            {"request_id": "A-001", "unit": "Alpha Company", "item": "5.56mm ammunition", "quantity_requested": 400, "urgency_stated": "critical", "location": "Grid 1100", "mission_criticality": "combat"},
            {"request_id": "B-001", "unit": "Bravo Medical", "item": "Medical bandages", "quantity_requested": 100, "urgency_stated": "high", "location": "Aid Station", "mission_criticality": "combat"},
            {"request_id": "C-001", "unit": "Charlie Comms", "item": "Radio batteries", "quantity_requested": 60, "urgency_stated": "high", "location": "Comms Hub", "mission_criticality": "combat"},
        ],
    },
]


# ===========================================================================
# GRADER FUNCTIONS (replaces Rubric classes)
# ===========================================================================

def grade_priority_classify(action: MilSupplyAction, observation: MilSupplyObservation) -> float:
    ground_truth: Dict[str, str] = observation.info.get("_ground_truth", {})
    classifications: Dict[str, str] = action.classifications or {}
    total = len(ground_truth)
    if total == 0:
        return 0.0
    correct = 0
    penalty = 0.0
    for req_id, truth in ground_truth.items():
        predicted = classifications.get(req_id, "").lower().strip()
        if predicted == truth:
            correct += 1
        elif truth == "critical" and predicted == "routine":
            penalty += 0.2
    score = max(0.0, (correct / total) - penalty)
    return round(min(score, 1.0), 4)


def grade_shortage_detect(action: MilSupplyAction, observation: MilSupplyObservation) -> float:
    truth: Set[str] = set(observation.info.get("_ground_truth_shortages", []))
    predicted: Set[str] = set(action.shortage_items or [])
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def grade_optimize_allocation(action: MilSupplyAction, observation: MilSupplyObservation) -> float:
    allocations: List[Dict[str, Any]] = action.allocations or []
    available: Dict[str, int] = observation.available_stock or {}
    units_data: List[Dict[str, Any]] = observation.info.get("_units_with_needed", [])

    alloc_map: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for entry in allocations:
        unit = entry.get("unit", "")
        item = entry.get("item", "")
        qty = int(entry.get("quantity_allocated", 0))
        if unit and item and qty > 0:
            alloc_map[unit][item] += qty

    used: Dict[str, int] = defaultdict(int)
    for unit_allocs in alloc_map.values():
        for item, qty in unit_allocs.items():
            used[item] += qty
    over_allocated = any(used[item] > available.get(item, 0) for item in used)

    total_personnel = sum(u.get("personnel", 0) for u in units_data)
    if total_personnel == 0:
        return 0.0

    weighted_score = 0.0
    for u in units_data:
        unit_name = u["unit"]
        needed: Dict[str, int] = u.get("_needed_qty", {})
        if not needed:
            continue
        item_scores = []
        for item, qty_needed in needed.items():
            qty_given = alloc_map[unit_name].get(item, 0)
            item_scores.append(min(qty_given / qty_needed, 1.0) if qty_needed > 0 else 0.0)
        unit_gain = sum(item_scores) / len(item_scores) if item_scores else 0.0
        weight = u["personnel"] / total_personnel
        weighted_score += unit_gain * weight

    score = round(min(max(weighted_score, 0.0), 1.0), 4)
    if over_allocated:
        score = round(score * 0.5, 4)
    return score


GRADERS = {
    "priority-classify": grade_priority_classify,
    "shortage-detect": grade_shortage_detect,
    "optimize-allocation": grade_optimize_allocation,
}


# ===========================================================================
# MAIN ENVIRONMENT CLASS
# ===========================================================================

class MilSupplyEnvironment(Environment):
    """
    Military Logistics & Supply Chain Environment.
    Implements 3 tasks with graders: priority-classify, shortage-detect, optimize-allocation.
    """

    def __init__(self):
        super().__init__()
        self._state = MilSupplyState()
        self._current_observation: MilSupplyObservation = None

    def reset(self, task: str = "priority-classify", seed: int = None) -> MilSupplyObservation:
        if seed is not None:
            random.seed(seed)

        self._state = MilSupplyState(active_task=task)

        if task == "priority-classify":
            obs = self._reset_priority_classify()
        elif task == "shortage-detect":
            obs = self._reset_shortage_detect()
        elif task == "optimize-allocation":
            obs = self._reset_optimize_allocation()
        else:
            obs = self._reset_priority_classify()

        self._current_observation = obs
        return obs

    def step(self, action: MilSupplyAction) -> MilSupplyObservation:
        task = action.task or self._state.active_task
        grader = GRADERS.get(task, grade_priority_classify)
        reward = grader(action, self._current_observation)

        self._state.step_count += 1
        self._state.episode_done = True
        self._state.last_reward = reward

        obs = MilSupplyObservation(
            task=task,
            context=self._current_observation.context,
            reward=reward,
            done=True,
            info={**self._current_observation.info, "reward": reward},
            supply_requests=self._current_observation.supply_requests,
            inventory=self._current_observation.inventory,
            pending_requests=self._current_observation.pending_requests,
            available_stock=self._current_observation.available_stock,
            units=self._current_observation.units,
        )
        self._current_observation = obs
        return obs

    @property
    def state(self) -> MilSupplyState:
        return self._state

    def _reset_priority_classify(self) -> MilSupplyObservation:
        scenario = random.choice(PRIORITY_SCENARIOS)
        ground_truth = {r["request_id"]: r["_ground_truth"] for r in scenario["requests"]}
        clean_requests = [{k: v for k, v in r.items() if not k.startswith("_")} for r in scenario["requests"]]
        return MilSupplyObservation(
            task="priority-classify",
            context=scenario["context"],
            supply_requests=clean_requests,
            info={"_ground_truth": ground_truth},
        )

    def _reset_shortage_detect(self) -> MilSupplyObservation:
        scenario = random.choice(SHORTAGE_SCENARIOS)
        return MilSupplyObservation(
            task="shortage-detect",
            context=scenario["context"],
            inventory=scenario["inventory"],
            pending_requests=scenario["pending_requests"],
            info={"_ground_truth_shortages": scenario["_ground_truth_shortages"]},
        )

    def _reset_optimize_allocation(self) -> MilSupplyObservation:
        scenario = random.choice(ALLOCATION_SCENARIOS)
        clean_units = [{k: v for k, v in u.items() if not k.startswith("_")} for u in scenario["units"]]
        return MilSupplyObservation(
            task="optimize-allocation",
            context=scenario["context"],
            available_stock=scenario["available_stock"],
            units=clean_units,
            pending_requests=scenario["pending_requests"],
            info={"_units_with_needed": scenario["units"]},
        )