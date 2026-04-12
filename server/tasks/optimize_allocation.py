"""
Task: optimize-allocation (HARD)
===================================
Agent must allocate limited supplies across multiple units to maximize
aggregate operational readiness.

Each unit has:
  - current_readiness_pct (0–100)
  - critical_items_needed (list of items)
  - personnel count (used to weight importance)

Scoring formula:
  For each unit:
    readiness_gain = sum of (qty_allocated / qty_needed) capped at 1.0,
                     per item the unit critically needs
    weighted by personnel / total_personnel

  Total score = weighted average readiness gain, capped [0, 1]

Penalties:
  - Over-allocating beyond available stock: score *= 0.5
  - Allocating to units items they didn't request: no bonus, just waste
"""

import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple


SCENARIOS: List[Dict[str, Any]] = [
    {
        "context": (
            "Theater command has released emergency resupply stocks. "
            "You must allocate the following available items across 4 active units "
            "to maximize overall operational readiness. "
            "Units with more personnel and combat roles should be prioritized. "
            "Do not exceed available quantities."
        ),
        "available_stock": {
            "5.56mm ammunition":   800,
            "Medical bandages":    150,
            "Radio batteries":     80,
            "MRE rations":         500,
        },
        "units": [
            {
                "unit": "Alpha Company",
                "location": "Grid 1100",
                "personnel": 120,
                "current_readiness_pct": 40.0,
                "critical_items_needed": ["5.56mm ammunition", "MRE rations"],
                "_needed_qty": {"5.56mm ammunition": 400, "MRE rations": 200},
            },
            {
                "unit": "Bravo Medical",
                "location": "Aid Station",
                "personnel": 30,
                "current_readiness_pct": 55.0,
                "critical_items_needed": ["Medical bandages", "MRE rations"],
                "_needed_qty": {"Medical bandages": 100, "MRE rations": 80},
            },
            {
                "unit": "Charlie Comms",
                "location": "Comms Hub",
                "personnel": 20,
                "current_readiness_pct": 60.0,
                "critical_items_needed": ["Radio batteries"],
                "_needed_qty": {"Radio batteries": 60},
            },
            {
                "unit": "Delta Support",
                "location": "Base Camp",
                "personnel": 50,
                "current_readiness_pct": 70.0,
                "critical_items_needed": ["MRE rations"],
                "_needed_qty": {"MRE rations": 150},
            },
        ],
        "pending_requests": [
            {"request_id": "A-001", "unit": "Alpha Company",  "item": "5.56mm ammunition", "quantity_requested": 400, "urgency_stated": "critical", "location": "Grid 1100", "mission_criticality": "combat"},
            {"request_id": "A-002", "unit": "Alpha Company",  "item": "MRE rations",        "quantity_requested": 200, "urgency_stated": "high",     "location": "Grid 1100", "mission_criticality": "combat"},
            {"request_id": "B-001", "unit": "Bravo Medical",  "item": "Medical bandages",   "quantity_requested": 100, "urgency_stated": "high",     "location": "Aid Station","mission_criticality": "combat"},
            {"request_id": "B-002", "unit": "Bravo Medical",  "item": "MRE rations",        "quantity_requested": 80,  "urgency_stated": "routine",  "location": "Aid Station","mission_criticality": "support"},
            {"request_id": "C-001", "unit": "Charlie Comms",  "item": "Radio batteries",    "quantity_requested": 60,  "urgency_stated": "high",     "location": "Comms Hub", "mission_criticality": "combat"},
            {"request_id": "D-001", "unit": "Delta Support",  "item": "MRE rations",        "quantity_requested": 150, "urgency_stated": "routine",  "location": "Base Camp", "mission_criticality": "support"},
        ],
    },
    {
        "context": (
            "An unexpected advance has stretched supply lines. Available stock is critically limited. "
            "Allocate the following items across 5 units. "
            "Prioritize combat effectiveness and minimize readiness gaps for high-personnel units. "
            "You cannot exceed available quantities for any item."
        ),
        "available_stock": {
            "Tank rounds (120mm)":  40,
            "IV saline bags":       60,
            "Smoke grenades":       25,
            "GPS units":            5,
            "Diesel fuel (liters)": 3000,
        },
        "units": [
            {
                "unit": "3rd Armor",
                "location": "Grid 7788",
                "personnel": 80,
                "current_readiness_pct": 35.0,
                "critical_items_needed": ["Tank rounds (120mm)", "Diesel fuel (liters)"],
                "_needed_qty": {"Tank rounds (120mm)": 30, "Diesel fuel (liters)": 1500},
            },
            {
                "unit": "Field Hospital",
                "location": "Aid Station",
                "personnel": 25,
                "current_readiness_pct": 50.0,
                "critical_items_needed": ["IV saline bags"],
                "_needed_qty": {"IV saline bags": 50},
            },
            {
                "unit": "Infantry Bn",
                "location": "Grid 7790",
                "personnel": 150,
                "current_readiness_pct": 55.0,
                "critical_items_needed": ["Smoke grenades", "Diesel fuel (liters)"],
                "_needed_qty": {"Smoke grenades": 20, "Diesel fuel (liters)": 1000},
            },
            {
                "unit": "Recon Plt",
                "location": "Grid 8800",
                "personnel": 15,
                "current_readiness_pct": 45.0,
                "critical_items_needed": ["GPS units", "Smoke grenades"],
                "_needed_qty": {"GPS units": 4, "Smoke grenades": 8},
            },
            {
                "unit": "Engineer Co",
                "location": "Bridge",
                "personnel": 40,
                "current_readiness_pct": 65.0,
                "critical_items_needed": ["Diesel fuel (liters)"],
                "_needed_qty": {"Diesel fuel (liters)": 500},
            },
        ],
        "pending_requests": [
            {"request_id": "T-001", "unit": "3rd Armor",     "item": "Tank rounds (120mm)",  "quantity_requested": 30,   "urgency_stated": "critical", "location": "Grid 7788", "mission_criticality": "combat"},
            {"request_id": "T-002", "unit": "3rd Armor",     "item": "Diesel fuel (liters)", "quantity_requested": 1500, "urgency_stated": "critical", "location": "Grid 7788", "mission_criticality": "combat"},
            {"request_id": "F-001", "unit": "Field Hospital","item": "IV saline bags",       "quantity_requested": 50,   "urgency_stated": "critical", "location": "Aid Station","mission_criticality": "combat"},
            {"request_id": "I-001", "unit": "Infantry Bn",   "item": "Smoke grenades",       "quantity_requested": 20,   "urgency_stated": "high",     "location": "Grid 7790", "mission_criticality": "combat"},
            {"request_id": "I-002", "unit": "Infantry Bn",   "item": "Diesel fuel (liters)", "quantity_requested": 1000, "urgency_stated": "high",     "location": "Grid 7790", "mission_criticality": "combat"},
            {"request_id": "R-001", "unit": "Recon Plt",     "item": "GPS units",            "quantity_requested": 4,    "urgency_stated": "critical", "location": "Grid 8800", "mission_criticality": "combat"},
            {"request_id": "R-002", "unit": "Recon Plt",     "item": "Smoke grenades",       "quantity_requested": 8,    "urgency_stated": "high",     "location": "Grid 8800", "mission_criticality": "combat"},
            {"request_id": "E-001", "unit": "Engineer Co",   "item": "Diesel fuel (liters)", "quantity_requested": 500,  "urgency_stated": "routine",  "location": "Bridge",    "mission_criticality": "support"},
        ],
    },
]


class OptimizeAllocationTask:
    TASK_NAME = "optimize-allocation"

    def __init__(self):
        self._scenario: Dict[str, Any] = {}
        self._attempts = 0

    def reset(self) -> Dict[str, Any]:
        self._scenario = random.choice(SCENARIOS)
        self._attempts = 0
        return self._build_observation()

    def _build_observation(self) -> Dict[str, Any]:
        units_clean = []
        for u in self._scenario["units"]:
            uc = {k: v for k, v in u.items() if not k.startswith("_")}
            units_clean.append(uc)
        return {
            "task": self.TASK_NAME,
            "context": self._scenario["context"],
            "available_stock": self._scenario["available_stock"],
            "units": units_clean,
            "pending_requests": self._scenario["pending_requests"],
        }

    def step(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict]:
        self._attempts += 1
        allocations: List[Dict[str, Any]] = payload.get("allocations", [])

        # Build allocation map: unit -> item -> qty
        alloc_map: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for entry in allocations:
            unit = entry.get("unit", "")
            item = entry.get("item", "")
            qty = int(entry.get("quantity_allocated", 0))
            if unit and item and qty > 0:
                alloc_map[unit][item] += qty

        # Check over-allocation
        used: Dict[str, int] = defaultdict(int)
        for unit_allocs in alloc_map.values():
            for item, qty in unit_allocs.items():
                used[item] += qty

        available = self._scenario["available_stock"]
        over_allocated = any(used[item] > available.get(item, 0) for item in used)

        # Compute weighted readiness gain
        units = self._scenario["units"]
        total_personnel = sum(u["personnel"] for u in units)
        weighted_score = 0.0

        unit_details = {}
        for u in units:
            unit_name = u["unit"]
            needed: Dict[str, int] = u.get("_needed_qty", {})
            if not needed:
                continue

            item_scores = []
            for item, qty_needed in needed.items():
                qty_given = alloc_map[unit_name].get(item, 0)
                item_score = min(qty_given / qty_needed, 1.0) if qty_needed > 0 else 0.0
                item_scores.append(item_score)

            unit_gain = sum(item_scores) / len(item_scores) if item_scores else 0.0
            weight = u["personnel"] / total_personnel
            weighted_score += unit_gain * weight
            unit_details[unit_name] = {
                "gain": round(unit_gain, 3),
                "weight": round(weight, 3),
                "allocated": dict(alloc_map[unit_name]),
            }

        score = round(min(max(weighted_score, 0.0), 1.0), 4)

        if over_allocated:
            score = round(score * 0.5, 4)

        return (
            self._build_observation(),
            score,
            True,
            {
                "weighted_readiness_score": score,
                "over_allocated": over_allocated,
                "stock_used": dict(used),
                "stock_available": available,
                "unit_details": unit_details,
            },
        )

    def state(self) -> Dict[str, Any]:
        return {
            "task": self.TASK_NAME,
            "attempts": self._attempts,
            "available_stock": self._scenario.get("available_stock", {}),
        }