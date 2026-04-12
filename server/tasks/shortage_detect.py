"""
Task: shortage-detect (MEDIUM)
================================
Agent receives current inventory levels and a list of pending supply requests.
It must identify which items are CRITICALLY SHORT — meaning:
  - quantity_available < reorder_threshold AND
  - days_until_resupply > 3 AND
  - there is a pending request for that item from a combat/high-criticality unit

Grader: Precision + Recall averaged (F1 score) against ground truth shortage list.
Score = F1(predicted_shortage_items, true_shortage_items)
"""

import random
from typing import Any, Dict, List, Set, Tuple


SCENARIOS: List[Dict[str, Any]] = [
    {
        "context": (
            "Theater logistics command has provided current inventory levels for FOB Delta. "
            "Multiple units have pending requests. Identify which items are critically short "
            "and need emergency resupply immediately."
        ),
        "inventory": [
            {"item": "5.56mm ammunition",       "quantity_available": 500,  "reorder_threshold": 2000, "days_until_resupply": 7},
            {"item": "MRE rations",              "quantity_available": 3000, "reorder_threshold": 1000, "days_until_resupply": 2},
            {"item": "Medical bandages",         "quantity_available": 80,   "reorder_threshold": 200,  "days_until_resupply": 5},
            {"item": "Diesel fuel (liters)",     "quantity_available": 5000, "reorder_threshold": 3000, "days_until_resupply": 1},
            {"item": "Night vision batteries",  "quantity_available": 20,   "reorder_threshold": 100,  "days_until_resupply": 6},
            {"item": "Satellite comms units",   "quantity_available": 3,    "reorder_threshold": 2,    "days_until_resupply": 10},
            {"item": "Morphine auto-injectors", "quantity_available": 10,   "reorder_threshold": 50,   "days_until_resupply": 8},
        ],
        "pending_requests": [
            {"request_id": "R-201", "unit": "1st Infantry",    "item": "5.56mm ammunition",       "quantity_requested": 1000, "urgency_stated": "critical", "location": "Grid 1122", "mission_criticality": "combat"},
            {"request_id": "R-202", "unit": "Support Base",    "item": "MRE rations",              "quantity_requested": 500,  "urgency_stated": "routine",  "location": "Base",      "mission_criticality": "support"},
            {"request_id": "R-203", "unit": "Medical Corps",   "item": "Medical bandages",         "quantity_requested": 300,  "urgency_stated": "high",     "location": "Aid Stn",   "mission_criticality": "combat"},
            {"request_id": "R-204", "unit": "Recon Team",      "item": "Night vision batteries",  "quantity_requested": 60,   "urgency_stated": "high",     "location": "Grid 3344", "mission_criticality": "combat"},
            {"request_id": "R-205", "unit": "Admin",           "item": "Satellite comms units",   "quantity_requested": 1,    "urgency_stated": "low",      "location": "HQ",        "mission_criticality": "support"},
            {"request_id": "R-206", "unit": "Combat Medics",   "item": "Morphine auto-injectors", "quantity_requested": 40,   "urgency_stated": "critical", "location": "Grid 1122", "mission_criticality": "combat"},
        ],
        # Critical short: below threshold + resupply >3 days + combat request pending
        "_ground_truth_shortages": [
            "5.56mm ammunition",       # 500 < 2000, 7 days, combat request
            "Medical bandages",         # 80 < 200, 5 days, combat request
            "Night vision batteries",  # 20 < 100, 6 days, combat request
            "Morphine auto-injectors", # 10 < 50, 8 days, combat request
        ],
    },
    {
        "context": (
            "FOB Eagle is supporting a rapid advance operation. Inventory has been drawn down heavily "
            "over the past 48 hours. Review current stock and pending unit requests to flag "
            "items at critical shortage levels requiring immediate escalation to theater command."
        ),
        "inventory": [
            {"item": "Tank rounds (120mm)",     "quantity_available": 15,   "reorder_threshold": 80,   "days_until_resupply": 5},
            {"item": "Engineer tape",           "quantity_available": 200,  "reorder_threshold": 50,   "days_until_resupply": 2},
            {"item": "IV saline bags",          "quantity_available": 30,   "reorder_threshold": 100,  "days_until_resupply": 9},
            {"item": "Radio batteries (AA)",    "quantity_available": 400,  "reorder_threshold": 500,  "days_until_resupply": 2},
            {"item": "Smoke grenades",          "quantity_available": 10,   "reorder_threshold": 50,   "days_until_resupply": 6},
            {"item": "Bottled water (cases)",   "quantity_available": 600,  "reorder_threshold": 200,  "days_until_resupply": 1},
            {"item": "GPS handheld units",      "quantity_available": 2,    "reorder_threshold": 10,   "days_until_resupply": 12},
        ],
        "pending_requests": [
            {"request_id": "R-301", "unit": "3rd Armor",       "item": "Tank rounds (120mm)",   "quantity_requested": 60,  "urgency_stated": "critical", "location": "Grid 7788", "mission_criticality": "combat"},
            {"request_id": "R-302", "unit": "Engineers",       "item": "Engineer tape",         "quantity_requested": 100, "urgency_stated": "routine",  "location": "Bridge",    "mission_criticality": "support"},
            {"request_id": "R-303", "unit": "Field Hospital",  "item": "IV saline bags",        "quantity_requested": 80,  "urgency_stated": "critical", "location": "Aid Stn",   "mission_criticality": "combat"},
            {"request_id": "R-304", "unit": "Signal Platoon",  "item": "Radio batteries (AA)",  "quantity_requested": 200, "urgency_stated": "high",     "location": "Comms",     "mission_criticality": "combat"},
            {"request_id": "R-305", "unit": "Infantry Co.",    "item": "Smoke grenades",        "quantity_requested": 40,  "urgency_stated": "high",     "location": "Grid 7790", "mission_criticality": "combat"},
            {"request_id": "R-306", "unit": "Recon",           "item": "GPS handheld units",    "quantity_requested": 8,   "urgency_stated": "critical", "location": "Grid 8800", "mission_criticality": "combat"},
        ],
        "_ground_truth_shortages": [
            "Tank rounds (120mm)",   # 15 < 80, 5 days, combat
            "IV saline bags",        # 30 < 100, 9 days, combat
            "Smoke grenades",        # 10 < 50, 6 days, combat
            "GPS handheld units",    # 2 < 10, 12 days, combat
        ],
    },
]


def _f1(predicted: Set[str], truth: Set[str]) -> float:
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class ShortageDetectTask:
    TASK_NAME = "shortage-detect"

    def __init__(self):
        self._scenario: Dict[str, Any] = {}
        self._ground_truth: Set[str] = set()
        self._attempts = 0

    def reset(self) -> Dict[str, Any]:
        self._scenario = random.choice(SCENARIOS)
        self._ground_truth = set(self._scenario["_ground_truth_shortages"])
        self._attempts = 0
        return self._build_observation()

    def _build_observation(self) -> Dict[str, Any]:
        return {
            "task": self.TASK_NAME,
            "context": self._scenario["context"],
            "inventory": self._scenario["inventory"],
            "pending_requests": [
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in self._scenario["pending_requests"]
            ],
        }

    def step(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict]:
        self._attempts += 1
        predicted: Set[str] = set(payload.get("shortage_items", []))

        score = round(_f1(predicted, self._ground_truth), 4)

        tp = predicted & self._ground_truth
        fp = predicted - self._ground_truth
        fn = self._ground_truth - predicted

        return (
            self._build_observation(),
            score,
            True,
            {
                "f1_score": score,
                "true_positives": list(tp),
                "false_positives": list(fp),
                "false_negatives": list(fn),
                "ground_truth": list(self._ground_truth),
            },
        )

    def state(self) -> Dict[str, Any]:
        return {
            "task": self.TASK_NAME,
            "attempts": self._attempts,
            "ground_truth_shortages": list(self._ground_truth),
        }