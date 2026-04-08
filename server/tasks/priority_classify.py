"""
Task: priority-classify (EASY)
================================
Agent receives a list of supply requests and must classify each as:
  - critical   (life-safety / mission-critical, needed within hours)
  - high       (operationally important, needed within 24h)
  - routine    (can wait 72h+)

Grader: F1-style partial credit per request.
Score = (correct classifications) / (total requests)
Partial penalty for critical misclassified as routine (-0.2 per such error).
"""

import random
from typing import Any, Dict, List, Tuple

from models import SupplyRequest


# ---------------------------------------------------------------------------
# Scenario bank
# ---------------------------------------------------------------------------

SCENARIOS: List[Dict[str, Any]] = [
    {
        "context": (
            "FOB Alpha is conducting active combat operations. "
            "Multiple units have submitted supply requests simultaneously. "
            "You must triage these requests by urgency."
        ),
        "requests": [
            {
                "request_id": "REQ-001",
                "unit": "1st Infantry",
                "item": "Blood plasma",
                "quantity_requested": 20,
                "urgency_stated": "urgent",
                "location": "Grid 4421",
                "mission_criticality": "combat",
                "_ground_truth": "critical",
            },
            {
                "request_id": "REQ-002",
                "unit": "Support Battalion",
                "item": "Printer paper",
                "quantity_requested": 500,
                "urgency_stated": "urgent",  # unit over-stated urgency
                "location": "Base Camp",
                "mission_criticality": "support",
                "_ground_truth": "routine",
            },
            {
                "request_id": "REQ-003",
                "unit": "2nd Armored",
                "item": "Tank ammunition (120mm)",
                "quantity_requested": 40,
                "urgency_stated": "routine",  # unit under-stated urgency
                "location": "Grid 4419",
                "mission_criticality": "combat",
                "_ground_truth": "critical",
            },
            {
                "request_id": "REQ-004",
                "unit": "Medical Corps",
                "item": "Surgical gloves",
                "quantity_requested": 200,
                "urgency_stated": "high",
                "location": "Field Hospital",
                "mission_criticality": "support",
                "_ground_truth": "high",
            },
            {
                "request_id": "REQ-005",
                "unit": "Signal Corps",
                "item": "Radio batteries",
                "quantity_requested": 50,
                "urgency_stated": "high",
                "location": "Comms Hub",
                "mission_criticality": "combat",
                "_ground_truth": "high",
            },
        ],
    },
    {
        "context": (
            "FOB Bravo has received 6 simultaneous supply requests during a training exercise "
            "that has unexpectedly escalated into a real emergency situation. "
            "Classify each request's true urgency regardless of what the unit stated."
        ),
        "requests": [
            {
                "request_id": "REQ-101",
                "unit": "Recon Platoon",
                "item": "Night vision goggles",
                "quantity_requested": 8,
                "urgency_stated": "routine",
                "location": "Forward OP",
                "mission_criticality": "combat",
                "_ground_truth": "critical",
            },
            {
                "request_id": "REQ-102",
                "unit": "HQ Staff",
                "item": "Coffee and rations",
                "quantity_requested": 100,
                "urgency_stated": "urgent",
                "location": "HQ Tent",
                "mission_criticality": "support",
                "_ground_truth": "routine",
            },
            {
                "request_id": "REQ-103",
                "unit": "Combat Engineers",
                "item": "C4 explosive charges",
                "quantity_requested": 15,
                "urgency_stated": "high",
                "location": "Grid 5530",
                "mission_criticality": "combat",
                "_ground_truth": "critical",
            },
            {
                "request_id": "REQ-104",
                "unit": "Logistics",
                "item": "Fuel (diesel)",
                "quantity_requested": 2000,
                "urgency_stated": "high",
                "location": "Motor Pool",
                "mission_criticality": "support",
                "_ground_truth": "high",
            },
            {
                "request_id": "REQ-105",
                "unit": "Medical",
                "item": "Morphine auto-injectors",
                "quantity_requested": 30,
                "urgency_stated": "urgent",
                "location": "Aid Station",
                "mission_criticality": "combat",
                "_ground_truth": "critical",
            },
            {
                "request_id": "REQ-106",
                "unit": "Admin",
                "item": "Laptop chargers",
                "quantity_requested": 5,
                "urgency_stated": "high",
                "location": "Admin Tent",
                "mission_criticality": "support",
                "_ground_truth": "routine",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Task state
# ---------------------------------------------------------------------------

class PriorityClassifyTask:
    TASK_NAME = "priority-classify"

    def __init__(self):
        self._scenario: Dict[str, Any] = {}
        self._ground_truth: Dict[str, str] = {}
        self._attempts = 0

    def reset(self) -> Dict[str, Any]:
        self._scenario = random.choice(SCENARIOS)
        self._ground_truth = {
            r["request_id"]: r["_ground_truth"]
            for r in self._scenario["requests"]
        }
        self._attempts = 0
        return self._build_observation()

    def _build_observation(self) -> Dict[str, Any]:
        requests = []
        for r in self._scenario["requests"]:
            req = {k: v for k, v in r.items() if not k.startswith("_")}
            requests.append(req)
        return {
            "task": self.TASK_NAME,
            "supply_requests": requests,
            "context": self._scenario["context"],
        }

    def step(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict]:
        self._attempts += 1
        classifications: Dict[str, str] = payload.get("classifications", {})

        total = len(self._ground_truth)
        correct = 0
        critical_miss_penalty = 0.0

        details = {}
        for req_id, truth in self._ground_truth.items():
            predicted = classifications.get(req_id, "").lower().strip()
            if predicted == truth:
                correct += 1
                details[req_id] = {"result": "correct", "truth": truth, "predicted": predicted}
            else:
                # Extra penalty for critical misclassified as routine (dangerous error)
                if truth == "critical" and predicted == "routine":
                    critical_miss_penalty += 0.2
                details[req_id] = {"result": "wrong", "truth": truth, "predicted": predicted}

        base_score = correct / total if total > 0 else 0.0
        score = max(0.0, base_score - critical_miss_penalty)
        score = round(min(score, 1.0), 4)

        return (
            self._build_observation(),
            score,
            True,  # single-step task
            {
                "correct": correct,
                "total": total,
                "critical_miss_penalty": critical_miss_penalty,
                "details": details,
            },
        )

    def state(self) -> Dict[str, Any]:
        return {
            "task": self.TASK_NAME,
            "attempts": self._attempts,
            "scenario_context": self._scenario.get("context", ""),
            "ground_truth": self._ground_truth,
        }
