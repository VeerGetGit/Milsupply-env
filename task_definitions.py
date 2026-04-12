"""
task_definitions.py — milsupply-env
=====================================
Graders as OpenEnv Rubric subclasses.
Scores clamped strictly between 0.001 and 0.999.
"""

from typing import Any, Dict, List
from openenv.core.rubrics import Rubric


def _clamp(score: float) -> float:
    return round(min(max(score, 0.001), 0.999), 4)


class PriorityClassifyRubric(Rubric):
    """Grader for priority-classify task."""

    def forward(self, action: Any, observation: Any) -> float:
        if isinstance(action, dict):
            action_dict = action
        else:
            action_dict = action.dict() if hasattr(action, "dict") else {}

        if isinstance(observation, dict):
            ground_truth = observation.get("info", {}).get("_ground_truth", {})
        else:
            ground_truth = observation.info.get("_ground_truth", {}) if hasattr(observation, "info") else {}

        classifications = action_dict.get("classifications", {})
        if not ground_truth:
            return 0.001

        total = len(ground_truth)
        correct = 0
        penalty = 0.0

        for req_id, true_label in ground_truth.items():
            predicted = classifications.get(req_id, "").lower().strip()
            true_label_norm = true_label.lower().strip()
            if predicted == true_label_norm:
                correct += 1
            elif true_label_norm == "critical" and predicted == "routine":
                penalty += 0.2

        return _clamp((correct / total) - penalty)


class ShortageDetectRubric(Rubric):
    """Grader for shortage-detect task."""

    def forward(self, action: Any, observation: Any) -> float:
        if isinstance(action, dict):
            action_dict = action
        else:
            action_dict = action.dict() if hasattr(action, "dict") else {}

        if isinstance(observation, dict):
            ground_truth = observation.get("info", {}).get("_ground_truth_shortages", [])
        else:
            ground_truth = observation.info.get("_ground_truth_shortages", []) if hasattr(observation, "info") else []

        predicted = set(item.strip() for item in action_dict.get("shortage_items", []))
        actual = set(item.strip() for item in ground_truth)

        if not actual and not predicted:
            return 0.999
        if not actual or not predicted:
            return 0.001

        tp = len(predicted & actual)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(actual) if actual else 0.0

        if precision + recall == 0:
            return 0.001

        f1 = 2 * precision * recall / (precision + recall)
        return _clamp(f1)


class OptimizeAllocationRubric(Rubric):
    """Grader for optimize-allocation task."""

    def forward(self, action: Any, observation: Any) -> float:
        if isinstance(action, dict):
            action_dict = action
        else:
            action_dict = action.dict() if hasattr(action, "dict") else {}

        if isinstance(observation, dict):
            available_stock = observation.get("available_stock", {}) or {}
            units_with_needed = observation.get("info", {}).get("_units_with_needed", [])
        else:
            available_stock = observation.available_stock or {} if hasattr(observation, "available_stock") else {}
            units_with_needed = observation.info.get("_units_with_needed", []) if hasattr(observation, "info") else []

        allocations = action_dict.get("allocations", [])

        allocated_totals: Dict[str, float] = {}
        for alloc in allocations:
            item = alloc.get("item", "")
            qty = float(alloc.get("quantity_allocated", 0))
            allocated_totals[item] = allocated_totals.get(item, 0.0) + qty

        over_allocated = any(
            allocated_totals.get(item, 0.0) > available_stock.get(item, 0)
            for item in allocated_totals
        )

        alloc_lookup: Dict[str, Dict[str, float]] = {}
        for alloc in allocations:
            unit = alloc.get("unit", "")
            item = alloc.get("item", "")
            qty = float(alloc.get("quantity_allocated", 0))
            if unit not in alloc_lookup:
                alloc_lookup[unit] = {}
            alloc_lookup[unit][item] = alloc_lookup[unit].get(item, 0.0) + qty

        total_personnel = sum(u.get("personnel", 1) for u in units_with_needed)
        if total_personnel == 0:
            return 0.001

        weighted_score = 0.0
        for unit in units_with_needed:
            unit_name = unit.get("unit", "")
            personnel = unit.get("personnel", 1)
            needed = unit.get("_needed_qty", {})
            if not needed:
                continue
            item_scores = []
            for item, needed_qty in needed.items():
                if needed_qty <= 0:
                    item_scores.append(1.0)
                    continue
                given = alloc_lookup.get(unit_name, {}).get(item, 0.0)
                item_scores.append(min(given / needed_qty, 1.0))
            unit_fulfillment = sum(item_scores) / len(item_scores)
            weight = personnel / total_personnel
            weighted_score += unit_fulfillment * weight

        if over_allocated:
            weighted_score *= 0.5

        return _clamp(weighted_score)


# Instantiated rubrics
priority_classify_rubric = PriorityClassifyRubric()
shortage_detect_rubric = ShortageDetectRubric()
optimize_allocation_rubric = OptimizeAllocationRubric()


# Legacy function-based graders (used internally by environment.py)
def grade_priority_classify(action: Dict[str, Any], ground_truth: Dict[str, str]) -> float:
    action_dict = {"classifications": action.get("classifications", {})}
    obs = {"info": {"_ground_truth": ground_truth}}
    return priority_classify_rubric.forward(action_dict, obs)


def grade_shortage_detect(action: Dict[str, Any], ground_truth_shortages: list) -> float:
    action_dict = {"shortage_items": action.get("shortage_items", [])}
    obs = {"info": {"_ground_truth_shortages": ground_truth_shortages}}
    return shortage_detect_rubric.forward(action_dict, obs)


def grade_optimize_allocation(action: Dict[str, Any], available_stock: Dict[str, int], units_with_needed: list) -> float:
    action_dict = {"allocations": action.get("allocations", [])}
    obs = {"available_stock": available_stock, "info": {"_units_with_needed": units_with_needed}}
    return optimize_allocation_rubric.forward(action_dict, obs)


GRADERS = {
    "priority-classify": grade_priority_classify,
    "shortage-detect": grade_shortage_detect,
    "optimize-allocation": grade_optimize_allocation,
}

TASKS = [
    {
        "name": "priority-classify",
        "grader": grade_priority_classify,
    },
    {
        "name": "shortage-detect",
        "grader": grade_shortage_detect,
    },
    {
        "name": "optimize-allocation",
        "grader": grade_optimize_allocation,
    },
]