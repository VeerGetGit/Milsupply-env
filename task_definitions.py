"""
task_definitions.py — milsupply-env
=====================================
Grader functions for all three tasks.
Scores are clamped strictly between 0.001 and 0.999.
Imported by environment.py via GRADERS dict.
"""

from typing import Any, Dict, List


def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) as required by the OpenEnv validator."""
    return round(min(max(score, 0.001), 0.999), 4)


# ---------------------------------------------------------------------------
# Task 1: priority-classify
# ---------------------------------------------------------------------------

def grade_priority_classify(
    action: Dict[str, Any],
    ground_truth: Dict[str, str],
) -> float:
    classifications = action.get("classifications", {})
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

    score = (correct / total) - penalty
    return _clamp(score)


# ---------------------------------------------------------------------------
# Task 2: shortage-detect
# ---------------------------------------------------------------------------

def grade_shortage_detect(
    action: Dict[str, Any],
    ground_truth_shortages: List[str],
) -> float:
    predicted = set(item.strip() for item in action.get("shortage_items", []))
    actual = set(item.strip() for item in ground_truth_shortages)

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


# ---------------------------------------------------------------------------
# Task 3: optimize-allocation
# ---------------------------------------------------------------------------

def grade_optimize_allocation(
    action: Dict[str, Any],
    available_stock: Dict[str, int],
    units_with_needed: List[Dict[str, Any]],
) -> float:
    allocations = action.get("allocations", [])

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
        needed: Dict[str, int] = unit.get("_needed_qty", {})

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


# ---------------------------------------------------------------------------
# Graders registry
# ---------------------------------------------------------------------------

GRADERS = {
    "priority-classify": grade_priority_classify,
    "shortage-detect": grade_shortage_detect,
    "optimize-allocation": grade_optimize_allocation,
}