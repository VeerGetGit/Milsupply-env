"""
task_definitions.py — milsupply-env
=====================================
Grader functions for all three tasks.
Imported by environment.py via GRADERS dict.
"""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Task 1: priority-classify
# ---------------------------------------------------------------------------

def grade_priority_classify(
    action: Dict[str, Any],
    ground_truth: Dict[str, str],
) -> float:
    """
    Score = (correct / total) - 0.2 per critical-to-routine misclassification.
    Clamped to [0.0, 1.0].
    """
    classifications = action.get("classifications", {})
    if not ground_truth:
        return 0.0

    total = len(ground_truth)
    correct = 0
    penalty = 0.0

    for req_id, true_label in ground_truth.items():
        predicted = classifications.get(req_id, "").lower().strip()
        true_label_norm = true_label.lower().strip()

        if predicted == true_label_norm:
            correct += 1
        elif true_label_norm == "critical" and predicted == "routine":
            # Dangerous misclassification — apply penalty
            penalty += 0.2

    score = (correct / total) - penalty
    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Task 2: shortage-detect
# ---------------------------------------------------------------------------

def grade_shortage_detect(
    action: Dict[str, Any],
    ground_truth_shortages: List[str],
) -> float:
    """
    F1 score between predicted shortage_items and ground truth set.
    """
    predicted = set(item.strip() for item in action.get("shortage_items", []))
    actual = set(item.strip() for item in ground_truth_shortages)

    if not actual and not predicted:
        return 1.0
    if not actual or not predicted:
        return 0.0

    tp = len(predicted & actual)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(actual) if actual else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(min(max(f1, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Task 3: optimize-allocation
# ---------------------------------------------------------------------------

def grade_optimize_allocation(
    action: Dict[str, Any],
    available_stock: Dict[str, int],
    units_with_needed: List[Dict[str, Any]],
) -> float:
    """
    Weighted readiness gain across all units, weighted by personnel count.
    Score is halved if any item is over-allocated beyond available stock.
    Clamped to [0.0, 1.0].
    """
    allocations = action.get("allocations", [])

    # Check for over-allocation
    allocated_totals: Dict[str, float] = {}
    for alloc in allocations:
        item = alloc.get("item", "")
        qty = float(alloc.get("quantity_allocated", 0))
        allocated_totals[item] = allocated_totals.get(item, 0.0) + qty

    over_allocated = any(
        allocated_totals.get(item, 0.0) > available_stock.get(item, 0)
        for item in allocated_totals
    )

    # Build lookup: {unit_name: {item: qty_allocated}}
    alloc_lookup: Dict[str, Dict[str, float]] = {}
    for alloc in allocations:
        unit = alloc.get("unit", "")
        item = alloc.get("item", "")
        qty = float(alloc.get("quantity_allocated", 0))
        if unit not in alloc_lookup:
            alloc_lookup[unit] = {}
        alloc_lookup[unit][item] = alloc_lookup[unit].get(item, 0.0) + qty

    # Compute weighted readiness gain
    total_personnel = sum(u.get("personnel", 1) for u in units_with_needed)
    if total_personnel == 0:
        return 0.0

    weighted_score = 0.0
    for unit in units_with_needed:
        unit_name = unit.get("unit", "")
        personnel = unit.get("personnel", 1)
        needed: Dict[str, int] = unit.get("_needed_qty", {})

        if not needed:
            continue

        # Fraction of each needed item fulfilled
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

    return round(min(max(weighted_score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Graders registry
# ---------------------------------------------------------------------------

GRADERS = {
    "priority-classify": grade_priority_classify,
    "shortage-detect": grade_shortage_detect,
    "optimize-allocation": grade_optimize_allocation,
}