import uvicorn
from fastapi import FastAPI
from openenv.core.env_server import create_app
from models import MilSupplyAction, MilSupplyObservation
from environment import MilSupplyEnvironment

# Pass the CLASS not an instance
app = create_app(MilSupplyEnvironment, MilSupplyAction, MilSupplyObservation)


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "priority-classify",
                "difficulty": "easy",
                "score_range": [0, 1],
                "max_steps": 1,
                "has_grader": True,
                "grader": "task_definitions.py::grade_priority_classify",
            },
            {
                "name": "shortage-detect",
                "difficulty": "medium",
                "score_range": [0, 1],
                "max_steps": 1,
                "has_grader": True,
                "grader": "task_definitions.py::grade_shortage_detect",
            },
            {
                "name": "optimize-allocation",
                "difficulty": "hard",
                "score_range": [0, 1],
                "max_steps": 1,
                "has_grader": True,
                "grader": "task_definitions.py::grade_optimize_allocation",
            },
        ]
    }

@app.post("/grader")
def grader(payload: dict):
    task = payload.get("task", "priority-classify")
    action = payload.get("action", {})
    observation = payload.get("observation", {})
    
    from task_definitions import grade_priority_classify, grade_shortage_detect, grade_optimize_allocation
    
    if task == "shortage-detect":
        ground_truth = observation.get("info", {}).get("_ground_truth_shortages", [])
        score = grade_shortage_detect(action, ground_truth)
    elif task == "optimize-allocation":
        available_stock = observation.get("available_stock", {})
        units = observation.get("info", {}).get("_units_with_needed", [])
        score = grade_optimize_allocation(action, available_stock, units)
    else:
        ground_truth = observation.get("info", {}).get("_ground_truth", {})
        score = grade_priority_classify(action, ground_truth)
    
    return {"score": score, "task": task}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()