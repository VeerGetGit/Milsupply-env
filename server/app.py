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


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()