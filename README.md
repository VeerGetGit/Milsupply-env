# milsupply-env 🪖📦

**Military Logistics & Supply Chain** — an OpenEnv environment where AI agents manage real-world defense supply operations.

> 🚀 **Live Demo:** `https://YOUR_HF_USERNAME-milsupply-env.hf.space`  
> Replace with your actual HuggingFace Space URL after deployment.

---

## Overview

`milsupply-env` simulates the logistics challenges faced by military supply chain officers:
- Triaging competing supply requests under pressure
- Detecting critical inventory shortages before they impact operations
- Optimally allocating scarce resources across multiple operational units

This is a domain where mistakes have real consequences — misclassifying a blood plasma request as "routine" or over-allocating ammunition to a training unit at the expense of a combat unit are the kinds of errors this environment is designed to surface and penalize.

---

## Tasks

| Task | Difficulty | Description | Score |
|------|-----------|-------------|-------|
| `priority-classify` | 🟢 Easy | Classify supply requests as `critical`, `high`, or `routine` | Accuracy − critical-miss penalty |
| `shortage-detect` | 🟡 Medium | Identify critically short items given inventory + pending requests | F1 score |
| `optimize-allocation` | 🔴 Hard | Allocate limited stock across units to maximize operational readiness | Weighted readiness gain |

### Task 1: `priority-classify` (Easy)

The agent receives a list of supply requests. Each request includes the item type, requesting unit, mission criticality, and a stated urgency (which may be wrong — units often over-state or under-state urgency).

**Action:**
```json
{"classifications": {"REQ-001": "critical", "REQ-002": "routine", "REQ-003": "high"}}
```

**Scoring:** `correct / total`, minus `0.2` per critical request misclassified as routine (a dangerous error).

---

### Task 2: `shortage-detect` (Medium)

The agent receives current inventory levels and pending unit requests. It must identify items that are **critically short** — meaning all three are true:
1. `quantity_available < reorder_threshold`
2. `days_until_resupply > 3`
3. A pending request exists from a combat/high-criticality unit

**Action:**
```json
{"shortage_items": ["5.56mm ammunition", "Morphine auto-injectors"]}
```

**Scoring:** F1 score against the ground-truth shortage set (balances precision and recall).

---

### Task 3: `optimize-allocation` (Hard)

The agent receives a fixed pool of available stock and must allocate it across multiple units. Units have different personnel counts, readiness levels, and critical item needs.

**Action:**
```json
{"allocations": [
    {"unit": "Alpha Company", "item": "5.56mm ammunition", "quantity_allocated": 400},
    {"unit": "Bravo Medical", "item": "Medical bandages", "quantity_allocated": 100}
]}
```

**Scoring:** Weighted readiness gain across units (weighted by personnel count). Halved if any item is over-allocated beyond available stock.

---

## Observation & Action Spaces

### Observation Space (JSON)

All observations include a `task` field and `context` string plus task-specific fields:

```json
{
  "task": "priority-classify",
  "context": "FOB Alpha is conducting active combat operations...",
  "supply_requests": [
    {
      "request_id": "REQ-001",
      "unit": "1st Infantry",
      "item": "Blood plasma",
      "quantity_requested": 20,
      "urgency_stated": "urgent",
      "location": "Grid 4421",
      "mission_criticality": "combat"
    }
  ]
}
```

### Action Space (JSON)

Task-specific JSON payload sent to `POST /step`:

```json
{
  "task": "priority-classify",
  "payload": {
    "classifications": {"REQ-001": "critical"}
  }
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `GET`  | `/tasks`  | List available tasks with difficulty and score range |
| `GET`  | `/state`  | Current environment state (active task, episode done, task internals) |
| `POST` | `/reset`  | Start new episode `{"task": "priority-classify"}` |
| `POST` | `/step`   | Take action `{"task": "...", "payload": {...}}` |

---

## Setup & Usage

### Prerequisites

- Docker
- Python 3.11+
- A Hugging Face account and API token (`HF_TOKEN`)

### Run with Docker

```bash
docker build -t milsupply-env .
docker run -p 7860:7860 milsupply-env
```

### Run locally (development)

```bash
cd server/
pip install -r requirements.txt
python main.py
```

Server will be available at `http://localhost:7860`.

### Use the deployed HuggingFace Space directly

Once deployed, you can interact with the environment at:

```
https://YOUR_HF_USERNAME-milsupply-env.hf.space
```

Example:
```bash
# Health check
curl https://YOUR_HF_USERNAME-milsupply-env.hf.space/health

# Reset to a task
curl -X POST https://YOUR_HF_USERNAME-milsupply-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "priority-classify"}'

# Submit action
curl -X POST https://YOUR_HF_USERNAME-milsupply-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"task": "priority-classify", "payload": {"classifications": {"REQ-001": "critical"}}}'
```

### Run inference baseline

```bash
pip install openai requests

export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token_here

# Point to local server or deployed Space
export MILSUPPLY_ENV_URL=http://localhost:7860

# Run a single task
export MILSUPPLY_TASK=priority-classify
python inference.py

# Run all three tasks
export MILSUPPLY_TASK=all
python inference.py
```

---

## Reward Design

The reward function provides **partial credit at every step**, not just binary success/failure:

- **priority-classify**: Fraction of correctly classified requests. Critical-to-routine misclassification adds a `−0.2` penalty per error (simulating the real cost of deprioritizing life-safety items).
- **shortage-detect**: F1 score ensures the agent is penalized for both false alarms (precision) and missed shortages (recall).
- **optimize-allocation**: Continuous readiness gain formula rewards any partial fulfillment of a unit's needs. The over-allocation penalty (`× 0.5`) discourages invalid solutions.

All scores are normalized to `[0.0, 1.0]`.

---

## Baseline Scores

> **Note:** Scenarios are selected randomly per episode. Scores below are approximate averages over multiple runs.

| Task | Model | Approximate Score |
|------|-------|-------|
| `priority-classify` | Qwen2.5-72B-Instruct | ~0.80 |
| `shortage-detect`   | Qwen2.5-72B-Instruct | ~0.70 |
| `optimize-allocation` | Qwen2.5-72B-Instruct | ~0.55 |

---

## Project Structure

```
milsupply-env/
├── Dockerfile                         ← HuggingFace Space entry point
├── server/
│   ├── main.py                        ← FastAPI OpenEnv server
│   ├── models.py                      ← Pydantic typed models
│   ├── app.py                         ← App entry point
│   ├── requirements.txt
│   └── tasks/
│       ├── __init__.py
│       ├── priority_classify.py       ← Easy task + grader
│       ├── shortage_detect.py         ← Medium task + grader
│       └── optimize_allocation.py     ← Hard task + grader
├── inference.py                       ← Baseline inference script
├── openenv.yaml                       ← OpenEnv metadata
└── README.md
```

---

## Motivation

Logistics failures are a leading cause of operational failure in military campaigns. Real-world supply chain decisions are made under time pressure with incomplete information — exactly the conditions that make them ideal for AI agent evaluation. Unlike toy environments, errors in this domain have cascading effects (a unit that runs out of ammunition becomes non-operational), which creates natural reward shaping with meaningful partial credit signals.

This environment is immediately applicable to:
- Evaluating LLM reasoning under resource constraints
- Training agents for supply chain optimization
- Benchmarking structured output quality for defense-adjacent applications