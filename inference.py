"""
Inference Script — milsupply-env
===================================
Military Logistics & Supply Chain OpenEnv Baseline

Environment variables required:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.

Usage:
  export API_BASE_URL=https://router.huggingface.co/v1
  export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
  export HF_TOKEN=hf_xxx
  export MILSUPPLY_TASK=priority-classify   # or shortage-detect / optimize-allocation
  export MILSUPPLY_ENV_URL=http://localhost:7860
  python inference.py
"""

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MILSUPPLY_TASK", "priority-classify")
ENV_URL = os.getenv("MILSUPPLY_ENV_URL", "http://localhost:7860").rstrip("/")
BENCHMARK = "milsupply-env"
MAX_STEPS = 1          # all tasks are single-step in this env
TEMPERATURE = 0.2      # low temp for deterministic structured output
MAX_TOKENS = 1024
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Logging helpers (competition format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Truncate action for log readability (keep it single-line)
    action_log = action.replace("\n", " ")[:200]
    print(
        f"[STEP] step={step} action={action_log} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------

def env_reset(task: str) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/reset", json={"task": task}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/step", json={"task": task, "payload": payload}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Task-specific prompts and response parsers
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "priority-classify": textwrap.dedent("""
        You are a military logistics officer responsible for triaging supply requests.
        You will receive a list of supply requests from various units.
        For each request, classify the true priority as: critical, high, or routine.
        
        Guidelines:
        - critical: life-safety items, combat ammunition, medical supplies in active combat zones, needed within hours
        - high: operationally important, needed within 24 hours
        - routine: administrative or non-urgent, can wait 72+ hours
        
        Note: The urgency stated by the requesting unit may be inaccurate (units tend to over-state urgency).
        Use your judgment based on the item type and mission_criticality field.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {"classifications": {"REQ-001": "critical", "REQ-002": "routine", ...}}
        
        No explanation, no markdown, no extra text. JSON only.
    """).strip(),

    "shortage-detect": textwrap.dedent("""
        You are a military supply chain analyst.
        You will receive current inventory levels and pending supply requests from units.
        
        Identify items that are CRITICALLY SHORT using these criteria:
        1. quantity_available < reorder_threshold (stock is below minimum)
        2. days_until_resupply > 3 (resupply is not imminent)
        3. There is a pending request for that item from a combat or high-criticality unit
        
        ALL THREE criteria must be met for an item to be critically short.
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {"shortage_items": ["item name 1", "item name 2", ...]}
        
        Use the exact item names as they appear in the inventory.
        No explanation, no markdown, no extra text. JSON only.
    """).strip(),

    "optimize-allocation": textwrap.dedent("""
        You are a theater logistics commander responsible for allocating scarce supplies.
        You will receive available stock and a list of units with their critical needs.
        
        Your goal: allocate supplies to MAXIMIZE overall operational readiness across all units.
        
        Rules:
        - You CANNOT exceed the available_stock quantity for any item
        - Prioritize units with more personnel and lower current readiness
        - Combat-mission units take priority over support units
        - Partial allocation is better than no allocation
        - You do not have to allocate all stock
        
        You MUST respond with ONLY a valid JSON object in this exact format:
        {"allocations": [
            {"unit": "Unit Name", "item": "item name", "quantity_allocated": 100},
            ...
        ]}
        
        Use exact unit names and item names as they appear in the observation.
        No explanation, no markdown, no extra text. JSON only.
    """).strip(),
}


def build_user_prompt(task: str, observation: Dict[str, Any]) -> str:
    """Convert observation dict to a readable prompt for the model."""
    obs_json = json.dumps(observation, indent=2)
    return f"Here is the current situation:\n\n{obs_json}\n\nProvide your response now."


def parse_model_response(task: str, text: str) -> Dict[str, Any]:
    """Parse the model's JSON response into a task payload."""
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract JSON from the text
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            # Return a safe fallback per task
            fallbacks = {
                "priority-classify": {"classifications": {}},
                "shortage-detect": {"shortage_items": []},
                "optimize-allocation": {"allocations": []},
            }
            return fallbacks.get(task, {})

    return data


def get_model_action(client: OpenAI, task: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM and return a parsed action payload."""
    user_prompt = build_user_prompt(task, observation)
    system_prompt = SYSTEM_PROMPTS[task]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_model_response(task, text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        fallbacks = {
            "priority-classify": {"classifications": {}},
            "shortage-detect": {"shortage_items": []},
            "optimize-allocation": {"allocations": []},
        }
        return fallbacks.get(task, {})


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg: Optional[str] = None

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        reset_resp = env_reset(task)
        observation = reset_resp.get("observation", {})
        done = reset_resp.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Get action from model
            payload = get_model_action(client, task, observation)
            action_str = json.dumps(payload)

            # Step environment
            try:
                step_resp = env_step(task, payload)
                observation = step_resp.get("observation", {})
                reward = float(step_resp.get("reward", 0.0))
                done = step_resp.get("done", True)
                info = step_resp.get("info", {})
                error_msg = None
            except Exception as e:
                reward = 0.0
                done = True
                error_msg = str(e)
                info = {}

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        score = rewards[-1] if rewards else 0.0
        score = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        error_msg = str(exc)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks_to_run = os.getenv("MILSUPPLY_TASK", "")
    if tasks_to_run == "all":
        task_list = ["priority-classify", "shortage-detect", "optimize-allocation"]
    elif tasks_to_run:
        task_list = [tasks_to_run]
    else:
        task_list = ["priority-classify"]

    for task in task_list:
        run_task(client, task)
        print()  # blank line between tasks


if __name__ == "__main__":
    main()
