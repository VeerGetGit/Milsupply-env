"""
Inference Script — milsupply-env
===================================
Military Logistics & Supply Chain OpenEnv Baseline
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

MAX_STEPS = 1
TEMPERATURE = 0.2
MAX_TOKENS = 1024
SUCCESS_SCORE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_log = action.replace("\n", " ")

    print(
        f"[STEP] step={step} action={action_log} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment API
# ---------------------------------------------------------------------------

def env_reset(task: str) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/reset", json={"task": task}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    payload["task"] = task
    resp = requests.post(f"{ENV_URL}/step", json={"action": payload}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "priority-classify": "You are a logistics officer. Return JSON only.",
    "shortage-detect": "You are a supply analyst. Return JSON only.",
    "optimize-allocation": "You are a commander. Return JSON only.",
}


def build_user_prompt(observation: Dict[str, Any]) -> str:
    return f"Situation:\n{json.dumps(observation, indent=2)}\n\nReturn JSON only."


def parse_model_response(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "").strip()

    try:
        return json.loads(text)
    except Exception:
        return {}


def get_model_action(client: OpenAI, task: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user", "content": build_user_prompt(observation)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        text = completion.choices[0].message.content or ""
        return parse_model_response(text)

    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task: str) -> None:
    rewards: List[float] = []
    steps_taken = 0

    score = 0.0
    success = False

    log_start(task, BENCHMARK, MODEL_NAME)

    try:
        reset_resp = env_reset(task)
        observation = reset_resp.get("observation", {})
        done = reset_resp.get("done", False)

        step = 1

        while not done and step <= MAX_STEPS:

            action = get_model_action(client, task, observation)

            step_resp = env_step(task, action)

            observation = step_resp.get("observation", {})
            reward = float(step_resp.get("reward", 0.0))
            done = step_resp.get("done", True)

            error_msg = step_resp.get("info", {}).get("error", None)

            rewards.append(reward)
            steps_taken = step

            log_step(step, json.dumps(action), reward, done, error_msg)

            step += 1

        # ✅ FIXED SCORE (IMPORTANT)
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(score, 1.0))

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Error: {exc}", flush=True)

    finally:
        log_end(success, steps_taken, score, rewards)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task = os.getenv("MILSUPPLY_TASK", "priority-classify")

    run_task(client, task)


if __name__ == "__main__":
    main()