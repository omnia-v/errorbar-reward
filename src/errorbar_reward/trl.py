"""TRL adapter: a GRPOTrainer `reward_funcs` callable backed by a reward session.

    from errorbar_reward import ErrorbarRewardClient, trl_reward_func
    client = ErrorbarRewardClient()
    session = client.create_session(reward={...}, reward_budget_usd=5)
    trainer = GRPOTrainer(..., reward_funcs=[trl_reward_func(client, session.id)])

TRL calls reward functions as f(prompts, completions, **kwargs) and accepts
None in the returned list (the sample is masked out of the loss). Prompts and
completions are conversational (lists of messages) or plain strings; both are
rendered to the reward server's contract. Request ids are deterministic per
(step, index) so a retried step re-settles the same ledger rows.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Optional

from .client import ErrorbarRewardClient


def _render_conversation(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '') if isinstance(m.get('content'), str) else json.dumps(m.get('content'))}"
            for m in prompt
            if isinstance(m, dict)
        )
    return json.dumps(prompt)


def _steps_from_completion(completion: Any) -> list[dict[str, Any]] | None:
    """Agentic completions are message lists with assistant and tool turns."""
    if not isinstance(completion, list):
        return None
    steps: list[dict[str, Any]] = []
    for m in completion:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("assistant", "tool"):
            continue
        content = m.get("content")
        if not isinstance(content, str):
            content = json.dumps(content) if content is not None else ""
        if role == "assistant" and m.get("tool_calls"):
            content = content + "\n" + json.dumps(m["tool_calls"])
        step: dict[str, Any] = {"role": role, "content": content}
        if role == "tool" and m.get("name"):
            step["toolName"] = str(m["name"])
        steps.append(step)
    return steps or None


def _final_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for m in reversed(completion):
            if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
                return m["content"]
    return json.dumps(completion)


def trl_reward_func(
    client: ErrorbarRewardClient,
    session_id: str,
    agentic: bool = False,
    step_of: Optional[Callable[[dict[str, Any]], str]] = None,
) -> Callable[..., list[Optional[float]]]:
    """Build the callable. `agentic=True` sends whole completions as steps
    (the session must be agentic). `step_of(kwargs)` derives the optimizer
    step tag; default reads kwargs["step"] when TRL passes it, else "s"."""

    def reward(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[Optional[float]]:
        step = step_of(kwargs) if step_of else str(kwargs.get("step", "s"))
        items = []
        for i, (p, c) in enumerate(zip(prompts, completions)):
            conv = _render_conversation(p)
            digest = hashlib.sha256(f"{step}:{i}:{conv[:2000]}".encode()).hexdigest()[:16]
            item: dict[str, Any] = {"requestId": f"{step}-{i}-{digest}", "conversation": conv}
            steps = _steps_from_completion(c) if agentic else None
            if steps:
                item["steps"] = steps
            else:
                item["response"] = _final_text(c)
            items.append(item)
        scores = client.score(session_id, items, step=step)
        return [s.grade for s in scores]

    reward.__name__ = "errorbar_certified_reward"
    return reward
