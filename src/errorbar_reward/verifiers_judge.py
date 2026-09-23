"""Prime Intellect verifiers (v1) adapter: an errorbar reward session as a Judge.

    from errorbar_reward.verifiers_judge import ErrorbarJudge, ErrorbarJudgeConfig
    judge = ErrorbarJudge(ErrorbarJudgeConfig(session_id="...", agentic=True))
    # attach via TaskConfig.judges, or call inside a @vf.reward:
    #   score = await judge.score(task, trace)

The Judge base is imported lazily so this module loads without verifiers
installed. `score` returns the certified grade in [0, 1]; a masked item
(unparseable judge) RAISES, matching verifiers' rule that a malformed verdict
is a judge failure and must error the rollout, not score the model.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .client import ErrorbarRewardClient


@dataclass
class ErrorbarJudgeConfig:
    session_id: str
    agentic: bool = False
    api_key: str | None = None
    base_url: str | None = None
    step: str = "s"
    id: str = "errorbar"


def _trace_to_item(trace: Any, agentic: bool, step: str) -> dict[str, Any]:
    """Render a verifiers Trace into the reward contract. Uses the public
    accessors documented on Trace (messages, assistant_messages, tool_messages,
    last_reply) and falls back to str() so an unexpected shape still scores."""
    messages = list(getattr(trace, "messages", []) or [])
    prompt_parts: list[str] = []
    steps: list[dict[str, Any]] = []
    for m in messages:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
        if not isinstance(content, str):
            content = json.dumps(content) if content is not None else ""
        if role in ("assistant", "tool"):
            tool_calls = getattr(m, "tool_calls", None) if not isinstance(m, dict) else m.get("tool_calls")
            if role == "assistant" and tool_calls:
                content = content + "\n" + json.dumps(tool_calls, default=str)
            steps.append({"role": role, "content": content})
        else:
            prompt_parts.append(f"{role or 'user'}: {content}")
    conversation = "\n".join(prompt_parts) or str(getattr(trace, "transcript", "") or "")
    last = getattr(trace, "last_reply", None)
    response = last if isinstance(last, str) else (getattr(last, "content", None) if last is not None else None)
    if not isinstance(response, str):
        response = steps[-1]["content"] if steps else ""
    key = getattr(getattr(trace, "task", None), "key", None) or hashlib.sha256(conversation.encode()).hexdigest()[:16]
    item: dict[str, Any] = {"requestId": f"{step}-{key}", "conversation": conversation}
    if agentic and steps:
        item["steps"] = steps
    else:
        item["response"] = response
    return item


def _make_judge_class():
    from verifiers.v1.judge import Judge  # type: ignore[import-not-found]

    class ErrorbarJudge(Judge):  # type: ignore[misc,valid-type]
        """A certified reward as a verifiers Judge. The certificate for the
        session's criteria is available as `judge.certificates`."""

        def __init__(self, config: ErrorbarJudgeConfig, client: ErrorbarRewardClient | None = None):
            self.config = config  # verifiers' base __init__ resolves configs it knows; ours is explicit
            self.client = client or ErrorbarRewardClient(api_key=config.api_key, base_url=config.base_url)
            self.session = self.client.get_session(config.session_id)
            self.certificates = self.session.certificates

        async def score(self, task: Any, trace: Any) -> float:
            item = _trace_to_item(trace, self.config.agentic, self.config.step)
            scores = await asyncio.to_thread(self.client.score, self.config.session_id, [item], self.config.step)
            grade = scores[0].grade
            if grade is None:
                raise ValueError(
                    f"errorbar judge could not parse a verdict for {item['requestId']} — masked, not scored"
                )
            return float(grade)

    return ErrorbarJudge


try:
    ErrorbarJudge = _make_judge_class()
except Exception:  # verifiers not installed: expose a plain class with the same surface
    class ErrorbarJudge:  # type: ignore[no-redef]
        def __init__(self, config: ErrorbarJudgeConfig, client: ErrorbarRewardClient | None = None):
            self.config = config
            self.client = client or ErrorbarRewardClient(api_key=config.api_key, base_url=config.base_url)
            self.session = self.client.get_session(config.session_id)
            self.certificates = self.session.certificates

        async def score(self, task: Any, trace: Any) -> float:
            item = _trace_to_item(trace, self.config.agentic, self.config.step)
            scores = await asyncio.to_thread(self.client.score, self.config.session_id, [item], self.config.step)
            grade = scores[0].grade
            if grade is None:
                raise ValueError(
                    f"errorbar judge could not parse a verdict for {item['requestId']} — masked, not scored"
                )
            return float(grade)
