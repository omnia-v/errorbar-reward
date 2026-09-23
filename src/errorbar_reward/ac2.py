"""Applied Compute AC2 adapter: errorbar's certified reward as an AC2 grader.

Two graders, one HTTP contract (POST /v1/reward/score against a reward session):

  ErrorbarGrader         the TRAINING grader — `TrainingConfig(method="grpo",
                         ac2_grader="ErrorbarGrader")`. Also runs the eval sidecar
                         (AC2 scores a fixed dataset at step 0 and every
                         `eval_interval` steps with the same grader).
  ErrorbarAnchorGrader   the SECOND grader for regrade jobs —
                         `ac2 grade run --grader ErrorbarAnchorGrader --job-id <train_id>`
                         re-scores the stored eval-sidecar traces with the
                         independent certified criterion and, when the step is
                         known, submits the batch to the training session's
                         anchor route so the hold rule runs.

AC2 facts this module is written against (docs.appliedcompute.com, read
2026-09-07; the SDK is private, so the surface below is quoted, not imported at
authoring time):

  - "A grader is a Python class in the customer project under src/, discovered
    by class name." Interface:
        class MyGrader(Grader):
            async def _grade(self, grader_params: dict | None, trace: Trace,
                             env: EnvironmentProtocol) -> GraderOutput
    returning `GraderOutput(score: float, reasoning: str)`.
  - `LLMGrader` adds `model_config = ModelConfiguration(model=...)` and
    `self.get_completion(messages)`. We call errorbar's API instead, so both
    classes subclass `Grader`, not `LLMGrader`.
  - "`trace` is a list of episodes; `trace[-1].get_items()` yields Message items
    with `.role` and `.content`; the last assistant message is the candidate's
    answer" (their SubstringGrader example).
  - "`grader_params` is per-task JSON from `Task.grader_params` (e.g.
    {"expected": "Paris"}) — may be `{}` during regrade jobs."
  - "Project/user secrets are available to graders" — configuration is read
    from the environment: ERRORBAR_API_KEY, ERRORBAR_SESSION_ID,
    ERRORBAR_ANCHOR_SESSION_ID, ERRORBAR_BASE_URL (see README-ac2.md).
  - Regrade: "`ac2 grade run --grader <Name> --job-id <train_id> [--step n]`
    re-scores stored traces with another grader, shown as a second score column
    `<Grader>@<version>`." A regrade job has no live environment: `env` may be
    None and `grader_params` may be `{}`.
  - `custom_reward_post_process_path`: an import path for reward
    post-processing in GRPO — see MASKED-GRADE POLICY below.

MASKED-GRADE POLICY. errorbar returns `grade: null` for an item whose verdict
could not be parsed — the item is MASKED, not scored, and every other adapter
in this package passes None through (TRL skips it; verifiers raises). AC2's
`GraderOutput.score` is a float, so None has nowhere to go. This adapter returns
the sentinel `MASKED_SCORE` (0.0) with reasoning that STARTS WITH
`MASKED_REASONING_PREFIX` ("masked: unparsed grader verdict"), so a
`custom_reward_post_process_path` function — or a human reading the score
column — can tell a masked item from a genuine zero by its reasoning. Why a
sentinel rather than raising: a raised exception in `_grade` fails the rollout
inside AC2's loop on the grader side, which is the one failure mode a customer
cannot see or retry from their trainer config; a 0.0 with a flagged reason is
visible in the score column and post-processable. The cost is honest and
documented: without a post-processor that drops flagged items, a masked sample
trains as a zero-reward sample (the exact thing the None contract exists to
prevent). Set ERRORBAR_MASKED_POLICY=raise to get the raising behaviour
instead.

409 (anchor hold). A held session refuses scoring until a human resumes it. The
training grader returns 0.0 for every item with reasoning
"errorbar anchor hold at step N — training reward withheld" and logs once per
step: a group of identical rewards has zero group-relative advantage, so GRPO
takes no update from a held step — the reward is withheld, not faked.
402 (budget/expiry) and any network error RAISE — never silently zero a batch.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Optional

import requests

from .client import ErrorbarRewardClient, RewardRefused, RewardScore

log = logging.getLogger("errorbar_reward.ac2")

# ── the AC2 base classes ──────────────────────────────────────────────────────
# The SDK is private; import from the documented runtime module when present
# and fall back to structurally identical stand-ins so this module imports (and
# is testable) without it. Discovery is by class name, so the fallback base is
# only ever used outside AC2.
try:  # pragma: no cover — exercised only inside an AC2 project
    from ac2.runtime import Grader, GraderOutput  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001 — any import failure means "no SDK here"
    class Grader:  # type: ignore[no-redef]
        """Stand-in for `ac2.runtime.Grader` when the SDK is not installed."""

        async def grade(self, grader_params: dict | None, trace: Any, env: Any = None):
            return await self._grade(grader_params, trace, env)

        async def _grade(self, grader_params: dict | None, trace: Any, env: Any):  # pragma: no cover
            raise NotImplementedError

    @dataclass
    class GraderOutput:  # type: ignore[no-redef]
        """Stand-in for `ac2.runtime.GraderOutput(score: float, reasoning: str)`."""

        score: float
        reasoning: str


MASKED_SCORE = 0.0
MASKED_REASONING_PREFIX = "masked: unparsed grader verdict"
HELD_SCORE = 0.0

# Server rule for `step` (app/api/v1/reward/score/route.ts STEP_TAG).
_STEP_TAG = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
_STEP_KEYS = ("errorbar_step", "step", "training_step", "global_step")
DEFAULT_STEP = "s"  # what the TRL adapter sends when the trainer gives no step
ANCHOR_ID_STEP = "anchor"  # the step slot of every anchor item id: one id per prompt, every check
# The anchor route accepts at most 24 items per PART (app/api/v1/reward/sessions/[id]/anchor,
# ANCHOR_ITEMS_MAX in lib/finetuning/reward-score.ts): every item is judged inside the one
# request. The client splits a larger frozen set into parts of 16, so the SET may be larger.
ANCHOR_PART_MAX = 24
ANCHOR_SET_MAX = 2000  # sanity cap on the buffered frozen set, not a server limit
ANCHOR_ITEMS_MAX = ANCHOR_SET_MAX  # kept for callers that imported the old name


# ── configuration ─────────────────────────────────────────────────────────────
@dataclass
class Ac2GraderConfig:
    """Everything read from the AC2 project's secrets/environment.

    ERRORBAR_API_KEY            workspace API key (required)
    ERRORBAR_SESSION_ID         the TRAINING reward session (required)
    ERRORBAR_ANCHOR_SESSION_ID  a session whose reward is the independent
                                certified criterion (ErrorbarAnchorGrader only)
    ERRORBAR_BASE_URL           default https://gateway.errorbar.ai/v1
    ERRORBAR_AGENTIC            "1" to send the assistant/tool turns as
                                structured `steps` (the session must be agentic)
    ERRORBAR_STEP               optimizer step tag when AC2 does not pass one
    ERRORBAR_MASKED_POLICY      "zero" (default) or "raise"
    ERRORBAR_ANCHOR_SET_SIZE    ErrorbarAnchorGrader: number of eval-sidecar
                                traces per step (the frozen set; sent to the
                                anchor route in parts of 16, at most 24 per
                                part). AC2 calls a grader once per trace, so
                                the grader buffers items per step and submits
                                once it has this many — once per step; later
                                items for a submitted step are ignored. Unset =
                                score only, never submit.
    """

    api_key: str
    session_id: str
    base_url: Optional[str] = None
    anchor_session_id: Optional[str] = None
    agentic: bool = False
    step: Optional[str] = None
    masked_policy: str = "zero"
    anchor_set_size: Optional[int] = None
    timeout_s: float = 300.0

    @classmethod
    def from_env(cls, env: Optional[dict[str, str]] = None, *, need_anchor: bool = False) -> "Ac2GraderConfig":
        e = os.environ if env is None else env
        api_key = e.get("ERRORBAR_API_KEY") or e.get("OMNIA_API_KEY")
        if not api_key:
            raise ValueError("ERRORBAR_API_KEY is not set — `ac2 secrets put --key ERRORBAR_API_KEY --value <key>`")
        session_id = e.get("ERRORBAR_SESSION_ID")
        if not session_id:
            raise ValueError("ERRORBAR_SESSION_ID is not set — the reward session created for this training run")
        anchor_session_id = e.get("ERRORBAR_ANCHOR_SESSION_ID") or None
        if need_anchor and not anchor_session_id:
            raise ValueError(
                "ERRORBAR_ANCHOR_SESSION_ID is not set — ErrorbarAnchorGrader scores with a session whose reward "
                "is the independent certified criterion"
            )
        policy = (e.get("ERRORBAR_MASKED_POLICY") or "zero").strip().lower()
        if policy not in ("zero", "raise"):
            raise ValueError(f"ERRORBAR_MASKED_POLICY must be 'zero' or 'raise', got {policy!r}")
        size_raw = e.get("ERRORBAR_ANCHOR_SET_SIZE")
        size: Optional[int] = None
        if size_raw:
            size = int(size_raw)
            if not 1 <= size <= ANCHOR_SET_MAX:
                raise ValueError(f"ERRORBAR_ANCHOR_SET_SIZE must be 1..{ANCHOR_SET_MAX}, got {size}")
        return cls(
            api_key=api_key,
            session_id=session_id,
            base_url=e.get("ERRORBAR_BASE_URL") or None,
            anchor_session_id=anchor_session_id,
            agentic=(e.get("ERRORBAR_AGENTIC") or "").strip().lower() in ("1", "true", "yes"),
            step=_clean_step(e.get("ERRORBAR_STEP")),
            masked_policy=policy,
            anchor_set_size=size,
        )


def _clean_step(value: Any) -> Optional[str]:
    if value is None or value == "" or value == {}:
        return None
    s = str(value).strip()
    if isinstance(value, float) and value.is_integer():
        s = str(int(value))
    return s if _STEP_TAG.match(s) else None


def resolve_step(grader_params: Optional[dict[str, Any]], trace: Any, config_step: Optional[str]) -> Optional[str]:
    """Where the optimizer step comes from, in order: `grader_params`
    (errorbar_step / step / training_step / global_step), ERRORBAR_STEP, then
    an attribute of the same names on the trace or its last episode. The AC2
    docs do not describe how a grader learns the current training step (the
    CLI's `--step n` selects which stored traces to regrade; whether it is
    forwarded to the grader is not stated), so this is best-effort and None is
    a legitimate answer."""
    if isinstance(grader_params, dict):
        for k in _STEP_KEYS:
            step = _clean_step(grader_params.get(k))
            if step:
                return step
    if config_step:
        return config_step
    for obj in (trace, _last_episode(trace)):
        if obj is None:
            continue
        for k in _STEP_KEYS:
            step = _clean_step(getattr(obj, k, None))
            if step:
                return step
    return None


# ── trace → reward item ───────────────────────────────────────────────────────
def _last_episode(trace: Any) -> Any:
    if trace is None:
        return None
    if isinstance(trace, (list, tuple)):
        return trace[-1] if trace else None
    return trace  # a single episode-like object


def _items_of(trace: Any) -> list[Any]:
    ep = _last_episode(trace)
    if ep is None:
        return []
    get_items = getattr(ep, "get_items", None)
    if callable(get_items):
        return list(get_items())
    if isinstance(ep, (list, tuple)):
        return list(ep)
    return list(getattr(ep, "items", None) or getattr(ep, "messages", None) or [])


def _field(m: Any, name: str) -> Any:
    return m.get(name) if isinstance(m, dict) else getattr(m, name, None)


def _text(content: Any) -> str:
    """Message content as text. Strings pass through; content-part lists keep
    their text parts; anything else is JSON so nothing is dropped silently."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(_field(p, "text"), str):
                parts.append(_field(p, "text"))
            else:
                parts.append(json.dumps(p, default=str))
        return "\n".join(parts)
    return json.dumps(content, default=str)


def flatten_conversation(messages: list[dict[str, str]]) -> str:
    """Same semantics as rl/omnia_grpo/trajectory.py flatten_conversation:
    one "role: content" line per message."""
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


@dataclass
class TraceView:
    prompt: list[dict[str, str]]           # messages before the candidate's turn(s)
    steps: list[dict[str, str]]            # assistant/tool turns (agentic instrument)
    response: str                          # last assistant message content
    conversation: str                      # flattened prompt


def view_trace(trace: Any, agentic: bool = False) -> TraceView:
    """Split an AC2 trace into the reward contract's parts.

    Non-agentic: everything before the LAST assistant message is the prompt
    side (rendered "role: content" per line) and that message is `response`.
    Agentic: everything before the FIRST assistant message is the prompt side;
    every assistant/tool turn after it is a structured step (assistant
    tool_calls appended as JSON, matching the TRL adapter) and `response` is
    still the last assistant message.
    """
    msgs: list[dict[str, Any]] = []
    for m in _items_of(trace):
        role = _field(m, "role")
        if role is None:
            continue  # not a Message item (AC2 episodes can hold other item kinds)
        role = str(role)
        content = _text(_field(m, "content"))
        tool_calls = _field(m, "tool_calls")
        if role == "assistant" and tool_calls:
            content = (content + "\n" if content else "") + json.dumps(tool_calls, default=str)
        entry: dict[str, Any] = {"role": role, "content": content}
        name = _field(m, "name") or _field(m, "tool_name")
        if role == "tool" and name:
            entry["toolName"] = str(name)
        msgs.append(entry)

    last_assistant = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i]["role"] == "assistant"), None)
    if last_assistant is None:
        raise ValueError("trace has no assistant message — nothing to grade")
    response = msgs[last_assistant]["content"]

    if agentic:
        first_assistant = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        prompt = msgs[:first_assistant]
        steps = [
            {k: v for k, v in m.items() if k in ("role", "content", "toolName")}
            for m in msgs[first_assistant:]
            if m["role"] in ("assistant", "tool")
        ]
    else:
        prompt = msgs[:last_assistant]
        steps = []
    prompt_plain = [{"role": m["role"], "content": m["content"]} for m in prompt]
    return TraceView(prompt=prompt_plain, steps=steps, response=response, conversation=flatten_conversation(prompt_plain))


def request_id(session_id: str, step: Optional[str], conversation: str, response: Optional[str]) -> str:
    """Deterministic per (session, step, sha256(conversation + response)).
    The same trace re-graded settles into the same ledger row (idempotent
    metering); two rollouts of one prompt within a step get distinct ids
    because their responses differ. `response=None` keys on the prompt alone
    — the anchor route wants the SAME ids for the frozen prompt set every
    step. Fits the server's 200-char cap."""
    h = hashlib.sha256()
    h.update(conversation.encode("utf-8"))
    if response is not None:
        h.update(b"\x00")
        h.update(response.encode("utf-8"))
    digest = h.hexdigest()[:24]
    return f"ac2-{session_id[:24]}-{step or DEFAULT_STEP}-{digest}"


def build_item(view: TraceView, *, session_id: str, step: Optional[str], agentic: bool, stable_id: bool = False) -> dict[str, Any]:
    """One POST /v1/reward/score item (camelCase, the client's dialect)."""
    conv = view.conversation or "(empty prompt)"
    item: dict[str, Any] = {
        "requestId": request_id(session_id, step, conv, None if stable_id else view.response),
        "conversation": conv,
    }
    if agentic and view.steps:
        item["steps"] = view.steps
    else:
        item["response"] = view.response
    return item


def _score_reasoning(score: RewardScore, extra: str = "") -> str:
    raw = score.raw
    flags: list[str] = []
    if raw.get("exec_unverified") or raw.get("execUnverified"):
        flags.append("execUnverified=true (sandbox checks could not run; those components scored 0)")
    if raw.get("attested") is False:
        flags.append("attested=false (no proxy attestation; trainer-reported sources used)")
    if raw.get("attestation_mismatch") or raw.get("attestationMismatch"):
        flags.append("attestationMismatch=true (trainer report contradicted by proxy; grade nulled)")
    if raw.get("sim_fraction") is not None or raw.get("simFraction") is not None:
        flags.append(f"simFraction={raw.get('sim_fraction', raw.get('simFraction'))}")
    parts = [f"errorbar verdict={score.verdict}", f"grade={'null' if score.grade is None else f'{score.grade:.4f}'}"]
    if flags:
        parts.append("; ".join(flags))
    if extra:
        parts.append(extra)
    return " | ".join(parts)


# ── the graders ───────────────────────────────────────────────────────────────
class ErrorbarGrader(Grader):  # type: ignore[misc]
    """errorbar's certified reward as the AC2 training grader.

        TrainingConfig(method="grpo", ac2_grader="ErrorbarGrader", ...)

    One trace → one item → POST /v1/reward/score on ERRORBAR_SESSION_ID →
    `GraderOutput(score=grade, reasoning="errorbar verdict=... | grade=... | flags")`.
    `grader_params` (e.g. {"expected": "Paris"}) is NOT sent: the session's
    certified criterion is the reward; only a step key is read from it.
    """

    session_env_key = "ERRORBAR_SESSION_ID"

    def __init__(self, config: Optional[Ac2GraderConfig] = None, client: Optional[ErrorbarRewardClient] = None, **kwargs: Any):
        try:
            super().__init__(**kwargs)
        except TypeError:  # the real base may take no kwargs
            super().__init__()
        self.config = config or Ac2GraderConfig.from_env()
        self.client = client or ErrorbarRewardClient(
            api_key=self.config.api_key, base_url=self.config.base_url, timeout_s=self.config.timeout_s
        )
        self._held_logged: set[str] = set()

    @property
    def scoring_session_id(self) -> str:
        return self.config.session_id

    async def _grade(self, grader_params: dict | None, trace: Any, env: Any = None) -> GraderOutput:  # type: ignore[override]
        step = resolve_step(grader_params, trace, self.config.step)
        view = view_trace(trace, agentic=self.config.agentic)
        item = build_item(view, session_id=self.scoring_session_id, step=step, agentic=self.config.agentic)
        out = await asyncio.to_thread(self._score_item, item, step)
        await self._after_grade(view, item, step)
        return out

    async def _after_grade(self, view: TraceView, item: dict[str, Any], step: Optional[str]) -> None:
        """Hook for subclasses (the anchor grader buffers here)."""

    def _score_item(self, item: dict[str, Any], step: Optional[str]) -> GraderOutput:
        try:
            scores = self.client.score(self.scoring_session_id, [item], step=step)
        except RewardRefused as e:
            if e.status == 409:
                tag = step or "unknown"
                if tag not in self._held_logged:
                    self._held_logged.add(tag)
                    log.warning("errorbar session %s is held (step %s): %s", self.scoring_session_id, tag, e)
                return GraderOutput(score=HELD_SCORE, reasoning=f"errorbar anchor hold at step {tag} — training reward withheld ({e})")
            if e.status == 402:
                raise RewardRefused(
                    402,
                    f"errorbar refused to score (budget exhausted or session expired) — "
                    f"top up / recreate session {self.scoring_session_id} and set ERRORBAR_SESSION_ID: {e}",
                ) from e
            raise
        except requests.RequestException as e:
            raise RuntimeError(f"errorbar reward server unreachable ({type(e).__name__}: {e}) — not scoring this trace as 0") from e
        score = scores[0]
        if score.grade is None:
            if self.config.masked_policy == "raise":
                raise ValueError(f"{MASKED_REASONING_PREFIX} for {item['requestId']} — masked, not scored")
            return GraderOutput(score=MASKED_SCORE, reasoning=_score_reasoning(score, f"{MASKED_REASONING_PREFIX} — sentinel {MASKED_SCORE}, not a score"))
        return GraderOutput(score=float(score.grade), reasoning=_score_reasoning(score))


class ErrorbarAnchorGrader(ErrorbarGrader):
    """The SECOND grader: an independent certified criterion over the
    eval-sidecar traces, plus the anchor hold rule.

        ac2 grade run --grader ErrorbarAnchorGrader --job-id <train_id> [--step n]

    Scores each trace with ERRORBAR_ANCHOR_SESSION_ID — a reward session whose
    reward is the independent certified criterion — so the regrade job's
    `ErrorbarAnchorGrader@<version>` column is a second opinion from a
    different instrument, not the training reward re-run.

    Anchor submission. When the step is known and ERRORBAR_ANCHOR_SET_SIZE is
    set, the grader buffers the step's items and, once it holds that many,
    POSTs them to `/v1/reward/sessions/{ERRORBAR_SESSION_ID}/anchor` — the
    TRAINING session's anchor route, which scores the same outputs with the
    training reward and with the session's configured anchor criterion and
    runs the hold rule (`held`, `pinned_step`, `underpowered`, `history`). The
    buffer exists because AC2 calls a grader once per trace while the anchor
    route wants the frozen set as one check (sent in parts of ≤24 items,
    ≥20 gradable to hold); each step is submitted once.
    Anchor item ids key on the prompt only (never the step), so the frozen set
    carries the same ids at every check and the server can pair them. errorbar
    holds the frozen set: before the first check the grader registers the first
    submitted batch's prompts with the session (or reads back a set registered
    earlier) and every later check sends that set's ids; a prompt outside it is
    refused by the server. The anchor verdict is logged and kept in
    `self.anchor_results`; a 409 there means the run is already held and is
    logged, not raised.

    Regrade jobs run without a live environment (`env` is None) and
    `grader_params` may be `{}` — the step then comes from ERRORBAR_STEP
    (set it to the `--step n` you regrade), otherwise scoring proceeds and
    the anchor is not submitted.
    """

    def __init__(self, config: Optional[Ac2GraderConfig] = None, client: Optional[ErrorbarRewardClient] = None, **kwargs: Any):
        super().__init__(config or Ac2GraderConfig.from_env(need_anchor=True), client, **kwargs)
        if not self.config.anchor_session_id:
            raise ValueError("ErrorbarAnchorGrader needs ERRORBAR_ANCHOR_SESSION_ID")
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        self._submitted: set[str] = set()  # steps already sent: a second batch would be a second history point
        self._lock = threading.Lock()
        self.anchor_results: list[dict[str, Any]] = []
        # The frozen set: prompt messages by anchor request id, and the registered
        # control ids once the session holds a set.
        self._prompts: dict[str, list[dict[str, str]]] = {}
        self.control_ids: Optional[set[str]] = None

    @property
    def scoring_session_id(self) -> str:
        return self.config.anchor_session_id  # type: ignore[return-value]

    async def _after_grade(self, view: TraceView, item: dict[str, Any], step: Optional[str]) -> None:
        if step is None or not self.config.anchor_set_size:
            return
        # One id per PROMPT across every check — the step is not part of it, or the
        # server could never pair a prompt's reads (it pairs by request id).
        anchor_item = build_item(view, session_id=self.config.session_id, step=ANCHOR_ID_STEP, agentic=self.config.agentic, stable_id=True)
        batch: Optional[list[dict[str, Any]]] = None
        with self._lock:
            self._prompts.setdefault(anchor_item["requestId"], view.prompt)
            if step in self._submitted:
                return
            buf = self._buffers.setdefault(step, [])
            if all(i["requestId"] != anchor_item["requestId"] for i in buf):
                buf.append(anchor_item)
            if len(buf) >= self.config.anchor_set_size:
                batch = self._buffers.pop(step)
                self._submitted.add(step)
        if batch is not None:
            await asyncio.to_thread(self.submit_anchor, step, batch)

    def submit_anchor(self, step: str, items: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """POST the frozen set for `step` to the training session's anchor
        route. Returns the server's snake_case verdict, or None on 409 (already
        held). 402 and network errors raise."""
        try:
            self._ensure_frozen_set(items)
            items = [i for i in items if i["requestId"] in (self.control_ids or ())]
            out = self.client.anchor(self.config.session_id, items, step=step)
        except RewardRefused as e:
            if e.status == 409:
                log.warning("errorbar anchor: session %s already held at step %s: %s", self.config.session_id, step, e)
                return None
            raise
        except requests.RequestException as e:
            raise RuntimeError(f"errorbar anchor route unreachable ({type(e).__name__}: {e})") from e
        self.anchor_results.append({"step": step, **out})
        if out.get("held"):
            log.warning(
                "errorbar anchor HELD session %s at step %s — %s; pinned step %s",
                self.config.session_id, step, out.get("reason"), out.get("pinned_step"),
            )
        elif out.get("underpowered"):
            log.info("errorbar anchor step %s: %s", step, out["underpowered"])
        else:
            log.info("errorbar anchor step %s: not held (%s)", step, out.get("reason"))
        return out

    def _ensure_frozen_set(self, items: list[dict[str, Any]]) -> None:
        """The session's frozen set: read back when one is registered (a restarted
        grader), else registered from this first batch's prompts. All control
        (confirm_share 0): the AC2 path decides on the checks, not on an
        end-of-run confirmation."""
        if self.control_ids is not None:
            return
        try:
            self.control_ids = set(self.client.frozen_set(self.config.session_id).get("control_ids") or [])
            return
        except RewardRefused as e:
            if e.status != 404:
                raise
        registration = [{"requestId": i["requestId"], "prompt": self._prompts.get(i["requestId"]) or "(empty prompt)"} for i in items]
        out = self.client.register_frozen_set(self.config.session_id, registration, confirm_share=0.0)
        self.control_ids = set(out.get("control_ids") or [])
        log.info("errorbar anchor: registered a frozen set of %d prompts with session %s", len(self.control_ids), self.config.session_id)

    def flush_anchor(self) -> list[dict[str, Any]]:
        """Submit whatever is buffered (a dataset smaller than
        ERRORBAR_ANCHOR_SET_SIZE, or a final partial batch). Returns the verdicts."""
        with self._lock:
            pending = list(self._buffers.items())
            self._buffers.clear()
            self._submitted.update(step for step, items in pending if items)
        return [r for step, items in pending if items and (r := self.submit_anchor(step, items)) is not None]


__all__ = [
    "Ac2GraderConfig",
    "ErrorbarGrader",
    "ErrorbarAnchorGrader",
    "GraderOutput",
    "MASKED_SCORE",
    "MASKED_REASONING_PREFIX",
    "HELD_SCORE",
    "TraceView",
    "view_trace",
    "build_item",
    "request_id",
    "resolve_step",
]
