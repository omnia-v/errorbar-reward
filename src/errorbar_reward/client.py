"""HTTP client for errorbar reward sessions.

Auth is the workspace API key (Bearer). The management API speaks snake_case;
this client returns snake_case dicts untouched and typed wrappers on top.

    client = ErrorbarRewardClient(api_key=os.environ["ERRORBAR_API_KEY"])
    session = client.create_session(
        reward={"mode": "single", "criterionId": "crit_..."},
        reward_budget_usd=5,
    )
    for cert in session.certificates:
        print(cert["criterion"]["name"], cert["calibration"])
    scores = client.score(session.id, items=[...], step="12")

Watching your OWN reward instead (training control): errorbar never scores
the reward; at each check you send the frozen-set outputs with the score your
reward gave each one, and errorbar reads it against an independent anchor.

    session = client.create_session(
        external_reward={"name": "rubric judge v3", "scale": "unit"},
        anchor_criterion_id="crit_anchor...",
        reward_budget_usd=25,
    )
    check = client.anchor(session.id, items=[{..., "rewardScore": 1.0}, ...], step="64")
    check["status"]  # continue | review | hold
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlencode
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests

DEFAULT_BASE_URL = "https://gateway.errorbar.ai/v1"


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "0"


class RewardRefused(RuntimeError):
    """The server refused to score: budget exhausted (402), session gone
    (404), reward invalid (422), auth (401/403). The trainer should crash
    loudly rather than continue on a silent zero reward."""

    def __init__(self, status: int, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"reward server {status}: {message}")
        self.status = status
        # The error object's other fields — on a hold (409): pinned_step and pin.
        self.details = details or {}


@dataclass
class RewardSession:
    id: str
    status: str
    agentic: bool
    budget_usd: float
    spent_usd: float
    expires_at: str
    certificates: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "RewardSession":
        return cls(
            id=d["id"],
            status=d["status"],
            agentic=bool(d.get("agentic", False)),
            budget_usd=float(d.get("budget_usd", 0)),
            spent_usd=float(d.get("spent_usd", 0)),
            expires_at=str(d.get("expires_at", "")),
            certificates=list(d.get("certificates") or []),
            raw=d,
        )


@dataclass
class RewardScore:
    request_id: str
    grade: Optional[float]
    verdict: str
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "RewardScore":
        g = d.get("grade")
        return cls(
            request_id=str(d["request_id"]),
            grade=None if g is None else float(g),
            verdict=str(d.get("verdict", "unparsed")),
            raw=d,
        )


Http = Callable[..., requests.Response]


class ErrorbarRewardClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 900.0,
        http: Http | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.api_key = api_key or os.environ.get("ERRORBAR_API_KEY") or os.environ.get("OMNIA_API_KEY")
        if not self.api_key:
            raise ValueError("ERRORBAR_API_KEY is required")
        self.base_url = (base_url or os.environ.get("ERRORBAR_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self._http: Http = http or requests.request
        self._sleep = sleep  # injectable so the anchor part backoff is tested without waiting

    # ── transport ──────────────────────────────────────────────────────────
    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        res = self._http(
            method,
            f"{self.base_url}{path}",
            data=None if body is None else json.dumps(body, separators=(",", ":")),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"errorbar-reward/{_version()}",
            },
            timeout=self.timeout_s,
        )
        if res.status_code >= 400:
            msg = res.text[:300]
            details: dict[str, Any] = {}
            try:
                err = res.json().get("error")
                if isinstance(err, dict):
                    msg = str(err.get("message", msg))
                    details = {k: v for k, v in err.items() if k not in ("message", "type")}
                elif isinstance(err, str):
                    msg = err
            except Exception:  # noqa: BLE001 — the body is diagnostic only
                pass
            raise RewardRefused(res.status_code, msg, details)
        return res

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._request(method, path, body).json()

    def _call_text(self, method: str, path: str) -> str:
        return self._request(method, path).text

    # ── sessions ───────────────────────────────────────────────────────────
    def create_session(
        self,
        reward: dict[str, Any] | None = None,
        reward_budget_usd: float = 0.0,
        agentic: bool = False,
        ttl_hours: float | None = None,
        label: str | None = None,
        external_reward: dict[str, Any] | None = None,
        anchor_criterion_id: str | None = None,
        anchor_min_improvement: float | None = None,
        instruments: list[dict[str, Any]] | None = None,
        frozen_set: list[dict[str, Any]] | None = None,
        confirm_share: float | None = None,
        mode: str | None = None,
    ) -> RewardSession:
        """reward: errorbar scores it (a validated criterion spec).
        external_reward: errorbar watches YOUR reward instead —
        {"name", "scale": "unit" | "threshold", "passThreshold"?}; requires
        anchor_criterion_id. Exactly one of the two. anchor_min_improvement: the
        smallest improvement over the baseline (truth scale, 0–1) the gate calls
        SUPPORTED (default 0). instruments: your other measurements —
        [{"name", "kind": "customer_eval" | "programmatic" | "judge", "scale":
        "unit" | "threshold", "passThreshold"?}]; send their per-item scores as
        item["measurements"] = {name: score} with each anchor check.
        frozen_set: the anchor's frozen control set, registered with the session —
        [{"requestId", "prompt": str | [{role, content}], "reference"?}]. errorbar
        keeps the prompts, seals confirm_share of them (default 0.25) for the
        end-of-run confirmation, and reads every check's prompts from its own copy;
        checks then send only requestId + response. Without it, register one with
        register_frozen_set() before the first check.
        mode: "shadow" — the gate decides every check but a hold is recorded on
        the scorecard, never enforced (prove it on your runs first); "active"
        (the default) — a hold stops the session. Promote with set_mode()."""
        if (reward is None) == (external_reward is None):
            raise ValueError("give exactly one of reward or external_reward")
        body: dict[str, Any] = {"rewardBudgetUsd": reward_budget_usd, "agentic": agentic}
        if reward is not None:
            body["reward"] = reward
        if external_reward is not None:
            body["externalReward"] = external_reward
        if anchor_criterion_id:
            body["anchor"] = {"criterionId": anchor_criterion_id}
            if anchor_min_improvement is not None:
                body["anchor"]["minImprovement"] = anchor_min_improvement
            if instruments:
                body["anchor"]["instruments"] = instruments
            if frozen_set:
                body["anchor"]["frozenSet"] = frozen_set
                if confirm_share is not None:
                    body["anchor"]["confirmShare"] = confirm_share
            elif confirm_share is not None:
                raise ValueError("confirm_share needs frozen_set")
            if mode is not None:
                if mode not in ("shadow", "active"):
                    raise ValueError('mode must be "shadow" or "active"')
                body["anchor"]["mode"] = mode
        elif anchor_min_improvement is not None or instruments or frozen_set or mode:
            raise ValueError("anchor_min_improvement, instruments, frozen_set and mode need anchor_criterion_id")
        if ttl_hours is not None:
            body["ttlHours"] = ttl_hours
        if label:
            body["label"] = label
        return RewardSession.from_json(self._call("POST", "/reward/sessions", body))

    def get_session(self, session_id: str) -> RewardSession:
        return RewardSession.from_json(self._call("GET", f"/reward/sessions/{session_id}"))

    def stop_session(self, session_id: str) -> RewardSession:
        return RewardSession.from_json(self._call("DELETE", f"/reward/sessions/{session_id}"))

    # ── the frozen control set ─────────────────────────────────────────────
    def register_frozen_set(
        self, session_id: str, items: list[dict[str, Any]], confirm_share: float | None = None
    ) -> dict[str, Any]:
        """Register the anchor's frozen control set — once, before the first check.
        items: [{"requestId", "prompt": str | [{role, content}], "reference"?}]
        ("reference" is required when the anchor was certified reading one). Returns
        {frozen_set: {control, confirm, confirm_share, registered_at},
        control_ids, confirm_ids}: checks read control_ids; confirm_ids are sealed
        for confirm(). Raises RewardRefused(409) when a set is already registered or
        a check already ran."""
        body: dict[str, Any] = {"items": items}
        if confirm_share is not None:
            body["confirmShare"] = confirm_share
        return self._call("POST", f"/reward/sessions/{session_id}/frozen-set", body)

    def frozen_set(self, session_id: str) -> dict[str, Any]:
        """The registered set: counts and each split's ids. RewardRefused(404) before registration."""
        return self._call("GET", f"/reward/sessions/{session_id}/frozen-set")

    # ── scoring ────────────────────────────────────────────────────────────
    def score(self, session_id: str, items: list[dict[str, Any]], step: str | None = None) -> list[RewardScore]:
        """items: [{requestId, conversation, response}] or, for agentic
        sessions, [{requestId, conversation, steps:[{role, content, source?,
        toolName?}], sessionId?}]. Returns scores in item order; grade None =
        masked (unparseable judge), never zero."""
        body: dict[str, Any] = {"sessionId": session_id, "items": items}
        if step is not None:
            body["step"] = step
        payload = self._call("POST", "/reward/score", body)
        by_id = {s["request_id"]: RewardScore.from_json(s) for s in payload["scores"]}
        return [by_id[i["requestId"]] for i in items]

    # ── anchor ─────────────────────────────────────────────────────────────
    # The anchor route judges every item of the part inside one request, under the
    # route's time budget (800 s; the score route has 300 s): 64-item parts timed out
    # in production once completions grew (live run tc-live-gate-s1, 2026-09-22). The
    # server refuses more than 24; the default is 16. The client's HTTP timeout
    # (timeout_s, default 900) must outlast the anchor route, or a slow part is
    # abandoned client-side while the server still finishes and bills it.
    ANCHOR_PART_MAX = 24
    # A part that fails transiently (HTTP 5xx, timeout, network error) is re-sent up to
    # this many times with these waits between tries. The server reuses the rows it
    # already judged for the open step, so a re-send costs only what never landed.
    ANCHOR_PART_RETRIES = 3
    ANCHOR_PART_BACKOFF_S = (5.0, 15.0, 45.0)

    def anchor(
        self,
        session_id: str,
        items: list[dict[str, Any]],
        step: str,
        part_size: int = 16,
    ) -> dict[str, Any]:
        """Submit the policy's outputs on the session's frozen control set:
        [{"requestId", "response", "rewardScore"?}] — no conversation; errorbar
        reads each prompt from the registered set (register_frozen_set). Returns
        the server's snake_case verdict: {status (continue | review | hold),
        held, state, reason, review_reasons, required_evidence, pinned_step,
        pin, improvement, point, history, counts} — counts per cell, never
        per-item verdicts (those are on anchor_items(), audit scope). `improvement` is this
        checkpoint's paired improvement over the baseline check ({anchor,
        reward}: observed + truth-scale estimate, fixed and look-corrected
        intervals, supported / contradicted); `pin` is the best-supported
        checkpoint ({step, tied_steps, best_step, supported, reason}) — keep it,
        not simply the latest. Raises RewardRefused(409) when the
        session is already held — resume deliberately with resume_session().

        part_size defaults to 16: each part is judged inside ONE request, and a
        reasoning anchor judging full solutions (1–2k tokens) runs ~60–120 s per
        call — 64 items (four waves of 16) outran the 300 s function limit in
        production (2026-09-21). One wave per request stays inside it.

        A set larger than part_size is sent in parts (every part but the last
        with final=false); the check is decided on the last part over all of
        them. Parts are sent in order and never in parallel. For a session
        that watches your own reward, every item carries "rewardScore" (the
        score your reward gave it; None when it failed to score).

        A part that fails transiently (HTTP 5xx, a timeout, a network error) is
        re-sent up to ANCHOR_PART_RETRIES times, waiting ANCHOR_PART_BACKOFF_S
        (5 / 15 / 45 s) between tries. The server keeps the rows it already
        judged for the open step and reuses them, so the re-send pays only for
        the items that never landed (the response carries `reused`). A 4xx
        refusal is never retried. When the retries are spent the attempt is
        reported to the session and the last error raised, as before."""
        if not items:
            raise ValueError("anchor needs at least one item")
        size = max(1, min(int(part_size), self.ANCHOR_PART_MAX))
        parts = [items[i : i + size] for i in range(0, len(items), size)]
        out: dict[str, Any] = {}
        for k, part in enumerate(parts):
            final = k == len(parts) - 1
            try:
                out = self._post_anchor_part(session_id, step, part, final, k + 1, len(parts))
            except Exception as exc:
                # A check that cannot complete must not leave the gate looking quiet:
                # tell the session why, then raise as before. The trainer decides what
                # to do about it; the platform's job is to stop claiming it is watching.
                self.report_anchor_attempt(session_id, step, f"{type(exc).__name__}: {exc}")
                raise
        return out

    @staticmethod
    def _is_transient(exc: BaseException) -> bool:
        """Worth a re-send: the server or the network failed, not the request."""
        if isinstance(exc, RewardRefused):
            return exc.status >= 500
        return isinstance(exc, requests.RequestException)

    def _post_anchor_part(self, session_id: str, step: str, part: list[dict[str, Any]], final: bool, index: int, count: int) -> dict[str, Any]:
        body = {"step": step, "items": part, "final": final}
        path = f"/reward/sessions/{session_id}/anchor"
        tries = 1 + self.ANCHOR_PART_RETRIES
        for attempt in range(1, tries + 1):
            try:
                out = self._call("POST", path, body)
            except Exception as exc:  # noqa: BLE001 — classified below
                if attempt >= tries or not self._is_transient(exc):
                    raise
                wait = self.ANCHOR_PART_BACKOFF_S[min(attempt - 1, len(self.ANCHOR_PART_BACKOFF_S) - 1)]
                print(
                    f"errorbar anchor step {step}: part {index}/{count} failed ({type(exc).__name__}: {exc}) — "
                    f"re-sending in {wait:g}s (try {attempt + 1}/{tries}); items already judged are reused server-side",
                    flush=True,
                )
                self._sleep(wait)
                continue
            reused = out.get("reused") if isinstance(out, dict) else None
            if attempt > 1 and reused:
                print(f"errorbar anchor step {step}: part {index}/{count} landed on try {attempt}; {reused} items reused, not re-billed", flush=True)
            return out
        raise RuntimeError("unreachable")  # pragma: no cover

    def report_anchor_attempt(self, session_id: str, step: str, error: str) -> dict[str, Any] | None:
        """Tell the session about a check that could not complete (a timeout, a
        refusal, a crashed part). Spends nothing and decides nothing: it moves the
        session's measurement state to retrying/unavailable so a gate that has gone
        blind says so. Best effort — it never raises, because it is called from the
        failure path of something that already went wrong."""
        try:
            return self._call(
                "POST",
                f"/reward/sessions/{session_id}/anchor/attempt",
                {"step": str(step), "error": str(error)[:2000]},
            )
        except Exception:
            return None

    def confirm(
        self,
        session_id: str,
        baseline_step: str,
        candidate_step: str,
        prompts: list[dict[str, Any]] | list[str],
        baseline_responses: list[str],
        candidate_responses: list[str],
        part_size: int = 16,
        candidate_weights_sha256: str | None = None,
        baseline_weights_sha256: str | None = None,
    ) -> dict[str, Any]:
        """The end-of-run Improvement Record on a FRESH sample: the session's
        sealed confirm split (frozen_set()["confirm_ids"]; request ids, or dicts
        with "requestId"), with the baseline's and the chosen checkpoint's output
        for each, in the same order. errorbar reads each prompt from its copy. The anchor
        judges both; returns {confirmation: {verdict: SUPPORTED | INSUFFICIENT |
        CONTRADICTED, n, observed, corrected, additional_pairs, reason, ...}}.
        Sent in parts of part_size prompts (both arms of a prompt stay in one
        part). Raises RewardRefused(422) if a prompt was used in a check, or if
        the sealed split is not covered; RewardRefused(409) for a candidate
        already confirmed (one decision per candidate). The answer's `document`
        is the signed Improvement Record — it binds candidate_weights_sha256
        (sha256 of the checkpoint's weights) when given; verify it with
        POST /v1/verify. confirmations() lists every signed record."""
        if not prompts:
            raise ValueError("confirm needs at least one prompt")
        if not (len(prompts) == len(baseline_responses) == len(candidate_responses)):
            raise ValueError("prompts, baseline_responses and candidate_responses must have the same length")
        size = max(1, min(int(part_size), self.ANCHOR_PART_MAX // 2))
        out: dict[str, Any] = {}
        for start in range(0, len(prompts), size):
            chunk = range(start, min(start + size, len(prompts)))
            items = []
            for i in chunk:
                p = prompts[i]
                rid = p if isinstance(p, str) else p["requestId"]
                items.append({"requestId": rid, "arm": "baseline", "response": baseline_responses[i]})
                items.append({"requestId": rid, "arm": "candidate", "response": candidate_responses[i]})
            final = chunk.stop >= len(prompts)
            body: dict[str, Any] = {"baselineStep": baseline_step, "candidateStep": candidate_step, "items": items, "final": final}
            if final and candidate_weights_sha256:
                body["candidateWeightsSha256"] = candidate_weights_sha256
            if final and baseline_weights_sha256:
                body["baselineWeightsSha256"] = baseline_weights_sha256
            out = self._call("POST", f"/reward/sessions/{session_id}/confirm", body)
        return out

    def confirmations(self, session_id: str) -> list[dict[str, Any]]:
        """Every signed Improvement Record of the session (exact keys, verifiable)."""
        return self._call("GET", f"/reward/sessions/{session_id}/confirm")["documents"]

    def review(self, session_id: str, step: str, verdicts: dict[str, str]) -> dict[str, Any]:
        """Human verdicts on a hold's review sample ({request_id: "pass" |
        "fail"}; the ids are on anchor_items(session_id, step)["required_evidence"],
        which needs an audit-scoped key — the check answer only counts them).
        Returns the reading: {conclusion: confirmed | refuted | inconclusive,
        anchor_right, reviewed, anchor_right_ci, reading, ...}. Never resumes
        the run — that stays resume_session()."""
        body = {"step": step, "verdicts": [{"request_id": k, "verdict": v} for k, v in verdicts.items()]}
        return self._call("POST", f"/reward/sessions/{session_id}/anchor/review", body)

    def preflight(self, session_id: str, group_size: int, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Qualify the reward before training, on the step-0 policy's rollouts:
        items [{"promptId", "rewardScore", "truth": "pass" | "fail" | None}]
        (several rollouts per prompt; rewardScore as your reward gave it, or the
        grade /score returned). Returns {preflight: {reward: {leniency,
        strictness, trust}, solvability, leakage, gates, reading}}. Spends nothing."""
        return self._call("POST", f"/reward/sessions/{session_id}/preflight", {"groupSize": int(group_size), "items": items})

    def set_mode(self, session_id: str, mode: str) -> dict[str, Any]:
        """Shadow ↔ active. From the next check; never retroactive. Admin key."""
        if mode not in ("shadow", "active"):
            raise ValueError('mode must be "shadow" or "active"')
        return self._call("POST", f"/reward/sessions/{session_id}/mode", {"mode": mode})

    def scorecard(self, session_id: str) -> dict[str, Any]:
        """The shadow scorecard: every shadow hold episode, the outcomes recorded,
        the alarms' precision with its interval, and a reading (misses are not
        measured)."""
        return self._call("GET", f"/reward/sessions/{session_id}/scorecard")

    def record_outcome(self, session_id: str, step: str, outcome: str, note: str | None = None) -> dict[str, Any]:
        """What your own evidence shows for the shadow hold that began at `step`:
        "real_problem" | "false_alarm" | "unknown". Returns the updated scorecard."""
        if outcome not in ("real_problem", "false_alarm", "unknown"):
            raise ValueError('outcome must be "real_problem", "false_alarm" or "unknown"')
        body: dict[str, Any] = {"step": str(step), "outcome": outcome}
        if note:
            body["note"] = note
        return self._call("POST", f"/reward/sessions/{session_id}/scorecard", body)

    def anchor_truth(self, session_id: str, step: str, labels: dict[str, str]) -> dict[str, Any]:
        """Your truth on a check's outputs ({request_id: "pass" | "fail"}) — the answer
        to the check's `transport_sample` (a random sample of its outputs). The
        anchor is re-measured on that checkpoint; returns {labelled, ignored,
        transport: {status: holds | drifted | underpowered | unmeasured, ...}}."""
        body = {"step": str(step), "labels": [{"requestId": k, "verdict": v} for k, v in labels.items()]}
        return self._call("POST", f"/reward/sessions/{session_id}/anchor/truth", body)

    def anchor_items(self, session_id: str, step: str | None = None) -> dict[str, Any]:
        """The anchor's per-item verdicts at a check, for human reviewers: {items:
        [{request_id, reward_grade, anchor_verdict, kind, measurements}],
        required_evidence (with the sampled ids), review}. Without step: the
        checked steps. Needs a key with the `audit:read` scope — keep it off the
        trainer's key, or the policy's operator can train against the anchor."""
        q = f"?{urlencode({'step': step})}" if step is not None else ""
        return self._call("GET", f"/reward/sessions/{session_id}/anchor/items{q}")

    def anchor_report(self, session_id: str) -> dict[str, Any]:
        """The anchor report — the study's three-line chart for this run, as a
        document: every stored check ({step, n, reward_rate, anchor_rate,
        anchor_ci, ...}), the holds ({step, at, reason, pinned_step}), the
        current hold state, the rule the session decides under, a summary,
        and the anchor judge's live certificate. Nothing is recomputed from
        rollouts. Print it with `python -m errorbar_reward.anchor_report <id>`."""
        return self._call("GET", f"/reward/sessions/{session_id}/anchor")

    def recalibrate_anchor(self, session_id: str, label_set: bool = True) -> dict[str, Any]:
        """Close the loop after a hold: turn this run's own evidence — your people's
        review verdicts and any groundTruth you sent with checks — into labels on the
        anchor criterion, on THIS policy's outputs (the population its certificate
        could not speak for).

        Writes nothing over a label you already have and adopts nothing: the answer
        names the three steps that follow (align, re-certify, resume), because
        pointing a judge at a new population is a human act."""
        return self._call(
            "POST",
            f"/reward/sessions/{session_id}/anchor/recalibrate",
            {"labelSet": bool(label_set)},
        )

    def resume_session(self, session_id: str, changed: str = "none", note: str | None = None) -> RewardSession:
        """Clear a hold, saying what changed: "reward" / "environment" (the hold
        rule restarts from the next check; the improvement over the original
        baseline keeps counting), "anchor" (refused 409 unless the anchor was
        re-certified since the hold; everything restarts) or "none" (the next
        check replays the same evidence). raw["resume"] carries the server's note."""
        if changed not in ("reward", "environment", "anchor", "none"):
            raise ValueError('changed must be "reward", "environment", "anchor" or "none"')
        body: dict[str, Any] = {"changed": changed}
        if note:
            body["note"] = note
        return RewardSession.from_json(self._call("POST", f"/reward/sessions/{session_id}/resume", body))

    # ── exports ────────────────────────────────────────────────────────────
    def export_session(
        self,
        session_id: str,
        fmt: str = "rl",
        path: str | None = None,
        step: str | None = None,
        min_grade: float | None = None,
    ) -> list[dict[str, Any]]:
        """The graded trajectories this session scored, as JSONL rows.

        fmt: "rl" (prompt / completion / reward per scored item), "sft"
        (passing winners as prompt / completion lines), "pairwise" (chosen /
        rejected pairs from the same prompt within a step). Capped server-side
        (X-Errorbar-Export-Capped) — pass step= to narrow. Writes the raw JSONL
        to path when given.
        """
        q: dict[str, str] = {"format": fmt}
        if step:
            q["step"] = step
        if min_grade is not None:
            q["min_grade"] = str(min_grade)
        text = self._call_text("GET", f"/reward/sessions/{session_id}/export?{urlencode(q)}")
        if path:
            Path(path).write_text(text)
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def environment(self, session_id: str) -> dict[str, Any]:
        """The signed environment bundle: reward spec, judge certificates,
        assertions, declared tools, tasks and adapter snippets."""
        return self._call("GET", f"/reward/sessions/{session_id}/environment")

    def export_tasks(self, session_id: str, path: str) -> int:
        """tasks.jsonl for rl/omnia_grpo/tasks.py (or any OpenEnv-style loader). Returns the row count."""
        text = self._call_text("GET", f"/reward/sessions/{session_id}/environment?format=tasks")
        Path(path).write_text(text)
        return sum(1 for line in text.splitlines() if line.strip())
