"""errorbar_reward.ac2 — the AC2 grader adapter against a fake SDK and a fake HTTP layer.

The ac2 SDK is private. A structurally identical `ac2.runtime` shim (Grader,
GraderOutput, Message, Trace) is injected via sys.modules BEFORE the adapter is
imported, so the module resolves its base class through the documented import
path exactly as it would inside an AC2 project.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── fake ac2.runtime ──────────────────────────────────────────────────────────
def _install_ac2_shim() -> types.ModuleType:
    ac2 = types.ModuleType("ac2")
    runtime = types.ModuleType("ac2.runtime")

    @dataclass
    class GraderOutput:
        score: float
        reasoning: str

    class Grader:
        """Mirrors the documented surface: subclasses implement
        `async _grade(grader_params, trace, env) -> GraderOutput`."""

        async def grade(self, grader_params, trace, env=None):
            return await self._grade(grader_params, trace, env)

    @dataclass
    class Message:
        role: str
        content: Any
        tool_calls: Any = None
        name: str | None = None

    @dataclass
    class Episode:
        items: list = field(default_factory=list)

        def get_items(self):
            return list(self.items)

    Trace = list  # "trace is a list of episodes; trace[-1].get_items() yields Message items"

    runtime.Grader, runtime.GraderOutput, runtime.Message, runtime.Episode, runtime.Trace = Grader, GraderOutput, Message, Episode, Trace
    ac2.runtime = runtime
    sys.modules["ac2"] = ac2
    sys.modules["ac2.runtime"] = runtime
    return runtime


rt = _install_ac2_shim()
sys.modules.pop("errorbar_reward.ac2", None)  # make sure the import below binds to the shim

from errorbar_reward import ErrorbarRewardClient, RewardRefused  # noqa: E402
from errorbar_reward import ac2 as adapter  # noqa: E402
from errorbar_reward.ac2 import (  # noqa: E402
    Ac2GraderConfig,
    ErrorbarAnchorGrader,
    ErrorbarGrader,
    MASKED_REASONING_PREFIX,
    MASKED_SCORE,
    request_id,
    resolve_step,
    view_trace,
)


# ── fake HTTP ─────────────────────────────────────────────────────────────────
class FakeResponse:
    def __init__(self, status: int, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeHttp:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.routes: dict[tuple[str, str], Any] = {}

    def __call__(self, method, url, data=None, headers=None, timeout=None):
        path = url.split("/v1", 1)[1]
        call = {"method": method, "path": path, "body": json.loads(data) if data else None, "headers": headers}
        self.calls.append(call)
        handler = self.routes.get((method, path))
        if handler is None:
            return FakeResponse(404, {"error": {"message": "Reward session not found"}})
        return handler(call)


def ok_scores(grade=0.8, verdict="pass", **extra):
    def handler(call):
        return FakeResponse(
            200,
            {
                "session_id": call["body"]["sessionId"],
                "scores": [{"request_id": i["requestId"], "grade": grade, "verdict": verdict, **extra} for i in call["body"]["items"]],
                "spend_micros": "1200",
                "criteria": [{"id": "crit_1", "name": "Answer is correct"}],
            },
        )

    return handler


def make_client(http):
    return ErrorbarRewardClient(api_key="sk_test", base_url="https://x.test/v1", http=http)


def cfg(**over) -> Ac2GraderConfig:
    base = dict(api_key="sk_test", session_id="sess_train", base_url="https://x.test/v1")
    base.update(over)
    return Ac2GraderConfig(**base)


def trace_of(*messages) -> list:
    """One episode holding Message items, the SubstringGrader-example shape."""
    return [rt.Episode(items=[rt.Message(*m) if isinstance(m, tuple) else m for m in messages])]


SIMPLE = trace_of(("system", "Answer in one word."), ("user", "Capital of France?"), ("assistant", "Paris"))


def run(coro):
    return asyncio.run(coro)


# ── tests ─────────────────────────────────────────────────────────────────────
def test_module_binds_to_the_sdk_base_classes():
    assert issubclass(ErrorbarGrader, rt.Grader) and adapter.GraderOutput is rt.GraderOutput


def test_happy_path_builds_the_documented_request_body():
    http = FakeHttp()
    http.routes[("POST", "/reward/score")] = ok_scores(0.83, "pass")
    grader = ErrorbarGrader(cfg(), client=make_client(http))
    out = run(grader.grade({"expected": "Paris", "step": 12}, SIMPLE, env=None))

    assert isinstance(out, rt.GraderOutput)
    assert out.score == pytest.approx(0.83)
    assert "verdict=pass" in out.reasoning and "grade=0.8300" in out.reasoning

    body = http.calls[0]["body"]
    assert http.calls[0]["headers"]["Authorization"] == "Bearer sk_test"
    assert body["sessionId"] == "sess_train" and body["step"] == "12"
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["conversation"] == "system: Answer in one word.\nuser: Capital of France?"
    assert item["response"] == "Paris"
    assert "steps" not in item
    assert item["requestId"] == request_id("sess_train", "12", item["conversation"], "Paris")
    assert item["requestId"].startswith("ac2-sess_train-12-")
    # grader_params content never leaves: only the step key is read from it
    assert "Paris" not in json.dumps({k: v for k, v in body.items() if k != "items"})
    assert "expected" not in json.dumps(body)


def test_request_id_is_deterministic_and_distinguishes_rollouts():
    a = request_id("sess", "3", "user: q", "answer one")
    assert a == request_id("sess", "3", "user: q", "answer one")
    assert a != request_id("sess", "3", "user: q", "answer two")  # two rollouts of one prompt in a group
    assert a != request_id("sess", "4", "user: q", "answer one")  # a different step re-meters
    assert a != request_id("other", "3", "user: q", "answer one")
    assert len(a) <= 200
    # prompt-only ids (the anchor's frozen set) are stable across responses
    assert request_id("sess", "3", "user: q", None) == request_id("sess", "3", "user: q", None)

    http = FakeHttp()
    http.routes[("POST", "/reward/score")] = ok_scores()
    grader = ErrorbarGrader(cfg(), client=make_client(http))
    run(grader.grade({}, SIMPLE, None))
    run(grader.grade({}, SIMPLE, None))
    assert http.calls[0]["body"]["items"][0]["requestId"] == http.calls[1]["body"]["items"][0]["requestId"]
    assert "step" not in http.calls[0]["body"]  # no step anywhere → server default namespace


def test_409_hold_returns_zero_with_hold_reasoning(caplog):
    http = FakeHttp()
    http.routes[("POST", "/reward/score")] = lambda c: FakeResponse(
        409, {"error": {"message": "Run is held by its anchor: reward rising without the anchor following", "code": "held"}}
    )
    grader = ErrorbarGrader(cfg(), client=make_client(http))
    with caplog.at_level("WARNING", logger="errorbar_reward.ac2"):
        out = run(grader.grade({"step": "40"}, SIMPLE, None))
    assert out.score == 0.0
    assert out.reasoning.startswith("errorbar anchor hold at step 40 — training reward withheld")
    assert any("held" in r.message for r in caplog.records)


def test_masked_null_grade_returns_sentinel_with_flagged_reasoning():
    http = FakeHttp()
    http.routes[("POST", "/reward/score")] = ok_scores(None, "unparsed")
    grader = ErrorbarGrader(cfg(), client=make_client(http))
    out = run(grader.grade({}, SIMPLE, None))
    assert out.score == MASKED_SCORE == 0.0
    assert MASKED_REASONING_PREFIX in out.reasoning and "grade=null" in out.reasoning

    strict = ErrorbarGrader(cfg(masked_policy="raise"), client=make_client(http))
    with pytest.raises(ValueError, match="masked"):
        run(strict.grade({}, SIMPLE, None))


def test_402_and_network_errors_raise_never_zero():
    http = FakeHttp()
    http.routes[("POST", "/reward/score")] = lambda c: FakeResponse(402, {"error": {"message": "Run budget exhausted", "code": "budget_exhausted"}})
    grader = ErrorbarGrader(cfg(), client=make_client(http))
    with pytest.raises(RewardRefused) as e:
        run(grader.grade({}, SIMPLE, None))
    assert e.value.status == 402 and "ERRORBAR_SESSION_ID" in str(e.value)

    import requests

    def down(*a, **k):
        raise requests.ConnectionError("boom")

    grader = ErrorbarGrader(cfg(), client=make_client(down))
    with pytest.raises(RuntimeError, match="unreachable"):
        run(grader.grade({}, SIMPLE, None))


def test_flags_surface_in_reasoning():
    http = FakeHttp()
    http.routes[("POST", "/reward/score")] = ok_scores(0.5, "fail", exec_unverified=True, attested=False, sim_fraction=0.1)
    out = run(ErrorbarGrader(cfg(), client=make_client(http)).grade({}, SIMPLE, None))
    assert "execUnverified=true" in out.reasoning and "attested=false" in out.reasoning and "simFraction=0.1" in out.reasoning
    assert "verdict=fail" in out.reasoning


def test_agentic_trace_sends_steps_with_tool_turns():
    trace = trace_of(
        ("system", "You can run code."),
        ("user", "What is 6*7?"),
        rt.Message(role="assistant", content="", tool_calls=[{"name": "python", "arguments": "6*7"}]),
        rt.Message(role="tool", content="42", name="python"),
        ("assistant", "42"),
    )
    http = FakeHttp()
    http.routes[("POST", "/reward/score")] = ok_scores(1.0)
    run(ErrorbarGrader(cfg(agentic=True), client=make_client(http)).grade({}, trace, None))
    item = http.calls[0]["body"]["items"][0]
    assert item["conversation"] == "system: You can run code.\nuser: What is 6*7?"
    assert [s["role"] for s in item["steps"]] == ["assistant", "tool", "assistant"]
    assert item["steps"][1]["toolName"] == "python" and "python" in item["steps"][0]["content"]
    assert "response" not in item

    # non-agentic view of the same trace: everything before the LAST assistant turn is the prompt
    v = view_trace(trace, agentic=False)
    assert v.response == "42" and v.conversation.splitlines()[-1] == "tool: 42"


def test_content_parts_and_non_message_items_are_handled():
    trace = trace_of(
        ("user", [{"type": "text", "text": "hello"}, {"type": "image", "url": "x"}]),
        object(),  # an episode item that is not a Message
        ("assistant", "hi"),
    )
    v = view_trace(trace)
    assert v.conversation.startswith("user: hello\n") and v.response == "hi"
    with pytest.raises(ValueError, match="no assistant message"):
        view_trace(trace_of(("user", "q")))


def test_resolve_step_order_and_sanitising():
    assert resolve_step({"step": 7}, None, None) == "7"
    assert resolve_step({"errorbar_step": "s.12"}, None, "99") == "s.12"
    assert resolve_step({}, None, "99") == "99"
    assert resolve_step({"step": "bad step!"}, None, None) is None  # fails the server's tag rule

    class Ep:
        training_step = 5.0

    assert resolve_step(None, [Ep()], None) == "5"
    assert resolve_step(None, [], None) is None


def test_config_from_env(monkeypatch):
    for k in ("ERRORBAR_API_KEY", "ERRORBAR_SESSION_ID", "ERRORBAR_ANCHOR_SESSION_ID", "ERRORBAR_ANCHOR_SET_SIZE", "ERRORBAR_STEP"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError, match="ERRORBAR_API_KEY"):
        Ac2GraderConfig.from_env()
    env = {"ERRORBAR_API_KEY": "k", "ERRORBAR_SESSION_ID": "s", "ERRORBAR_STEP": "20", "ERRORBAR_AGENTIC": "1", "ERRORBAR_ANCHOR_SET_SIZE": "30"}
    c = Ac2GraderConfig.from_env(env)
    assert (c.step, c.agentic, c.anchor_set_size, c.masked_policy) == ("20", True, 30, "zero")
    with pytest.raises(ValueError, match="ERRORBAR_ANCHOR_SESSION_ID"):
        Ac2GraderConfig.from_env(env, need_anchor=True)
    with pytest.raises(ValueError, match="1..2000"):
        Ac2GraderConfig.from_env({**env, "ERRORBAR_ANCHOR_SET_SIZE": "2001"})  # a frozen set larger than one part is fine; the client splits it


# ── anchor grader ─────────────────────────────────────────────────────────────
def anchor_ok(held=False):
    def handler(call):
        n = len(call["body"]["items"])
        return FakeResponse(
            200,
            {
                "session_id": "sess_train",
                "held": held,
                "reason": "reward rising without the anchor following" if held else "within tolerance",
                "pinned_step": "10" if held else None,
                "point": {"step": call["body"]["step"], "n": n, "reward_rate": 0.7, "anchor_rate": 0.6, "anchor_ci": [0.4, 0.8], "masked": 0},
                "underpowered": None if n >= 20 else f"Anchor needs at least 20 gradable prompts to hold a run (got {n}).",
                "history": [],
                "items": [
                    {"request_id": it["requestId"], "reward_grade": 1.0 if i % 3 else 0.0, "anchor_verdict": "fail" if i % 3 == 1 else "pass",
                     "kind": "seam" if i % 3 == 1 else ("false_fail" if i % 3 == 0 else "agree")}
                    for i, it in enumerate(call["body"]["items"])
                ],
                "counts": {"seam": sum(1 for i in range(n) if i % 3 == 1), "false_fail": sum(1 for i in range(n) if i % 3 == 0), "agree": sum(1 for i in range(n) if i % 3 == 2), "masked": 0},
                "spend_micros": "4000",
            },
        )

    return handler


def frozen_routes(http):
    """The session's frozen-set routes: 404 until registered, then the registered ids."""
    state: dict[str, Any] = {}

    def register(call):
        state["items"] = call["body"]["items"]
        state["share"] = call["body"].get("confirmShare")
        ids = [i["requestId"] for i in state["items"]]
        return FakeResponse(200, {"frozen_set": {"control": len(ids), "confirm": 0}, "control_ids": ids, "confirm_ids": []})

    def read(call):
        if "items" not in state:
            return FakeResponse(404, {"error": {"message": "This session has no frozen set yet"}})
        return FakeResponse(200, {"control_ids": [i["requestId"] for i in state["items"]], "confirm_ids": []})

    http.routes[("POST", "/reward/sessions/sess_train/frozen-set")] = register
    http.routes[("GET", "/reward/sessions/sess_train/frozen-set")] = read
    return state


def prompts(n):
    return [trace_of(("user", f"question {i}"), ("assistant", f"answer {i}")) for i in range(n)]


def test_anchor_grader_scores_with_anchor_session_and_posts_frozen_set_with_step():
    http = FakeHttp()
    http.routes[("POST", "/reward/score")] = ok_scores(0.6, "fail")
    http.routes[("POST", "/reward/sessions/sess_train/anchor")] = anchor_ok()
    frozen = frozen_routes(http)
    grader = ErrorbarAnchorGrader(cfg(anchor_session_id="sess_anchor", anchor_set_size=3), client=make_client(http))

    outs = [run(grader.grade({}, t, None)) for t in prompts(3)]  # regrade job: no env, grader_params == {}
    assert all(o.score == pytest.approx(0.6) for o in outs)
    assert not any(c["path"].endswith("/anchor") for c in http.calls), "no step known → score only, never submit"

    # step from grader_params (or ERRORBAR_STEP): the third trace completes the set and triggers ONE anchor POST
    outs = [run(grader.grade({"step": 40}, t, None)) for t in prompts(3)]
    score_calls = [c for c in http.calls if c["path"] == "/reward/score"]
    anchor_calls = [c for c in http.calls if c["path"] == "/reward/sessions/sess_train/anchor"]
    assert all(c["body"]["sessionId"] == "sess_anchor" for c in score_calls), "second column scores on the anchor session"
    assert len(anchor_calls) == 1
    body = anchor_calls[0]["body"]
    assert body["step"] == "40" and len(body["items"]) == 3
    # errorbar holds the frozen set: the first check's prompts were registered first (all control).
    assert frozen["share"] == 0.0
    assert frozen["items"][0] == {"requestId": request_id("sess_train", "anchor", "user: question 0", None), "prompt": [{"role": "user", "content": "question 0"}]}
    assert body["items"][0] == {
        "requestId": request_id("sess_train", "anchor", "user: question 0", None),
        "conversation": "user: question 0",
        "response": "answer 0",
    }
    assert grader.anchor_results[0]["step"] == "40" and grader.anchor_results[0]["held"] is False
    assert grader._buffers == {}
    # No seam list on the trainer's side: per-item anchor verdicts stay on the server.
    assert not hasattr(grader, "seam_items") and not hasattr(grader, "write_seam")

    # the frozen set carries the SAME request ids at the next step's check
    [run(grader.grade({"step": 50}, t, None)) for t in prompts(3)]
    # — the WHOLE id, not only its digest: the server pairs reads by request id.
    ids_40 = [i["requestId"] for i in body["items"]]
    ids_50 = [i["requestId"] for i in http.calls[-1]["body"]["items"]]
    assert ids_40 == ids_50
    assert sum(1 for c in http.calls if c["path"].endswith("/frozen-set") and c["method"] == "POST") == 1, "registered once"


def test_anchor_grader_reads_back_a_registered_set_after_a_restart():
    http = FakeHttp()
    http.routes[("POST", "/reward/score")] = ok_scores(0.6)
    http.routes[("POST", "/reward/sessions/sess_train/anchor")] = anchor_ok()
    frozen = frozen_routes(http)
    frozen["items"] = [{"requestId": request_id("sess_train", "anchor", f"user: question {i}", None)} for i in range(2)]
    grader = ErrorbarAnchorGrader(cfg(anchor_session_id="sess_anchor", anchor_set_size=3), client=make_client(http))
    [run(grader.grade({"step": 60}, t, None)) for t in prompts(3)]
    assert not any(c["method"] == "POST" and c["path"].endswith("/frozen-set") for c in http.calls)
    sent = [c for c in http.calls if c["path"].endswith("/anchor")][0]["body"]["items"]
    assert len(sent) == 2, "a prompt outside the registered set is not sent"


def test_anchor_grader_flush_and_held_handling(caplog):
    http = FakeHttp()
    http.routes[("POST", "/reward/score")] = ok_scores(0.9)
    http.routes[("POST", "/reward/sessions/sess_train/anchor")] = anchor_ok(held=True)
    frozen_routes(http)
    grader = ErrorbarAnchorGrader(cfg(anchor_session_id="sess_anchor", anchor_set_size=10), client=make_client(http))
    [run(grader.grade({"step": 8}, t, None)) for t in prompts(2)]
    assert not any(c["path"].endswith("/anchor") for c in http.calls)
    with caplog.at_level("WARNING", logger="errorbar_reward.ac2"):
        verdicts = grader.flush_anchor()  # partial set, explicit flush
    assert len(verdicts) == 1 and verdicts[0]["held"] is True and verdicts[0]["pinned_step"] == "10"
    assert any("HELD" in r.message for r in caplog.records)

    # once the session is held, the anchor route answers 409: logged, not raised; scoring on the anchor session continues
    http.routes[("POST", "/reward/sessions/sess_train/anchor")] = lambda c: FakeResponse(409, {"error": {"message": "Session is held: anchor divergence", "code": "held"}})
    grader2 = ErrorbarAnchorGrader(cfg(anchor_session_id="sess_anchor", anchor_set_size=1), client=make_client(http))
    out = run(grader2.grade({"step": 9}, prompts(1)[0], None))
    assert out.score == pytest.approx(0.9) and grader2.anchor_results == []


def test_anchor_grader_submits_each_step_once():
    # Once a step's frozen set has been POSTed, more traces for that step must not
    # start a second batch: a second final:true check for the same step would be a
    # second history point in the session (2026-09-23 audit).
    http = FakeHttp()
    http.routes[("POST", "/reward/score")] = ok_scores(0.7)
    http.routes[("POST", "/reward/sessions/sess_train/anchor")] = anchor_ok()
    frozen_routes(http)
    grader = ErrorbarAnchorGrader(cfg(anchor_session_id="sess_anchor", anchor_set_size=2), client=make_client(http))
    [run(grader.grade({"step": 12}, t, None)) for t in prompts(5)]
    anchor_calls = [c for c in http.calls if c["path"] == "/reward/sessions/sess_train/anchor"]
    assert len(anchor_calls) == 1, "the set submits once; later traces for the step are ignored"
    assert grader.flush_anchor() == [], "nothing buffered for a submitted step"
    # a new step starts a new buffer
    [run(grader.grade({"step": 13}, t, None)) for t in prompts(2)]
    anchor_calls = [c for c in http.calls if c["path"] == "/reward/sessions/sess_train/anchor"]
    assert [c["body"].get("step") for c in anchor_calls] == ["12", "13"]


def test_anchor_grader_requires_anchor_session():
    with pytest.raises(ValueError, match="ERRORBAR_ANCHOR_SESSION_ID"):
        ErrorbarAnchorGrader(cfg(), client=make_client(FakeHttp()))
