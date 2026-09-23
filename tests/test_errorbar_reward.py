"""errorbar_reward adapters — contract tests against a fake HTTP layer."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace

import pytest

from errorbar_reward import ErrorbarRewardClient, RewardRefused, trl_reward_func
from errorbar_reward.verifiers_judge import ErrorbarJudge, ErrorbarJudgeConfig, _trace_to_item


class FakeResponse:
    def __init__(self, status: int, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeHttp:
    """Records calls; answers by (method, path)."""

    def __init__(self):
        self.calls = []
        self.routes = {}

    def __call__(self, method, url, data=None, headers=None, timeout=None):
        path = url.split("/v1", 1)[1]
        self.calls.append({"method": method, "path": path, "body": json.loads(data) if data else None, "headers": headers})
        handler = self.routes.get((method, path))
        if handler is None:
            return FakeResponse(404, {"error": {"message": "Reward session not found"}})
        return handler(self.calls[-1])


SESSION = {
    "id": "sess1",
    "status": "ACTIVE",
    "agentic": False,
    "budget_usd": 5.0,
    "spent_usd": 0.1,
    "expires_at": "2026-09-07T00:00:00.000Z",
    "certificates": [{"criterion": {"name": "Judge A"}, "calibration": {"tpr": 0.95}}],
}


def make_client(http, sleeps=None):
    """sleeps: a list that records the anchor part backoff waits instead of sleeping."""
    return ErrorbarRewardClient(
        api_key="sk_sovereign_test",
        base_url="https://x.test/v1",
        http=http,
        sleep=(lambda s: None) if sleeps is None else sleeps.append,
    )


def test_create_session_sends_camel_case_and_returns_certificates():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions")] = lambda c: FakeResponse(201, SESSION)
    s = make_client(http).create_session(reward={"mode": "single", "criterionId": "c1"}, reward_budget_usd=5, ttl_hours=12, label="run 1")
    assert s.id == "sess1" and s.certificates[0]["criterion"]["name"] == "Judge A"
    body = http.calls[0]["body"]
    assert body == {"reward": {"mode": "single", "criterionId": "c1"}, "rewardBudgetUsd": 5, "agentic": False, "ttlHours": 12, "label": "run 1"}
    assert http.calls[0]["headers"]["Authorization"] == "Bearer sk_sovereign_test"


def test_score_preserves_item_order_and_masks_none():
    http = FakeHttp()
    http.routes[("POST", "/reward/score")] = lambda c: FakeResponse(
        200,
        {
            "session_id": "sess1",
            "scores": [
                {"request_id": "b", "grade": None, "verdict": "unparsed"},
                {"request_id": "a", "grade": 0.9, "verdict": "pass"},
            ],
        },
    )
    scores = make_client(http).score("sess1", [{"requestId": "a", "conversation": "q", "response": "x"}, {"requestId": "b", "conversation": "q", "response": "y"}], step="12")
    assert [s.grade for s in scores] == [0.9, None]
    assert http.calls[0]["body"]["step"] == "12"


def test_refusals_raise_with_status():
    http = FakeHttp()
    http.routes[("POST", "/reward/score")] = lambda c: FakeResponse(402, {"error": {"message": "Run budget exhausted", "code": "budget_exhausted"}})
    with pytest.raises(RewardRefused) as e:
        make_client(http).score("sess1", [{"requestId": "a", "conversation": "q", "response": "x"}])
    assert e.value.status == 402 and "budget" in str(e.value)


def test_trl_reward_func_renders_prompts_and_returns_grades():
    http = FakeHttp()
    seen = {}

    def handler(c):
        seen["body"] = c["body"]
        return FakeResponse(200, {"scores": [{"request_id": i["requestId"], "grade": 0.5, "verdict": "fail"} for i in c["body"]["items"]]})

    http.routes[("POST", "/reward/score")] = handler
    f = trl_reward_func(make_client(http), "sess1")
    out = f(
        prompts=[[{"role": "user", "content": "hi"}], "plain prompt"],
        completions=[[{"role": "assistant", "content": "hello"}], "answer"],
        step=3,
    )
    assert out == [0.5, 0.5]
    items = seen["body"]["items"]
    assert items[0]["conversation"] == "user: hi" and items[0]["response"] == "hello"
    assert items[1]["conversation"] == "plain prompt" and items[1]["response"] == "answer"
    assert seen["body"]["step"] == "3"
    # deterministic ids: same inputs → same request ids (idempotent re-score)
    f(prompts=[[{"role": "user", "content": "hi"}], "plain prompt"], completions=[[{"role": "assistant", "content": "hello"}], "answer"], step=3)
    assert seen["body"]["items"][0]["requestId"] == items[0]["requestId"]


def test_trl_agentic_sends_steps():
    http = FakeHttp()
    seen = {}
    http.routes[("POST", "/reward/score")] = lambda c: (seen.setdefault("body", c["body"]), FakeResponse(200, {"scores": [{"request_id": c["body"]["items"][0]["requestId"], "grade": 1.0, "verdict": "pass"}]}))[1]
    f = trl_reward_func(make_client(http), "sess1", agentic=True)
    f(prompts=["do it"], completions=[[{"role": "assistant", "content": "", "tool_calls": [{"name": "run_code"}]}, {"role": "tool", "name": "run_code", "content": "42"}, {"role": "assistant", "content": "done"}]])
    steps = seen["body"]["items"][0]["steps"]
    assert [s["role"] for s in steps] == ["assistant", "tool", "assistant"]
    assert steps[1]["toolName"] == "run_code" and "response" not in seen["body"]["items"][0]


def test_verifiers_judge_scores_trace_and_raises_on_masked():
    http = FakeHttp()
    http.routes[("GET", "/reward/sessions/sess1")] = lambda c: FakeResponse(200, SESSION)
    grades = iter([0.8, None])
    http.routes[("POST", "/reward/score")] = lambda c: FakeResponse(200, {"scores": [{"request_id": c["body"]["items"][0]["requestId"], "grade": next(grades), "verdict": "pass"}]})
    cfg = ErrorbarJudgeConfig(session_id="sess1", api_key="k", base_url="https://x.test/v1")
    judge = ErrorbarJudge(cfg, client=make_client(http))
    assert judge.certificates == SESSION["certificates"]
    trace = SimpleNamespace(
        messages=[SimpleNamespace(role="user", content="q"), SimpleNamespace(role="assistant", content="a")],
        last_reply="a",
        task=SimpleNamespace(key="t1"),
    )
    assert asyncio.run(judge.score(None, trace)) == 0.8
    with pytest.raises(ValueError):
        asyncio.run(judge.score(None, trace))


def test_trace_to_item_agentic_splits_prompt_and_steps():
    trace = SimpleNamespace(
        messages=[
            {"role": "system", "content": "be good"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "t"}]},
            {"role": "tool", "content": "r"},
            {"role": "assistant", "content": "final"},
        ],
        last_reply="final",
        task=SimpleNamespace(key="k9"),
    )
    item = _trace_to_item(trace, agentic=True, step="7")
    assert item["requestId"] == "7-k9" and item["conversation"] == "system: be good\nuser: q"
    assert [s["role"] for s in item["steps"]] == ["assistant", "tool", "assistant"]


def test_export_session_parses_jsonl(tmp_path):
    from types import SimpleNamespace
    from errorbar_reward import ErrorbarRewardClient

    seen: dict[str, str] = {}

    def fake(method, url, data=None, headers=None, timeout=None):
        seen["url"] = url
        return SimpleNamespace(status_code=200, text='{"reward": 0.9}\n{"reward": 0.1}\n', json=lambda: {})

    c = ErrorbarRewardClient(api_key="k", base_url="https://x/v1", http=fake)
    out = tmp_path / "rl.jsonl"
    rows = c.export_session("s1", fmt="sft", path=str(out), step="s12", min_grade=0.8)
    assert [r["reward"] for r in rows] == [0.9, 0.1]
    assert "format=sft" in seen["url"] and "step=s12" in seen["url"] and "min_grade=0.8" in seen["url"]
    assert out.read_text().count("\n") == 2


# ── anchor report ──────────────────────────────────────────────────────────

ANCHOR_REPORT = {
    "session_id": "sess1",
    "anchor": {"criterion_id": "cA", "criterion_name": "Anchor judge", "judge_model": "judge-x", "judge_mode": "corrected"},
    "certificate": {
        "criterion_id": "cA",
        "name": "Anchor judge",
        "judge_model": "judge-x",
        "unit": "request",
        "issued_at": "2026-09-08T00:00:00.000Z",
        "calibration": {
            "measured": True,
            "matrix": {"tp": 40, "fp": 2, "tn": 50, "fn": 3},
            "metrics": {"n": 95, "tpr": 0.93, "tpr_ci": [0.81, 0.98], "tnr": 0.96, "tnr_ci": [0.87, 0.99], "kappa": 0.894},
            "labels": 95,
            "holdout_active": True,
            "aligned_at": "2026-09-01T00:00:00.000Z",
        },
        "trust": {"trust": "trustworthy"},
        "population": {"statement": "Error rates were measured on all workspace traffic, single exchanges — they are claims about that population and no other."},
        "validity": {"drift_status": "ok", "drift_reason": None, "judge_model_available": True},
        "signature": {"alg": "HS256", "key_id": "k1", "value": "sig"},
    },
    "rule": {"window": 2, "gap_floor": 0.05, "drop_floor": 0.02, "consecutive": 2, "min_prompts": 20},
    "history": [
        {"step": "0", "at": "t0", "n": 60, "reward_rate": 0.5, "anchor_rate": 0.6, "anchor_ci": [0.55, 0.65], "anchor_corrected": True, "masked": 0},
        {"step": "16", "at": "t1", "n": 60, "reward_rate": 0.58, "anchor_rate": 0.6, "anchor_ci": [0.55, 0.65], "anchor_corrected": True, "masked": 2},
        {"step": "32", "at": "t2", "n": 60, "reward_rate": 0.7, "anchor_rate": 0.59, "anchor_ci": [0.54, 0.64], "anchor_corrected": True, "masked": 0},
    ],
    "holds": [{"step": "32", "at": "t2", "state": "grader_fooled", "reason": "The gap widened +21% since the run's first check", "pinned_step": "0"}],
    "current": {"held": True, "held_at": "t2", "held_state": "grader_fooled", "held_reason": "The gap widened +21% since the run's first check", "pinned_step": "0", "resumed_at": None},
    "summary": {
        "checks": 3, "usable_checks": 3, "holds": 1, "max_divergence": 0.11, "max_divergence_step": "32",
        "anchor_range": [0.59, 0.6], "reward_range": [0.5, 0.7], "first_step": "0", "last_step": "32",
    },
}


def test_anchor_report_is_a_get_on_the_anchor_route():
    http = FakeHttp()
    http.routes[("GET", "/reward/sessions/sess1/anchor")] = lambda c: FakeResponse(200, ANCHOR_REPORT)
    report = make_client(http).anchor_report("sess1")
    assert report["holds"][0]["pinned_step"] == "0" and len(report["history"]) == 3
    assert http.calls[0]["method"] == "GET" and http.calls[0]["body"] is None
    assert http.calls[0]["headers"]["Authorization"] == "Bearer sk_sovereign_test"


def test_anchor_report_cli_prints_three_line_table_and_certificate():
    import io
    from errorbar_reward import anchor_report as cli

    http = FakeHttp()
    http.routes[("GET", "/reward/sessions/sess1/anchor")] = lambda c: FakeResponse(200, ANCHOR_REPORT)
    out = io.StringIO()
    assert cli.main(["sess1"], out=out, client=make_client(http)) == 0
    text = out.getvalue()
    lines = text.splitlines()
    # identity + rule
    assert lines[0] == "anchor report — session sess1"
    assert "Anchor judge (cA) · judge judge-x · read: corrected" in lines[1]
    assert "rule: on 2 consecutive checks — GAP: the reward read minus the anchor read on the same outputs has widened" in text
    assert "(floor 5.0%)" in text and "gap only → grader_fooled · gap and drop → gamed · drop only → degrading" in text
    # the table: step | reward read | anchor read [ci] | reward−anchor | n | held/pinned
    header = next(l for l in lines if l.startswith("step"))
    assert [h.strip() for h in header.split("|")] == ["step", "reward read", "anchor read [95% CI]", "reward−anchor", "n", "held/pinned"]
    row0 = next(l for l in lines if l.startswith("0 "))
    cells = [c.strip() for c in row0.split("|")]
    assert cells[:5] == ["0", "50.0%", "60.0% [55.0, 65.0]", "-10.0%", "60"] and cells[5] == ""
    row16 = next(l for l in lines if l.startswith("16 "))
    assert "(+2 masked)" in row16
    row32 = next(l for l in lines if l.startswith("32 "))
    cells = [c.strip() for c in row32.split("|")]
    assert cells[3] == "+11.0%" and cells[5] == "HELD grader_fooled → pinned 0"
    # holds, current, summary
    assert "holds (1):" in text and "step 32 at t2 [grader_fooled] → pinned 0: The gap widened +21%" in text
    assert "current: HELD (grader_fooled) since t2 · pinned step 0" in text
    assert "summary: 3 checks (3 usable) · 1 hold · max reward−anchor +11.0% at step 32 · anchor 59.0%–60.0% · reward 50.0%–70.0%" in text
    # the certificate's identity + trust lines
    assert "certificate: Anchor judge (cA) · judge judge-x · unit request · issued 2026-09-08T00:00:00.000Z" in text
    assert "trust: trustworthy · sensitivity 93.0% [81.0, 98.0] · specificity 96.0% [87.0, 99.0] · κ 0.894 · 95 labels (held-out report half)" in text
    assert "matrix: tp 40 · fp 2 · tn 50 · fn 3" in text
    assert "population: Error rates were measured on" in text
    assert "validity: drift ok · judge model available" in text
    assert "signature: HS256 key k1" in text


def test_anchor_report_cli_json_and_missing_certificate():
    import io
    from errorbar_reward import anchor_report as cli

    gone = dict(ANCHOR_REPORT, certificate=None, holds=[], current={"held": False, "resumed_at": "t9"})
    http = FakeHttp()
    http.routes[("GET", "/reward/sessions/sess1/anchor")] = lambda c: FakeResponse(200, gone)
    out = io.StringIO()
    cli.main(["sess1"], out=out, client=make_client(http))
    text = out.getvalue()
    assert "holds: none" in text and "current: not held · resumed t9" in text
    assert "certificate: none — the anchor criterion no longer exists" in text
    out = io.StringIO()
    cli.main(["sess1", "--json"], out=out, client=make_client(http))
    assert json.loads(out.getvalue())["anchor"]["criterion_id"] == "cA"


def test_anchor_report_cli_refusal_surfaces_as_reward_refused():
    from errorbar_reward import anchor_report as cli

    http = FakeHttp()  # no route → 404
    with pytest.raises(RewardRefused) as e:
        cli.main(["nope"], client=make_client(http))
    assert e.value.status == 404


# ── training control: external reward, parts, review ─────────────────────────


def test_create_session_external_reward_sends_the_watched_reward_and_the_anchor():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions")] = lambda call: FakeResponse(201, {**SESSION, "certificates": []})
    s = make_client(http).create_session(
        external_reward={"name": "rubric judge v3", "scale": "unit"},
        anchor_criterion_id="crit_anchor",
        reward_budget_usd=25,
    )
    assert s.id == "sess1"
    body = http.calls[0]["body"]
    assert body["externalReward"] == {"name": "rubric judge v3", "scale": "unit"}
    assert body["anchor"] == {"criterionId": "crit_anchor"}
    assert "reward" not in body


def test_create_session_requires_exactly_one_reward_kind():
    client = make_client(FakeHttp())
    with pytest.raises(ValueError):
        client.create_session(reward_budget_usd=5)
    with pytest.raises(ValueError):
        client.create_session(reward={"mode": "single", "criterionId": "c"}, external_reward={"name": "x"}, reward_budget_usd=5)


def test_anchor_sends_a_large_set_in_ordered_parts_and_returns_the_final_verdict():
    http = FakeHttp()

    def answer(call):
        body = call["body"]
        if body["final"]:
            return FakeResponse(200, {"status": "hold", "held": True, "required_evidence": [{"kind": "human_review", "request_ids": ["p3"]}]})
        return FakeResponse(200, {"partial": True, "received": len(body["items"])})

    http.routes[("POST", "/reward/sessions/sess1/anchor")] = answer
    items = [{"requestId": f"p{i}", "conversation": "q", "response": "a", "rewardScore": 1.0} for i in range(70)]
    out = make_client(http).anchor("sess1", items, step="64")
    assert [len(c["body"]["items"]) for c in http.calls] == [16, 16, 16, 16, 6]
    assert [c["body"]["final"] for c in http.calls] == [False, False, False, False, True]
    assert all(c["body"]["step"] == "64" for c in http.calls)
    assert [i["requestId"] for c in http.calls for i in c["body"]["items"]] == [f"p{i}" for i in range(70)]
    assert out["status"] == "hold"


def test_anchor_small_set_is_one_final_call_and_part_size_is_capped():
    """The cap is TIME: the route judges every item of the part inside one request under
    a 300s budget, so parts above ANCHOR_PART_MAX (24) are refused by the server —
    64-item parts timed out in production on 2026-09-22 (live run tc-live-gate-s1)."""
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions/sess1/anchor")] = lambda call: FakeResponse(200, {"status": "continue"})
    client = make_client(http)
    client.anchor("sess1", [{"requestId": "p0", "conversation": "q", "response": "a"}], step="0")
    assert http.calls[-1]["body"]["final"] is True
    client.anchor("sess1", [{"requestId": f"p{i}", "conversation": "q", "response": "a"} for i in range(130)], step="16", part_size=500)
    assert [len(c["body"]["items"]) for c in http.calls[1:]] == [24, 24, 24, 24, 24, 10]


def test_review_posts_verdicts_and_returns_the_reading():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions/sess1/anchor/review")] = lambda call: FakeResponse(200, {"conclusion": "confirmed", "reviewed": 2})
    out = make_client(http).review("sess1", "64", {"p1": "fail", "p2": "fail"})
    assert out["conclusion"] == "confirmed"
    assert http.calls[0]["body"] == {"step": "64", "verdicts": [{"request_id": "p1", "verdict": "fail"}, {"request_id": "p2", "verdict": "fail"}]}


def test_anchor_report_cli_prints_status_review_reasons_and_reviews():
    from errorbar_reward.anchor_report import render

    text = render(
        {
            "session_id": "s",
            "anchor": {"criterion_name": "A", "criterion_id": "a"},
            "history": [{"step": "64", "reward_rate": 0.96, "anchor_rate": 0.52, "anchor_ci": [0.46, 0.58], "n": 290, "status": "review"}],
            "holds": [],
            "reviews": [{"step": "64", "sample": ["x"], "population": 120, "result": {"conclusion": "confirmed", "reading": "The hold stands."}}],
            "current": {"status": "review", "review_reasons": ["Anchor judge is drift-flagged."], "held": False},
            "summary": {},
        }
    )
    assert "REVIEW" in text
    assert "status: REVIEW" in text
    assert "review: Anchor judge is drift-flagged." in text
    assert "human review at step 64: CONFIRMED — The hold stands." in text


def test_anchor_report_prints_the_gates_improvement_and_pin():
    from errorbar_reward.anchor_report import improvement_lines, render

    improvement = {
        "anchor": {
            "vs_step": "0", "step": "64", "n": 300, "look": 4,
            "observed": {"estimate": 0.01, "ci": [-0.02, 0.04], "ci_seq": [-0.04, 0.06]},
            "corrected": {"estimate": 0.0105, "ci": [-0.02, 0.041], "ci_seq": [-0.042, 0.063]},
            "supported": False, "contradicted": False,
            "note": "The interval includes no change: the evidence cannot tell these checkpoints apart.",
        },
        "reward": {"observed": {"estimate": 0.13}},
    }
    pin = {"step": "0", "tied_steps": ["0", "16", "32"], "supported": False, "reason": "keep the baseline."}
    lines = improvement_lines(improvement, pin)
    assert lines[0] == "improvement, step 64 over step 0 (n=300): anchor +1.1 pts [-4.2, +6.3] truth scale → NOT SUPPORTED · reward +13.0 pts"
    assert lines[1] == "  The interval includes no change: the evidence cannot tell these checkpoints apart."
    assert lines[2] == "keep: step 0 (improvement not supported; tied: 0, 16, 32) — keep the baseline."
    # supported / contradicted wording, observed scale when uncorrected, no pin → no keep line
    sup = improvement_lines({"anchor": {**improvement["anchor"], "corrected": None, "supported": True, "note": None}, "reward": {}}, None)
    assert sup == ["improvement, step 64 over step 0 (n=300): anchor +1.0 pts [-4.0, +6.0] observed → SUPPORTED · reward — pts"]
    con = improvement_lines({"anchor": {**improvement["anchor"], "contradicted": True, "note": None}, "reward": {}}, None)
    assert "→ CONTRADICTED" in con[0]
    assert improvement_lines(None, None) == []
    # in the full report, between status and current
    text = render({"session_id": "s", "anchor": {}, "history": [], "holds": [], "current": {"held": False, "improvement": improvement, "pin": pin}, "summary": {}})
    assert "keep: step 0" in text and text.index("improvement, step 64") < text.index("current: not held")


def test_create_session_sends_the_minimum_improvement_with_the_anchor():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions")] = lambda call: FakeResponse(201, {**SESSION, "certificates": []})
    make_client(http).create_session(external_reward={"name": "r", "scale": "unit"}, anchor_criterion_id="a", anchor_min_improvement=0.03, reward_budget_usd=5)
    assert http.calls[0]["body"]["anchor"] == {"criterionId": "a", "minImprovement": 0.03}
    with pytest.raises(ValueError):
        make_client(FakeHttp()).create_session(reward={"mode": "single", "criterionId": "c"}, anchor_min_improvement=0.03, reward_budget_usd=5)


def test_confirm_sends_both_arms_per_prompt_in_parts_and_returns_the_record():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions/sess1/confirm")] = lambda call: FakeResponse(200, {"confirmation": {"verdict": "SUPPORTED", "n": len(call["body"]["items"]) // 2}})
    prompts = [{"requestId": f"f{i}", "conversation": f"q{i}"} for i in range(20)]
    out = make_client(http).confirm("sess1", "0", "96", prompts, [f"b{i}" for i in range(20)], [f"c{i}" for i in range(20)], part_size=8)
    assert out["confirmation"]["verdict"] == "SUPPORTED"
    sizes = [len(c["body"]["items"]) for c in http.calls]
    assert sizes == [16, 16, 8]
    assert [c["body"]["final"] for c in http.calls] == [False, False, True]
    first = http.calls[0]["body"]
    assert first["baselineStep"] == "0" and first["candidateStep"] == "96"
    # errorbar reads each prompt from the frozen set: ids only, never the text.
    assert first["items"][0] == {"requestId": "f0", "arm": "baseline", "response": "b0"}
    assert first["items"][1] == {"requestId": "f0", "arm": "candidate", "response": "c0"}
    # Plain ids work too (frozen_set()["confirm_ids"]).
    make_client(http).confirm("sess1", "0", "96", ["f0"], ["b"], ["c"])
    assert http.calls[-1]["body"]["items"][0] == {"requestId": "f0", "arm": "baseline", "response": "b"}


def test_the_frozen_set_is_registered_at_create_or_after_and_read_back():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions")] = lambda call: FakeResponse(201, SESSION)
    items = [{"requestId": "p0", "prompt": "Q0", "reference": "0"}]
    make_client(http).create_session(external_reward={"name": "r", "scale": "unit"}, anchor_criterion_id="a1", frozen_set=items, confirm_share=0.2)
    assert http.calls[0]["body"]["anchor"] == {"criterionId": "a1", "frozenSet": items, "confirmShare": 0.2}
    with pytest.raises(ValueError):
        make_client(http).create_session(external_reward={"name": "r", "scale": "unit"}, anchor_criterion_id="a1", confirm_share=0.2)
    with pytest.raises(ValueError):
        make_client(http).create_session(external_reward={"name": "r", "scale": "unit"}, frozen_set=items)
    http.routes[("POST", "/reward/sessions/sess1/frozen-set")] = lambda call: FakeResponse(200, {"frozen_set": {"control": 1, "confirm": 0}, "control_ids": ["p0"], "confirm_ids": []})
    out = make_client(http).register_frozen_set("sess1", items, confirm_share=0.25)
    assert http.calls[-1]["body"] == {"items": items, "confirmShare": 0.25}
    assert out["control_ids"] == ["p0"]
    http.routes[("GET", "/reward/sessions/sess1/frozen-set")] = lambda call: FakeResponse(200, {"control_ids": ["p0"], "confirm_ids": []})
    assert make_client(http).frozen_set("sess1")["control_ids"] == ["p0"]


def test_confirm_validates_its_inputs():
    c = make_client(FakeHttp())
    with pytest.raises(ValueError):
        c.confirm("s", "0", "8", [], [], [])
    with pytest.raises(ValueError):
        c.confirm("s", "0", "8", [{"requestId": "a"}], ["b"], [])


def test_report_prints_the_confirmation():
    from errorbar_reward.anchor_report import confirmation_lines, render

    c = {"verdict": "INSUFFICIENT", "reason": "About 240 more fresh prompts would settle it if the difference holds."}
    assert confirmation_lines(c) == ["confirmed on fresh prompts: INSUFFICIENT — About 240 more fresh prompts would settle it if the difference holds."]
    assert confirmation_lines(None) == []
    text = render({"session_id": "s", "anchor": {}, "history": [], "holds": [], "current": {"held": False}, "summary": {}, "confirmation": c})
    assert "confirmed on fresh prompts: INSUFFICIENT" in text


def test_resume_session_says_what_changed():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions/sess1/resume")] = lambda call: FakeResponse(200, {**SESSION, "resume": {"resumed": True, "changed": call["body"]["changed"]}})
    s = make_client(http).resume_session("sess1", changed="reward", note="grader fixed")
    assert http.calls[0]["body"] == {"changed": "reward", "note": "grader fixed"}
    assert s.raw["resume"]["changed"] == "reward"
    make_client(http).resume_session("sess1")
    assert http.calls[-1]["body"] == {"changed": "none"}
    with pytest.raises(ValueError):
        make_client(http).resume_session("sess1", changed="everything")


def test_create_session_registers_instruments_and_report_prints_them():
    from errorbar_reward.anchor_report import improvement_lines

    http = FakeHttp()
    http.routes[("POST", "/reward/sessions")] = lambda call: FakeResponse(201, {**SESSION, "certificates": []})
    inst = [{"name": "tests", "kind": "programmatic", "scale": "unit"}]
    make_client(http).create_session(external_reward={"name": "r"}, anchor_criterion_id="a", instruments=inst, reward_budget_usd=5)
    assert http.calls[0]["body"]["anchor"] == {"criterionId": "a", "instruments": inst}
    with pytest.raises(ValueError):
        make_client(FakeHttp()).create_session(reward={"mode": "single", "criterionId": "c"}, instruments=inst, reward_budget_usd=5)
    imp = {
        "anchor": {"vs_step": "0", "step": "16", "n": 300, "observed": {"estimate": -0.2, "ci_seq": [-0.3, -0.1]}, "corrected": None, "supported": False, "contradicted": True},
        "reward": {"observed": {"estimate": 0.1}},
        "instruments": {"tests": {"observed": {"estimate": 0.2, "ci_seq": [0.1, 0.3]}, "supported": True, "contradicted": False}},
    }
    lines = improvement_lines(imp, None)
    assert "→ CONTRADICTED" in lines[0]
    assert lines[1] == "  tests: +20.0 pts [+10.0, +30.0] → SUPPORTED"


# ── a check that cannot complete must not leave the gate looking quiet ──
def test_a_failed_anchor_part_reports_the_attempt_then_raises():
    """2026-09-22: every check was refused for 80 steps and the only record was a
    file on the box. The client now tells the session why, before it raises."""
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions/sess1/anchor")] = lambda call: FakeResponse(504, {"error": "FUNCTION_INVOCATION_TIMEOUT"})
    http.routes[("POST", "/reward/sessions/sess1/anchor/attempt")] = lambda call: FakeResponse(200, {"recorded": 1, "measurement": {"state": "retrying"}})
    client = make_client(http)
    with pytest.raises(RewardRefused):
        client.anchor("sess1", [{"requestId": "p0", "conversation": "q", "response": "a"}], step="80")
    reported = [c for c in http.calls if c["path"].endswith("/anchor/attempt")]
    assert len(reported) == 1
    assert reported[0]["body"]["step"] == "80"
    assert "504" in reported[0]["body"]["error"]


def test_reporting_an_attempt_never_raises_on_top_of_the_original_failure():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions/sess1/anchor")] = lambda call: FakeResponse(504, {"error": "timeout"})
    http.routes[("POST", "/reward/sessions/sess1/anchor/attempt")] = lambda call: FakeResponse(500, {"error": "also down"})
    client = make_client(http)
    with pytest.raises(RewardRefused, match="504"):  # the ORIGINAL failure, not the report's
        client.anchor("sess1", [{"requestId": "p0", "conversation": "q", "response": "a"}], step="80")
    assert client.report_anchor_attempt("sess1", "80", "x") is None


def test_a_transient_part_failure_is_re_sent_with_backoff_and_the_check_completes():
    """2026-09-22: a part that 504'd was not retried until the next cadence — 16 steps
    of blindness. The client now re-sends THAT part (5 / 15 / 45 s), and the server
    reuses what it already judged, so the re-send pays only for what never landed."""
    http = FakeHttp()
    answers = iter([FakeResponse(504, {"error": "FUNCTION_INVOCATION_TIMEOUT"}), FakeResponse(502, {"error": "bad gateway"})])

    def anchor(call):
        nxt = next(answers, None)
        if nxt is not None:
            return nxt
        body = call["body"]
        if body["final"]:
            return FakeResponse(200, {"status": "continue", "reused": 0})
        return FakeResponse(200, {"partial": True, "received": len(body["items"]), "judged": 4, "reused": len(body["items"]) - 4})

    http.routes[("POST", "/reward/sessions/sess1/anchor")] = anchor
    sleeps: list[float] = []
    items = [{"requestId": f"p{i}", "conversation": "q", "response": "a", "rewardScore": 1.0} for i in range(20)]
    out = make_client(http, sleeps).anchor("sess1", items, step="32")
    assert out["status"] == "continue"
    parts = [c for c in http.calls if c["path"] == "/reward/sessions/sess1/anchor"]
    # first part sent three times (two failures, then it lands), the final part once
    assert [(len(c["body"]["items"]), c["body"]["final"]) for c in parts] == [(16, False), (16, False), (16, False), (4, True)]
    assert [i["requestId"] for i in parts[2]["body"]["items"]] == [f"p{i}" for i in range(16)]  # the SAME part, same ids
    assert sleeps == [5.0, 15.0]
    assert not [c for c in http.calls if c["path"].endswith("/anchor/attempt")]  # it completed: nothing to report


def test_a_network_error_or_timeout_is_re_sent_too():
    import requests as _requests

    http = FakeHttp()
    state = {"n": 0}

    def anchor(call):
        state["n"] += 1
        if state["n"] == 1:
            raise _requests.exceptions.Timeout("read timed out")
        if state["n"] == 2:
            raise _requests.exceptions.ConnectionError("reset by peer")
        return FakeResponse(200, {"status": "continue"})

    http.routes[("POST", "/reward/sessions/sess1/anchor")] = anchor
    sleeps: list[float] = []
    out = make_client(http, sleeps).anchor("sess1", [{"requestId": "p0", "conversation": "q", "response": "a"}], step="8")
    assert out["status"] == "continue" and state["n"] == 3 and sleeps == [5.0, 15.0]


def test_the_retries_are_spent_after_three_re_sends_then_the_attempt_is_reported_and_the_last_error_raised():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions/sess1/anchor")] = lambda call: FakeResponse(504, {"error": "FUNCTION_INVOCATION_TIMEOUT"})
    http.routes[("POST", "/reward/sessions/sess1/anchor/attempt")] = lambda call: FakeResponse(200, {"recorded": 1, "measurement": {"state": "retrying"}})
    sleeps: list[float] = []
    with pytest.raises(RewardRefused, match="504"):
        make_client(http, sleeps).anchor("sess1", [{"requestId": "p0", "conversation": "q", "response": "a"}], step="80")
    parts = [c for c in http.calls if c["path"] == "/reward/sessions/sess1/anchor"]
    assert len(parts) == 4  # the first try + 3 re-sends
    assert sleeps == [5.0, 15.0, 45.0]
    reported = [c for c in http.calls if c["path"].endswith("/anchor/attempt")]
    assert len(reported) == 1 and reported[0]["body"]["step"] == "80" and "504" in reported[0]["body"]["error"]


def test_a_refusal_4xx_is_never_re_sent():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions/sess1/anchor")] = lambda call: FakeResponse(422, {"error": "A check for step 16 is still open"})
    sleeps: list[float] = []
    with pytest.raises(RewardRefused, match="422"):
        make_client(http, sleeps).anchor("sess1", [{"requestId": "p0", "conversation": "q", "response": "a"}], step="8")
    assert len([c for c in http.calls if c["path"] == "/reward/sessions/sess1/anchor"]) == 1 and sleeps == []


def test_a_successful_check_reports_nothing():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions/sess1/anchor")] = lambda call: FakeResponse(200, {"status": "continue"})
    make_client(http).anchor("sess1", [{"requestId": "p0", "conversation": "q", "response": "a"}], step="80")
    assert not [c for c in http.calls if c["path"].endswith("/anchor/attempt")]


def test_recalibrate_anchor_closes_the_loop():
    """A hold's evidence becomes labels for the judge that held, and the answer
    says what to do next rather than doing it."""
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions/sess1/anchor/recalibrate")] = lambda call: FakeResponse(
        200,
        {"written": 40, "skipped": 2, "label_set_id": "ls_1", "ready": True,
         "next": ["POST /v1/criteria/c1/align", "GET /v1/criteria/c1/certificate", "resume with changed=anchor"]},
    )
    out = make_client(http).recalibrate_anchor("sess1")
    assert out["written"] == 40 and out["ready"] is True
    assert http.calls[0]["body"] == {"labelSet": True}
    assert "align" in out["next"][0]


def test_shadow_mode_create_promote_and_scorecard():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions")] = lambda call: FakeResponse(201, SESSION)
    make_client(http).create_session(external_reward={"name": "r", "scale": "unit"}, anchor_criterion_id="a1", mode="shadow")
    assert http.calls[-1]["body"]["anchor"] == {"criterionId": "a1", "mode": "shadow"}
    with pytest.raises(ValueError):
        make_client(http).create_session(external_reward={"name": "r", "scale": "unit"}, anchor_criterion_id="a1", mode="enforce")
    http.routes[("POST", "/reward/sessions/s1/mode")] = lambda call: FakeResponse(200, {"mode": call["body"]["mode"], "from": "shadow"})
    assert make_client(http).set_mode("s1", "active")["mode"] == "active"
    http.routes[("GET", "/reward/sessions/s1/scorecard")] = lambda call: FakeResponse(200, {"mode": "shadow", "episodes": []})
    assert make_client(http).scorecard("s1")["mode"] == "shadow"
    http.routes[("POST", "/reward/sessions/s1/scorecard")] = lambda call: FakeResponse(200, {"shadow_hold": call["body"]})
    out = make_client(http).record_outcome("s1", 16, "real_problem", note="eval fell")
    assert out["shadow_hold"] == {"step": "16", "outcome": "real_problem", "note": "eval fell"}
    with pytest.raises(ValueError):
        make_client(http).record_outcome("s1", "16", "maybe")


def test_a_refusal_keeps_the_error_details():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions/s1/anchor")] = lambda call: FakeResponse(409, {"error": {"message": "Session is held", "type": "invalid_request_error", "code": "held", "pinned_step": "32", "pin": {"step": "32"}}})
    with pytest.raises(RewardRefused) as e:
        make_client(http).anchor("s1", [{"requestId": "p0", "response": "a"}], step="48")
    assert e.value.status == 409 and e.value.details == {"code": "held", "pinned_step": "32", "pin": {"step": "32"}}


def test_preflight_sends_the_rollouts_and_group_size():
    http = FakeHttp()
    http.routes[("POST", "/reward/sessions/s1/preflight")] = lambda call: FakeResponse(200, {"preflight": {"gates": {"reward": "red"}}})
    out = make_client(http).preflight("s1", 8, [{"promptId": "p0", "rewardScore": 1, "truth": "fail"}])
    assert http.calls[-1]["body"] == {"groupSize": 8, "items": [{"promptId": "p0", "rewardScore": 1, "truth": "fail"}]}
    assert out["preflight"]["gates"]["reward"] == "red"
