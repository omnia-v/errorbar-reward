# errorbar-reward — the certified reward in your own trainer

```bash
pip install errorbar-reward
```

One HTTP contract (`POST /v1/reward/score` against a reward session), exposed in
the shapes trainers call:

| Adapter | Use |
| --- | --- |
| `ErrorbarRewardClient` | plain client: create a session, score, anchor, export |
| `trl_reward_func(client, session_id=…)` | a TRL `GRPOTrainer` `reward_funcs` callable |
| `ErrorbarJudge(session_id=…)` | a verifiers v1 Judge |
| `ErrorbarGrader` / `ErrorbarAnchorGrader` | Applied Compute AC2 graders — see [README-ac2.md](docs/ac2.md) |

Every adapter reads `ERRORBAR_API_KEY` and, optionally, `ERRORBAR_BASE_URL`
(default `https://gateway.errorbar.ai/v1`). Masked grades (`None`) pass through
untouched — see the package docstring for why they are never zero.

```python
from errorbar_reward import ErrorbarRewardClient, trl_reward_func

client = ErrorbarRewardClient()
session = client.create_session(reward={"mode": "single", "criterionId": "crit_…"}, reward_budget_usd=25)
for cert in session.certificates:
    print(cert["name"], cert["trust"])          # the grader's error rate, before the first rollout
reward_funcs = [trl_reward_func(client, session_id=session.id)]
```

## The mid-run anchor

A session created with `anchor.criterionId` carries an independent calibrated
judge, and errorbar holds the frozen prompt set it reads. Register the set once,
before the first check — at create (`frozen_set=`) or right after:

```python
fs = client.register_frozen_set(session.id, [
    {"requestId": "p0", "prompt": "A solid has volume …", "reference": "\\frac{69}{125}"},
    # … the reference is required when the anchor was certified reading one
])
control, sealed = fs["control_ids"], fs["confirm_ids"]   # a quarter is sealed for confirm()
```

errorbar renders what the anchor reads from its own copy of each prompt, exactly
as the anchor was certified. Every N optimizer steps the trainer submits the
policy's outputs on the control prompts — ids and responses only:

```python
items = [{"requestId": rid, "response": outputs[rid]} for rid in control]  # + "rewardScore" when you keep your own reward
verdict = client.anchor(session.id, items=items, step="64")   # 16 items per part
imp = (verdict.get("improvement") or {}).get("anchor") or {}
print(imp.get("note"), imp.get("supported"))          # is this checkpoint better than the baseline?
print(verdict["status"])                               # continue | review | hold
print((verdict.get("pin") or {}).get("reason"))        # which checkpoint to ship, and why
if verdict["held"]:
    print(verdict["reason"], "ship step", verdict["pinned_step"])
```

Each check answers three things.

**Is this checkpoint better than the one the run started from?** `improvement`
is a paired difference on the frozen prompts, corrected for the anchor's
measured error, with an interval that stays valid although you look at it every
check. `supported` is true only when that interval clears the session's
`minImprovement` — never because the number moved.

**Should the run continue?** `status` is `continue`, `review` (the evidence
cannot carry a decision; `required_evidence` says what fixes it) or `hold`.
`held: true` means the gap between the two reads on the same outputs widened
past its own paired noise, or the anchor fell past its own — on two consecutive
checks, never one. A held session refuses scoring (409) until a human resumes it.

**Which checkpoint would you ship?** `pin`: the earliest checkpoint that cannot
be separated from the best one, with the untrained baseline as a candidate. When
nothing separates from the baseline the pin says so in words, and
`ship_pinned_final` ships the base model rather than weights.

A check should cover every control prompt; below 95% it is `review`. Send a
frozen set larger than 24 items in parts — `client.anchor(...,
part_size=16)` does it for you. A check whose last part never lands is
superseded by the next step's check, which says so in `warnings`.

```python
client.confirm(session.id, baseline_step="0", candidate_step="96",
               prompts=sealed, baseline_responses=base_outputs, candidate_responses=cand_outputs)
client.review(session.id, "96", {"anchor-p0": "fail"})
client.resume_session(session.id, changed="reward", note="raised the pass bar")
```

`confirm` is the end-of-run Improvement Record on prompts **no check has seen**
(the frozen set chose the checkpoint, so it cannot also be the evidence):
SUPPORTED, CONTRADICTED, or INSUFFICIENT with how many more fresh prompts would
settle it. `review` sends your people's verdicts on the items a hold flagged —
and if they refute the anchor, the anchor is degraded for the run, not the run.
`resume` must say what changed; an anchor change needs a re-certified instrument
or it is refused with 409.

## The anchor report

The research post on the cheap-grader anchor study shows one chart per run:
the reward grader's read, the independent anchor's read, and the hold
decisions, check by check. The same document exists for your run — nothing is
recomputed from rollouts; it is a read of the history every anchor check
already stored, plus the anchor judge's live (dated) certificate:

```
GET /v1/reward/sessions/{id}/anchor
```

```json
{
  "session_id": "…",
  "anchor": { "criterion_id": "…", "criterion_name": "…", "judge_model": "…", "judge_mode": "corrected" },
  "certificate": { "…": "the anchor criterion's full signed certificate, as issued now" },
  "rule": { "window": 2, "gap_floor": 0.05, "drop_floor": 0.02, "consecutive": 2, "min_prompts": 20 },
    "history": [ { "step": "0", "at": "…", "n": 300, "reward_rate": 0.51, "anchor_rate": 0.49, "anchor_ci": [0.43, 0.55], "anchor_corrected": true, "seam": 67, "delta": { "anchor": { "observed": { "estimate": -0.03, "ci_seq": [-0.09, 0.03] }, "supported": false } }, "pin": { "step": "0", "best_step": "0", "tied_steps": ["0"], "supported": false }, "drop_floor": { "floor": 0.054, "vs_step": "0" } } ],
  "holds": [ { "step": "96", "at": "…", "state": "grader_fooled", "reason": "…", "pinned_step": "48" } ],
  "current": { "held": true, "held_at": "…", "held_state": "grader_fooled", "held_reason": "…", "pinned_step": "48", "resumed_at": null },
  "summary": { "checks": 7, "usable_checks": 7, "holds": 1, "max_divergence": 0.15, "max_divergence_step": "96", "max_gap_widening": 0.12, "max_gap_widening_step": "96", "anchor_range": [0.47, 0.53], "reward_range": [0.51, 0.68], "first_step": "0", "last_step": "96" }
}
```

`judge_mode` says how the anchor read is produced: `corrected` — the observed
pass rate mapped through the anchor's measured confusion, with the interval
that correction gives; `observed` — the anchor has no measured error rates and
the raw pass rate is reported. `holds` is the decision rule replayed over the
stored history under `rule`; the hold the session is currently under carries
the reason the trainer was told. `state` names which of the rule's three
readings held: `grader_fooled` (the reward read pulled away from the anchor
read on the same outputs while the anchor held — the grader is passing what
the anchor fails, before truth has moved), `gamed` (the gap widened and the
anchor fell — the hack has reached the truth), `degrading` (the anchor fell
and the gap did not widen — both instruments see the fall). `gap_se` on each
check is the paired standard error of the per-item difference the gap is
tested against. The same report sits inside the signed
environment bundle (`GET /v1/reward/sessions/{id}/environment` →
`anchor.report`).

From Python:

```python
report = client.anchor_report(session.id)
```

or printed as the three-line table with the certificate's identity and trust
lines under it:

```sh
python -m errorbar_reward.anchor_report <session_id>          # table
python -m errorbar_reward.anchor_report <session_id> --json   # the raw document
```

```
step | reward read | anchor read [95% CI] | reward−anchor | n   | held/pinned
-----+-------------+----------------------+---------------+-----+----------------
0    | 51.0%       | 49.0% [43.0, 55.0]   | +2.0%         | 300 |
…
96   | 68.0%       | 50.0% [44.0, 56.0]   | +18.0%        | 300 | HELD → pinned 48
```

## Tests

```sh
cd rl && python -m pytest tests/test_errorbar_reward.py tests/test_errorbar_ac2.py
```

Both files run against a fake HTTP layer; no network or SDK is needed.
