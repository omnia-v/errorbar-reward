# errorbar certified reward on Applied Compute AC2

Two graders in `errorbar_reward/ac2.py`, one API contract. Both run inside your
AC2 project and call errorbar's reward API over HTTPS.

| Grader | Where it runs | What it scores with |
| --- | --- | --- |
| `ErrorbarGrader` | training (`ac2_grader`) and the eval sidecar | your training reward session (`ERRORBAR_SESSION_ID`) |
| `ErrorbarAnchorGrader` | regrade jobs over the eval-sidecar traces | an independent certified criterion (`ERRORBAR_ANCHOR_SESSION_ID`), plus the anchor hold rule on the training session |

## Two commands

1. Train with the certified reward as the grader:

   ```python
   TrainingConfig(method="grpo", ac2_grader="ErrorbarGrader", ...)
   ```

   AC2's eval sidecar scores your fixed eval dataset at step 0 and every
   `eval_interval` steps with the same grader.

2. Add the second score column after (or during) the run:

   ```sh
   ac2 grade run --grader ErrorbarAnchorGrader --job-id <train_id> [--step n]
   ```

   The stored eval-sidecar traces are re-scored with the independent criterion
   and appear as `ErrorbarAnchorGrader@<version>` next to the training score.

   The server keeps the same two columns per anchor check, with the hold
   decisions, on the training session. Fetch them as one document — every
   check, the holds, the rule, and the anchor judge's certificate — with
   `GET /v1/reward/sessions/{ERRORBAR_SESSION_ID}/anchor`, or print the
   three-line table:

   ```sh
   python -m errorbar_reward.anchor_report $ERRORBAR_SESSION_ID
   ```

   See [README.md § The anchor report](README.md#the-anchor-report).

## Install

Copy `errorbar_reward/` into your project under `src/` (or `pip install` the
`rl/` package with the `ac2` extra — it adds no dependencies: the AC2 SDK is
provided by your project environment, and the only runtime requirement is
`requests`). AC2 discovers graders by class name, so `ErrorbarGrader` and
`ErrorbarAnchorGrader` must be importable from your `src/` tree — a one-line
module is enough:

```python
# src/graders.py
from errorbar_reward.ac2 import ErrorbarGrader, ErrorbarAnchorGrader  # noqa: F401
```

## Secrets

Graders read their configuration from the environment. Store the values as
project secrets:

```sh
ac2 secrets put --key ERRORBAR_API_KEY --value sk_...
ac2 secrets put --key ERRORBAR_SESSION_ID --value rs_...          # the training reward session
ac2 secrets put --key ERRORBAR_ANCHOR_SESSION_ID --value rs_...   # ErrorbarAnchorGrader only
```

Optional:

| Variable | Meaning |
| --- | --- |
| `ERRORBAR_BASE_URL` | default `https://gateway.errorbar.ai/v1` |
| `ERRORBAR_AGENTIC` | `1` to send assistant/tool turns as structured steps (the session must be agentic) |
| `ERRORBAR_STEP` | optimizer step tag when AC2 does not pass one (see "The step") |
| `ERRORBAR_MASKED_POLICY` | `zero` (default) or `raise` — see "Masked grades" |
| `ERRORBAR_ANCHOR_SET_SIZE` | `ErrorbarAnchorGrader`: number of eval-sidecar traces per step (the frozen set, sent in parts of 16); each step is submitted once; unset = score only, never submit to the anchor route |

Create the sessions with the errorbar API or dashboard: the training session's
reward is the certified criterion you train on; give it `anchor.criterionId`
pointing at the independent criterion, and create a second session whose
reward is that same independent criterion for the anchor grader to score with.

## What each grader sends

One trace becomes one item of `POST /v1/reward/score`:

```json
{
  "sessionId": "rs_...",
  "step": "40",
  "items": [{
    "requestId": "ac2-rs_...-40-<sha256(conversation, response)[:24]>",
    "conversation": "system: ...\nuser: ...",
    "response": "<last assistant message>"
  }]
}
```

- `conversation` is every message before the candidate's answer, one
  `role: content` line each. `response` is the last assistant message. With
  `ERRORBAR_AGENTIC=1` the assistant/tool turns after the first assistant
  message are sent as `steps` instead of `response`.
- `requestId` is deterministic: re-grading the same trace settles into the same
  metering row; two rollouts of one prompt get distinct ids.
- `grader_params` (e.g. `{"expected": "Paris"}`) is **not** sent. The
  session's certified criterion is the reward. Only a step key
  (`errorbar_step`, `step`, `training_step`, `global_step`) is read from it.

The score column shows `GraderOutput(score=<grade in [0, 1]>, reasoning=
"errorbar verdict=pass | grade=0.8300 | <flags>")`. Flags carried through from
the API: `execUnverified=true` (sandbox checks could not run; those components
scored 0), `attested=false`, `attestationMismatch=true`, `simFraction=…`.

`ErrorbarAnchorGrader` additionally buffers the step's items and, once it
holds `ERRORBAR_ANCHOR_SET_SIZE` of them, posts the whole frozen set to
`POST /v1/reward/sessions/{ERRORBAR_SESSION_ID}/anchor`:

```json
{ "step": "40", "items": [{ "requestId": "ac2-rs_...-40-<sha256(conversation)[:24]>", "conversation": "user: ...", "response": "..." }] }
```

Item ids there key on the prompt alone so the frozen set carries the same ids
at every check. The answer (`held`, `state`, `reason`, `pinned_step`,
`underpowered`, `history`, `counts`) is logged and kept on
`grader.anchor_results`. A held session refuses further training scores with
409 until you resume it; the pinned step is the checkpoint to ship. Fewer than
20 gradable prompts cannot hold a run (`underpowered` says so). Call
`grader.flush_anchor()` to submit a partial set.

**No per-item verdicts come back.** The answer counts how many outputs fell
in each cell — `seam` (the reward passed it, the anchor failed it), `false_fail`,
`agree`, `masked` — but never which. A trainer that knew which outputs the
independent anchor failed could train on them, and the anchor would become a
second reward the policy learns to satisfy. The items live on the server, for
the people who review them: `GET /v1/reward/sessions/{id}/anchor/items?step=…`
with a key carrying the `audit:read` scope — keep that scope off the key your
trainer uses.

## Data boundary

Graders run inside AC2, in your project environment. What leaves it is the
trace text — the prompt-side messages and the candidate's answer (and tool
turns, if agentic) — sent to errorbar's API with your API key. `grader_params`,
environment state, model weights, and the rest of the AC2 job never leave.
errorbar keeps the graded items for your session's export and metering, under
the retention terms of your workspace.

## Masked grades

errorbar returns `grade: null` for an item whose verdict could not be parsed:
the item is masked, not scored. AC2's `GraderOutput.score` is a float, so by
default the grader returns **0.0 with reasoning starting `masked: unparsed
grader verdict`**. That sentinel is visible in the score column and can be
dropped by a `custom_reward_post_process_path` function that inspects the
reasoning. Without such a post-processor a masked sample trains as a
zero-reward sample; if you would rather the rollout fail loudly, set
`ERRORBAR_MASKED_POLICY=raise`.

Related, and different: when the training session is **held** by its anchor,
every item scores 0.0 with reasoning `errorbar anchor hold at step N — training
reward withheld`. A group of identical rewards has no group-relative advantage,
so GRPO takes no update from a held step. Budget exhaustion or session expiry
(402) and network failures raise — a whole batch is never silently zeroed.

## The step

The AC2 docs do not describe how a grader learns the current optimizer step.
The grader looks, in order, at `grader_params` (`errorbar_step`, `step`,
`training_step`, `global_step`), `ERRORBAR_STEP`, then attributes of the same
names on the trace. Without a step, scoring proceeds under the session's
default namespace and the anchor grader scores but does not submit to the
anchor route. For a regrade job, set `ERRORBAR_STEP` to the `--step n` you are
regrading.

## Tests

```sh
cd rl && python -m pytest tests/test_errorbar_ac2.py
```

The tests run against a small stand-in for the `ac2.runtime` module and a fake
HTTP layer; no SDK or network is needed.
