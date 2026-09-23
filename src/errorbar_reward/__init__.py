"""errorbar certified reward — adapters for your own trainer.

One HTTP contract (POST /v1/reward/score against a reward session) exposed in
the three shapes trainers actually call:

  - ErrorbarRewardClient        plain client (create session, score, stop)
  - trl_reward_func(...)        a TRL GRPOTrainer `reward_funcs` callable
  - ErrorbarJudge               a Prime Intellect verifiers v1 Judge
  - ErrorbarGrader /            Applied Compute AC2 graders (errorbar_reward.ac2;
    ErrorbarAnchorGrader        imported on demand — see README-ac2.md)

Masking contract: an item the judge could not parse comes back with
grade None. Every adapter passes None through — GRPOTrainer skips None
rewards; verifiers raises, because a malformed verdict must never score the
model. Treating None as 0.0 would train the policy to confuse the parser.
(AC2's GraderOutput.score is a float, so the ac2 adapter returns a 0.0
SENTINEL with a flagged reasoning string, or raises with
ERRORBAR_MASKED_POLICY=raise — the trade-off is written up in ac2.py.)
"""

from .client import ErrorbarRewardClient, RewardSession, RewardScore, RewardRefused
from .trl import trl_reward_func

__all__ = [
    "ErrorbarRewardClient",
    "RewardSession",
    "RewardScore",
    "RewardRefused",
    "trl_reward_func",
]

__version__ = "0.1.0"
