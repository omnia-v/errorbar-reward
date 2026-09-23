"""Print a reward session's anchor report — the three-line table of the
cheap-grader anchor study (reward read, independent anchor read, hold
decisions) for YOUR run, with the anchor judge's certificate attached.

    python -m errorbar_reward.anchor_report <session_id> [--json] [--base-url URL]

Reads ERRORBAR_API_KEY (and ERRORBAR_BASE_URL) like the rest of the package.
Nothing is computed here: the server returns the stored history of every
mid-run check and the live certificate; this module only renders them.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Iterable, TextIO

from .client import ErrorbarRewardClient


def _num(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _pct(x: Any, signed: bool = False) -> str:
    v = _num(x)
    if v is None:
        return "—"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def _ci(ci: Any) -> str:
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return ""
    lo, hi = _num(ci[0]), _num(ci[1])
    if lo is None or hi is None:
        return ""
    return f"[{lo * 100:.1f}, {hi * 100:.1f}]"


def _table(rows: Iterable[list[str]], header: list[str]) -> list[str]:
    rows = [header, *rows]
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    out = []
    for k, r in enumerate(rows):
        out.append(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip())
        if k == 0:
            out.append("-+-".join("-" * w for w in widths))
    return out


def history_rows(report: dict[str, Any]) -> list[list[str]]:
    """One row per stored check: step | reward read | anchor read [ci] | reward−anchor | n | held/pinned."""
    holds_by_step = {str(h.get("step")): h for h in report.get("holds") or []}
    rows: list[list[str]] = []
    for p in report.get("history") or []:
        step = str(p.get("step"))
        reward, anchor = _num(p.get("reward_rate")), _num(p.get("anchor_rate"))
        gap = None if reward is None or anchor is None else reward - anchor
        held = holds_by_step.get(step)
        state = f"HELD {held.get('state') or ''} → pinned {held.get('pinned_step')}".replace("  ", " ") if held else ""
        if not held and p.get("status") == "review":
            state = "REVIEW"
        n = str(p.get("n", ""))
        masked = p.get("masked") or 0
        if masked:
            n = f"{n} (+{masked} masked)"
        rows.append([step, _pct(reward), f"{_pct(anchor)} {_ci(p.get('anchor_ci'))}".rstrip(), _pct(gap, signed=True), n, state])
    return rows


def rule_lines(rule: dict[str, Any] | None) -> list[str]:
    if not rule:
        return []
    if "gap_floor" not in rule and "rise_margin" in rule:  # a report from before the three-state readout
        return [
            "rule: hold when, on {c} consecutive checks, the reward has risen ≥ {rise} since the run's first usable check while the anchor moved ≤ {follow},".format(
                c=rule.get("consecutive"), rise=_pct(rule.get("rise_margin"), signed=True), follow=_pct(rule.get("follow_tolerance"), signed=True)
            ),
            "      or the anchor sits below its best by more than its own noise (floor {floor}); a check with fewer than {n} gradable prompts holds nothing.".format(
                floor=_pct(rule.get("drop_floor")), n=rule.get("min_prompts")
            ),
        ]
    return [
        "rule: on {c} consecutive checks — GAP: the reward read minus the anchor read on the same outputs has widened since the run's first usable check".format(c=rule.get("consecutive")),
        "      by more than its paired noise (floor {gap}); DROP: the anchor sits below its best by more than its own noise (floor {drop}).".format(
            gap=_pct(rule.get("gap_floor")), drop=_pct(rule.get("drop_floor"))
        ),
        "      gap only → grader_fooled · gap and drop → gamed · drop only → degrading; every state holds and pins. Fewer than {n} gradable prompts holds nothing.".format(
            n=rule.get("min_prompts")
        ),
    ]


def certificate_lines(cert: dict[str, Any] | None) -> list[str]:
    """Identity + trust lines of the anchor judge's certificate."""
    if not cert:
        return ["certificate: none — the anchor criterion no longer exists in this workspace; nothing about the anchor read is certified."]
    cal = cert.get("calibration") or {}
    metrics = cal.get("metrics") or {}
    trust = cert.get("trust") or {}
    validity = cert.get("validity") or {}
    pop = cert.get("population") or {}
    sig = cert.get("signature") or {}
    lines = [
        f"certificate: {cert.get('name')} ({cert.get('criterion_id')}) · judge {cert.get('judge_model')} · unit {cert.get('unit')} · issued {cert.get('issued_at')}",
    ]
    if cal.get("measured"):
        m = cal.get("matrix") or {}
        lines.append(
            "  trust: {t} · sensitivity {tpr} {tpr_ci} · specificity {tnr} {tnr_ci} · κ {k} · {labels} labels{holdout}".format(
                t=trust.get("trust"),
                tpr=_pct(metrics.get("tpr")),
                tpr_ci=_ci(metrics.get("tpr_ci")),
                tnr=_pct(metrics.get("tnr")),
                tnr_ci=_ci(metrics.get("tnr_ci")),
                k="—" if metrics.get("kappa") is None else f"{float(metrics['kappa']):.3f}",
                labels=cal.get("labels"),
                holdout=" (held-out report half)" if cal.get("holdout_active") else "",
            )
        )
        lines.append(f"  matrix: tp {m.get('tp')} · fp {m.get('fp')} · tn {m.get('tn')} · fn {m.get('fn')} · aligned {cal.get('aligned_at')}")
    else:
        lines.append("  trust: unmeasured — no measured error rates; the anchor read is the observed pass rate, uncorrected.")
    if pop.get("statement"):
        lines.append(f"  population: {pop['statement']}")
    drift = validity.get("drift_status")
    avail = validity.get("judge_model_available")
    lines.append(
        "  validity: drift {d}{r} · judge model {a}".format(
            d=drift,
            r=f" ({validity.get('drift_reason')})" if validity.get("drift_reason") else "",
            a="available" if avail else "UNAVAILABLE" if avail is False else "?",
        )
    )
    if sig:
        lines.append(f"  signature: {sig.get('alg')} key {sig.get('key_id')} — verify with POST /v1/verify")
    elif cert.get("unsigned"):
        lines.append("  signature: none (server has no signing secret configured)")
    return lines


def _pts(x: Any) -> str:
    return "—" if x is None else f"{x * 100:+.1f}"


def improvement_lines(improvement: dict | None, pin: dict | None) -> list[str]:
    """The gate's two answers: how much the latest checkpoint improved over the
    baseline (anchor on the truth scale when corrected, and the run's own
    reward), and which checkpoint to keep."""
    out: list[str] = []
    if improvement:
        a = improvement.get("anchor") or {}
        r = improvement.get("reward") or {}
        scale = a.get("corrected") or a.get("observed") or {}
        seq = scale.get("ci_seq") or [None, None]
        verdict = "SUPPORTED" if a.get("supported") else "CONTRADICTED" if a.get("contradicted") else "NOT SUPPORTED"
        out.append(
            f"improvement, step {a.get('step')} over step {a.get('vs_step')} (n={a.get('n')}): "
            f"anchor {_pts(scale.get('estimate'))} pts [{_pts(seq[0])}, {_pts(seq[1])}] {'truth scale' if a.get('corrected') else 'observed'} → {verdict}"
            f" · reward {_pts((r.get('observed') or {}).get('estimate'))} pts"
        )
        if a.get("note"):
            out.append(f"  {a['note']}")
        for name, d in (improvement.get("instruments") or {}).items():
            obs = d.get("observed") or {}
            seq_i = obs.get("ci_seq") or [None, None]
            v = "SUPPORTED" if d.get("supported") else "CONTRADICTED" if d.get("contradicted") else "NOT SUPPORTED"
            out.append(f"  {name}: {_pts(obs.get('estimate'))} pts [{_pts(seq_i[0])}, {_pts(seq_i[1])}] → {v}")
    if pin:
        tied = ", ".join(str(t) for t in pin.get("tied_steps") or [])
        out.append(f"keep: step {pin.get('step')} ({'supported' if pin.get('supported') else 'improvement not supported'}; tied: {tied}) — {pin.get('reason')}")
    return out


def confirmation_lines(c: dict | None) -> list[str]:
    """The end-of-run Improvement Record on a fresh sample."""
    if not c:
        return []
    return [f"confirmed on fresh prompts: {c.get('verdict')} — {c.get('reason')}"]


def render(report: dict[str, Any]) -> str:
    anchor = report.get("anchor") or {}
    current = report.get("current") or {}
    summary = report.get("summary") or {}
    lines = [
        f"anchor report — session {report.get('session_id')}",
        f"anchor: {anchor.get('criterion_name')} ({anchor.get('criterion_id')}) · judge {anchor.get('judge_model') or '?'} · read: {anchor.get('judge_mode')}",
        *rule_lines(report.get("rule")),
        "",
        *_table(history_rows(report), ["step", "reward read", "anchor read [95% CI]", "reward−anchor", "n", "held/pinned"]),
        "",
    ]
    holds = report.get("holds") or []
    if holds:
        lines.append(f"holds ({len(holds)}):")
        for h in holds:
            lines.append(f"  step {h.get('step')} at {h.get('at')} [{h.get('state') or '?'}] → pinned {h.get('pinned_step')}: {h.get('reason')}")
    else:
        lines.append("holds: none")
    status = current.get("status")
    if status:
        lines.append(f"status: {str(status).upper()}")
        for r in current.get("review_reasons") or []:
            lines.append(f"  review: {r}")
    for rv in report.get("reviews") or []:
        res = rv.get("result") or {}
        read = res.get("reading") or f"requested ({len(rv.get('sample') or [])} of {rv.get('population')} seam outputs), no verdicts yet"
        lines.append(f"human review at step {rv.get('step')}: {str(res.get('conclusion', 'open')).upper()} — {read}")
    lines.extend(improvement_lines(current.get("improvement"), current.get("pin")))
    lines.extend(confirmation_lines(report.get("confirmation")))
    if current.get("held"):
        lines.append(f"current: HELD ({current.get('held_state') or '?'}) since {current.get('held_at')} · pinned step {current.get('pinned_step')} — scoring refuses 409 until resumed")
    else:
        resumed = f" · resumed {current['resumed_at']}" if current.get("resumed_at") else ""
        lines.append(f"current: not held{resumed}")
    if summary:
        ar, rr = summary.get("anchor_range"), summary.get("reward_range")
        lines.append(
            "summary: {c} checks ({u} usable) · {h} hold{hs} · max reward−anchor {d}{ds} · anchor {a} · reward {r}".format(
                c=summary.get("checks"),
                u=summary.get("usable_checks"),
                h=summary.get("holds"),
                hs="" if summary.get("holds") == 1 else "s",
                d=_pct(summary.get("max_divergence"), signed=True),
                ds=f" at step {summary.get('max_divergence_step')}" if summary.get("max_divergence_step") is not None else "",
                a=f"{_pct(ar[0])}–{_pct(ar[1])}" if ar else "—",
                r=f"{_pct(rr[0])}–{_pct(rr[1])}" if rr else "—",
            )
        )
    lines.append("")
    lines.extend(certificate_lines(report.get("certificate")))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, out: TextIO = sys.stdout, client: ErrorbarRewardClient | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m errorbar_reward.anchor_report", description=__doc__.split("\n\n")[0])
    ap.add_argument("session_id")
    ap.add_argument("--json", action="store_true", help="print the raw report document instead of the table")
    ap.add_argument("--base-url", default=None, help="override ERRORBAR_BASE_URL")
    a = ap.parse_args(argv)
    c = client or ErrorbarRewardClient(base_url=a.base_url)
    report = c.anchor_report(a.session_id)
    if a.json:
        out.write(json.dumps(report, indent=1) + "\n")
    else:
        out.write(render(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
