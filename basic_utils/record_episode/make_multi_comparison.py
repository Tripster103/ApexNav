#!/usr/bin/env python3
"""
Human plus several ApexNav conditions on the same episodes, side by side.

make_comparison.py's N-way sibling. Reads

  results/<dataset>/human/<playlist>/continue.txt
  results/<dataset>/apexnav/<condition>/<playlist>/continue.txt   (one per --conditions)

and writes results/<dataset>/comparison/<name>/{continue.txt,record.txt,summary.txt}.
Each episode block's table carries a Human column then one column per condition,
in the order given. The first condition is the reference: every other condition
gets a "vs <ref>" verdict per episode --

  fixed  (ref failed, this succeeded) | broke (ref succeeded, this failed) | same

-- and summary.txt tallies those as a paired (McNemar) comparison, which is the
right test here: every condition played the identical 500 episodes.

Per-episode values come from differencing continue.txt (make_subset.py's parser),
episodes are matched on (scene_id, episode_id), and the run aborts unless every
side covers exactly the same set. Play order is the human run's.

  make_multi_comparison.py --dataset ovon_val_seen --playlist ovon_combined500 \\
      --conditions baseline=GDINO ovon_gdino_rc=GDINO-rc ovon_owlv2=OWLv2 \\
      --name ovon_combined500_detectors

Authored by Claude (Anthropic Claude Opus 5.5) for Broden Tripcony.
"""

import argparse
import os
import sys
from collections import Counter
from math import comb

from prettytable import PrettyTable

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(REPO)  # FYP/, which holds results/
sys.path.insert(0, HERE)
from make_comparison import index_by_episode  # noqa: E402
from make_subset import fmt_average, fmt_total  # noqa: E402


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value on the discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def verdict(ref_ok, ok):
    if ok and not ref_ok:
        return "fixed"
    if ref_ok and not ok:
        return "broke"
    return "same"


def table(metric_names, labels, totals, header, fmt):
    t = PrettyTable(["Metric"] + labels)
    t.align["Metric"] = "c"
    for name in metric_names:
        t.add_row([name.replace("Total ", header, 1)]
                  + [fmt(name, totals[lab][name]) for lab in labels])
    return t


def main():
    p = argparse.ArgumentParser(
        description="Merge a human run and several ApexNav conditions into one record pair.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--playlist", required=True)
    p.add_argument("--conditions", nargs="+", required=True, metavar="DIR=LABEL",
                   help="condition dirs under results/<dataset>/apexnav/, first is the reference")
    p.add_argument("--name", help="output folder under comparison/ (default: <playlist>_multi)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    base = os.path.join(ROOT, "results", args.dataset)
    sides = [("Human", os.path.join(base, "human", args.playlist, "continue.txt"))]
    for c in args.conditions:
        cdir, _, label = c.partition("=")
        sides.append((label or cdir,
                      os.path.join(base, "apexnav", cdir, args.playlist, "continue.txt")))
    labels = [lab for lab, _ in sides]
    if len(set(labels)) != len(labels):
        sys.exit(f"duplicate labels: {labels}")

    runs, metric_names = {}, None
    for lab, path in sides:
        if not os.path.isfile(path):
            sys.exit(f"no {os.path.relpath(path, ROOT)}")
        runs[lab], names = index_by_episode(path)
        if metric_names is None:
            metric_names = names
        elif names != metric_names:
            sys.exit(f"{lab}: metric rows differ from Human's")
    keys = set(runs["Human"])
    for lab in labels[1:]:
        if set(runs[lab]) != keys:
            sys.exit(f"{lab}: episode set differs from Human's "
                     f"({len(keys - set(runs[lab]))} missing, "
                     f"{len(set(runs[lab]) - keys)} extra)")

    order = sorted(keys, key=lambda k: runs["Human"][k]["no"])
    ref, others = labels[1], labels[2:]
    totals = {lab: {m: 0.0 for m in metric_names} for lab in labels}
    ok = {lab: [] for lab in labels}
    results = {lab: Counter() for lab in labels}
    cont_blocks, rec_blocks = [], []

    for n, key in enumerate(order, start=1):
        h = runs["Human"][key]
        for lab in labels:
            ep = runs[lab][key]
            for m in metric_names:
                totals[lab][m] += ep["delta"][m]
            ok[lab].append(ep["result"] == "success")
            results[lab][ep["result"]] += 1

        head = f"\nScene ID: {h['scene_id']}\nEpisode ID: {h['episode_id']}\n"
        tail = "".join(f"{lab.lower()} result: {runs[lab][key]['result']}\n" for lab in labels)
        tail += "".join(f"{lab.lower()} vs {ref.lower()}: "
                        f"{verdict(ok[ref][-1], ok[lab][-1])}\n" for lab in others)
        tail += f"target to find is {h['target']}\nNo.{n} task is finished\n"
        cont_blocks.append(head + f"{table(metric_names, labels, totals, 'Total ', fmt_total)}\n" + tail)
        rec_blocks.append(head + f"{table(metric_names, labels, totals, 'Average ', lambda m, v: fmt_average(m, v, n))}\n" + tail)

    N = len(order)
    out = [f"{args.dataset} / {args.playlist}: {N} episodes, every side played the same set",
           f"reference condition: {ref}", ""]
    out.append(str(table(metric_names, labels, totals, "Average ",
                         lambda m, v: fmt_average(m, v, N))))
    out += ["", f"Paired vs {ref} (fixed = {ref} failed & this succeeded; broke = the reverse):"]
    for lab in others:
        v = Counter(verdict(r, o) for r, o in zip(ok[ref], ok[lab]))
        out.append(f"  {lab:<10} fixed {v['fixed']:>3}  broke {v['broke']:>3}  "
                   f"net {v['fixed'] - v['broke']:+d}  ({100 * (v['fixed'] - v['broke']) / N:+.1f}pp)  "
                   f"exact McNemar p = {mcnemar_p(v['fixed'], v['broke']):.3g}")
    for i, a in enumerate(others):
        for b in others[i + 1:]:
            v = Counter(verdict(x, y) for x, y in zip(ok[a], ok[b]))
            out.append(f"  {b} vs {a}: fixed {v['fixed']}  broke {v['broke']}  "
                       f"net {v['fixed'] - v['broke']:+d}  "
                       f"exact McNemar p = {mcnemar_p(v['fixed'], v['broke']):.3g}")
    out += ["", "Human vs each condition (episodes):"]
    for lab in labels[1:]:
        h_only = sum(hh and not a for hh, a in zip(ok["Human"], ok[lab]))
        a_only = sum(a and not hh for hh, a in zip(ok["Human"], ok[lab]))
        out.append(f"  {lab:<10} human-only success {h_only:>3}   {lab}-only success {a_only:>3}")
    any_agent = sum(any(ok[lab][i] for lab in labels[1:]) for i in range(N))
    out += ["", f"Solved by at least one condition: {any_agent}/{N} ({100 * any_agent / N:.1f}%)",
            "", "Outcomes per side:"]
    for lab in labels:
        out.append(f"  {lab:<10} " + ", ".join(f"{c}x {r}" for r, c in results[lab].most_common()))
    summary = "\n".join(out) + "\n"
    print(summary)

    out_dir = os.path.join(base, "comparison", args.name or f"{args.playlist}_multi")
    if args.dry_run:
        print(f"[dry run] would write {os.path.relpath(out_dir, ROOT)}/")
        return
    if os.path.exists(out_dir) and not args.force:
        sys.exit(f"{os.path.relpath(out_dir, ROOT)} exists (use --force)")
    os.makedirs(out_dir, exist_ok=True)
    # write_record() prepends, so both files read newest-first.
    for fname, blocks in (("continue.txt", cont_blocks), ("record.txt", rec_blocks)):
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fh:
            fh.write("".join(b + "\n" for b in reversed(blocks)))
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(f"wrote {os.path.relpath(out_dir, ROOT)}/{{continue,record,summary}}.txt")


if __name__ == "__main__":
    main()
