#!/usr/bin/env python3
"""
Human vs ApexNav on the same episodes, side by side in the record format.

Reads the two runs a playlist produced --

  results/apexnav/<dataset>/human/<playlist>/continue.txt      played by hand
  results/apexnav/<dataset>/baseline/<playlist>/continue.txt   carved out of the
                                                               full eval

-- and writes one merged pair to results/apexnav/<dataset>/comparison/<playlist>/.
Same block-per-episode shape as any other record file, except each block's table
carries a Human column and an ApexNav column, and the outcome lines name both
sides plus how they agree:

  both success | agent-only failure | human-only failure | both failed

continue.txt holds running totals, record.txt running averages -- the same split
the sweep writes. Per-episode values are recovered by differencing, so this is
make_subset.py's parser doing the reading.

Episodes are matched on (scene_id, episode_id), not position, and the run aborts
unless both sides cover exactly the same set.

Authored by Claude (Anthropic Claude Opus 5) for Broden Tripcony.
"""

import argparse
import os
import sys
from collections import Counter

from prettytable import PrettyTable

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(REPO)  # FYP/, which holds results/
sys.path.insert(0, HERE)
from make_subset import fmt_average, fmt_total, is_distance, parse_continue  # noqa: E402

DATASETS = ["hm3dv1", "hm3dv2", "mp3d", "ovon_val_seen"]
AGREEMENT = ["both success", "agent-only failure", "human-only failure", "both failed"]


def index_by_episode(path):
    episodes, metric_names = parse_continue(path)
    out = {}
    for ep in episodes:
        key = (ep["scene_id"], str(ep["episode_id"]))
        if key in out:
            sys.exit(f"{path}: {key} appears twice")
        out[key] = ep
    return out, metric_names


def agreement_of(human_ok, agent_ok):
    if human_ok and agent_ok:
        return "both success"
    if human_ok and not agent_ok:
        return "agent-only failure"
    if agent_ok and not human_ok:
        return "human-only failure"
    return "both failed"


def side_table(metric_names, human_vals, agent_vals, header, fmt):
    t = PrettyTable(["Metric", "Human", "ApexNav"])
    t.align["Metric"] = "c"
    for name in metric_names:
        label = name.replace("Total ", header, 1)
        t.add_row([label, fmt(name, human_vals[name]), fmt(name, agent_vals[name])])
    return t


def build(dataset, playlist, condition, out_root, dry_run, force):
    # results/<dataset>/{human,comparison}/<playlist>/ sit directly under the
    # dataset; the agent's conditions hang off an apexnav/ level beside them.
    base = os.path.join(ROOT, "results", dataset)
    human_path = os.path.join(base, "human", playlist, "continue.txt")
    agent_path = os.path.join(base, "apexnav", condition, playlist, "continue.txt")
    for p in (human_path, agent_path):
        if not os.path.isfile(p):
            return None, f"{dataset}/{playlist}: no {os.path.relpath(p, ROOT)}"

    human, metric_names = index_by_episode(human_path)
    agent, agent_metrics = index_by_episode(agent_path)
    if metric_names != agent_metrics:
        return None, f"{dataset}: metric rows differ between the two runs"
    if set(human) != set(agent):
        return None, (
            f"{dataset}/{playlist}: episode sets differ "
            f"({len(set(human) - set(agent))} only human, "
            f"{len(set(agent) - set(human))} only agent) -- not the same episodes"
        )

    # Play order: the human run is the one a person actually sat through, so its
    # No. ordering is the one worth preserving.
    order = sorted(human, key=lambda k: human[k]["no"])

    totals = {"human": {m: 0.0 for m in metric_names},
              "agent": {m: 0.0 for m in metric_names}}
    cont_blocks, rec_blocks, tally, rows = [], [], Counter(), []

    for n, key in enumerate(order, start=1):
        h, a = human[key], agent[key]
        for m in metric_names:
            totals["human"][m] += h["delta"][m]
            totals["agent"][m] += a["delta"][m]

        h_ok = h["result"] == "success"
        a_ok = a["result"] == "success"
        agree = agreement_of(h_ok, a_ok)
        tally[agree] += 1
        rows.append((h["result"], a["result"], agree, h["target"]))

        head = (f"\nScene ID: {h['scene_id']}\n"
                f"Episode ID: {h['episode_id']}\n")
        tail = (f"human result: {h['result']}\n"
                f"apexnav result: {a['result']}\n"
                f"agreement: {agree}\n"
                f"target to find is {h['target']}\n"
                f"No.{n} task is finished\n")

        cont_blocks.append(
            head
            + f"{side_table(metric_names, totals['human'], totals['agent'], 'Total ', fmt_total)}\n"
            + tail)
        rec_blocks.append(
            head
            + f"{side_table(metric_names, totals['human'], totals['agent'], 'Average ', lambda m, v: fmt_average(m, v, n))}\n"
            + tail)

    out_dir = (os.path.join(out_root, playlist) if out_root
               else os.path.join(base, "comparison", playlist))
    summary = {"dataset": dataset, "playlist": playlist, "n": len(order),
               "tally": tally, "rows": rows, "out_dir": out_dir,
               "totals": totals, "metric_names": metric_names}

    if dry_run:
        return summary, None
    if os.path.exists(out_dir) and not force:
        return None, f"{os.path.relpath(out_dir, ROOT)} exists (use --force)"
    os.makedirs(out_dir, exist_ok=True)
    # write_record() prepends, so both files read newest-first.
    for name, blocks in (("continue.txt", cont_blocks), ("record.txt", rec_blocks)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write("".join(b + "\n" for b in reversed(blocks)))
    return summary, None


def print_summary(summaries):
    print(f"\n{'dataset':<16}{'n':>4}{'both ok':>9}{'agent-only':>12}"
          f"{'human-only':>12}{'both fail':>11}   human/agent SR")
    print("-" * 82)
    for s in summaries:
        t = s["tally"]
        n = s["n"]
        h_sr = 100 * (t["both success"] + t["agent-only failure"]) / n
        a_sr = 100 * (t["both success"] + t["human-only failure"]) / n
        print(f"{s['dataset']:<16}{n:>4}{t['both success']:>9}"
              f"{t['agent-only failure']:>12}{t['human-only failure']:>12}"
              f"{t['both failed']:>11}   {h_sr:.0f}% / {a_sr:.0f}%")

    print("\nIs the set of human failures a subset of ApexNav's?")
    for s in summaries:
        only = s["tally"]["human-only failure"]
        verdict = "yes" if only == 0 else f"no -- {only} episode(s) only the human failed"
        print(f"  {s['dataset']:<16}{verdict}")

    print("\nFailure modes on the episodes each side failed:")
    for s in summaries:
        h_modes = Counter(h for h, a, ag, t in s["rows"] if h != "success")
        a_modes = Counter(a for h, a, ag, t in s["rows"] if a != "success")
        print(f"\n  {s['dataset']}  (n={s['n']})")
        for who, modes in (("human  ", h_modes), ("apexnav", a_modes)):
            body = ", ".join(f"{v}x {k}" for k, v in modes.most_common()) or "none"
            print(f"    {who}  {body}")


def main():
    p = argparse.ArgumentParser(
        description="Merge a human run and an ApexNav run into one side-by-side record pair.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--playlist", help="playlist name; default: each dataset's only human run")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--condition", default="baseline", help="agent condition dir (default baseline)")
    p.add_argument("--out-root", help="override the output parent (default: alongside the inputs)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    summaries, problems = [], []
    for dataset in args.datasets:
        human_root = os.path.join(ROOT, "results", dataset, "human")
        if args.playlist:
            playlists = [args.playlist]
        elif os.path.isdir(human_root):
            playlists = sorted(d for d in os.listdir(human_root)
                               if os.path.isdir(os.path.join(human_root, d)))
        else:
            playlists = []
        if len(playlists) != 1 and not args.playlist:
            problems.append(f"{dataset}: {len(playlists)} human runs -- pass --playlist")
            continue
        s, err = build(dataset, playlists[0], args.condition,
                       args.out_root, args.dry_run, args.force)
        (problems.append(err) if err else summaries.append(s))

    for msg in problems:
        print(f"skip: {msg}", file=sys.stderr)
    if not summaries:
        sys.exit("nothing to compare")

    print_summary(summaries)
    if args.dry_run:
        print("\n[dry run] nothing written")
    else:
        print("\nwrote:")
        for s in summaries:
            print(f"  {os.path.relpath(s['out_dir'], ROOT)}/{{continue,record}}.txt")


if __name__ == "__main__":
    main()
