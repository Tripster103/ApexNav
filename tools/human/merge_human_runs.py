#!/usr/bin/env python3
"""Concatenate several human runs into one combined run.

Each playlist round (e.g. ovon_random50_v2A/B/C, then ovon_random350) is played
into its own results/<dataset>/human/<playlist>/ directory, and every downstream
tool -- make_subset.py, make_comparison.py, compare_human_agent.py -- takes one
playlist at a time. This joins disjoint rounds into a single run directory so
those tools can score them as one evaluation:

  results/<dataset>/human/<name>/
    continue.txt   running totals   -- as if one uninterrupted run had played
    record.txt     running averages    the concatenated playlist, in the order
    playlist.json  the concatenated    the runs were given
                   playlist

Per-episode values are recovered by differencing each run's continue.txt
(make_subset.py's parser) and re-accumulated in order, rendered by make_subset's
own block/table formatters, so the output is byte-for-byte the record format.

playlist.json carries selection.indices_0based + per_episode, i.e. it IS a
playlist: point make_subset.py --playlist at it and the matched agent subset comes
out in the same No. order, so No.N is the same episode on both sides.

Checks before writing anything:
  * each run's episodes are in its source playlist's play order (the source is
    the path human_play.py pinned in the run's playlist.json);
  * all runs share one metric set (--common-metrics: keep the intersection);
  * no episode appears twice across runs (by scene+episode id AND by index).

    PY=/scratch2/ml20/btripcon/FYP/conda/envs/apexnav/bin/python
    H=/scratch2/ml20/btripcon/FYP/results/ovon_val_seen/human
    $PY tools/human/merge_human_runs.py \\
        $H/ovon_random50_v2A $H/ovon_random50_v2B $H/ovon_random50_v2C $H/ovon_random350 \\
        --name ovon_combined500

No container needed -- pure file processing, login-node safe.

Authored by Claude (Anthropic Claude Opus 5.5) for Broden Tripcony.
"""
import argparse
import json
import os
import sys
from datetime import datetime

# This file sits two levels below the ApexNav root; make_subset.py lives in
# basic_utils/record_episode/, which is not a package import path from here.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "basic_utils", "record_episode"))
from make_subset import (average_table, compact_json, parse_continue,  # noqa: E402
                         render_block, render_file, total_table)


def load_run(run_dir):
    """-> (episodes, metric_names, pin, source_playlist) for one human run dir."""
    cont = os.path.join(run_dir, "continue.txt")
    pin_path = os.path.join(run_dir, "playlist.json")
    for f in (cont, pin_path):
        if not os.path.isfile(f):
            sys.exit(f"{run_dir}: no {os.path.basename(f)} -- not a human run directory?")
    episodes, metric_names = parse_continue(cont)
    with open(pin_path) as fh:
        pin = json.load(fh)
    src = pin.get("source")
    # Older pins predate the move to per-dataset playlist dirs (e7e91c4) and
    # point at playlists/<name>.json; look under playlists/<dataset>/ too.
    if src and not os.path.isfile(src):
        moved = os.path.join(REPO_ROOT, "playlists", pin.get("dataset", ""), os.path.basename(src))
        src = moved if os.path.isfile(moved) else src
    if not src or not os.path.isfile(src):
        sys.exit(f"{pin_path}: source playlist {src!r} not found")
    with open(src) as fh:
        source = json.load(fh)
    return episodes, metric_names, pin, source


def main():
    p = argparse.ArgumentParser(
        description="Concatenate disjoint human runs into one combined run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("runs", nargs="+", help="human run directories, in the order to concatenate")
    p.add_argument("--name", required=True, help="output folder name, e.g. ovon_combined500")
    p.add_argument("--out-dir", help="parent dir for the output (default: the first run's parent)")
    p.add_argument("--force", action="store_true", help="overwrite an existing output folder")
    p.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    p.add_argument("--common-metrics", action="store_true",
                   help="keep only metric rows present in every run (e.g. a run played "
                        "before SoftStepSPL existed) instead of refusing to merge")
    args = p.parse_args()

    runs = [os.path.abspath(r.rstrip("/")) for r in args.runs]
    if len(runs) < 2:
        p.error("give at least two runs to merge")

    episodes, indices, per_episode, parts = [], [], [], []
    metric_names = dataset = split = n_total = None
    for run in runs:
        eps, names, pin, source = load_run(run)
        part = os.path.basename(run)

        if metric_names is None:
            metric_names = names
        elif names != metric_names:
            if not args.common_metrics:
                sys.exit(f"{part}: metric rows {names} differ from {metric_names} "
                         "-- pass --common-metrics to keep only the shared ones")
            dropped = [m for m in metric_names + names if m not in metric_names or m not in names]
            print(f"  {part}: dropping metrics not in every run: {sorted(set(dropped))}")
            metric_names = [m for m in metric_names if m in names]
        ds = (source.get("source") or {}).get("dataset")
        if dataset is None:
            dataset, split = ds, pin.get("split")
            n_total = (source.get("source") or {}).get("n_episodes_total")
        elif (ds, pin.get("split")) != (dataset, split):
            sys.exit(f"{part}: dataset/split {ds}/{pin.get('split')} != {dataset}/{split}")

        idx = source["selection"]["indices_0based"]
        rows = {r["index"]: r for r in source.get("per_episode") or []}
        if len(eps) != len(idx):
            sys.exit(f"{part}: {len(eps)} episodes played, playlist has {len(idx)} -- unfinished?")
        if [str(rows[i]["episode_id"]) for i in idx] != [str(e["episode_id"]) for e in eps]:
            sys.exit(f"{part}: run order != its playlist's play order")

        for i, ep in zip(idx, eps):
            episodes.append(ep)
            indices.append(i)
            per_episode.append(dict(rows[i], part=part))
        parts.append((part, run, len(eps)))

    keys = [(e["scene_id"], str(e["episode_id"])) for e in episodes]
    if len(set(keys)) != len(keys) or len(set(indices)) != len(indices):
        sys.exit(f"runs overlap: {len(keys) - len(set(keys))} duplicate episodes")

    totals = {m: 0.0 for m in metric_names}
    cum_seconds = 0.0
    cont_blocks, rec_blocks = [], []
    for no, ep in enumerate(episodes, start=1):
        for m in metric_names:
            totals[m] += ep["delta"][m]
        cum_seconds += ep["seconds"]
        cont_blocks.append(render_block(ep, total_table(metric_names, totals), no, cum_seconds))
        rec_blocks.append(render_block(ep, average_table(metric_names, totals, no), no, cum_seconds))

    n = len(episodes)
    for part, _, k in parts:
        print(f"  {k:5d}  {part}")
    print(f"  {n:5d}  -> {args.name}  ({dataset} {split})")
    print(average_table(metric_names, totals, n))

    out_dir = os.path.join(args.out_dir or os.path.dirname(runs[0]), args.name)
    if args.dry_run:
        print(f"[dry run] would write {out_dir}")
        return
    if os.path.exists(out_dir) and not args.force:
        sys.exit(f"{out_dir} exists -- pass --force to overwrite")

    playlist = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "tools/human/merge_human_runs.py",
        "dataset": dataset,
        "split": split,
        "n": n,
        "source": {
            "dataset": dataset,
            "n_episodes_total": n_total,
            "combined_from": [{"run": run, "n": k} for _, run, k in parts],
        },
        "selection": {
            "name": args.name,
            "n_selected": n,
            "indexing": "0-based test_epi_num into the habitat episode iterator",
            "indices_0based": indices,
        },
        # human_play.py's pin keeps indices at the top level; mirror it.
        "indices_0based": indices,
        "per_episode": per_episode,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "continue.txt"), "w", encoding="utf-8") as fh:
        fh.write(render_file(cont_blocks))
    with open(os.path.join(out_dir, "record.txt"), "w", encoding="utf-8") as fh:
        fh.write(render_file(rec_blocks))
    with open(os.path.join(out_dir, "playlist.json"), "w") as fh:
        fh.write(compact_json(playlist) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
