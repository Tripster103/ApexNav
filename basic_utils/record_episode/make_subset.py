#!/usr/bin/env python3
"""
Build a subset run from an existing continue.txt.

continue.txt logs *cumulative* totals per episode (see habitat_evaluation.py
table2), so each episode's individual contribution is recovered by differencing
consecutive blocks. That makes it possible to re-derive SR / SPL / Soft SPL /
StepSPL / Distance to Goal over any subset of episodes, and to re-emit a
continue.txt + record.txt pair that look exactly like a real run of just those
episodes.

record.txt logs *averages* instead, so it is the lossier of the two files
(recovering a cumulative from an average means multiplying the rounding error by
the episode index). Everything here reads continue.txt only; the subset's
record.txt is regenerated from it.

Precision: continue.txt cumulatives are pre-rounded (f"{spl_all:.2f}"), so a
recovered per-episode value carries +/-0.01. Rounding errors telescope across a
subset, leaving ~0.03-0.14pp of error on a subset mean -- two orders of
magnitude below the sampling noise at those subset sizes. Success counts are
integers and come back exactly.

Usage
-----
  # first 100 episodes (0-based, inclusive ranges)
  make_subset.py results/apexnav/baseline/hm3dv1/continue.txt --indices 0-99

  # every chair episode
  make_subset.py .../continue.txt --target chair

  # 50 random episodes, reproducible
  make_subset.py .../continue.txt --sample 50 --seed 0 --name human_pilot

  # at most 2 episodes per scene -- for human runs, where revisiting a floorplan
  # hands the player a map the agent never had
  make_subset.py .../continue.txt --max-per-scene 2 --seed 0

  # a failure-mode arm: only the episodes ApexNav called a false positive
  make_subset.py .../continue.txt --result "false positive" --max-per-scene 2

  # indices from a file, one per line
  make_subset.py .../continue.txt --index-file episodes.txt

Writes <dir-of-continue.txt>/<name>/{continue.txt,record.txt,metadata.json}.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime

from prettytable import PrettyTable

BLOCK_RE = re.compile(r"Scene ID:")
ROW_RE = re.compile(r"^\|\s*(Total [^|]*?)\s*\|\s*([-\d.eE+]+)\s*\|\s*$", re.M)
SCENE_RE = re.compile(r"^Scene ID:\s*(.*?)\s*$", re.M)
EPISODE_RE = re.compile(r"^Episode ID:\s*(.*?)\s*$", re.M)
RESULT_RE = re.compile(r"^success or not:\s*(.*?)\s*$", re.M)
LABEL_RE = re.compile(r"^target to find is\s*(.*?)\s*$", re.M)
NUM_RE = re.compile(r"^No\.(\d+) task is finished\s*$", re.M)
TIME_RE = re.compile(r"^([\d.]+) seconds spend in this task\s*$", re.M)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_continue(path):
    """Parse continue.txt into a list of episodes ordered by No. (ascending).

    Returns (episodes, metric_names). Each episode carries the cumulative
    totals as written in the file; per-episode values are differenced later.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    starts = [m.start() for m in BLOCK_RE.finditer(text)]
    if not starts:
        sys.exit(f"no episode blocks found in {path}")
    starts.append(len(text))

    by_num, metric_names = {}, None
    for i in range(len(starts) - 1):
        chunk = text[starts[i] : starts[i + 1]]
        num = NUM_RE.search(chunk)
        if not num:
            continue
        rows = ROW_RE.findall(chunk)
        if not rows:
            continue
        names = [n for n, _ in rows]
        if metric_names is None:
            metric_names = names
        elif names != metric_names:
            sys.exit(f"inconsistent metric rows at No.{num.group(1)}: {names}")

        time_m = TIME_RE.search(chunk)
        n = int(num.group(1))
        if n in by_num:
            sys.exit(f"duplicate No.{n} in {path}")
        by_num[n] = {
            "no": n,
            "scene_id": SCENE_RE.search(chunk).group(1),
            "episode_id": EPISODE_RE.search(chunk).group(1),
            "result": RESULT_RE.search(chunk).group(1),
            "target": LABEL_RE.search(chunk).group(1),
            "cum": {name: float(val) for name, val in rows},
            "cum_seconds": float(time_m.group(1)) if time_m else None,
        }

    expected = set(range(1, max(by_num) + 1))
    missing = sorted(expected - set(by_num))
    if missing:
        sys.exit(
            f"{path} is missing No.{missing[:5]}{'...' if len(missing) > 5 else ''} "
            "-- differencing needs a contiguous run"
        )

    episodes = [by_num[n] for n in sorted(by_num)]

    # Difference the cumulatives to recover each episode's own contribution.
    prev = {name: 0.0 for name in metric_names}
    prev_seconds = 0.0
    for ep in episodes:
        ep["delta"] = {name: ep["cum"][name] - prev[name] for name in metric_names}
        prev = ep["cum"]
        if ep["cum_seconds"] is None:
            ep["seconds"] = 0.0
        else:
            ep["seconds"] = max(0.0, ep["cum_seconds"] - prev_seconds)
            prev_seconds = ep["cum_seconds"]
    return episodes, metric_names


# --------------------------------------------------------------------------
# formatting -- mirrors habitat_evaluation.py table1/table2 exactly
# --------------------------------------------------------------------------
def is_distance(name):
    return "Distance" in name


def is_count(name):
    return name == "Total Success"


def fmt_total(name, value):
    if is_count(name):
        return f"{int(round(value))}"
    if is_distance(name):
        return f"{value:.4f}"
    return f"{value:.2f}"


def fmt_average(name, total, n):
    """Averages are percentages for every metric except Distance to Goal."""
    if is_distance(name):
        return f"{total / n:.4f}"
    return f"{total / n * 100:.2f}%"


def total_table(metric_names, totals):
    t = PrettyTable(["Metric", "Total"])
    for name in metric_names:
        t.add_row([name, fmt_total(name, totals[name])])
    return t


def average_table(metric_names, totals, n):
    t = PrettyTable(["Metric", "Average"])
    for name in metric_names:
        t.add_row([name.replace("Total ", "Average ", 1), fmt_average(name, totals[name], n)])
    return t


def render_block(ep, table, new_no, cum_seconds):
    """Reproduce write_record()'s block layout (leading + trailing newline)."""
    return (
        f"\nScene ID: {ep['scene_id']}\n"
        f"Episode ID: {ep['episode_id']}\n"
        f"{table}\n"
        f"success or not: {ep['result']}\n"
        f"target to find is {ep['target']}\n"
        f"No.{new_no} task is finished\n"
        f"{cum_seconds:.2f} seconds spend in this task\n"
    )


def compact_json(obj):
    """indent=2, but keep the long index arrays on a single line."""
    text = json.dumps(obj, indent=2)
    return re.sub(
        r"\[\s*\n\s*((?:-?[\d.]+,\s*\n\s*)*-?[\d.]+)\s*\n\s*\]",
        lambda m: "[" + ", ".join(m.group(1).replace(",", " ").split()) + "]",
        text,
    )


def render_file(blocks):
    """write_record() prepends each block, so the file reads newest-first."""
    return "".join(b + "\n" for b in reversed(blocks))


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def parse_index_spec(spec, n_total):
    """'0,5,10-20,900-' -> [0, 5, 10..20, 900..n_total-1] (0-based, inclusive)."""
    out = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            lo = int(lo) if lo else 0
            hi = int(hi) if hi else n_total - 1
            if lo > hi:
                sys.exit(f"bad range '{part}': start > end")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


def dedupe(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def select(episodes, args):
    n_total = len(episodes)

    if args.indices:
        idx = parse_index_spec(args.indices, n_total)
    elif args.index_file:
        idx = []
        with open(args.index_file) as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if line:
                    idx.extend(parse_index_spec(line, n_total))
    else:
        idx = list(range(n_total))

    bad = [i for i in idx if not 0 <= i < n_total]
    if bad:
        sys.exit(f"index out of range 0..{n_total - 1}: {sorted(set(bad))[:10]}")
    idx = dedupe(idx)

    def keep(i):
        ep = episodes[i]
        if args.target and ep["target"].lower() not in args.target:
            return False
        if args.result and not any(r in ep["result"].lower() for r in args.result):
            return False
        if args.scene and not any(s in ep["scene_id"] for s in args.scene):
            return False
        return True

    idx = [i for i in idx if keep(i)]
    if not idx:
        sys.exit("selection is empty")

    # Per-scene cap. These val splits have very few scenes (mp3d 11, hm3dv1 20,
    # hm3dv2 36) for 1000-2200 episodes, so an uncapped subset makes a human
    # player revisit the same floorplan many times and score with a map the agent
    # never had. Drawn with the seeded RNG rather than taking the first N, since
    # within-scene order is not arbitrary. Applied BEFORE --sample so the sample
    # draws from an already scene-balanced pool.
    if args.max_per_scene is not None:
        by_scene = {}
        for i in idx:
            by_scene.setdefault(episodes[i]["scene_id"], []).append(i)
        rng = random.Random(args.seed)
        capped = []
        for scene in sorted(by_scene):
            pool = by_scene[scene]
            capped.extend(
                pool if len(pool) <= args.max_per_scene
                else rng.sample(pool, args.max_per_scene)
            )
        idx = sorted(capped)

    if args.sample is not None:
        if args.sample > len(idx):
            sys.exit(f"--sample {args.sample} exceeds the {len(idx)} selected episodes")
        idx = sorted(random.Random(args.seed).sample(idx, args.sample))
    if args.shuffle:
        random.Random(args.seed).shuffle(idx)
    return idx


def default_name(args, idx):
    tokens = []
    if args.indices:
        tokens.append("idx" + args.indices.replace(" ", "").replace(",", "_"))
    if args.index_file:
        tokens.append("file-" + os.path.splitext(os.path.basename(args.index_file))[0])
    if args.target:
        tokens.append("target-" + "_".join(sorted(args.target)))
    if args.result:
        tokens.append("result-" + "_".join(sorted(args.result)))
    if args.scene:
        tokens.append("scene-" + "_".join(sorted(args.scene)))
    if args.max_per_scene is not None:
        tokens.append(f"maxscene{args.max_per_scene}")
    if args.sample is not None:
        tokens.append(f"sample{args.sample}-seed{args.seed}")
    if args.shuffle:
        tokens.append(f"shuf-seed{args.seed}")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", "_".join(tokens)) if tokens else "all"
    name = f"subset_{stem}_n{len(idx)}"
    if len(name) > 80:  # long specs collapse to a stable hash
        digest = hashlib.sha1(",".join(map(str, idx)).encode()).hexdigest()[:8]
        name = f"subset_custom_n{len(idx)}_{digest}"
    return name


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Recover a subset run (SR/SPL/SoftSPL/StepSPL/DTG) from continue.txt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----\n", 1)[1],
    )
    p.add_argument("continue_file", help="path to the source continue.txt")
    p.add_argument("--indices", help="0-based indices/ranges, e.g. '0-99,150,200-249'")
    p.add_argument("--index-file", help="file of 0-based indices, one per line ('#' comments ok)")
    p.add_argument("--target", nargs="+", type=str.lower, help="keep only these target categories")
    p.add_argument("--result", nargs="+", type=str.lower, help="keep episodes whose result contains any of these")
    p.add_argument("--scene", nargs="+", help="keep episodes whose scene id contains any of these")
    p.add_argument("--max-per-scene", type=int, help="keep at most N episodes per scene (seeded draw)")
    p.add_argument("--sample", type=int, help="randomly draw this many from the selection")
    p.add_argument("--shuffle", action="store_true", help="shuffle the subset order")
    p.add_argument("--seed", type=int, default=0, help="seed for --sample/--shuffle (default 0)")
    p.add_argument("--name", help="output folder name (default: derived from the selection)")
    p.add_argument("--out-dir", help="parent dir for the output folder (default: alongside continue.txt)")
    p.add_argument("--force", action="store_true", help="overwrite an existing output folder")
    p.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    p.add_argument("--list", action="store_true", help="list available targets/results and exit")
    args = p.parse_args()

    episodes, metric_names = parse_continue(args.continue_file)

    if args.list:
        for field in ("target", "result"):
            counts = {}
            for ep in episodes:
                counts[ep[field]] = counts.get(ep[field], 0) + 1
            print(f"\n{field}s ({len(counts)}):")
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"  {v:6d}  {k}")
        print(f"\n{len(episodes)} episodes, metrics: {', '.join(metric_names)}")
        return

    idx = select(episodes, args)
    name = args.name or default_name(args, idx)
    parent = args.out_dir or os.path.dirname(os.path.abspath(args.continue_file))
    out_dir = os.path.join(parent, name)

    # Re-accumulate the subset in its selected order.
    totals = {m: 0.0 for m in metric_names}
    cum_seconds = 0.0
    cont_blocks, rec_blocks, per_episode = [], [], []
    for new_no, i in enumerate(idx, start=1):
        ep = episodes[i]
        for m in metric_names:
            totals[m] += ep["delta"][m]
        cum_seconds += ep["seconds"]
        cont_blocks.append(
            render_block(ep, total_table(metric_names, totals), new_no, cum_seconds)
        )
        rec_blocks.append(
            render_block(ep, average_table(metric_names, totals, new_no), new_no, cum_seconds)
        )
        per_episode.append(
            {
                "new_no": new_no,
                "source_index_0based": i,
                "source_no_1based": ep["no"],
                "scene_id": ep["scene_id"],
                "episode_id": ep["episode_id"],
                "target": ep["target"],
                "result": ep["result"],
                "seconds": round(ep["seconds"], 2),
                "metrics": {m: round(ep["delta"][m], 4) for m in metric_names},
            }
        )

    n = len(idx)
    summary = {}
    for m in metric_names:
        key = m.replace("Total ", "").replace(" ", "_").lower()
        summary[f"{key}_total"] = round(totals[m], 4)
        summary[f"{key}_" + ("mean" if is_distance(m) else "pct")] = round(
            totals[m] / n * (1 if is_distance(m) else 100), 4
        )

    outcomes, targets, scenes = {}, {}, {}
    for i in idx:
        outcomes[episodes[i]["result"]] = outcomes.get(episodes[i]["result"], 0) + 1
        targets[episodes[i]["target"]] = targets.get(episodes[i]["target"], 0) + 1
        scene = os.path.basename(episodes[i]["scene_id"])
        scenes[scene] = scenes.get(scene, 0) + 1

    metadata = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": {
            "continue_file": os.path.abspath(args.continue_file),
            "n_episodes": len(episodes),
            "metrics": metric_names,
        },
        "selection": {
            "name": name,
            "n_selected": n,
            "indexing": "0-based into the source run, ordered by No.",
            "indices": args.indices,
            "index_file": os.path.abspath(args.index_file) if args.index_file else None,
            "target": args.target,
            "result": args.result,
            "scene": args.scene,
            "max_per_scene": args.max_per_scene,
            "sample": args.sample,
            "shuffle": args.shuffle,
            "seed": args.seed
            if (args.sample is not None or args.shuffle or args.max_per_scene is not None)
            else None,
            "indices_0based": idx,
            "source_no_1based": [episodes[i]["no"] for i in idx],
        },
        "results": summary,
        "outcome_counts": dict(sorted(outcomes.items(), key=lambda kv: -kv[1])),
        "target_counts": dict(sorted(targets.items(), key=lambda kv: -kv[1])),
        "scene_counts": dict(sorted(scenes.items(), key=lambda kv: -kv[1])),
        "precision_note": (
            "Per-episode values are recovered by differencing continue.txt's pre-rounded "
            "cumulative totals (2dp for SPL-family metrics, 4dp for Distance to Goal). "
            "Success counts are exact; SPL-family subset means carry roughly 0.03-0.14 "
            "percentage points of rounding error, far below sampling noise at these n."
        ),
        "per_episode": per_episode,
    }

    print(f"source : {args.continue_file}  ({len(episodes)} episodes)")
    print(f"subset : {n} episodes -> {out_dir}")
    print(average_table(metric_names, totals, n))
    print(total_table(metric_names, totals))

    if args.dry_run:
        print("\n[dry run] nothing written")
        return

    if os.path.exists(out_dir) and not args.force:
        sys.exit(f"{out_dir} already exists (use --force to overwrite)")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "continue.txt"), "w", encoding="utf-8") as fh:
        fh.write(render_file(cont_blocks))
    with open(os.path.join(out_dir, "record.txt"), "w", encoding="utf-8") as fh:
        fh.write(render_file(rec_blocks))
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        fh.write(compact_json(metadata) + "\n")
    print(f"\nwrote continue.txt, record.txt, metadata.json to {out_dir}")


if __name__ == "__main__":
    main()
