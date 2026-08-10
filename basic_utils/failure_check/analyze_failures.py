"""
Failure-mode analysis over ApexNav record.txt files.

Mirrors the "Failure Cause Analysis" done in the ApexNav paper (Sec. V-C,
Fig. 7): per-dataset percentage breakdown of outcome categories, collapsed
into their A-F taxonomy (Success / Different Floor / False Positive /
No Frontier / Stepout / Missing Target) for direct comparison against their
reported numbers.

Goes one step further than the paper by also cross-tabulating failure rate
against target object category (`label`, already logged per-episode in
record.txt via write_record()) -- the paper only discusses this
qualitatively ("Missing Target cases ... mostly involve small or ambiguous
objects like plants, clothes, or pictures"); this produces the actual
per-category numbers.

Usage:
    python analyze_failures.py --record /path/to/record.txt --dataset hm3dv1 --out results/hm3dv1
    python analyze_failures.py --record r1.txt hm3dv1 --record r2.txt hm3dv2 --record r3.txt mp3d --out results/combined

Each --record takes two values: the file path and a dataset label.
Multiple --record flags can be combined for a cross-dataset summary.

Output (written to --out, plus printed to stdout):
    - episodes_<dataset>.csv        raw per-episode rows (scene_id, episode_id, result_text, label, apexnav_category)
    - failure_dist_<dataset>.csv    our 9-way + ApexNav's 6-way (A-F) distribution, counts and %
    - by_object_<dataset>.csv       per target-object-category success rate and failure breakdown, sorted worst-first
    - failure_dist.png / by_object_worst.png   bar charts (only if matplotlib is available; skipped otherwise)
"""

import argparse
import csv
import os
import re
from collections import Counter, defaultdict

# Exact order from params.py RESULT_TYPES -- kept in sync manually.
RESULT_TYPES = [
    "success",
    "infeasible",
    "no frontier",
    "false positive",
    "stepout true negative",
    "stepout feasible",
    "stucking",
    "[no frontier] false negative",
    "[stucking] false negative",
    "[stepout] false negative",
]

# Collapse our 9 categories down to ApexNav's own published A-F taxonomy
# (Sec. V-C / Fig. 7), so our per-dataset percentages line up against their
# reported numbers directly. See docs/ApexNav_Model_Components.md and
# failure_check.py for the source conditions behind each of our categories.
APEXNAV_TAXONOMY = {
    "success": "A: Success",
    "infeasible": "B: Different Floor",
    "false positive": "C: False Positive",
    "no frontier": "D: No Frontier",
    "stepout feasible": "E: Stepout",
    "stucking": "E: Stepout",  # ApexNav's paper has no separate "stuck" bucket;
    # functionally identical to stepout for their taxonomy (passive stop,
    # never reached target). We keep the finer distinction in our own 9-way
    # table since it's diagnostically useful (collision/planning stuck vs.
    # ran out of steps while still moving are different bugs to chase).
    "stepout true negative": "F: Missing Target",  # stopped near target passively, didn't recognize it
    "[no frontier] false negative": "F: Missing Target",
    "[stucking] false negative": "F: Missing Target",
    "[stepout] false negative": "F: Missing Target",
}
APEXNAV_ORDER = [
    "A: Success",
    "B: Different Floor",
    "C: False Positive",
    "D: No Frontier",
    "E: Stepout",
    "F: Missing Target",
]

# Matches one write_record() block. Blocks are prepended (newest first) and
# may be duplicated if a job was resumed mid-dataset -- we keep only the
# first (=newest) occurrence per (scene_id, episode_id).
BLOCK_RE = re.compile(
    r"Scene ID:\s*(?P<scene_id>.+?)\s*\n"
    r"Episode ID:\s*(?P<episode_id>.+?)\s*\n"
    r".*?"
    r"success or not:\s*(?P<result_text>.+?)\s*\n"
    r"target to find is\s*(?P<label>.+?)\s*\n",
    re.DOTALL,
)


def parse_record_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    seen = set()
    rows = []
    for m in BLOCK_RE.finditer(text):
        scene_id = m.group("scene_id").strip()
        episode_id = m.group("episode_id").strip()
        key = (scene_id, episode_id)
        if key in seen:
            continue
        seen.add(key)

        result_text = m.group("result_text").strip()
        label = m.group("label").strip()

        if result_text not in APEXNAV_TAXONOMY:
            print(
                f"WARNING: unrecognised result_text {result_text!r} in {path} "
                f"(scene={scene_id}, ep={episode_id}) -- check RESULT_TYPES is still in sync with params.py"
            )
            continue

        rows.append(
            {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "result_text": result_text,
                "apexnav_category": APEXNAV_TAXONOMY[result_text],
                "label": label,
            }
        )
    return rows


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def failure_distribution(rows):
    n = len(rows)
    fine = Counter(r["result_text"] for r in rows)
    coarse = Counter(r["apexnav_category"] for r in rows)
    fine_rows = [
        {"category": c, "count": fine.get(c, 0), "pct": round(100 * fine.get(c, 0) / n, 2) if n else 0.0}
        for c in RESULT_TYPES
    ]
    coarse_rows = [
        {"category": c, "count": coarse.get(c, 0), "pct": round(100 * coarse.get(c, 0) / n, 2) if n else 0.0}
        for c in APEXNAV_ORDER
    ]
    return fine_rows, coarse_rows


def by_object_category(rows):
    per_label = defaultdict(lambda: Counter())
    for r in rows:
        per_label[r["label"]][r["apexnav_category"]] += 1

    out = []
    for label, counts in per_label.items():
        total = sum(counts.values())
        success = counts.get("A: Success", 0)
        out.append(
            {
                "label": label,
                "n_episodes": total,
                "success_rate_pct": round(100 * success / total, 2) if total else 0.0,
                **{cat.split(": ")[1].replace(" ", "_"): counts.get(cat, 0) for cat in APEXNAV_ORDER},
            }
        )
    # Worst success rate first -- these are the object categories worth a closer (video) look.
    out.sort(key=lambda r: r["success_rate_pct"])
    return out


def maybe_plot(coarse_rows, dataset, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping chart, CSVs still written.")
        return

    labels = [r["category"] for r in coarse_rows]
    pcts = [r["pct"] for r in coarse_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, pcts, color="#4C72B0")
    ax.set_ylabel("% of episodes")
    ax.set_title(f"Failure Cause Statistics -- {dataset} (cf. ApexNav Fig. 7)")
    for i, p in enumerate(pcts):
        ax.text(i, p + 0.5, f"{p:.1f}%", ha="center", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, f"failure_dist_{dataset}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--record",
        nargs=2,
        action="append",
        metavar=("PATH", "DATASET_LABEL"),
        required=True,
        help="Path to a record.txt plus the dataset label to tag it with. Repeatable.",
    )
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_rows = []

    for path, dataset in args.record:
        rows = parse_record_file(path)
        for r in rows:
            r["dataset"] = dataset
        all_rows.extend(rows)

        write_csv(
            os.path.join(args.out, f"episodes_{dataset}.csv"),
            ["scene_id", "episode_id", "result_text", "apexnav_category", "label"],
            rows,
        )

        fine_rows, coarse_rows = failure_distribution(rows)
        write_csv(os.path.join(args.out, f"failure_dist_fine_{dataset}.csv"), ["category", "count", "pct"], fine_rows)
        write_csv(os.path.join(args.out, f"failure_dist_{dataset}.csv"), ["category", "count", "pct"], coarse_rows)

        print(f"\n=== {dataset} ({len(rows)} episodes) -- ApexNav-comparable taxonomy ===")
        for r in coarse_rows:
            print(f"  {r['category']:<22} {r['count']:>5}  ({r['pct']:>5.2f}%)")

        obj_rows = by_object_category(rows)
        write_csv(
            os.path.join(args.out, f"by_object_{dataset}.csv"),
            ["label", "n_episodes", "success_rate_pct"] + [c.split(": ")[1].replace(" ", "_") for c in APEXNAV_ORDER],
            obj_rows,
        )
        print(f"  Worst 5 object categories by success rate ({dataset}):")
        for r in obj_rows[:5]:
            print(f"    {r['label']:<25} n={r['n_episodes']:<4} success={r['success_rate_pct']}%")

        maybe_plot(coarse_rows, dataset, args.out)

    if len(args.record) > 1:
        write_csv(
            os.path.join(args.out, "episodes_all.csv"),
            ["dataset", "scene_id", "episode_id", "result_text", "apexnav_category", "label"],
            all_rows,
        )
        print(f"\nWrote combined episodes_all.csv ({len(all_rows)} episodes across {len(args.record)} datasets)")


if __name__ == "__main__":
    main()
