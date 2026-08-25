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

Authored by Claude (Anthropic Claude Opus 5) for Broden Tripcony.
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
# Palette: slots 1-6 of the validated categorical order (blue, orange, aqua,
# yellow, magenta, green), assigned to A-F in fixed order and never cycled --
# a category keeps its hue across every chart and every dataset. Verified with
# a Python port of the dataviz validator on the adjacent pairlist (stacked and
# grouped bars use adjacent pairs): worst CVD dE 9.1 (protan, target >=8),
# worst normal-vision dE 19.6 (floor 15), light surface #fcfcfb.
#
# Aqua/yellow/magenta land under 3:1 contrast on that surface, which triggers
# the documented relief rule -- so every chart below carries visible direct
# labels, and the same numbers are written to CSV as a table view. Do not drop
# the labels without re-checking that rule.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e4e3df"

CATEGORY_COLOR = {
    "A: Success": "#2a78d6",
    "B: Different Floor": "#eb6834",
    "C: False Positive": "#1baf7a",
    "D: No Frontier": "#eda100",
    "E: Stepout": "#e87ba4",
    "F: Missing Target": "#008300",
}
# Single-hue slot for one-series magnitude charts (no identity to encode, so no
# legend and no categorical hue -- see the dataviz form heuristic).
SINGLE_HUE = "#2a78d6"


def _plt():
    """Import matplotlib with the Agg backend, or None if unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("matplotlib not available -- skipping charts, CSVs still written.")
        return None


def _style(ax, plt):
    """Recessive axes: no box, muted ticks, grid behind the marks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.set_axisbelow(True)


def plot_stacked_composition(per_dataset, out_dir):
    """Headline chart: outcome composition per dataset, one stacked row each.

    Composition of a whole -> stacked bar. Horizontal so the dataset names and
    the segment labels both read without rotation.
    """
    plt = _plt()
    if plt is None:
        return

    datasets = list(per_dataset.keys())
    fig, ax = plt.subplots(figsize=(11, 1.15 * len(datasets) + 2.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, plt)

    for row, ds in enumerate(datasets):
        coarse = {r["category"]: r["pct"] for r in per_dataset[ds]["coarse"]}
        left = 0.0
        for cat in APEXNAV_ORDER:
            pct = coarse.get(cat, 0.0)
            if pct <= 0:
                continue
            # 2px surface gap between adjacent segments (skill: mark spacers).
            ax.barh(row, pct, left=left, height=0.52, color=CATEGORY_COLOR[cat],
                    edgecolor=SURFACE, linewidth=2, zorder=3)
            # Direct label -- required by the relief rule for the low-contrast
            # slots, and useful everywhere. Only label segments with room.
            if pct >= 4.5:
                ax.text(left + pct / 2, row, f"{pct:.1f}", ha="center", va="center",
                        color="#ffffff", fontsize=8.5, fontweight="bold", zorder=4)
            left += pct

    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(
        [f"{ds}\n{per_dataset[ds]['n']} eps" for ds in datasets],
        color=INK_PRIMARY, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of episodes", color=INK_SECONDARY, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_title("ApexNav outcome composition by dataset",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=14)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOR[c]) for c in APEXNAV_ORDER]
    ax.legend(handles, APEXNAV_ORDER, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=3, frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    path = os.path.join(out_dir, "outcome_composition.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_fine_categories(per_dataset, out_dir):
    """Diagnostic chart: the full 10-way check_failure() taxonomy, grouped.

    The coarse A-F view collapses distinctions that matter for debugging (a
    stuck agent and an out-of-steps agent are different bugs). Grouped bars so
    one category can be compared straight across datasets.
    """
    plt = _plt()
    if plt is None:
        return

    import numpy as np

    datasets = list(per_dataset.keys())
    shown = [c for c in RESULT_TYPES
             if any(r["count"] for ds in datasets
                    for r in per_dataset[ds]["fine"] if r["category"] == c)]

    y = np.arange(len(shown))
    h = 0.8 / len(datasets)
    fig, ax = plt.subplots(figsize=(10, 0.30 * len(shown) * len(datasets) + 1.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, plt)

    # Datasets are the series here, so they take the fixed categorical order.
    ds_colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for k, ds in enumerate(datasets):
        fine = {r["category"]: r["pct"] for r in per_dataset[ds]["fine"]}
        vals = [fine.get(c, 0.0) for c in shown]
        pos = y + (k - (len(datasets) - 1) / 2) * h
        ax.barh(pos, vals, height=h * 0.86, color=ds_colors[k % len(ds_colors)],
                label=ds, zorder=3)
        for yy, v in zip(pos, vals):
            if v > 0:
                ax.text(v + 0.6, yy, f"{v:.1f}", va="center", ha="left",
                        color=INK_SECONDARY, fontsize=7.5, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(shown, color=INK_PRIMARY, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% of episodes", color=INK_SECONDARY, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_title("check_failure() outcome categories (full 10-way taxonomy)",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")

    path = os.path.join(out_dir, "fine_categories.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_by_object(dataset, obj_rows, out_dir, min_episodes=15, top_n=20):
    """Per target-object success rate -- one series, so one hue and no legend.

    Categories with tiny n are dropped: a 0% success rate over 3 episodes is
    noise, and mixing it with a 0% over 90 episodes hides the real signal.
    """
    plt = _plt()
    if plt is None:
        return

    rows = [r for r in obj_rows if r["n_episodes"] >= min_episodes][:top_n]
    if not rows:
        print(f"  (no object category with n >= {min_episodes} in {dataset} -- skipping chart)")
        return

    fig, ax = plt.subplots(figsize=(9, 0.36 * len(rows) + 2.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, plt)

    labels = [f"{r['label']}  (n={r['n_episodes']})" for r in rows]
    vals = [r["success_rate_pct"] for r in rows]
    ax.barh(range(len(rows)), vals, height=0.62, color=SINGLE_HUE, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", ha="left",
                color=INK_SECONDARY, fontsize=8, zorder=4)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, color=INK_PRIMARY, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(100, max(vals) + 10))
    ax.set_xlabel("success rate (%)", color=INK_SECONDARY, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    # pad has to clear the subtitle line below it -- at pad=10 the two collide.
    ax.set_title(f"Success rate by target object -- {dataset}",
                 color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.012, f"worst first, categories with n >= {min_episodes} only",
            transform=ax.transAxes, color=INK_MUTED, fontsize=8.5, va="bottom")

    path = os.path.join(out_dir, f"by_object_{dataset}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
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
    per_dataset = {}

    for path, dataset in args.record:
        rows = parse_record_file(path)
        for r in rows:
            r["dataset"] = dataset
        all_rows.extend(rows)

        write_csv(
            os.path.join(args.out, f"episodes_{dataset}.csv"),
            # "dataset" is tagged onto every row above, so it has to be declared
            # here too -- DictWriter raises on any key missing from fieldnames.
            ["dataset", "scene_id", "episode_id", "result_text", "apexnav_category", "label"],
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

        per_dataset[dataset] = {"fine": fine_rows, "coarse": coarse_rows, "n": len(rows)}
        plot_by_object(dataset, obj_rows, args.out)

    plot_stacked_composition(per_dataset, args.out)
    plot_fine_categories(per_dataset, args.out)

    if len(args.record) > 1:
        write_csv(
            os.path.join(args.out, "episodes_all.csv"),
            ["dataset", "scene_id", "episode_id", "result_text", "apexnav_category", "label"],
            all_rows,
        )
        print(f"\nWrote combined episodes_all.csv ({len(all_rows)} episodes across {len(args.record)} datasets)")


if __name__ == "__main__":
    main()
