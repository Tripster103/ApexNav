#!/usr/bin/env python3
"""Sample a playlist of episodes straight from an ObjectNav dataset.

This is the *dataset*-side selector, and it is deliberately not
basic_utils/record_episode/make_subset.py. make_subset.py carves a slice out of
an agent run's continue.txt, so it can only ever select episodes some agent
already played, and its selection criteria are agent outcomes ("every episode
ApexNav called a false positive"). This reads the dataset itself, so it can
select episodes nothing has ever run, before any agent result exists.

Both emit the same metadata.json shape -- specifically selection.indices_0based
-- so human_play.py --playlist consumes either without knowing the difference.

Indices are `test_epi_num`: 0-based positions in the EPISODE ITERATOR ORDER, not
episode_ids. Those two are unrelated numbers, and the iterator order is not file
order (group_by_scene defaults to True, so episodes come out clustered by scene).
Rather than reimplement that ordering, this imports find_episode_index.py's
build_iterator(), which rebuilds exactly the iterator habitat.Env would build,
without constructing a simulator -- a few seconds, no GPU.

Sampling: stratified round-robin over scenes. Shuffle the scenes, take one random
episode from each in turn, and keep looping until --n are collected. No val split
has 50 scenes (mp3d 11, hm3dv1 20, hm3dv2 36, ovon 36), so an --n of 50 cannot be
one-per-scene; round-robin spreads it as thinly as the data allows -- at most
ceil(n/scenes) from any one scene -- instead of leaving it to chance. This matters
for human runs specifically: replaying a floorplan hands the player a map the
agent never had.

The collected episodes are then shuffled into playback order, so the few
same-scene pairs that stratification cannot avoid do not land back to back.
human_play.py walks the playlist in the order written here (it sets
iterator_options.cycle=True so the iterator can wrap to reach any index).

Usage (inside the apexnav container; runs from any directory):
    python tools/human/make_playlist.py --dataset hm3dv1        # 50 episodes, seed 0
    python tools/human/make_playlist.py --dataset ovon --n 100 --seed 7
    python tools/human/make_playlist.py --dataset mp3d --target chair sofa
    python tools/human/make_playlist.py --dataset hm3dv2 --dry-run

Writes <repo>/playlists/<dataset>_<name>.json. Play it with:
    bash jobs/run_human_play.sh --dataset hm3dv1 \\
        --playlist /scratch2/ml20/btripcon/FYP/ApexNav/playlists/hm3dv1_random50_seed0.json

Note on datasets: OVON's val splits reuse HM3Dv2's 36 scenes exactly -- the
seen/unseen axis is object *categories* (79 vs 49, zero overlap), not scenes. So
an ovon playlist and an hm3dv2 playlist can name the same scenes while sharing no
episodes.

Authored by Claude (Anthropic Claude Opus 5) for Broden Tripcony.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime

from find_episode_index import build_iterator

# Repo root resolved from this file, so --out-dir defaults somewhere stable no
# matter which directory this is launched from. find_episode_index is a sibling
# in tools/human/, which `python tools/human/make_playlist.py` puts on sys.path
# automatically -- but it does NOT put the repo root there, hence the explicit
# path here rather than a bare relative default.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def scan(dataset):
    """Walk the iterator once, recording every episode's index and identity.

    Returns rows in iterator order. This is the only place the dataset is read;
    everything downstream is plain Python on these rows.
    """
    d, it = build_iterator(dataset)
    rows = []
    for i, ep in enumerate(it):
        rows.append(
            {
                "index": i,
                "scene": os.path.basename(ep.scene_id).split(".")[0],
                "scene_id": ep.scene_id,
                "episode_id": str(ep.episode_id),
                "category": ep.object_category,
            }
        )
    if len(rows) != len(d.episodes):
        # Not fatal, but it would mean the iterator is not walking the dataset
        # one-for-one, which would invalidate every index below.
        print(
            f"WARNING: iterator yielded {len(rows)} episodes but the dataset holds "
            f"{len(d.episodes)} -- indices may not mean what you think",
            file=sys.stderr,
        )
    return rows


def stratified_sample(rows, n, rng):
    """Round-robin over scenes, one random episode per scene per pass.

    Scenes are visited in a shuffled order and each scene's own episodes are
    shuffled, so the draw is uniform within a scene and the *first* pass is a
    uniform sample of scenes. Passes continue until n episodes are collected or
    every episode is exhausted.
    """
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r["scene"]].append(r)

    scenes = sorted(by_scene)  # sort first so the shuffle is seed-reproducible
    rng.shuffle(scenes)
    for s in scenes:
        rng.shuffle(by_scene[s])

    picked = []
    while len(picked) < n:
        progressed = False
        for s in scenes:
            if not by_scene[s]:
                continue
            picked.append(by_scene[s].pop())
            progressed = True
            if len(picked) == n:
                break
        if not progressed:
            break  # every scene exhausted
    return picked


def main():
    ap = argparse.ArgumentParser(
        description="Sample a playlist of episodes from an ObjectNav dataset."
    )
    ap.add_argument(
        "--dataset", required=True, choices=["hm3dv1", "hm3dv2", "mp3d", "ovon"]
    )
    ap.add_argument("--n", type=int, default=50, help="episodes to draw (default 50)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    ap.add_argument(
        "--target", nargs="+", type=str.lower, help="restrict to these goal categories"
    )
    ap.add_argument(
        "--scene", nargs="+", help="restrict to scenes whose id contains any of these"
    )
    ap.add_argument("--name", help="output basename (default: random<N>_seed<S>)")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="where to write the json (default: <repo>/playlists/)",
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the summary, write nothing"
    )
    args = ap.parse_args()

    if args.n < 1:
        sys.exit("--n must be at least 1")

    rows = scan(args.dataset)
    total = len(rows)
    total_scenes = len({r["scene"] for r in rows})

    if args.target:
        rows = [r for r in rows if r["category"].lower() in args.target]
    if args.scene:
        rows = [r for r in rows if any(s in r["scene_id"] for s in args.scene)]
    if not rows:
        sys.exit("no episodes left after --target/--scene filtering")

    n_scenes = len({r["scene"] for r in rows})
    if args.n > len(rows):
        sys.exit(
            f"asked for {args.n} episodes but only {len(rows)} are available "
            f"after filtering ({total} in the dataset)"
        )

    rng = random.Random(args.seed)
    picked = stratified_sample(rows, args.n, rng)
    # Playback order. Stratification decides *which* episodes; this decides the
    # order they are played in, breaking up the same-scene pairs that a
    # 36-scene split cannot avoid when n=50.
    rng.shuffle(picked)

    scene_counts = Counter(r["scene"] for r in picked)
    cat_counts = Counter(r["category"] for r in picked)
    name = args.name or f"random{args.n}_seed{args.seed}"
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "playlists")
    out_path = os.path.join(out_dir, f"{args.dataset}_{name}.json")

    print(
        f"dataset : {args.dataset}  ({total} episodes, {total_scenes} scenes)"
        + (f"  -> {len(rows)} episodes, {n_scenes} scenes after filtering"
           if len(rows) != total else "")
    )
    print(f"playlist: {len(picked)} episodes across {len(scene_counts)} scenes")
    print(
        f"per-scene: min={min(scene_counts.values())} max={max(scene_counts.values())}"
    )
    print(f"categories: {len(cat_counts)}  {dict(cat_counts.most_common(8))}")
    print(f"first 10 indices (play order): {[r['index'] for r in picked[:10]]}")

    if args.dry_run:
        print("\n[dry run] nothing written")
        return

    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "generator": "make_playlist.py",
        "source": {
            "dataset": args.dataset,
            "n_episodes_total": total,
            "n_scenes_total": total_scenes,
        },
        "selection": {
            "name": name,
            "n_selected": len(picked),
            "indexing": "0-based test_epi_num into the habitat episode iterator",
            "strategy": "stratified round-robin over scenes, then shuffled for play order",
            "n": args.n,
            "seed": args.seed,
            "target": args.target,
            "scene": args.scene,
            "indices_0based": [r["index"] for r in picked],
        },
        "scene_counts": dict(scene_counts.most_common()),
        "target_counts": dict(cat_counts.most_common()),
        "per_episode": picked,
    }

    if os.path.exists(out_path) and not args.force:
        sys.exit(f"{out_path} already exists (use --force to overwrite)")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwrote {out_path}")
    print(
        f"play it with:\n  bash /scratch2/ml20/btripcon/FYP/jobs/run_human_play.sh "
        f"--dataset {args.dataset} \\\n      --playlist {os.path.abspath(out_path)}"
    )


if __name__ == "__main__":
    main()
