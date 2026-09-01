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
episode from each in turn, and keep looping until --n are collected. Every val
split has fewer than 50 scenes (mp3d 11, hm3dv1 20, hm3dv2 36, ovon 36), so with
the default --n 50 the first pass alone reaches every scene: **a playlist always
covers 100% of the split's scenes**, and no scene contributes more than
ceil(n/scenes). This is checked and recorded in the output as
selection.scene_coverage; --n below the scene count is refused unless you pass
--allow-partial-coverage.

Play order: NOT a plain shuffle. A uniform shuffle regularly puts two episodes
from the same scene back to back -- every seed-1 playlist generated before
2026-09-01 had a worst same-scene gap of 1 -- and replaying a floorplan you just
walked hands the player a map the agent never had. spread_play_order() instead
deals the episodes out under the widest same-scene cooldown the counts allow,
picking at random among the eligible scenes with the most episodes left. That
reaches the spacing ceiling the shape permits (mp3d 10, hm3dv1 16, hm3dv2/ovon
25+ slots between repeats) while staying non-cyclic -- a different --seed gives a
different sequence, so the player cannot anticipate which scene comes next.

--sets N carves N mutually disjoint playlists in one go (this is what the missing
split_playlists.py used to do, and did wrong: it dealt a flat shuffled pool, so
each set covered only 26-31 of 36 scenes). Dealing is scene-aware -- each scene's
pool episodes are dealt round-robin across the sets -- so **every set covers every
scene**, and each set gets its own spread ordering.

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


def spread_play_order(picked, rng, attempts=200):
    """Order the chosen episodes so repeats of a scene sit as far apart as possible.

    A plain rng.shuffle() leaves same-scene episodes adjacent surprisingly often
    (p(min gap == 1) was ~1 for a 50/36 split across 400 trials). Replaying a
    floorplan back to back is the one thing a human playlist must not do: the
    second run is not a navigation trial any more, it is a recall trial, and the
    agent never had that advantage.

    Greedy with an adaptive cooldown. At each slot, take the widest cooldown k
    that still leaves some scene playable, then choose at random among the
    eligible scenes holding the most unplayed episodes. The cooldown does the
    spreading -- it is what pushes the minimum gap to the ceiling the counts
    allow -- and the random tiebreak is what stops the result being a fixed
    round-robin cycle the player could anticipate. Best-of-`attempts` on
    (min gap, then mean gap).

    Verified against the four real split shapes at n=50: mp3d reaches a minimum
    gap of 10 (ceiling 50//5), hm3dv1 16 (ceiling 50//3), hm3dv2 27 and ovon 26
    (ceiling 50//2 = 25, beaten because only some scenes repeat). Five seeds give
    five distinct orderings.
    """
    by_scene = defaultdict(list)
    for r in picked:
        by_scene[r["scene"]].append(r)
    counts = {s: len(v) for s, v in by_scene.items()}

    def one_pass():
        rem = dict(counts)
        order, lastpos = [], {}
        for i in range(len(picked)):
            live = [s for s, c in rem.items() if c]
            for k in range(len(live) - 1, -1, -1):
                elig = [s for s in live if s not in lastpos or i - lastpos[s] > k]
                if elig:
                    break
            top = max(rem[s] for s in elig)
            s = rng.choice(sorted(x for x in elig if rem[x] == top))
            order.append(s)
            rem[s] -= 1
            lastpos[s] = i
        return order

    best = None
    for _ in range(attempts):
        order = one_pass()
        score = same_scene_gaps(order)
        if best is None or score > best[0]:
            best = (score, order)

    pool = {s: list(v) for s, v in by_scene.items()}
    for v in pool.values():
        rng.shuffle(v)
    return [pool[s].pop() for s in best[1]], best[0]


def same_scene_gaps(scenes):
    """(minimum, mean) slot distance between consecutive repeats of a scene.

    A gap of 1 means the same floorplan is played twice in a row. Returns the
    playlist length when nothing repeats, so a fully-unique playlist scores best.
    """
    lastpos, gaps = {}, []
    for i, s in enumerate(scenes):
        if s in lastpos:
            gaps.append(i - lastpos[s])
        lastpos[s] = i
    if not gaps:
        return (len(scenes), float(len(scenes)))
    return (min(gaps), sum(gaps) / len(gaps))


def deal_disjoint_sets(picked, n_sets, rng):
    """Split a pool into n_sets disjoint playlists, dealing scene by scene.

    Replaces the (lost) split_playlists.py, which dealt the flat shuffled pool
    and so let whole scenes miss a set entirely -- the hm3dv2 seed-1 sets covered
    26, 30 and 26 of 36 scenes. Dealing each scene's episodes round-robin instead
    means a scene holding >= n_sets pool episodes lands in every set, which is
    what --n * --sets stratified over the split always produces here.

    Two-level balance, because coverage alone is not enough. Each episode goes
    to a set holding the fewest of ITS OWN scene (that is what spreads a scene
    across every set), and among those, to the set that is globally smallest
    (that is what keeps the sets the same size). A plain rotating-offset deal
    gets the first property but not the second -- it produced 49/50/51 rather
    than 50/50/50 on a 150-episode pool -- and unequal sets mean the per-set
    averages a human benchmark reports are over different N.
    """
    by_scene = defaultdict(list)
    for r in picked:
        by_scene[r["scene"]].append(r)
    sets = [[] for _ in range(n_sets)]
    per_scene = [Counter() for _ in range(n_sets)]
    for scene in sorted(by_scene):
        eps = by_scene[scene]
        rng.shuffle(eps)
        for ep in eps:
            fewest = min(per_scene[i][scene] for i in range(n_sets))
            cands = [i for i in range(n_sets) if per_scene[i][scene] == fewest]
            smallest = min(len(sets[i]) for i in cands)
            i = rng.choice([c for c in cands if len(sets[c]) == smallest])
            sets[i].append(ep)
            per_scene[i][scene] += 1
    return sets


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
        "--sets",
        type=int,
        default=1,
        help="write N mutually disjoint playlists (setA, setB, ...) instead of "
        "one; each still covers every scene (default 1)",
    )
    ap.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help="permit a playlist that misses some scenes (refused by default)",
    )
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
    if args.sets < 1:
        sys.exit("--sets must be at least 1")
    need = args.n * args.sets
    if need > len(rows):
        sys.exit(
            f"asked for {args.n} x {args.sets} = {need} episodes but only "
            f"{len(rows)} are available after filtering ({total} in the dataset)"
        )
    if args.n < n_scenes and not args.allow_partial_coverage:
        sys.exit(
            f"--n {args.n} is below the {n_scenes} scenes in this split, so a "
            "playlist cannot cover them all. Raise --n to at least "
            f"{n_scenes}, or pass --allow-partial-coverage if that is intended."
        )

    rng = random.Random(args.seed)
    # Stratify over the whole pool first, then deal. Stratifying decides *which*
    # episodes; spread_play_order below decides the order they are played in.
    pool = stratified_sample(rows, need, rng)
    groups = (
        deal_disjoint_sets(pool, args.sets, rng) if args.sets > 1 else [pool]
    )

    base = args.name or f"random{args.n}_seed{args.seed}"
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "playlists")
    labels = (
        [""]
        if args.sets == 1
        else [f"_set{chr(ord('A') + i)}" for i in range(args.sets)]
    )
    paths = [
        os.path.join(out_dir, f"{args.dataset}_{base}{lab}.json") for lab in labels
    ]

    print(
        f"dataset : {args.dataset}  ({total} episodes, {total_scenes} scenes)"
        + (f"  -> {len(rows)} episodes, {n_scenes} scenes after filtering"
           if len(rows) != total else "")
    )
    if args.sets > 1:
        print(f"pool    : {len(pool)} episodes dealt scene-wise into {args.sets} "
              "disjoint sets")

    written = []
    for lab, path, picked in zip(labels, paths, groups):
        picked, (min_gap, mean_gap) = spread_play_order(picked, rng)
        scene_counts = Counter(r["scene"] for r in picked)
        cat_counts = Counter(r["category"] for r in picked)
        covered = len(scene_counts)

        tag = f"set{lab[-1]}" if lab else "playlist"
        print(
            f"\n{tag}: {len(picked)} episodes | scenes {covered}/{n_scenes}"
            f" | per-scene {min(scene_counts.values())}-{max(scene_counts.values())}"
            f" | same-scene gap min={min_gap} mean={mean_gap:.1f}"
        )
        print(f"  categories: {len(cat_counts)}  {dict(cat_counts.most_common(8))}")
        print(f"  first 10 indices (play order): {[r['index'] for r in picked[:10]]}")

        if covered < n_scenes and not args.allow_partial_coverage:
            missing = sorted(set(r["scene"] for r in rows) - set(scene_counts))
            sys.exit(
                f"{tag} covers only {covered}/{n_scenes} scenes, missing "
                f"{', '.join(missing)} -- refusing to write. This should not "
                "happen with --n >= scene count; pass --allow-partial-coverage "
                "to override."
            )
        if min_gap == 1:
            print(
                f"  WARNING: {tag} plays a scene twice back to back -- the "
                "second run is a recall trial, not a navigation trial",
                file=sys.stderr,
            )

        meta = {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "generator": "make_playlist.py",
            "source": {
                "dataset": args.dataset,
                "n_episodes_total": total,
                "n_scenes_total": total_scenes,
            },
            "selection": {
                "name": f"{base}{lab}",
                "n_selected": len(picked),
                "indexing": "0-based test_epi_num into the habitat episode iterator",
                "strategy": (
                    "stratified round-robin over scenes"
                    + (f", dealt scene-wise into {args.sets} disjoint sets"
                       if args.sets > 1 else "")
                    + ", then max-spread play order (adaptive same-scene cooldown,"
                      " random among eligible)"
                ),
                "n": args.n,
                "seed": args.seed,
                "sets": args.sets,
                "target": args.target,
                "scene": args.scene,
                "scene_coverage": {
                    "scenes_covered": covered,
                    "scenes_in_split": n_scenes,
                    "complete": covered == n_scenes,
                },
                "play_order": {
                    "min_same_scene_gap": min_gap,
                    "mean_same_scene_gap": round(mean_gap, 2),
                    "note": "slots between consecutive episodes of the same scene; 1 == back to back",
                },
                "indices_0based": [r["index"] for r in picked],
            },
            "scene_counts": dict(scene_counts.most_common()),
            "target_counts": dict(cat_counts.most_common()),
            "per_episode": picked,
        }
        if args.sets > 1:
            meta["selection"]["disjoint_with"] = [
                os.path.basename(q) for q in paths if q != path
            ]

        if args.dry_run:
            continue
        if os.path.exists(path) and not args.force:
            sys.exit(f"{path} already exists (use --force to overwrite)")
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        written.append(path)

    if args.dry_run:
        print("\n[dry run] nothing written")
        return

    print("\nwrote:")
    for path in written:
        print(f"  {path}")
    print(
        f"play it with:\n  bash /scratch2/ml20/btripcon/FYP/jobs/run_human_play.sh "
        f"--dataset {args.dataset} \\\n      --playlist {os.path.abspath(written[0])}"
    )


if __name__ == "__main__":
    main()
