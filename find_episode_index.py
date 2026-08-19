"""Resolve a (scene_id, episode_id) pair to the test_epi_num index that selects it.

habitat_evaluation.py's `test_epi_num` is a 0-based INDEX INTO THE EPISODE
ITERATOR ORDER, not a dataset episode_id. The two are unrelated numbers: an
episode_id is only unique within its scene and is assigned by the dataset
generator, while the index depends on how EpisodeIterator groups the episode
list (group_by_scene defaults to True, so episodes come out clustered by scene,
not in file order).

This script rebuilds the exact same iterator habitat.Env would build --
env.py:139-148 composes iterator_options from the Hydra config and adds
habitat.seed -- and walks it WITHOUT constructing a simulator, so it costs a
couple of seconds and no GPU.

The index it prints is also (No. task number - 1) in the record.txt written by a
full sweep: habitat_evaluation.py increments num_total before write_record, so
iterator index i is logged as "No.(i+1) task".

Usage (inside the apexnav container, from the ApexNav directory):
    python find_episode_index.py --dataset hm3dv1 \
        --scene 00839-zt1RVoi7PcG --episode-id 118
    python find_episode_index.py --dataset hm3dv1 --index 1999   # reverse lookup
    python find_episode_index.py --dataset hm3dv1 --list | head

--scene is matched as a substring of the full scene_id path, so the short
"00839-zt1RVoi7PcG" form from a record.txt line is enough.

Authored by Claude (Anthropic Claude Opus 5) for Broden Tripcony.
"""
import argparse
import sys

from hydra import initialize, compose
from habitat.datasets import make_dataset


def build_iterator(dataset):
    """Mirror env.py:_setup_episode_iterator so indices match habitat.Env exactly."""
    if dataset == "ovon":
        # Same registration side effect habitat_evaluation.py:749 needs -- without
        # it make_dataset("OVON-v1", ...) raises "Could not find dataset OVON-v1".
        import ovon  # noqa: F401

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=f"habitat_eval_{dataset}")

    d = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
    iter_opts = dict(cfg.habitat.environment.iterator_options)
    iter_opts["seed"] = cfg.habitat.seed
    return d, d.get_episode_iterator(**iter_opts)


def describe(index, ep):
    print(f"test_epi_num={index}   (record.txt calls this No.{index + 1} task)")
    print(f"  scene_id        : {ep.scene_id}")
    print(f"  episode_id      : {ep.episode_id}")
    print(f"  object_category : {ep.object_category}")
    print(f"  start_position  : {ep.start_position}")
    print(f"  start_rotation  : {ep.start_rotation}")
    print(f"  goals           : {len(ep.goals)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", required=True, choices=["hm3dv1", "hm3dv2", "mp3d", "ovon"]
    )
    ap.add_argument("--scene", help="substring of scene_id, e.g. 00839-zt1RVoi7PcG")
    ap.add_argument("--episode-id", help="episode_id within that scene, e.g. 118")
    ap.add_argument("--index", type=int, help="reverse lookup: describe this index")
    ap.add_argument(
        "--list", action="store_true", help="print every index as TSV and exit"
    )
    args = ap.parse_args()

    d, it = build_iterator(args.dataset)
    print(f"# {args.dataset}: {len(d.episodes)} episodes", file=sys.stderr)

    if args.list:
        for i, ep in enumerate(it):
            print(f"{i}\t{ep.scene_id}\t{ep.episode_id}\t{ep.object_category}")
        return

    if args.index is not None:
        for i, ep in enumerate(it):
            if i == args.index:
                describe(i, ep)
                return
        sys.exit(f"index {args.index} is past the end of {args.dataset}")

    if not (args.scene and args.episode_id):
        sys.exit("need --scene and --episode-id together, or --index, or --list")

    for i, ep in enumerate(it):
        if args.scene in ep.scene_id and str(ep.episode_id) == str(args.episode_id):
            describe(i, ep)
            return
    sys.exit(f"no episode {args.episode_id} in a scene matching '{args.scene}'")


if __name__ == "__main__":
    main()
