#!/usr/bin/env python3
"""Derive ApexNav's depth-camera params from the SAME composed Hydra config that
habitat_evaluation.py evaluates against.

Why this exists: the C++ mapper gets its intrinsics from roslaunch args, and the
camera height comes from habitat2ros/habitat_publisher.py -- neither of which ever
saw the Habitat sensor config. Both were hardcoded to the HM3D-ObjectNav LoCoBot
body (640x480, hfov 79, camera y=0.88). The 2026-08-11 OVON run (job 58962960)
used OVON's Stretch body (360x640, hfov 42, camera y=1.31) and scored 4.9% SR:
every projected point was rotated ~19.8 deg in yaw / ~10.7 deg in pitch and placed
0.43 m too low. Deriving here means changing an embodiment in
config/habitat_eval_<dataset>.yaml can no longer silently desync the mapper.

For the LoCoBot datasets (hm3dv1/hm3dv2/mp3d) this reproduces the launch-file
defaults 320 / 240 / 388.1910413097385 / 422.0475153598262 bit-for-bit, so wiring
it up is a no-op for the three already-validated datasets.

Emits exactly ONE line on stdout, e.g.:
    habitat_config:=habitat_eval_ovon cx:=180.0 cy:=320.0 fx:=... fy:=... \
        left_angle:=... right_angle:=...
Everything the imports emit is diverted to stderr -- see _load() below.

Authored by Claude (Anthropic Claude Opus 5) for Broden Tripcony.
"""
import argparse
import contextlib
import math
import os
import sys

import numpy as np
from hydra import initialize_config_dir, compose


# Repo paths resolved from this file, not the working directory. This script sits
# two levels below the ApexNav root, and hydra's initialize(config_path=...)
# resolves relative to the *calling file* -- a bare "config" would look under
# tools/ovon/ now that this no longer lives in the root. An absolute
# initialize_config_dir is immune to that and to wherever the caller is cd'd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")


def _load(dataset, overrides):
    """Compose the dataset's config, keeping stdout clean.

    Both imports are noisy on STDOUT, not stderr: `import ovon` prints a
    multi-line 'frontier_exploration package not installed' warning. Since the
    caller captures our stdout with $(...) and hands it straight to roslaunch,
    a single stray line there becomes a bogus roslaunch arg. Divert stdout to
    stderr for the whole import+compose phase; only the final result line, printed
    by main() after this returns, is allowed onto real stdout.
    """
    with contextlib.redirect_stdout(sys.stderr):
        # Registers habitat-lab's Hydra search-path plugin, without which the
        # `defaults: /benchmark/nav/objectnav` entry in every habitat_eval_*.yaml
        # cannot resolve (MissingConfigException).
        import habitat  # noqa: F401

        # Same ConfigStore override habitat_evaluation.py performs, and for the same
        # reason -- it must precede compose() or OVON's own config cannot resolve.
        if dataset == "ovon":
            import ovon  # noqa: F401

        with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
            return compose(config_name=f"habitat_eval_{dataset}", overrides=overrides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", required=True, choices=["hm3dv1", "hm3dv2", "mp3d", "ovon"]
    )
    ap.add_argument("--emit", default="roslaunch",
                    choices=["roslaunch", "camera_height"])
    args, overrides = ap.parse_known_args()

    cfg = _load(args.dataset, overrides)

    d = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    w, h, hfov = int(d.width), int(d.height), float(d.hfov)

    # Identical expression to basic_utils/object_point_cloud_utils/
    # object_point_cloud.py:43-44. Keep them identical: the occupancy map and the
    # object point cloud have to land in the same metric frame.
    fx = w / (2 * math.tan(hfov * np.pi / 360.0))
    fy = h / (2 * math.tan(hfov / w * h * np.pi / 360.0))
    cx, cy = w / 2.0, h / 2.0

    if args.emit == "camera_height":
        print(repr(float(d.position[1])))
        return

    # Half angle of the view cone ApexNav assumes, in radians.
    #
    # frontier_map2d.cpp:280 retires a frontier to dormant once it falls inside this
    # cone and an unoccluded raycast reaches it; dormant frontiers stop being
    # exploration targets, and draining the active list is what ends an episode as
    # "no frontier". So a cone WIDER than the real sensor retires frontiers the camera
    # never imaged. Upstream's 30 deg was picked against the LoCoBot's 79 deg hfov
    # (76% of its 39.5 deg true half angle, conservative); the same 30 deg against
    # OVON's 42 deg hfov claims 143% of what the Stretch camera can see, and 44.1% of
    # that run's episodes ended as "no frontier".
    #
    # min() keeps the LoCoBot datasets at exactly 30 deg while clamping ovon to its
    # true 21 deg half hfov. Deliberately NOT rescaled to preserve upstream's 76%
    # margin -- that would take ovon to 15.95 deg and slow exploration further against
    # an unchanged 500 step cap.
    half_angle_deg = min(30.0, hfov / 2.0)
    # 3.1415926, not math.pi: this reproduces algorithm.xml's own
    # $(eval 30 * 3.1415926 / 180.0) bit-for-bit, so passing the value explicitly is a
    # true no-op for hm3dv1/hm3dv2/mp3d rather than merely equal to nine decimals.
    half_angle_rad = half_angle_deg * 3.1415926 / 180.0

    # repr() to round-trip the double exactly -- for the LoCoBot body this
    # reproduces the in-tree literals 388.1910413097385 / 422.0475153598262 bitwise.
    print(
        f"habitat_config:=habitat_eval_{args.dataset} "
        f"cx:={cx!r} cy:={cy!r} fx:={fx!r} fy:={fy!r} "
        f"left_angle:={half_angle_rad!r} right_angle:={half_angle_rad!r}"
    )


if __name__ == "__main__":
    main()
