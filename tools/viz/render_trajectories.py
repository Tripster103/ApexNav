#!/usr/bin/env python3
"""Draw human / ApexNav / shortest-path trajectories for one ObjectNav episode.

Reads the per-step pose logs written by human_play.py
(<out>/trajectories/NNN_<scene>_<id>.json) and by a single-episode
habitat_evaluation.py run (videos/.../single/<run>/<scene>_<id>_traj.json), adds
the geodesic shortest path from the start to the nearest goal view point, and
writes two figures, one panel per trajectory (PIRLNav Fig. 2 style):

  <out>_map.png       flat navmesh top-down map (navigable area in grey)
  <out>_textured.png  orthographic RGB render of the scene from above, with the
                      ceiling cut away by placing the camera --cut metres above
                      the floor (only geometry below the camera is drawn)

Both mark the start (blue square), the goal instances on this floor (magenta
squares), their view points = the success area (green), and each route.

Needs the GPU (habitat-sim renders through EGL even for the navmesh):
    bash scripts/render_trajectories.sh --dataset mp3d \\
        --traj Human=<human_traj.json> --traj ApexNav=<agent_traj.json> \\
        --out paper/figs/traj_<scene>_<id>

Multi-floor episodes: everything is drawn for one floor, the start floor by
default. Rerun with --floor-y <goal floor height> for the other floor; route
points more than 1 m off the drawn floor are faded.

Authored by Claude (Anthropic Claude Opus 5.5) for Broden Tripcony.
"""
import argparse
import gzip
import json
import math
import os

print("[render] importing", flush=True)
import magnum as mn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import habitat_sim  # noqa: E402

print("[render] imports done", flush=True)

CONTENT = {
    "mp3d": "data/datasets/objectnav/mp3d/v1/val/content",
    "hm3dv1": "data/datasets/objectnav/hm3d/v1/val/content",
    "hm3dv2": "data/datasets/objectnav/hm3d/v2/val/content",
}
# Same agent navmesh as the eval (default_agent_navmesh rebuilds it for this agent).
NAVMESH = dict(agent_radius=0.18, agent_height=0.88, agent_max_climb=0.2, agent_max_slope=45.0)
COLOURS = {"Human": "#e8743b", "ApexNav": "#2f7fd6", "Shortest path": "#1fa37a"}
EXTRA_COLOURS = ["#9b59b6", "#c0392b", "#16a085"]
OFF_FLOOR_M = 1.0


def load_episode(dataset, scene_id, episode_id, category):
    scene = os.path.basename(scene_id).split(".")[0]
    with gzip.open(os.path.join(CONTENT[dataset], f"{scene}.json.gz")) as fh:
        d = json.load(fh)
    # habitat-lab renumbers episode_id to the file index (object_nav_dataset.py:131).
    ep = d["episodes"][int(episode_id)]
    if ep["object_category"] != category:
        raise SystemExit(f"episode {episode_id} in {scene} is a {ep['object_category']}, "
                         f"trajectory says {category} -- wrong --dataset?")
    goals = d["goals_by_category"][f"{os.path.basename(ep['scene_id'])}_{category}"]
    return ep, goals


def make_sim(scene_path, res, cut_pos, ortho_scale):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_path
    backend.gpu_device_id = 0
    agent = habitat_sim.agent.AgentConfiguration()
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = "top"
    spec.sensor_type = habitat_sim.SensorType.COLOR
    spec.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    spec.resolution = [res, res]
    spec.position = [0.0, 0.0, 0.0]
    spec.orientation = [-math.pi / 2, 0.0, 0.0]  # look straight down
    spec.ortho_scale = ortho_scale
    agent.sensor_specifications = [spec]
    print(f"[render] loading scene {scene_path}", flush=True)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    print("[render] scene loaded, rebuilding navmesh", flush=True)

    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    for k, v in NAVMESH.items():
        setattr(ns, k, v)
    sim.recompute_navmesh(sim.pathfinder, ns)

    st = habitat_sim.AgentState()
    st.position = np.array(cut_pos, dtype=np.float32)
    sim.get_agent(0).set_state(st)
    return sim


def shortest_path(pathfinder, start, goals):
    req = habitat_sim.MultiGoalShortestPath()
    req.requested_start = np.array(start, dtype=np.float32)
    req.requested_ends = [np.array(vp["agent_state"]["position"], dtype=np.float32)
                          for g in goals for vp in g["view_points"]]
    if not pathfinder.find_path(req):
        return None, float("inf")
    return np.array([list(p) for p in req.points]), req.geodesic_distance


def draw_panels(bg, to_px, routes, start, goals, floor_y, title, out_png):
    fig, axes = plt.subplots(1, len(routes), figsize=(4.2 * len(routes), 4.6))
    axes = np.atleast_1d(axes)
    on_floor = [g for g in goals if abs(g["position"][1] - floor_y) < 2.0]
    vps = np.array([vp["agent_state"]["position"] for g in on_floor for vp in g["view_points"]
                    if abs(vp["agent_state"]["position"][1] - floor_y) < OFF_FLOOR_M])
    for ax, (label, pts, colour) in zip(axes, routes):
        ax.imshow(bg)
        if len(vps):
            u, v = to_px(vps)
            ax.scatter(u, v, s=4, c="#2ecc71", alpha=0.35, linewidths=0, label="Success area")
        if on_floor:
            u, v = to_px(np.array([g["position"] for g in on_floor]))
            ax.scatter(u, v, s=70, marker="s", facecolors="none", edgecolors="#ff2bd6",
                       linewidths=1.8, label="Goal object")
        if pts is not None and len(pts):
            u, v = to_px(pts)
            near = np.abs(pts[:, 1] - floor_y) < OFF_FLOOR_M
            ax.plot(u, v, color=colour, lw=2.2, alpha=0.25)
            ax.plot(np.where(near, u, np.nan), np.where(near, v, np.nan), color=colour, lw=2.2)
            ax.scatter(u[-1:], v[-1:], s=60, marker="X", c=colour, edgecolors="white",
                       linewidths=0.8, zorder=5, label="Stop")
        u, v = to_px(np.array([start]))
        ax.scatter(u, v, s=90, marker="s", c="#3b3bff", edgecolors="white", linewidths=1,
                   zorder=6, label="Start")
        ax.set_xlim(0, bg.shape[1])
        ax.set_ylim(bg.shape[0], 0)
        ax.set_title(label, fontsize=11)
        ax.set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"wrote {out_png} (+ .pdf)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(CONTENT))
    p.add_argument("--traj", action="append", default=[], metavar="LABEL=PATH",
                   help="a pose log, repeatable; LABEL is the panel title (Human, ApexNav, ...)")
    p.add_argument("--no-shortest", action="store_true", help="omit the shortest-path panel")
    p.add_argument("--out", required=True, help="output prefix, e.g. paper/figs/traj_X_29")
    p.add_argument("--floor-y", type=float, help="floor height to draw (default: start height)")
    p.add_argument("--cut", type=float, default=1.8, help="camera height above the floor (m)")
    p.add_argument("--off-floor", type=float, default=1.0,
                   help="route points further than this from --floor-y are faded (m); raise it for split levels")
    p.add_argument("--margin", type=float, default=1.5, help="crop margin around the routes (m)")
    p.add_argument("--res", type=int, default=1600, help="textured render resolution (px)")
    p.add_argument("--mpp", type=float, default=0.02, help="navmesh map metres per pixel")
    args = p.parse_args()
    if not args.traj:
        p.error("give at least one --traj LABEL=PATH")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    global OFF_FLOOR_M
    OFF_FLOOR_M = args.off_floor

    trajs = []
    for item in args.traj:
        label, path = item.split("=", 1)
        with open(path) as fh:
            t = json.load(fh)
        trajs.append((label, t))
    ref = trajs[0][1]
    for label, t in trajs[1:]:
        if (t["scene_id"], str(t["episode_id"])) != (ref["scene_id"], str(ref["episode_id"])):
            raise SystemExit(f"{label} is a different episode from {trajs[0][0]}")

    ep, goals = load_episode(args.dataset, ref["scene_id"], ref["episode_id"], ref["category"])
    start = ep["start_position"]
    floor_y = start[1] if args.floor_y is None else args.floor_y

    routes = []
    for i, (label, t) in enumerate(trajs):
        pts = np.array([pose["p"] for pose in t["poses"]])
        colour = COLOURS.get(label, EXTRA_COLOURS[i % len(EXTRA_COLOURS)])
        routes.append((f"{label} ({t['result']}, {t['steps']} steps)", pts, colour))

    # Crop: every route point on this floor, plus the start, plus margin; squared.
    xz = [np.array([start])[:, [0, 2]]] + [r[1][np.abs(r[1][:, 1] - floor_y) < OFF_FLOOR_M][:, [0, 2]]
                                            for r in routes if len(r[1])]
    xz = np.concatenate(xz)
    # ...plus goal instances on this floor within 3 m of a route, so the goal
    # and its success area are in frame rather than clipped.
    for g in goals:
        gp = np.array(g["position"])
        if abs(gp[1] - floor_y) < 2.0 and np.min(np.linalg.norm(xz - gp[[0, 2]], axis=1)) < 3.0:
            xz = np.vstack([xz, gp[[0, 2]]])
    lo, hi = xz.min(0) - args.margin, xz.max(0) + args.margin
    centre, extent = (lo + hi) / 2, float(max(hi - lo))

    scene_path = ref["scene_id"]
    sim = make_sim(scene_path, args.res, [centre[0], floor_y + args.cut, centre[1]], 1.0 / extent)
    try:
        if not args.no_shortest:
            sp, geo = shortest_path(sim.pathfinder, start, goals)
            routes.append((f"Shortest path ({geo:.2f} m)", sp, COLOURS["Shortest path"]))
            if sp is not None:  # widen the crop if the shortest path leaves it
                more = sp[np.abs(sp[:, 1] - floor_y) < OFF_FLOOR_M][:, [0, 2]]
                if len(more) and ((more < lo).any() or (more > hi).any()):
                    print("note: shortest path leaves the crop; pass a larger --margin to include it")

        title = (f"{args.dataset} {os.path.basename(scene_path)} ep {ref['episode_id']}"
                 f" -- find '{ref['category']}'")

        # 1) Flat navmesh map. get_topdown_view's grid starts at the navmesh lower bound.
        lb = sim.pathfinder.get_bounds()[0]
        nav = sim.pathfinder.get_topdown_view(args.mpp, floor_y)
        c0, r0 = [int((lo[0] - lb[0]) / args.mpp), int((lo[1] - lb[2]) / args.mpp)]
        n = int(extent / args.mpp)
        crop = np.zeros((n, n), dtype=bool)
        rs, cs = max(r0, 0), max(c0, 0)
        sub = nav[rs:r0 + n, cs:c0 + n]
        crop[rs - r0:rs - r0 + sub.shape[0], cs - c0:cs - c0 + sub.shape[1]] = sub
        bg_map = np.where(crop[..., None], [0.80, 0.80, 0.78], [1.0, 1.0, 1.0])

        def map_px(pts):
            return (pts[:, 0] - lo[0]) / args.mpp, (pts[:, 2] - lo[1]) / args.mpp

        print("[render] drawing navmesh map", flush=True)
        draw_panels(bg_map, map_px, routes, start, goals, floor_y, title, args.out + "_map.png")

        # 2) Textured orthographic render. Pixel coords come from the render
        #    camera's own matrices, so the overlay is right whatever the extent
        #    convention of ortho_scale turns out to be.
        print("[render] rendering textured view", flush=True)
        rgb = sim.get_sensor_observations()["top"][..., :3]
        cam = sim._sensors["top"]._sensor_object.render_camera
        world_w = 2.0 / cam.projection_matrix[0][0]
        if abs(world_w - extent) / extent > 0.05:
            print(f"note: render covers {world_w:.2f} m, crop is {extent:.2f} m "
                  "(ortho_scale convention differs); overlay is still exact")

        def tex_px(pts):
            u, v = [], []
            for q in pts:
                ndc = cam.projection_matrix.transform_point(
                    cam.camera_matrix.transform_point(mn.Vector3(*[float(c) for c in q])))
                u.append((ndc.x + 1) / 2 * args.res)
                v.append((1 - ndc.y) / 2 * args.res)
            return np.array(u), np.array(v)

        draw_panels(rgb, tex_px, routes, start, goals, floor_y, title, args.out + "_textured.png")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
