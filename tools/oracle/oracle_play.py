#!/usr/bin/env python3
"""Drive an ObjectNav episode with a geodesic ORACLE and score it like ApexNav.

Why this exists: when ApexNav (or a human) fails an episode, "the agent was bad"
and "the episode is unwinnable as we run it" look identical in record.txt. This
removes the ambiguity. It hands the oracle the answer -- the goal view-point
coordinates -- and then walks there using EXACTLY the agent's action space, step
size, turn angle, embodiment, navmesh and success radius. Whatever it scores is
the ceiling for that episode under this configuration:

  * oracle succeeds  -> the episode is winnable; the failure was the agent's.
  * oracle fails     -> the episode is not winnable as we run it, and the agent's
                        failure should not be counted against it uncaveated.

It prints a REACHABILITY table before moving: the geodesic distance from the
episode start to every goal view point, next to the dataset's own recorded
geodesic_distance. `inf` rows are the smoking gun -- they mean our recomputed
navmesh (config/habitat_eval_ovon.yaml's navmesh_settings block replaces each
scene's shipped one) put that view point on a different island from the start,
so no action sequence whatsoever reaches it. Agreement between the dataset's
recorded geodesic_distance and the best finite row is the converse check: it
says our navmesh reproduces theirs on the route that matters.

TWO oracles run, because they fail for different reasons and the difference is
diagnostic:

  planner  -- A* over the discrete state space (x cell, z cell, heading), with
              agent.act() transitions on the real navmesh, so collisions and
              allow_sliding:false are honoured. Optimal in ACTION COUNT, and
              complete: "no plan" means no action sequence exists within the
              step budget, not that a controller gave up. This is the real
              ceiling. (Geodesic-greedy was tried first and is useless here --
              facing a wall with sliding off, forward/left/right all score the
              same and it deadlocks.)
  follower -- habitat's stock ShortestPathFollower to one view point. Kept
              because it is what tools/human/human_play.py's StepSPL oracle
              uses; when it fails and the planner succeeds, StepSPL's t* for
              that episode is wrong, which is worth knowing.

NOTE on the follower: ShortestPathFollower defaults to stop_on_error=True, which
converts a GreedyFollowerError into a STOP. That makes a planner failure look
exactly like a successful arrival. It is constructed with stop_on_error=False
here so the failure is reported as a failure.

Deliberately does NOT touch the ROS stack, and does NOT write
record.txt/continue.txt -- an oracle is a diagnostic, not a baseline, and must
never land in a results directory analyze_failures.py reads as a run. Output is
a printed report plus, with --video, one mp4 and a sidecar json.

Needs a GPU node (habitat-sim renders through EGL). Use the wrapper, which
carries the 9-bind NVIDIA/EGL set:

    bash /scratch2/ml20/btripcon/FYP/scripts/run_oracle_play.sh \
        --dataset ovon --episode 1053 --video

--episode is a 0-based ITERATOR INDEX, not an episode_id -- the same index
human_play.py and the playlists use. Resolve one with
tools/human/find_episode_index.py.

Authored by Claude (Anthropic Claude Opus 5) for Broden Tripcony.
"""
import argparse
import heapq
import itertools
import json
import math
import os
import sys
import time

import numpy as np

# Same path bootstrap as tools/human/human_play.py: this file sits two levels
# below the ApexNav root, so neither hydra's config_path nor `import basic_utils`
# resolves without it.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

from hydra import compose, initialize_config_dir

import habitat
import habitat_sim
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
from prettytable import PrettyTable

# The three body actions, as the integer keys habitat-lab installs into the
# habitat_sim agent's action_space (habitat_simulator.py:406). agent.act() takes
# those keys directly, which is what lets the planner probe a move without
# rendering. look_up/look_down are excluded on purpose: pitch moves the camera,
# never the base, so it cannot change distance_to_goal.
BODY_ACTIONS = (
    HabitatSimActions.move_forward,
    HabitatSimActions.turn_left,
    HabitatSimActions.turn_right,
)
ACTION_NAMES = {
    HabitatSimActions.move_forward: "F",
    HabitatSimActions.turn_left: "L",
    HabitatSimActions.turn_right: "R",
    HabitatSimActions.stop: "STOP",
}


def candidate_view_points(env):
    """Every position the success measure will accept, in metric order.

    Mirrors compute_oracle_step_count() in tools/human/human_play.py, which in
    turn mirrors habitat_evaluation.py:233 and OVONDistanceToGoal.reset_metric
    (ovon/measurements/nav.py:45-59): for OVON the candidate set is extended with
    the view points of the episode's `children_object_categories`, so a `table`
    episode also succeeds on a `desk`. The getattr guards make the extension a
    no-op for hm3dv1/hm3dv2/mp3d.

    Returns (label, position) pairs so the reachability table can say which goal
    object each view point belongs to.
    """
    episode = env.current_episode
    out = []
    for goal in episode.goals:
        oid = getattr(goal, "object_id", "?")
        for i, vp in enumerate(goal.view_points):
            out.append((f"{oid}#vp{i}", np.asarray(vp.agent_state.position, dtype=np.float64)))

    child_categories = getattr(episode, "children_object_categories", None) or []
    goals_by_category = getattr(getattr(env, "_dataset", None), "goals_by_category", None)
    if child_categories and goals_by_category is not None:
        scene_key = episode.scene_id.split("/")[-1]
        for child_category in child_categories:
            child_goals = goals_by_category.get(f"{scene_key}_{child_category}")
            if child_goals is None:
                continue
            for goal in child_goals:
                oid = getattr(goal, "object_id", "?")
                for i, vp in enumerate(goal.view_points):
                    out.append((
                        f"{child_category}:{oid}#vp{i}",
                        np.asarray(vp.agent_state.position, dtype=np.float64),
                    ))
    return out


def reachability_report(env, candidates):
    """Geodesic distance from the episode start to each candidate view point.

    sim.geodesic_distance returns inf when the two points sit on different
    navmesh islands -- i.e. when no action sequence connects them. That is the
    only definitive "impossible" signal available, and it costs milliseconds, so
    it runs before any rollout rather than after a 500-step failure.
    """
    start = env.current_episode.start_position
    rows = [
        (label, pos, float(env.sim.geodesic_distance(start, [list(map(float, pos))], None)))
        for label, pos in candidates
    ]
    rows.sort(key=lambda r: (not math.isfinite(r[2]), r[2]))
    return rows


def _goal_distance(sim, position, goal_positions, cache=None, key=None):
    """Geodesic distance from `position` to the NEAREST candidate view point.

    The same quantity the success measure thresholds, from the same pathfinder
    call, so searching on it is searching on the metric itself rather than a
    proxy. `cache` keys on the discretised state so A* pays for each cell once.
    """
    if cache is not None and key in cache:
        return cache[key]
    d = float(sim.geodesic_distance(position, goal_positions, None))
    if cache is not None:
        cache[key] = d
    return d


def _heading_index(rotation, n_headings):
    """Yaw as an exact integer, because every turn is a whole turn_angle.

    Discretising heading is what makes the A* state space finite. It is exact,
    not approximate: the only rotations reachable from the start pose are
    multiples of turn_angle from it, so rounding never merges two genuinely
    different orientations."""
    fwd = quaternion_rotate_vector(rotation, np.array([0.0, 0.0, -1.0]))
    ang = math.atan2(float(fwd[0]), -float(fwd[2]))
    return int(round(ang / (2 * math.pi / n_headings))) % n_headings


def astar_plan(sim, start_state, goal_positions, success_distance, forward_step,
               n_headings, grid, max_expansions, max_actions):
    """Complete search for an action sequence that ends inside the success radius.

    THIS is the oracle. Earlier drafts used geodesic-greedy descent and it was
    useless here: with allow_sliding:false, an agent facing a wall gets the same
    goal distance from forward (blocked, no movement), left and right, so every
    candidate scores identically and greedy deadlocks -- observed on ovon 1053,
    stuck at 7.2 m for 60 steps with an 11.9 m path clearly available. Any
    distance-descent rule has that failure; a search does not.

    State = (x cell, z cell, heading index). Transitions are agent.act() on the
    real navmesh, so collisions and no-sliding are honoured exactly as env.step()
    honours them -- act() just skips rendering, which is what makes 10^5
    expansions affordable. g = actions taken, h = geodesic distance / step size,
    which is admissible (no action can close more ground than one forward step),
    so the first solution popped is optimal in ACTION COUNT -- the same quantity
    StepSPL normalises by.

    Returns (actions, info). actions is None when no sequence exists within
    max_actions and max_expansions; info says which of those limits bit, so
    "impossible" is never confused with "gave up".
    """
    agent = sim.get_agent(0)
    outer = agent.get_state()

    def restore(pos, rot):
        st = habitat_sim.AgentState()
        st.position = pos
        st.rotation = rot
        agent.set_state(st, reset_sensors=False)

    def key_of(pos, rot):
        return (int(round(float(pos[0]) / grid)), int(round(float(pos[2]) / grid)),
                _heading_index(rot, n_headings))

    dist_cache = {}
    start_key = key_of(start_state.position, start_state.rotation)
    start_d = _goal_distance(sim, start_state.position, goal_positions, dist_cache, start_key)
    if start_d <= success_distance:
        restore(outer.position, outer.rotation)
        return [], {"expansions": 0, "reason": "already_inside_radius", "optimal_actions": 0}

    counter = itertools.count()
    open_heap = [(start_d / forward_step, 0, next(counter), start_key,
                  np.array(start_state.position, dtype=np.float32), start_state.rotation, None)]
    best_g = {start_key: 0}
    parents = {}          # key -> (parent_key, action)
    closest = (start_d, start_key)
    expansions = 0
    reason = "exhausted_search_space"

    while open_heap:
        f, g, _, key, pos, rot, _p = heapq.heappop(open_heap)
        if g > best_g.get(key, math.inf):
            continue
        expansions += 1
        if expansions > max_expansions:
            reason = f"expansion_limit_{max_expansions}"
            break
        if g >= max_actions:
            continue

        for action in BODY_ACTIONS:
            restore(pos, rot)
            agent.act(action)
            st = agent.get_state()
            nkey = key_of(st.position, st.rotation)
            if nkey == key:
                continue  # blocked forward, or a turn that changed nothing
            ng = g + 1
            if ng >= best_g.get(nkey, math.inf):
                continue
            nd = _goal_distance(sim, st.position, goal_positions, dist_cache, nkey)
            best_g[nkey] = ng
            parents[nkey] = (key, action)
            if nd < closest[0]:
                closest = (nd, nkey)
            if nd <= success_distance:
                actions = []
                cur = nkey
                while cur in parents:
                    cur, a = parents[cur]
                    actions.append(a)
                actions.reverse()
                restore(outer.position, outer.rotation)
                return actions, {"expansions": expansions, "reason": "solved",
                                 "optimal_actions": len(actions)}
            heapq.heappush(open_heap, (ng + nd / forward_step, ng, next(counter), nkey,
                                       np.array(st.position, dtype=np.float32),
                                       st.rotation, key))

    restore(outer.position, outer.rotation)
    return None, {"expansions": expansions, "reason": reason,
                  "closest_found": closest[0]}


def rollout_actions(env, obs0, actions, success_distance, need_video):
    """Replay a planned action sequence through the REAL env.

    env.step (not agent.act) on purpose: only the env updates the measurements,
    so the success/SPL/distance_to_goal this returns come off exactly the code
    path that scored ApexNav's run. Stops early the moment the measure says we
    are inside the radius -- the plan was searched on the same quantity, so this
    normally fires on the last action."""
    frames = [_frame(obs0, env.get_metrics(), 0)] if need_video else []
    steps = 0
    closest = float(env.get_metrics()["distance_to_goal"])
    for action in actions:
        obs = env.step(action)
        steps += 1
        info = env.get_metrics()
        closest = min(closest, float(info["distance_to_goal"]))
        if need_video:
            frames.append(_frame(obs, info, steps))
        if closest <= success_distance:
            break
    status = "within_radius" if closest <= success_distance else "plan_replayed_not_inside"
    return dict(status=status, steps=steps, closest=closest, frames=frames,
                actions=[ACTION_NAMES[a] for a in actions[:steps]])


def rollout_follower(env, obs0, target, max_steps, success_distance, need_video):
    """habitat's stock ShortestPathFollower toward ONE view point.

    stop_on_error=False on purpose -- see the module docstring. The default True
    turns a GreedyFollowerError into a STOP action, which is indistinguishable
    from the follower deciding it has arrived."""
    follower = ShortestPathFollower(env.sim, success_distance, False, stop_on_error=False)
    frames = [_frame(obs0, env.get_metrics(), 0)] if need_video else []
    steps = 0
    closest = float(env.get_metrics()["distance_to_goal"])

    while steps < max_steps - 1:
        try:
            action = follower.get_next_action(target)
        except Exception as e:  # noqa: BLE001 -- GreedyFollowerError and friends
            return dict(status=f"follower_error:{type(e).__name__}", steps=steps,
                        closest=closest, frames=frames, actions=[])
        if action is None or action == HabitatSimActions.stop:
            return dict(status="follower_called_stop", steps=steps, closest=closest,
                        frames=frames, actions=[])
        obs = env.step(action)
        steps += 1
        info = env.get_metrics()
        closest = min(closest, float(info["distance_to_goal"]))
        if need_video:
            frames.append(_frame(obs, info, steps))

    return dict(status="step_budget_exhausted", steps=steps, closest=closest,
                frames=frames, actions=[])


def _frame(obs, info, steps):
    info = dict(info)
    img = observations_to_image(obs, info)
    info.pop("top_down_map", None)
    d = float(info.get("distance_to_goal", 0.0))
    return overlay_frame(img, {"step": steps, "d2g": round(d, 3) if math.isfinite(d) else "inf"})


def build_config(dataset, with_map):
    """Config composition identical to tools/human/human_play.py's main(), so the
    oracle runs the same embodiment, navmesh, step size and success radius the
    sweep did. Any divergence here would make the ceiling it reports meaningless."""
    if dataset == "ovon":
        import ovon  # noqa: F401  -- registers OVON-v1
        import ovon.measurements.nav  # noqa: F401  -- registers OVONDistanceToGoal

    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=f"habitat_eval_{dataset}")
    cfg = patch_config(cfg)

    # OVONDatasetV1 has no category_to_task_category_id, so the stock
    # ObjectGoalSensor cannot be constructed -- same drop human_play.py does.
    if cfg.habitat.dataset.type == "OVON-v1":
        from omegaconf import OmegaConf, open_dict
        was_readonly = OmegaConf.is_readonly(cfg)
        OmegaConf.set_readonly(cfg, False)
        with open_dict(cfg):
            cfg.habitat.task.lab_sensors.pop("objectgoal_sensor", None)
        OmegaConf.set_readonly(cfg, was_readonly)

    with habitat.config.read_write(cfg):
        # Cyclic iterator so any index is reachable, same as human_play.py.
        cfg.habitat.environment.iterator_options.cycle = True
        if with_map:
            # Unlike the human tool, the oracle SHOULD draw the route and the
            # view points -- seeing where the goal is and whether a path exists
            # is the entire point.
            cfg.habitat.task.measurements.update({
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    # OFF, and not a style choice: TopDownMap._draw_goals_aabb
                    # (habitat/tasks/nav/nav.py:772) indexes the SEMANTIC scene by
                    # goal object_id. HM3D's .basis.scn files fail to load here
                    # ("SSD Load Failure", printed at sim init), so sem_scene.objects
                    # is empty and the lookup dies with IndexError before the
                    # episode starts. draw_goal_positions + draw_view_points mark
                    # the goal from the episode data instead, needing no semantics.
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(draw=False, visibility_dist=5.0, fov=79),
                )
            })
    return cfg


def seek(env, target_index):
    """Advance the episode iterator to a 0-based index and reset onto it.

    The iterator is RELATIVE, so this tracks how far it has been consumed --
    exactly what human_play.py's load() keeps in self.consumed. Without that
    counter the first call is correct and every later one advances
    target_index MORE episodes from wherever it already sits, silently landing on
    an unrelated episode. That bit here on 2026-09-15: the reachability table was
    computed on ovon 1053 but every rollout after it ran on a different episode,
    starting 8 m from a goal that was not the wardrobe. The symptom was an A* plan
    that "solved" an 11.9 m route in 43 actions -- impossible at 0.25 m a step,
    which is what gave it away.

    Two-part dance, same as load(): walk the iterator, then assign
    current_episode unconditionally, because the setter clears
    _episode_from_iter_on_reset so reset() lands on this episode rather than
    advancing past it.
    """
    n = len(env.episodes)
    if not 0 <= target_index < n:
        sys.exit(f"episode index {target_index} out of range 0..{n - 1} for this dataset")
    # Env.__init__ consumes index 0, so the iterator starts there.
    consumed = getattr(env, "_oracle_consumed", 0)
    ep_obj = env.current_episode
    for _ in range((target_index - consumed) % n):
        ep_obj = next(env.episode_iterator)
    env.current_episode = ep_obj
    env._oracle_consumed = target_index
    obs = env.reset()
    # Cheap invariant: a wrong episode is silent otherwise, and every number this
    # tool prints is worthless if it is scored against one.
    assert np.allclose(env.sim.get_agent_state().position,
                       env.current_episode.start_position, atol=1e-3), (
        f"reset() left the agent at {env.sim.get_agent_state().position}, not the "
        f"episode start {env.current_episode.start_position}")
    return obs


def _finish(env, res, need_video):
    """Call STOP and read the metrics off. Success is only ever awarded on the
    stop action, for the oracle exactly as for the agent."""
    obs = env.step(HabitatSimActions.stop)
    res["steps"] += 1
    m = env.get_metrics()
    if need_video:
        res["frames"].append(_frame(obs, m, res["steps"]))
    res["metrics"] = {k: v for k, v in m.items() if k != "top_down_map"}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ovon",
                    choices=["hm3dv1", "hm3dv2", "mp3d", "ovon"])
    ap.add_argument("--episode", type=int, required=True,
                    help="0-based iterator index (NOT episode_id)")
    ap.add_argument("--video", action="store_true", help="write an mp4 of the best rollout")
    ap.add_argument("--panels", default="all", choices=["rgb", "all"],
                    help="all (default) = rgb+depth+top-down map with the route drawn")
    ap.add_argument("--out", help="output dir (default: <repo>/videos/oracle/test_<dataset>_<split>)")
    ap.add_argument("--grid", type=float, default=0.05,
                    help="A* position-cell size in metres (dedup resolution)")
    ap.add_argument("--max-expansions", type=int, default=300000,
                    help="A* node budget; exceeding it is reported, never silently "
                         "treated as 'impossible'")
    ap.add_argument("--follower-candidates", type=int, default=3,
                    help="how many nearest reachable view points to try with the "
                         "stock ShortestPathFollower (0 disables that comparison)")
    args = ap.parse_args()

    cfg = build_config(args.dataset, args.panels == "all")
    max_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance

    env = habitat.Env(cfg)
    seek(env, args.episode)
    episode = env.current_episode
    scene = os.path.basename(episode.scene_id)
    ep_info = getattr(episode, "info", {}) or {}

    print("=" * 78)
    print(f"ORACLE  dataset={args.dataset}  split={cfg.habitat.dataset.split}")
    print(f"  iterator index : {args.episode}  (record.txt calls this No.{args.episode + 1} task)")
    print(f"  episode_id     : {episode.episode_id}")
    print(f"  scene          : {scene}")
    print(f"  target         : {episode.object_category}")
    print(f"  start          : {np.round(episode.start_position, 4).tolist()}")
    print(f"  success radius : {success_distance} m   step budget: {max_steps}")
    print(f"  dataset geodesic_distance  : {ep_info.get('geodesic_distance')}")
    print(f"  dataset euclidean_distance : {ep_info.get('euclidean_distance')}")
    print("=" * 78)

    candidates = candidate_view_points(env)
    rows = reachability_report(env, candidates)
    t = PrettyTable(["view point", "position", "geodesic from start"])
    t.align = "l"
    for label, pos, d in rows:
        t.add_row([label, np.round(pos, 3).tolist(),
                   "inf" if not math.isfinite(d) else f"{d:.3f}"])
    print(t)
    reachable = [(label, pos, d) for label, pos, d in rows if math.isfinite(d)]
    print(f"reachable view points: {len(reachable)}/{len(rows)}")
    if not reachable:
        print("\nVERDICT: IMPOSSIBLE -- every goal view point is on a different navmesh "
              "island from the start. No agent could have solved this episode.")
        env.close()
        return

    goal_positions = [list(map(float, p)) for _, p, _ in reachable]
    attempts = []

    n_headings = int(round(360.0 / cfg.habitat.simulator.turn_angle))
    forward_step = float(cfg.habitat.simulator.forward_step_size)
    print(f"\n--- A* planner (grid {args.grid} m, {n_headings} headings, all "
          f"{len(goal_positions)} reachable view points) ---", flush=True)
    obs0 = seek(env, args.episode)
    t0 = time.time()
    # max_steps - 1 leaves a step for STOP, which counts against the budget too.
    plan, plan_info = astar_plan(
        env.sim, env.sim.get_agent_state(), goal_positions, success_distance,
        forward_step, n_headings, args.grid, args.max_expansions, max_steps - 1)
    print(f"    search: {plan_info['reason']}  expansions={plan_info['expansions']}  "
          f"({time.time() - t0:.1f}s)")
    if plan is None:
        res = dict(status=f"no_plan:{plan_info['reason']}", steps=0,
                   closest=float(plan_info.get("closest_found", math.inf)),
                   frames=[_frame(obs0, env.get_metrics(), 0)] if args.video else [],
                   actions=[])
    else:
        res = rollout_actions(env, obs0, plan, success_distance, args.video)
    res = _finish(env, res, args.video)
    res.update(name="A* planner", target_label="all reachable view points",
               seconds=time.time() - t0, plan_info=plan_info)
    attempts.append(res)
    m = res["metrics"]
    print(f"    {res['status']}  steps={res['steps']}  closest d2g={res['closest']:.3f}  "
          f"final d2g={m['distance_to_goal']:.3f}  success={bool(m['success'])}  "
          f"spl={m['spl']:.4f}  ({res['seconds']:.1f}s)")

    for label, pos, d in reachable[: args.follower_candidates]:
        print(f"\n--- ShortestPathFollower -> {label} (geodesic {d:.3f} m) ---", flush=True)
        obs0 = seek(env, args.episode)
        t0 = time.time()
        res = rollout_follower(env, obs0, list(map(float, pos)), max_steps,
                               success_distance, args.video)
        res = _finish(env, res, args.video)
        res.update(name=f"follower->{label}", target_label=label, seconds=time.time() - t0)
        attempts.append(res)
        m = res["metrics"]
        print(f"    {res['status']}  steps={res['steps']}  closest d2g={res['closest']:.3f}  "
              f"final d2g={m['distance_to_goal']:.3f}  success={bool(m['success'])}  "
              f"spl={m['spl']:.4f}  ({res['seconds']:.1f}s)")

    # Best = succeeded first, then closest approach. Ties broken toward fewer steps.
    best = min(attempts, key=lambda r: (not bool(r["metrics"]["success"]),
                                        r["closest"], r["steps"]))
    m = best["metrics"]

    print("\n" + "=" * 78)
    summary = PrettyTable(["attempt", "status", "steps", "closest d2g", "success", "spl"])
    summary.align = "l"
    for r in attempts:
        summary.add_row([r["name"], r["status"], r["steps"], f"{r['closest']:.3f}",
                         bool(r["metrics"]["success"]), f"{r['metrics']['spl']:.4f}"])
    print(summary)

    if m["success"]:
        verdict = (f"POSSIBLE -- '{best['name']}' reached the goal and stopped inside the "
                   f"{success_distance} m radius in {best['steps']} of {max_steps} steps. "
                   "This episode is winnable as configured; the agent's failure was its own.")
    elif plan is None and plan_info["reason"] == "exhausted_search_space":
        verdict = (f"IMPOSSIBLE -- A* searched the reachable state space to exhaustion "
                   f"({plan_info['expansions']} expansions) and no action sequence ends "
                   f"inside the {success_distance} m radius, closest {plan_info['closest_found']:.3f} m. "
                   "The view points are geodesically reachable but no legal STOP pose "
                   "scores inside the radius, so no agent could have passed this episode.")
    elif plan is None:
        verdict = (f"UNDECIDED -- A* hit its {plan_info['reason']} before finding a plan "
                   f"(closest {plan_info['closest_found']:.3f} m). Raise --max-expansions "
                   "before concluding anything about this episode.")
    else:
        verdict = (f"NOT REACHED -- a plan was found but replaying it left the agent "
                   f"{best['closest']:.3f} m away, against a {success_distance} m radius. "
                   "That is a mismatch between the searched dynamics and env.step, which "
                   "should not happen -- inspect the video.")
    print(f"\nVERDICT: {verdict}")
    print("=" * 78)

    if args.video:
        out_dir = args.out or os.path.join(
            REPO_ROOT, "videos", "oracle", f"test_{args.dataset}_{cfg.habitat.dataset.split}")
        os.makedirs(out_dir, exist_ok=True)
        name = f"oracle_{scene}_{episode.episode_id}_idx{args.episode}"
        images_to_video(best["frames"], out_dir, name, fps=6, quality=9)
        meta = {
            "dataset": args.dataset, "split": cfg.habitat.dataset.split,
            "episode_index_0based": args.episode, "episode_id": str(episode.episode_id),
            "scene": scene, "target": episode.object_category,
            "success_distance": success_distance, "max_episode_steps": max_steps,
            "dataset_geodesic_distance": ep_info.get("geodesic_distance"),
            "view_points": [
                {"label": l, "position": [float(x) for x in p],
                 "geodesic_from_start": (d if math.isfinite(d) else None)}
                for l, p, d in rows
            ],
            "reachable_view_points": len(reachable), "total_view_points": len(rows),
            "attempts": [
                {"name": r["name"], "status": r["status"], "steps": r["steps"],
                 "closest_distance_to_goal": r["closest"],
                 "metrics": {k: float(v) for k, v in r["metrics"].items()
                             if isinstance(v, (int, float))}}
                for r in attempts
            ],
            "best_attempt": best["name"],
            "astar": plan_info,
            "astar_actions": "".join(attempts[0]["actions"]) if attempts[0]["actions"] else None,
            "video_is_of": best["name"],
            "verdict_success": bool(m["success"]),
        }
        with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        print(f"\nvideo  -> {os.path.join(out_dir, name + '.mp4')}   (shows: {best['name']})")
        print(f"report -> {os.path.join(out_dir, name + '.json')}")

    env.close()


if __name__ == "__main__":
    main()
