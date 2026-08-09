# StepSPL: Implementation Plan for ApexNav

Notes from 2026-07-14, captured after confirming how ApexNav's evaluation pipeline actually computes SR/SPL, and scoping out adding the custom StepSPL metric defined in the Initial Paper (`FYP Part A/Initial Paper/final/draft2.tex`, eq. `stepspl`).

## Background: SR/SPL are Habitat's, not ApexNav's

Confirmed by reading `habitat_evaluation.py` and `config/habitat_eval_hm3dv1.yaml` directly (`Robotics-STAR-Lab/ApexNav`, `main` branch):

- `habitat_eval_hm3dv1.yaml` inherits `defaults: - /benchmark/nav/objectnav: - objectnav_hm3d` — Habitat-Lab's own standard ObjectNav benchmark config. ApexNav only overrides one field on top of it: `task.measurements.success.success_distance: 0.2`.
- `spl`, `success`, `soft_spl`, `distance_to_goal`, `distance_to_goal_reward` are **not defined anywhere in ApexNav's own code** — they come from Habitat-Lab's built-in measurement classes (`habitat.tasks.nav.nav`), registered as part of the inherited `objectnav_hm3d` task config.
- `habitat_evaluation.py` is a pure consumer: `info = env.get_metrics()`, then reads `info["spl"]`, `info["success"]`, etc. It only *adds* two measurements of its own on top of the inherited set — `top_down_map` and `collisions` — via `cfg.habitat.task.measurements.update({...})`.

This matters for StepSPL: adding it means following the same pattern (compute post-hoc in `habitat_evaluation.py` from `env`/episode state), not modifying Habitat-Lab's measurement framework itself.

## The metric (from Initial Paper, eq. `stepspl`)

```
StepSPL = (1/N) * sum_i [ S_i * t_i* / max(t_i, t_i*) ]
```

- `S_i` — success indicator for episode `i` (0 or 1).
- `t_i` — total actions taken by the agent in episode `i`, **including rotations** (i.e. every discrete action: forward, turn left/right, look up/down, stop).
- `t_i*` — minimum number of actions required to traverse the shortest geodesic path.

Direct analog of standard SPL, but measured in discrete action-steps instead of continuous path length — intended to also penalize excessive turning/scanning behavior, which standard SPL (pure translational path length) doesn't capture.

## Feasibility: high, cleanly additive, no core algorithm changes

All three components map onto data that's either already tracked or one standard Habitat-Lab utility call away:

| Term | Source | Status |
|---|---|---|
| `S_i` | `info["success"]` | Already available |
| `t_i` | `count_steps` in the existing per-episode loop | **Already tracked verbatim** — increments once per action, every action type, matching the definition exactly |
| `t_i*` | `habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` | Needs one new call per episode — standard Habitat-Lab utility, not custom pathfinding code |

**Config values already confirmed** (from `habitat_eval_hm3dv1.yaml`), relevant because they mean `ShortestPathFollower`'s oracle action sequence will be directly comparable to the agent's real `t_i` with no unit conversion needed:
```yaml
simulator:
  forward_step_size: 0.25   # meters per MOVE_FORWARD
  turn_angle: 30             # degrees per TURN_LEFT/TURN_RIGHT
```

`ShortestPathFollower` only reasons about position/heading (forward+turn actions), not camera pitch — this actually matches the metric's intent, since LOOK_UP/LOOK_DOWN actions inflate `t_i` with no "optimal" counterpart, appropriately penalizing extra scanning behavior beyond a straight-line walk to the goal.

## Implementation plan (in `habitat_evaluation.py`)

Mirror the existing SPL/SoftSPL accumulation pattern exactly — same file, same loop structure, same reporting/persistence calls.

1. **Import**: `from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower`
2. **Per-episode setup** (right after `env.reset()`, alongside where `label`/`llm_answer` are set up): instantiate a `ShortestPathFollower` for the episode's `env.sim`, then query the oracle action sequence length to the nearest goal viewpoint (episodes can have multiple valid goal positions — same multi-viewpoint handling SPL/`distance_to_goal` already deal with internally, worth checking how Habitat's own SPL measurement resolves "nearest viewpoint" and reusing the same logic/goal list rather than reinventing it).

   **Correction (2026-08-09):** the first implementation used `goal.position` (the raw object location) directly, on the assumption that's what Habitat's own measures use. Confirmed wrong via `[StepSPL DEBUG]` prints on a real hm3dv1 episode: `env.sim.geodesic_distance(start_position, [goal.position], None)` returned `inf` for every goal, even though the real ROS-driven agent completed the episode successfully — so a path clearly existed, but not to the object's raw position, which is frequently off the navmesh. Habitat-lab's own `DistanceToGoal`/`Success` measures actually resolve to the nearest of each goal's `view_points` (`goal.view_points[j].agent_state.position`), precomputed navigable points near the object for exactly this reason. Fixed by iterating `view_point.agent_state.position` across all goals' `view_points` instead of `goal.position`.
3. **New accumulator**: `stepspl_all = 0` initialized alongside `spl_all`, `soft_spl_all` (near the top of `main()`, and in `read_record`'s return tuple if `continue.txt`/`record.txt` resumability should cover it too — check `basic_utils/record_episode/{read_record,write_record}.py` for how the existing accumulators are (de)serialized).
4. **Per-episode calculation**, alongside where `spl`/`soft_spl` are pulled from `info` after each episode completes:
   ```python
   t_i_star = len(shortest_path_follower_actions)  # oracle sequence length
   t_i = count_steps  # already tracked
   step_spl = success * (t_i_star / max(t_i, t_i_star))
   stepspl_all += step_spl
   ```
5. **Reporting**: add a row to both `PrettyTable` outputs (`table1` average, `table2` total) next to the existing SPL/Soft SPL rows.
6. **Persistence**: include in both `write_record` calls (`record_file_path`, `continue_path`) so it survives resumed runs like the other metrics.

## Open questions to resolve during implementation

- Exact API signature for `ShortestPathFollower` in the pinned `habitat-lab v0.3.1` (constructor args, whether it needs `goal_radius` matching `success_distance: 0.2`) — check the installed package source directly on M3 rather than assuming, since APIs can drift between Habitat-Lab versions.
- How to handle episodes where no valid path exists (e.g. `is_on_same_floor` already filters some of these in the existing loop) — likely just exclude from the StepSPL average the same way, or check ApexNav's own `check_failure`/`is_on_same_floor` logic for the existing precedent.
- Whether `t_i*` should be computed from the agent's true start pose or the episode's canonical start (should be the same thing, but worth confirming against `env.current_episode.start_position`).

## Status

Not yet implemented. Planned for after task 5 (single-episode end-to-end sanity test) confirms the baseline pipeline works, so StepSPL is added against a known-working baseline rather than debugging both at once.
