# ApexNav — embodiment and action space

What body ApexNav drives, and exactly what it can do with it. Two embodiments
are in use: **LoCoBot** for hm3dv1 / hm3dv2 / mp3d, and **Stretch** for ovon.
They share an action space; they do not share a body.

Everything below is read out of the configs and source in this repo, with file
and line references. Where a value is derived rather than written down, the
expression that derives it is given.

Verified 2026-09-08.

---

## 1. Which body runs where

| dataset | embodiment | selected by |
|---|---|---|
| hm3dv1 | LoCoBot | `config/habitat_eval_hm3dv1.yaml` → `objectnav_hm3d.yaml` |
| hm3dv2 | LoCoBot | `config/habitat_eval_hm3dv2.yaml` → `objectnav_hm3d.yaml` |
| mp3d | LoCoBot | `config/habitat_eval_mp3d.yaml` → `objectnav_mp3d.yaml` |
| ovon | **Stretch** | `config/habitat_eval_ovon.yaml`, `agents.main_agent` block |

The three LoCoBot configs are byte-identical apart from the dataset path, the
LLM answer file and the video output path — same body, same action space, same
step sizes. `diff config/habitat_eval_hm3dv1.yaml config/habitat_eval_hm3dv2.yaml`
returns two lines, both paths.

OVON inherits the same LoCoBot body from
`config/benchmark/nav/objectnav/objectnav_ovon.yaml` and then **overrides it**
with Stretch in `habitat_eval_ovon.yaml`. That override is deliberate: the
HM3D-OVON paper's Table III was produced on Stretch, so it is the only body
whose numbers sit next to their published results.

---

## 2. Body breakdown

| | **LoCoBot** (hm3dv1/v2, mp3d) | **Stretch** (ovon) |
|---|---|---|
| agent height | 0.88 m | **1.41 m** |
| agent radius | 0.18 m | **0.17 m** |
| camera height | 0.88 m | **1.31 m** |
| RGB / depth resolution | 640 × 480 (landscape) | **360 × 640 (portrait)** |
| horizontal FOV | 79° | **42°** |
| vertical FOV (derived) | 59.25° | **74.67°** |
| depth range | 0.0 – 5.0 m | 0.0 – 5.0 m |
| depth normalised | true | true |
| `allow_sliding` | false | false |

Sources: `config/benchmark/nav/objectnav/objectnav_ovon.yaml` and habitat-lab's
`objectnav_hm3d.yaml` / `objectnav_mp3d.yaml` for LoCoBot;
`config/habitat_eval_ovon.yaml` `agents.main_agent` for Stretch.

Two things worth stating plainly:

- **The Stretch sensor is portrait**, 360 wide by 640 tall. Not a typo and not
  reversed. Combined with the 42° hfov it is a much narrower, much taller view
  than the LoCoBot's.
- **`min_depth` is 0.0 in our configs, not habitat's stock 0.5.** All four
  `habitat_eval_*.yaml` set it. That is a method choice, not a benchmark one.

### Derived camera intrinsics

The C++ mapper needs intrinsics, and they must match whichever body is active.
`tools/ovon/derive_camera_params.py` computes them from the composed Hydra
config so they cannot drift:

```
fx = w / (2 · tan(hfov · π / 360))
fy = h / (2 · tan((hfov · h / w) · π / 360))
cx, cy = w/2, h/2
half_cone = min(30°, hfov/2)
```

| | LoCoBot | Stretch |
|---|---|---|
| cx, cy | 320.0, 240.0 | 180.0, 320.0 |
| fx | 388.1910413097385 | 468.9160316448843 |
| fy | 422.0475153598262 | 419.5534660656559 |
| frontier half-cone | 30° | **21°** |

The LoCoBot values reproduce the hardcoded literals in
`src/planner/exploration_manager/launch/exploration.launch` bit-for-bit, so
deriving them is a no-op for the three validated datasets.

The half-cone is not cosmetic. `frontier_map2d.cpp:280` retires a frontier to
dormant once it falls inside this cone with an unoccluded raycast, and draining
the active frontier list is what ends an episode as `no frontier`. Upstream's
fixed 30° is 76% of the LoCoBot's true 39.5° half-hfov — conservative. Against
Stretch's 21° half-hfov the same 30° claims **143% of what the camera can
see**, retiring frontiers that were never imaged; 44.1% of that run's episodes
ended as `no frontier`. Hence the `min()`.

### Getting an embodiment wrong is silent and catastrophic

Running OVON's Stretch body against the LoCoBot intrinsic defaults rotated
every projected point ~19.8° in yaw and ~10.7° in pitch and placed it 0.43 m too
low. It scored **4.9% SR** (job 58962960, 2026-08-11) and threw no error.
Camera height is a separate path — `habitat2ros/habitat_publisher.py:13` — and
was hardcoded to 0.88 until 2026-08-18. Both are now derived.

---

## 3. Action space

Identical for both bodies. Six discrete actions, from habitat-lab's
`habitat/task/objectnav.yaml`:

| # | action | effect | magnitude |
|---|---|---|---|
| 0 | `stop` | ends the episode; success is judged here | — |
| 1 | `move_forward` | translate along heading | **0.25 m** |
| 2 | `turn_left` | yaw left | **30°** |
| 3 | `turn_right` | yaw right | **30°** |
| 4 | `look_up` | pitch camera up | **30°** |
| 5 | `look_down` | pitch camera down | **30°** |

- `forward_step_size: 0.25` and `turn_angle: 30` are set in all four
  `habitat_eval_*.yaml`.
- `tilt_angle: 30` comes from `objectnav.yaml`; habitat's dataclass default is
  15, so this is an explicit override. It matches ApexNav's hardcoded
  `np.pi / 6.0` in `habitat_evaluation.py:581,584`.
- **Do not add `tilt_angle` under `simulator:`.** It lives on
  `DiscreteNavigationActionConfig`, not `SimulatorConfig`. Setting it there
  fails config composition outright (killed job 58962681).
- `allow_sliding: false` — a blocked `move_forward` yields **no displacement at
  all**, rather than sliding along the obstacle. This is what makes the
  stuck-escape routine below necessary.
- Episode cap: **500 steps** (`max_episode_steps`), all four datasets.

Internally ApexNav uses its own enum (`params.py:17`) which the ROS planner
publishes as an `Int32` on `/habitat/plan_action`; `habitat_evaluation.py:572-586`
maps it onto `HabitatSimActions`:

```
STOP=0  MOVE_FORWARD=1  TURN_LEFT=2  TURN_RIGHT=3  TURN_DOWN=4  TURN_UP=5
                                                   (look_down)  (look_up)
```

Note `TURN_DOWN`/`TURN_UP` are pitch, not yaw — the names are misleading.

### The human baseline uses the same six actions

`tools/human/human_play.py:128-132` binds `W`/`A`/`D`/`R`/`F` to
`move_forward`/`turn_left`/`turn_right`/`look_up`/`look_down`, and `Space` to
`stop`. Same magnitudes, same 500-step cap, same `allow_sliding: false`. The
human-vs-agent comparison is therefore over an identical action space — no
continuous control, no larger steps.

---

## 4. How a continuous plan becomes discrete actions

The planner produces a target yaw; `ExplorationFSM::decideNextAction`
(`exploration_fsm.cpp:478-496`) discretises it:

```cpp
if (std::fabs(yaw_diff) > FSMConstants::ACTION_ANGLE / 1.9)
    next_action = (yaw_diff > 0) ? TURN_LEFT : TURN_RIGHT;
else
    next_action = MOVE_FORWARD;
```

`ACTION_ANGLE = π/6` (`exploration_fsm.h:39`), so the threshold is
**π/6 / 1.9 ≈ 15.79°**. Any heading error above that spends a whole action on a
30° turn that makes no forward progress. There is no combined move-and-turn.

### The 26-action startup calibration

Before any navigation, every episode runs a fixed opening sequence
(`exploration_fsm.cpp:85-96`):

| count | action | purpose |
|---|---|---|
| 1 | `look_down` | pitch to −30° |
| 12 | `turn_left` | full 360° sweep, pitched down |
| 1 | `look_up` | pitch back to 0° |
| 12 | `turn_left` | full 360° sweep, level |

**26 actions, of which 24 are pure rotation and 2 are pitch. Zero translation.**
That is 5.2% of the 500-step budget spent before the agent has moved, on every
episode, unconditionally.

This matters for interpreting StepSPL. SPL is normalised by *distance*
travelled, StepSPL by *actions* taken — so the calibration sweep contributes
nothing to SPL's denominator and 26 to StepSPL's. For an episode that uses
`a` actions total it inflates actions-per-metre by a factor of roughly
`a / (a − 26)`: about **+35% on a 100-action episode**. The human baseline has
no equivalent prologue; a person starts navigating immediately.

See `analysis/human_vs_agent/efficiency_ratio.png`, which plots SPL/StepSPL —
each side's actions-per-metre as a multiple of the oracle's. ApexNav sits at
1.54–1.96 and the human at 0.96–1.22, so ApexNav spends roughly **1.6× the
actions per metre a human does** (and 1.5–2.0× the oracle's rate). This sweep
is a fixed, quantifiable part of that.

**ApexNav cannot score 1.0 on that metric even in principle.** Two things
contribute, and they are different in kind:

- The 26 calibration actions are in `count_steps` but add nothing to path
  length. A perfectly straight, turn-free run to the goal — SPL exactly 1.0 —
  would still take `o + 26` actions against an oracle's `o`. **This is the
  method, not an artifact.** A 360° scan at two pitches before planning is a
  design choice ApexNav makes; an agent that skipped it would score better, and
  StepSPL registering that is the metric doing its job. It is exactly the
  "excessive turning/scanning" the metric was added to capture.
- The step counter includes the terminating `stop`, while the oracle rollout's
  `while action != HabitatSimActions.stop` loop does not. **This applies to
  both sides identically**: `habitat_evaluation.py:593` increments before every
  dispatched action including `stop`, and `human_play.py:456-457` does
  `self.steps += 1` inside its own `stop` branch. Both oracles exclude it
  (`habitat_evaluation.py:286-288`, `human_play.py:204-206`). So it is +1 on
  every rollout, agent and human alike — it raises both floors equally and
  explains **none** of the human/agent gap.

So the agent's floor is `(o + 27) / o` — about 1.68 for a 40-action oracle,
1.45 for 60, 1.34 for 80 — of which the +26 is method and the +1 is bookkeeping
that cancels against the human's own `(o + 1) / o`. Against observed values of
1.54–1.96 (agent) and 0.96–1.22 (human), the opening sweep is a large and
legitimate share of the gap. It is a real cost ApexNav pays on every episode,
not a measurement artifact, and StepSPL is the metric that surfaces it.

Quantifying that share exactly is not possible from the record files as they
stand: `record.txt` carries only the finished SPL and StepSPL, not the step
count and oracle count. Both sides already compute the pair —
`step_spl_overlay_lines()` (`habitat_evaluation.py:301`) for the video overlay,
and `human_play.py:436-443` for the browser HUD — so logging it per episode
would let you state "N of ApexNav's M actions were the opening sweep" directly.

Note the oracle is defined twice: `human_play.py:137` deliberately mirrors
`habitat_evaluation.py` rather than importing it, to avoid dragging in `rospy`
(see that function's docstring). If the StepSPL definition changes, both must
change, or the human and agent columns stop being comparable.

### The stuck-escape routine

With `allow_sliding: false`, a `move_forward` into geometry produces no motion.
If displacement is under `STUCKING_DISTANCE = 0.05 m` after a `move_forward`,
a 10-action scripted escape fires (`exploration_fsm.cpp:203-225`):

```
TURN_RIGHT, MOVE_FORWARD, TURN_RIGHT, MOVE_FORWARD, TURN_LEFT,
TURN_LEFT, TURN_LEFT, MOVE_FORWARD, TURN_LEFT, MOVE_FORWARD
```

If that fails, the cell is force-marked occupied and the episode is recorded as
`stucking`. Another fixed block of actions that yields little or no distance —
the same StepSPL/SPL asymmetry as the startup sweep, but episode-dependent.

---

## 5. What differs, at a glance

Same across all four datasets: action set, 0.25 m step, 30° turn, 30° tilt,
500-step cap, `allow_sliding: false`, depth range 0–5 m, the 26-action startup
sweep, the stuck-escape routine, the 15.79° turn threshold.

Differs on ovon only: body height 0.88 → 1.41 m, radius 0.18 → 0.17 m, camera
0.88 → 1.31 m, sensor 640×480 landscape → 360×640 portrait, hfov 79° → 42°,
frontier half-cone 30° → 21°, plus `success_distance` 0.2 → 0.25 m and a
recomputed navmesh (`agent_max_climb` 0.1, `cell_height` 0.05).

Because `success_distance` differs, **SR is not directly comparable between the
ovon column and the other three** — that is each benchmark using its own
protocol, and it needs stating in any write-up.

---

Authored by Claude (Anthropic Claude Opus 5) for Broden Tripcony.
