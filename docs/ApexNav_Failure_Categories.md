# ApexNav Failure Categories — What Each Label Actually Means

_Authored by Claude (Anthropic Claude Opus 5) for Broden Tripcony._

Traced from source on 2026-08-19: `basic_utils/failure_check/failure_check.py`
(`check_failure()`), `params.py` (`RESULT_TYPES`, `FINAL_RESULT`, `EXPL_RESULT`),
`src/planner/exploration_manager/src/exploration_fsm.cpp` and
`exploration_manager.cpp`. Written because the labels in `record.txt` do **not**
mean what their names suggest — most importantly `no frontier`.

## The two inputs that decide everything

`check_failure()` is called only when `success == 0`. It sees seven arguments,
but the branch it takes is driven by two ROS values latched from the C++ planner:

| Python arg | ROS topic | Set by |
|---|---|---|
| `final_state` | `/ros/expl_state` | `FINAL_RESULT` enum |
| `expl_result` | `/ros/expl_result` | `EXPL_RESULT` enum |

plus three Python-side booleans from `habitat_evaluation.py`:

- `step_out` — `count_steps == max_step` (500). Ran out of budget.
- `pass_object` — was **ever** within `success_distance` of a goal during the episode.
- `near_object` — was within `success_distance` **at the final step**.

## Decision tree, in evaluation order

```
1. is_feasible == 0                                   -> "infeasible"
2. else, ACTIVE stop (not step_out and not STUCKING):
     final_state == NO_FRONTIER
       or expl_result == SEARCH_EXTREME
         + pass_object                                -> "[no frontier] false negative"
         + not pass_object                            -> "no frontier"
     final_state == REACH_OBJECT and not near_object  -> "false positive"
     otherwise                                        -> "unknown failure (active stop)"
3. else, PASSIVE stop (step_out or STUCKING):
     near_object                                      -> "stepout true negative"
     final_state == STUCKING + pass_object            -> "[stucking] false negative"
     final_state == STUCKING                          -> "stucking"
     pass_object                                      -> "[stepout] false negative"
     otherwise                                        -> "stepout feasible"
```

## `no frontier` is three different conditions — this is the big one

The label fires on `final_state == NO_FRONTIER` **OR**
`expl_result == SEARCH_EXTREME`. Tracing both back into C++, that is **three
distinct planner outcomes collapsed into one label**, and only one of them
actually means "there are no frontiers":

| C++ return | Where | What actually happened |
|---|---|---|
| `NO_COVERABLE_FRONTIER` | `exploration_manager.cpp:158`, guarded by `ed_->frontiers_.empty()` | **Genuinely exhausted.** The frontier list is empty. |
| `NO_PASSABLE_FRONTIER` | `exploration_manager.cpp:162`, the `else` of that same guard | **Frontiers exist but none is reachable.** The planner returned an empty path to every one of them. A *planning/reachability* failure, not exploration exhaustion. |
| `SEARCH_EXTREME` | `exploration_manager.cpp:131,139,150` | **Not a frontier condition at all.** Normal frontier policy came back empty, the suspicious-object fallback failed, and the dormant-frontier retry failed — so the planner fell through to `searchObjectPathExtreme()` against progressively lower-confidence object clouds. Returning `SEARCH_EXTREME` means one of those **succeeded** and the agent had a target to drive at. |

`exploration_fsm.cpp:283-286` maps the first two onto
`FINAL_RESULT::NO_FRONTIER`; `failure_check.py` then folds `SEARCH_EXTREME` in
alongside them. So an episode labelled `no frontier` may have had plenty of
frontiers, or may have been actively chasing a low-confidence object detection
when the step budget ran out.

**This is not an artifact of the broken OVON run** — it is the shipped logic and
applies identically to the hm3dv1/hm3dv2/mp3d baselines. It does mean the
4.9%-SR OVON run's "1322/3000 episodes ended as no frontier" figure is softer
evidence for the view-cone hypothesis than it first looks: some unknown share of
those were `NO_PASSABLE_FRONTIER` or `SEARCH_EXTREME`.

**To disambiguate you must add logging** — `record.txt` carries only the final
label. The cheapest fix is to publish `expl_result` alongside `final_state` into
`write_record()`, since `habitat_evaluation.py` already subscribes to both.

## `infeasible` is a 2 m height window, not a reachability test

Evaluated **first**, before any agent behaviour is considered:

```python
ref = episode.start_position[1]
on_same_floor = (ref <= goal.position[1] < ref + 2.0)
```

If **no** goal satisfies this for any goal in the episode, the episode is
`infeasible` and no other branch is ever reached. Consequences:

- It is measured against the **episode start height**, not the agent's final
  position, and the window is one-sided — a goal *below* the start (sunken
  lounge, agent spawned on a landing) reads as "different floor".
- It says nothing about the navmesh. A goal genuinely disconnected from the
  agent's component but at the same height is **not** flagged infeasible — this
  is exactly the case that produced the `inf`/NaN SPL values guarded in
  `habitat_evaluation.py`.
- An `infeasible` episode reports **nothing about agent behaviour**. It is a
  property of the episode definition, so it belongs in a denominator discussion,
  not a failure-mode discussion.

That mp3d shows 13.85% `infeasible` against hm3dv2's 0.30% is therefore mostly a
statement about the multi-floor structure of the MP3D scenes.

## Every label, end to end

All ten declared `RESULT_TYPES`, in the order `check_failure()` can reach them.
"Passive stop" means the episode ended because the agent ran out of steps or was
declared stuck; "active stop" means the planner itself chose to stop.

Twelve strings are reachable: `success` (set in `habitat_evaluation.py:598`)
plus the eleven `check_failure()` can return. Only ten are declared in
`params.py`'s `RESULT_TYPES` — the last two rows are reachable in code but have
no results folder and are **dropped with a warning** by `analyze_failures.py`
rather than counted.

| Label | Source | In `RESULT_TYPES` | Fires when | What it actually means |
|---|---|---|---|---|
| `success` | `habitat_evaluation.py:598` | yes | Habitat scored it — agent called STOP within `success_distance` of a goal | The only outcome where `check_failure()` is never called at all. |
| `infeasible` | `failure_check.py:69` | yes | No goal height in `[start_y, start_y + 2.0)` | **Evaluated first**, before any agent behaviour, so it says nothing about what the agent did — it is a property of the episode. One-sided window, so a goal *below* the spawn reads as "different floor". Not a navmesh test. |
| `false positive` | `failure_check.py:85` | yes | Active stop, `REACH_OBJECT`, not within `success_distance` | Agent declared arrival on the wrong object. Perception **precision** failure. The dominant failure on every dataset. |
| `no frontier` | `failure_check.py:82` | yes | Active stop, never passed a goal, and `NO_FRONTIER` **or** `SEARCH_EXTREME` | Three planner outcomes in one label — exhausted, unreachable, or actively chasing a low-confidence detection. See the section above. |
| `stucking` | `failure_check.py:99` | yes | `stucking_action_count_ >= 25`, never passed a goal | Moved less than `STUCKING_DISTANCE` for 25 consecutive planning cycles. Control/collision failure, not perception. |
| `stepout feasible` | `failure_check.py:106` | yes | Hit the 500-step cap, never passed a goal, goal on same floor | Budget exhausted while still exploring. The "just too slow" bucket. |
| `[stepout] false negative` | `failure_check.py:103` | yes | Hit 500 steps, **did** pass within `success_distance` | Perception **recall** failure — the target was seen and not recognised. |
| `[stucking] false negative` | `failure_check.py:97` | yes | Stuck, and had passed the target | Same recall failure, compounded by a control failure. |
| `[no frontier] false negative` | `failure_check.py:80` | yes | The `no frontier` conditions, and had passed the target | Same recall failure; inherits the three-way ambiguity above. |
| `stepout true negative` | `failure_check.py:92` | yes | Passive stop, ended within `success_distance` | **Never fires.** Would need a passive stop ending inside the success radius, which that path cannot produce. Effectively dead code — see counts below. |
| `unknown failure (active stop)` | `failure_check.py:87` | **no** | Active stop, on the same floor, but `final_state` was neither `NO_FRONTIER`/`SEARCH_EXTREME` nor `REACH_OBJECT` | The planner stopped for a reason the taxonomy has no bucket for — e.g. it reported `EXPLORE` or `SEARCH_OBJECT` at the moment the episode ended. A genuine gap in the taxonomy, not an error path. |
| `unkonwn failure` | `failure_check.py:61` | **no** | The initialiser, if no branch below it ever assigns | Unreachable in practice: every path through the tree assigns something. Typo is in the upstream source; kept here verbatim because that is the string a grep would have to match. |

### Observed counts

Episode counts per label across every completed run on this machine. `ovon` is
the pre-fix 2026-08-11 run (job 58962960) kept for contrast, not a baseline.

| Label | hm3dv1 (2000) | hm3dv2 (1000) | mp3d (2195) | ovon-dodgey (3000) |
|---|---:|---:|---:|---:|
| `success` | 1180 | 761 | 853 | 147 |
| `infeasible` | 370 | 3 | 304 | 47 |
| `false positive` | 274 | 113 | 423 | 402 |
| `no frontier` | 55 | 10 | 107 | 1322 |
| `stucking` | 35 | 20 | 90 | 1076 |
| `stepout feasible` | 43 | 43 | 256 | 1 |
| `[stepout] false negative` | 29 | 37 | 117 | 0 |
| `[stucking] false negative` | 12 | 4 | 26 | 2 |
| `[no frontier] false negative` | 2 | 9 | 19 | 3 |
| `stepout true negative` | **0** | **0** | **0** | **0** |

Nine of the ten declared labels occur; `stepout true negative` is 0 across all
8,195 episodes, confirming the dead-code reading above.

The ovon column is worth reading against the other three. `stucking` at 35.9%
(1076/3000) is ~18x the baselines — a larger anomaly than the much-discussed
`no frontier` at 44.1%. Stuck means the agent physically failed to move, which
points at the depth-intrinsics bug fixed in `5644168` (a principal point 140 px
off in u projects obstacles into the wrong place, so the planner drives into
geometry it believes is clear) rather than at the view-cone width fixed in
`ea6d703`.

## If a parsed total comes up short

`analyze_failures.py` skips any episode whose `result_text` is not in
`RESULT_TYPES`, printing a warning per episode. The only two strings that can
trigger that are the last two rows above. All 8,195 episodes here parsed
cleanly, so neither has fired yet — but if a future run's parsed total is below
its episode count, those warnings are where to look.
