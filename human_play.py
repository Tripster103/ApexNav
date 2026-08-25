#!/usr/bin/env python3
"""Play a single ObjectNav episode by hand, in a browser, and score it like ApexNav.

Why this exists: every number in results/ is the agent's. There is no human
reference on the same episodes, so there is no way to say whether a 59% SR is
close to the ceiling or nowhere near it. This drives habitat.Env directly with
keyboard input and emits a record.txt block in the SAME format write_record()
produces, so a human run drops straight into analyze_failures.py alongside the
three baselines.

Deliberately does NOT touch the ROS stack. No roscore, no exploration_manager,
no four VLM servers -- just habitat.Env in one process. It still needs a GPU
node, because habitat-sim renders through EGL and the login node has no driver.

Do NOT hand-type an `apptainer exec --bind ...` line for this. Use the wrapper,
which carries the full 9-bind NVIDIA/EGL set and the cache env vars:

    # on a GPU node (inside your own allocation, or a session that already has one):
    bash /scratch2/ml20/btripcon/jobs/run_human_play.sh --dataset hm3dv1 --episode 1999

    # or a whole subset, played back to back:
    bash /scratch2/ml20/btripcon/jobs/run_human_play.sh --dataset hm3dv1 \\
        --playlist ../results/apexnav/baseline/hm3dv1/subset_.../metadata.json

Build a playlist with basic_utils/record_episode/make_subset.py, which selects on
failure mode / target / scene / random sample out of an existing agent run and
emits the 0-based indices as selection.indices_0based. Because both this and the
sweep walk the same shuffle:false iterator, those indices mean the same episodes
here. cycle:false means the playlist is walked in ascending index order only.

Results land in videos/human/test_<dataset>_<split>/ as record.txt (running
averages) + continue.txt (running totals), the same pair habitat_evaluation.py
writes, plus playlist.json pinning which episodes the run covers. Re-running the
same command resumes from continue.txt rather than starting over.

`--nv` on its own is NOT enough: habitat-sim opens a windowless EGL context, and
without the glvnd bind + the host driver's EGL libs on LD_LIBRARY_PATH it dies at
`cannot get default EGL display: EGL_BAD_PARAMETER`. That is exactly why the bind
list lives in one place -- see containers/apexnav/apptainer_run.sh's header for
the 2026-08-04 incident that motivated the rule.

Then open http://localhost:8080 -- VS Code forwards the port automatically.

--panels rgb   (default) shows only what the agent's RGB camera sees. This is
               the fair comparison: ApexNav gets no free map either.
--panels all   adds depth and the top-down map with fog of war. Useful for
               debugging an episode, but the human then sees strictly more than
               the agent did, so do not report those runs as a human baseline.

Authored by Claude (Anthropic Claude Opus 5) for Broden Tripcony.
"""
import argparse
import base64
import io as _io
import json
import math
import os
import sys
import time

import numpy as np

# Keep habitat's own logging off stdout so the Flask banner stays readable.
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

from flask import Flask, Response, jsonify, request
from werkzeug.serving import WSGIRequestHandler
from hydra import compose, initialize
from PIL import Image

import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import observations_to_image, overlay_frame

from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from prettytable import PrettyTable

# Browser key -> (habitat action, camera pitch delta). Mirrors the ACTION enum
# habitat_evaluation.py maps ROS commands through, so the human is restricted to
# exactly the agent's action space -- 0.25 m steps and 30 deg turns come from the
# dataset config, not from here.
KEYMAP = {
    "w": ("move_forward", 0.0),
    "a": ("turn_left", 0.0),
    "d": ("turn_right", 0.0),
    "r": ("look_up", math.pi / 6.0),
    "f": ("look_down", -math.pi / 6.0),
    " ": ("stop", 0.0),
}


def compute_oracle_step_count(env, success_distance, max_episode_steps):
    """Oracle shortest-path action count (t_i*) for StepSPL.

    NOTE: this mirrors habitat_evaluation.py:180 rather than importing it --
    importing that module executes its top-level `import rospy`, which would drag
    the whole ROS stack into a tool that deliberately does not use it. If the
    StepSPL definition changes there, change it here too. (Better fix once no
    sweep is running: lift both into basic_utils/.)
    """
    episode = env.current_episode
    start_position, start_rotation = episode.start_position, episode.start_rotation

    candidate_positions = [
        vp.agent_state.position for goal in episode.goals for vp in goal.view_points
    ]
    if not candidate_positions:
        env.sim.set_agent_state(start_position, start_rotation)
        return max_episode_steps

    distances = [
        env.sim.geodesic_distance(start_position, [p], None)
        for p in candidate_positions
    ]
    nearest = candidate_positions[int(np.argmin(distances))]

    follower = ShortestPathFollower(env.sim, success_distance, False)
    oracle_steps = 0
    try:
        action = follower.get_next_action(nearest)
        while action != HabitatSimActions.stop and oracle_steps < max_episode_steps:
            env.sim.step(action)
            oracle_steps += 1
            action = follower.get_next_action(nearest)
    except Exception as e:  # noqa: BLE001 -- same graceful degrade as the sweep
        print(f"[StepSPL WARNING] oracle rollout failed ({type(e).__name__}: {e})")
        oracle_steps = max_episode_steps

    env.sim.set_agent_state(start_position, start_rotation)
    return oracle_steps


def is_on_same_floor(height, episode, ceiling_height=2.0):
    """Same one-sided 2 m window failure_check.py uses -- see
    docs/ApexNav_Failure_Categories.md for why this is not a reachability test."""
    ref = episode.start_position[1]
    return ref <= height < ref + ceiling_height


def load_playlist(path):
    """Read a subset spec into a list of 0-based iterator indices.

    Accepts make_subset.py's metadata.json (selection.indices_0based) or a plain
    text file of one index per line ('#' comments allowed) -- the latter so a
    playlist can be hand-written without running make_subset.py at all.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    if path.endswith(".json"):
        meta = json.loads(text)
        try:
            idx = meta["selection"]["indices_0based"]
        except (KeyError, TypeError):
            sys.exit(f"{path}: no selection.indices_0based -- not a make_subset.py metadata.json")
    else:
        idx = []
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                idx.extend(int(tok) for tok in line.replace(",", " ").split())

    if not idx:
        sys.exit(f"{path}: playlist is empty")

    # The episode iterator is configured cycle:false, shuffle:false, so it only
    # ever moves forward and never wraps -- a playlist can only be walked in
    # ascending order. Sorting here (rather than erroring on an out-of-order
    # spec) is safe because play order does not affect any per-episode metric.
    # It does mean make_subset.py's --shuffle has no effect on playback.
    ordered = sorted(dict.fromkeys(idx))
    if ordered != list(idx):
        print(f"note: playlist re-ordered ascending ({len(idx)} -> {len(ordered)} unique)",
              flush=True)
    return ordered


class HumanSession:
    """Owns the env and walks a playlist of episodes, accumulating the same
    cumulative totals habitat_evaluation.py keeps, so the record.txt/continue.txt
    pair it writes is byte-comparable with a real sweep's.

    Every method runs on Flask's single request thread -- habitat-sim is not
    thread-safe, which is why app.run(threaded=False)."""

    def __init__(self, cfg, playlist, panels, out_dir, no_save=False):
        self.panels = panels
        self.playlist = playlist
        self.out_dir = out_dir
        self.no_save = no_save
        self.max_steps = cfg.habitat.environment.max_episode_steps
        self.success_distance = cfg.habitat.task.measurements.success.success_distance

        if panels == "all":
            with habitat.config.read_write(cfg):
                cfg.habitat.task.measurements.update(
                    {
                        "top_down_map": TopDownMapMeasurementConfig(
                            map_padding=3,
                            map_resolution=256,
                            draw_source=True,
                            draw_border=True,
                            # Off by default: drawing the oracle path would hand
                            # the human the answer outright.
                            draw_shortest_path=False,
                            draw_view_points=False,
                            draw_goal_positions=False,
                            draw_goal_aabbs=False,
                            fog_of_war=FogOfWarConfig(
                                draw=True, visibility_dist=5.0, fov=79
                            ),
                        )
                    }
                )

        self.env = habitat.Env(cfg)
        # Iterator index currently assigned to env.current_episode. Env.__init__
        # consumes index 0, so we start there.
        self.consumed = 0

        self.record_path = os.path.join(out_dir, "record.txt")
        self.continue_path = os.path.join(out_dir, "continue.txt")

        # Resume exactly the way habitat_evaluation.py does: continue.txt's newest
        # block carries the cumulative totals, and No.N tells us N episodes are
        # already done, so play resumes at playlist[N].
        (
            self.num_total,
            self.num_success,
            self.spl_all,
            self.soft_spl_all,
            self.distance_to_goal_all,
            self.distance_to_goal_reward_all,
            self.step_spl_all,
            self.cum_seconds,
        ) = read_record(self.continue_path, flag_once=no_save)

        if self.num_total > len(self.playlist):
            sys.exit(
                f"{self.continue_path} has {self.num_total} episodes but the playlist "
                f"has only {len(self.playlist)} -- wrong --out directory for this playlist?"
            )
        self.position = self.num_total
        if self.position:
            print(f"resuming at playlist entry {self.position + 1}/{len(self.playlist)}",
                  flush=True)
        self.load(self.position)

    def load(self, position):
        """Advance the iterator to playlist[position] and reset onto it."""
        self.position = position
        self.all_done = position >= len(self.playlist)
        if self.all_done:
            self.finished = True
            return

        target = self.playlist[position]
        if target < self.consumed:
            sys.exit(
                f"playlist index {target} is behind the iterator (at {self.consumed}); "
                "iterator_options.cycle is false so it cannot rewind"
            )
        # Walk forward, then assign unconditionally -- the current_episode setter
        # clears _episode_from_iter_on_reset (env.py:161), so the reset() below
        # lands on exactly this episode instead of advancing past it. The
        # assignment matters even when the walk is zero-length, because a previous
        # reset() will have set that flag back to True (env.py:257).
        ep_obj = self.env.current_episode
        for _ in range(target - self.consumed):
            ep_obj = next(self.env.episode_iterator)
        self.env.current_episode = ep_obj
        self.consumed = target

        self.obs = self.env.reset()
        self.oracle_steps = compute_oracle_step_count(
            self.env, self.success_distance, self.max_steps
        )
        self.episode = self.env.current_episode
        self.target = self.episode.object_category
        self.steps = 0
        self.camera_pitch = 0.0
        self.pass_object = False
        self.finished = False
        self.result = None
        self.started_at = time.time()
        return self.state()

    def frame_png_b64(self):
        info = self.env.get_metrics()
        if self.panels == "all":
            img = observations_to_image(self.obs, info)
            info.pop("top_down_map", None)
            img = overlay_frame(img, {"steps": self.steps, "target": self.target})
        else:
            img = self.obs["rgb"]
        buf = _io.BytesIO()
        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def state(self):
        if self.all_done:
            return {"all_done": True, "n_played": self.num_total,
                    "n_playlist": len(self.playlist)}
        info = self.env.get_metrics()
        return {
            "frame": self.frame_png_b64(),
            "target": self.target,
            "steps": self.steps,
            "max_steps": self.max_steps,
            "distance_to_goal": round(float(info.get("distance_to_goal", 0.0)), 3),
            "scene": os.path.basename(self.episode.scene_id),
            "episode_id": str(self.episode.episode_id),
            "finished": self.finished,
            "result": self.result,
            "oracle_steps": self.oracle_steps,
            "all_done": False,
            "position": self.position + 1,
            "n_playlist": len(self.playlist),
            "epi_index": self.playlist[self.position],
        }

    def act(self, key):
        if self.finished or key not in KEYMAP:
            return self.state()
        name, pitch_delta = KEYMAP[key]

        if name == "stop":
            self.obs = self.env.step(HabitatSimActions.stop)
            self.steps += 1
            return self._finish()

        self.obs = self.env.step(getattr(HabitatSimActions, name))
        self.steps += 1
        self.camera_pitch += pitch_delta

        info = self.env.get_metrics()
        if info["distance_to_goal"] <= self.success_distance:
            self.pass_object = True

        if self.steps >= self.max_steps:
            return self._finish()
        return self.state()

    def _finish(self):
        """Classify with the same vocabulary failure_check.py uses.

        check_failure() itself is not callable here: its branches key off
        final_state/expl_result, which are published by the C++ planner and have
        no meaning for a human. This maps the human's equivalent situations onto
        the same label set so the output stays comparable -- see
        docs/ApexNav_Failure_Categories.md.
        """
        info = self.env.get_metrics()
        self.finished = True
        success = int(info["success"])
        near = info["distance_to_goal"] <= self.success_distance

        if success:
            self.result = "success"
        elif not any(is_on_same_floor(g.position[1], self.episode)
                     for g in self.episode.goals):
            self.result = "infeasible"
        elif self.steps < self.max_steps:
            # Human chose to STOP and habitat did not score it.
            self.result = "false positive"
        elif self.pass_object or near:
            self.result = "[stepout] false negative"
        else:
            self.result = "stepout feasible"

        self.step_spl = success * (
            self.oracle_steps / max(self.steps, self.oracle_steps, 1)
        )
        self.metrics = {
            "success": success,
            "spl": float(info["spl"]) if math.isfinite(info["spl"]) else 0.0,
            "soft_spl": float(info["soft_spl"]) if math.isfinite(info["soft_spl"]) else 0.0,
            "step_spl": self.step_spl,
            "distance_to_goal": float(info["distance_to_goal"]),
        }
        self._accumulate_and_save()
        s = self.state()
        s["metrics"] = self.metrics
        s["totals"] = {
            "n": self.num_total,
            "success_pct": self.num_success / self.num_total * 100,
            "spl_pct": self.spl_all / self.num_total * 100,
            "step_spl_pct": self.step_spl_all / self.num_total * 100,
        }
        return s

    def _accumulate_and_save(self):
        """Fold this episode into the running totals and rewrite both files.

        Same non-finite guard as habitat_evaluation.py:673 and for the same
        reason: habitat's SPL/soft_spl/distance_to_goal go inf/NaN when the goal
        is unreachable via the navmesh, and these are running sums, so one
        non-finite value would poison every later average. The episode still
        counts in num_total.
        """
        m = self.metrics
        self.num_total += 1
        self.num_success += m["success"]
        self.spl_all += m["spl"] if math.isfinite(m["spl"]) else 0.0
        self.soft_spl_all += m["soft_spl"] if math.isfinite(m["soft_spl"]) else 0.0
        self.step_spl_all += m["step_spl"]
        self.distance_to_goal_all += (
            m["distance_to_goal"] if math.isfinite(m["distance_to_goal"]) else 0.0
        )
        # write_record()'s time field is cumulative across the whole run, not
        # per-episode -- habitat_evaluation.py builds it as (now - start + last_time)
        # where last_time came from continue.txt. Same convention here.
        self.cum_seconds += time.time() - self.started_at

        n = self.num_total
        table1 = PrettyTable(["Metric", "Average"])
        table1.add_row(["Average Success", f"{self.num_success / n * 100:.2f}%"])
        table1.add_row(["Average SPL", f"{self.spl_all / n * 100:.2f}%"])
        table1.add_row(["Average Soft SPL", f"{self.soft_spl_all / n * 100:.2f}%"])
        table1.add_row(["Average StepSPL", f"{self.step_spl_all / n * 100:.2f}%"])
        table1.add_row(["Average Distance to Goal", f"{self.distance_to_goal_all / n:.4f}"])

        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Total Success", f"{self.num_success}"])
        table2.add_row(["Total SPL", f"{self.spl_all:.2f}"])
        table2.add_row(["Total Soft SPL", f"{self.soft_spl_all:.2f}"])
        table2.add_row(["Total StepSPL", f"{self.step_spl_all:.2f}"])
        table2.add_row(["Total Distance to Goal", f"{self.distance_to_goal_all:.4f}"])

        # Totals above are still accumulated under --no-save so the on-screen
        # score and the per-episode console line stay correct; only the two
        # record files are withheld.
        if self.no_save:
            return

        for table, path in ((table1, self.record_path), (table2, self.continue_path)):
            write_record(
                self.episode.scene_id, self.episode.episode_id, table, self.result,
                self.target, self.num_total, self.cum_seconds, path,
            )

    def next_episode(self):
        """Advance to the next playlist entry (no-op if the current one is live)."""
        if not self.finished:
            return self.state()
        return self.load(self.position + 1) or self.state()


PAGE = """<!doctype html><meta charset=utf-8><title>ApexNav human play</title>
<style>
 body{margin:0;background:#14140f;color:#e8e7df;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;
      display:flex;flex-direction:column;align-items:center;gap:12px;padding:16px}
 img{max-width:min(92vw,900px);image-rendering:pixelated;border-radius:8px;background:#000}
 .bar{display:flex;gap:20px;flex-wrap:wrap;justify-content:center;align-items:baseline}
 .k{color:#8a8983;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
 .v{font-size:19px;font-weight:600}
 .target{color:#eda100}
 kbd{background:#2a2a24;border:1px solid #3d3d34;border-radius:4px;padding:1px 6px;font-size:12px}
 #done{display:none;background:#1d1d17;border:1px solid #3d3d34;border-radius:8px;padding:12px 18px;
       max-width:min(92vw,900px);width:100%}
 table{border-collapse:collapse;width:100%} td{padding:3px 8px;border-bottom:1px solid #2a2a24}
 td:last-child{text-align:right;font-variant-numeric:tabular-nums}
</style>
<div class=bar>
  <div><div class=k>find</div><div class="v target" id=target>-</div></div>
  <div><div class=k>steps</div><div class=v id=steps>0</div></div>
  <div><div class=k>scene</div><div class=v id=scene>-</div></div>
  <div><div class=k>episode</div><div class=v id=epi>-</div></div>
  <div><div class=k>progress</div><div class=v id=prog>-</div></div>
</div>
<img id=view>
<div class=bar>
 <span><kbd>W</kbd> forward</span><span><kbd>A</kbd> left</span><span><kbd>D</kbd> right</span>
 <span><kbd>R</kbd> look up</span><span><kbd>F</kbd> look down</span>
 <span><kbd>Space</kbd> stop here</span>
</div>
<div id=done></div>
<script>
let busy=false, over=false;
function paint(s){
  if(s.all_done){
    over=true;
    document.getElementById('view').style.display='none';
    const d=document.getElementById('done');
    d.style.display='block';
    d.innerHTML='<b>playlist complete</b><p>'+s.n_played+' / '+s.n_playlist+
      ' episodes recorded. record.txt and continue.txt are written — '+
      'you can close this tab.</p>';
    return;
  }
  document.getElementById('view').src='data:image/png;base64,'+s.frame;
  document.getElementById('target').textContent=s.target;
  document.getElementById('steps').textContent=s.steps+' / '+s.max_steps;
  document.getElementById('scene').textContent=s.scene;
  document.getElementById('epi').textContent=s.episode_id;
  document.getElementById('prog').textContent=s.position+' / '+s.n_playlist;
  if(s.finished){
    over=true;
    const m=s.metrics||{}, d=document.getElementById('done');
    d.style.display='block';
    d.innerHTML='<b>'+s.result+'</b><table>'+
      '<tr><td>success</td><td>'+m.success+'</td></tr>'+
      '<tr><td>SPL</td><td>'+(100*m.spl).toFixed(2)+'%</td></tr>'+
      '<tr><td>Soft SPL</td><td>'+(100*m.soft_spl).toFixed(2)+'%</td></tr>'+
      '<tr><td>StepSPL</td><td>'+(100*m.step_spl).toFixed(2)+'%</td></tr>'+
      '<tr><td>your steps / oracle</td><td>'+s.steps+' / '+s.oracle_steps+'</td></tr>'+
      '<tr><td>distance to goal</td><td>'+m.distance_to_goal.toFixed(3)+' m</td></tr>'+
      '</table><p>Run so far: <b>'+(s.totals?s.totals.n:0)+'</b> episodes, SR <b>'+
      (s.totals?s.totals.success_pct.toFixed(1):'0')+'%</b>, SPL '+
      (s.totals?s.totals.spl_pct.toFixed(1):'0')+'%.'+
      ' Written to record.txt / continue.txt. Press <kbd>N</kbd> for the next episode.</p>';
  }
}
async function send(k){
  if(busy) return; busy=true;
  try{
    const r=await fetch('/act',{method:'POST',headers:{'Content-Type':'application/json'},
                               body:JSON.stringify({key:k})});
    paint(await r.json());
  } finally { busy=false; }
}
async function next(){
  busy=true;
  try{
    document.getElementById('done').style.display='none'; over=false;
    const r=await fetch('/next',{method:'POST'}); paint(await r.json());
  } finally { busy=false; }
}
addEventListener('keydown',e=>{
  const k=e.key.toLowerCase();
  if(k==='n'&&over){e.preventDefault();next();return;}
  if(over) return;
  if(['w','a','d','r','f',' '].includes(k)){e.preventDefault();send(k===' '?' ':k);}
});
fetch('/state').then(r=>r.json()).then(paint);
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hm3dv1",
                    choices=["hm3dv1", "hm3dv2", "mp3d", "ovon"])
    ap.add_argument("--episode", type=int,
                    help="single 0-based iterator index (NOT episode_id -- use find_episode_index.py)")
    ap.add_argument("--playlist",
                    help="make_subset.py metadata.json, or a text file of 0-based indices")
    ap.add_argument("--panels", default="rgb", choices=["rgb", "all"])
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--out",
                    help="output dir (default: videos/human/test_<dataset>_<split>)")
    ap.add_argument("--no-save", action="store_true",
                    help="practice run: score on screen but write nothing to disk, "
                         "and ignore any existing results in the output dir")
    args = ap.parse_args()

    if args.episode is not None and args.playlist:
        ap.error("--episode and --playlist are mutually exclusive")

    if args.dataset == "ovon":
        import ovon  # noqa: F401 -- registers OVON-v1, same as habitat_evaluation.py

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=f"habitat_eval_{args.dataset}")

    # Same as habitat_evaluation.py:328, and required for the same reason: a raw
    # compose() leaves habitat.simulator.agents_order as MISSING, so Env.__init__
    # dies in get_agent_config() with MissingMandatoryValue. Must run before the
    # panels edits and before habitat.Env().
    cfg = patch_config(cfg)

    playlist = load_playlist(args.playlist) if args.playlist else [args.episode or 0]

    # Mirrors the agent's videos/test_<dataset>_<split>/ layout so human and agent
    # runs sit side by side and analyze_failures.py can read either.
    out_dir = args.out or os.path.join(
        "videos", "human", f"test_{args.dataset}_{cfg.habitat.dataset.split}"
    )

    # --no-save is for practice/probing runs: play an episode, see its score on
    # screen, leave no trace. Skipping makedirs as well as the writes means the
    # default results directory is not even created as a side effect, so a probe
    # can never be mistaken later for a real (empty) baseline run.
    if not args.no_save:
        os.makedirs(out_dir, exist_ok=True)

    # Pin the playlist next to its results. A continue.txt only means anything
    # against the playlist it was produced from -- resuming with a different one
    # would silently attribute episode N's metrics to a different episode.
    # Nothing is pinned or checked under --no-save: with no results being written
    # there is no continue.txt for a mismatched playlist to corrupt, and a probe
    # must not be blocked by (or overwrite) an unrelated real run's pin.
    pin_path = os.path.join(out_dir, "playlist.json")
    pin = {"dataset": args.dataset, "split": cfg.habitat.dataset.split,
           "source": os.path.abspath(args.playlist) if args.playlist else None,
           "n": len(playlist), "indices_0based": playlist}
    if args.no_save:
        pass
    elif os.path.exists(pin_path):
        with open(pin_path, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
        if existing.get("indices_0based") != playlist:
            # Show the indices, not just the counts -- the common case is two
            # single-episode runs, where "1 episodes vs 1" says nothing.
            def _brief(idx):
                idx = list(idx or [])
                shown = ", ".join(str(i) for i in idx[:6])
                return f"[{shown}{', ...' if len(idx) > 6 else ''}] ({len(idx)})"

            done = read_record(os.path.join(out_dir, "continue.txt"))[0]
            sys.exit(
                f"{pin_path} pins {_brief(existing.get('indices_0based'))}\n"
                f"but this run asks for {_brief(playlist)}.\n\n"
                f"That directory holds {done} completed episode(s). Either pass a "
                f"different --out for this run, or, if those results are no longer "
                f"wanted, delete {out_dir} to start over."
            )
    else:
        with open(pin_path, "w", encoding="utf-8") as fh:
            json.dump(pin, fh, indent=2)

    if args.no_save:
        print(f"Playlist: {len(playlist)} episode(s) -> NOT SAVED (--no-save)", flush=True)
    else:
        print(f"Playlist: {len(playlist)} episode(s) -> {out_dir}", flush=True)
    ep = HumanSession(cfg, playlist, args.panels, out_dir, no_save=args.no_save)
    if ep.all_done:
        print(f"Nothing to do: all {len(playlist)} episodes already in continue.txt",
              flush=True)
        return
    print(f"Ready: find a [{ep.target}] in {os.path.basename(ep.episode.scene_id)}",
          flush=True)

    app = Flask(__name__)
    # Flask's dev-server request log would bury the episode output.
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/state")
    def state():
        return jsonify(ep.state())

    @app.post("/act")
    def act():
        was_finished = ep.finished
        s = ep.act((request.json or {}).get("key", ""))
        if s["finished"] and not was_finished:
            print(f"[{ep.position + 1}/{len(playlist)}] idx={playlist[ep.position]} "
                  f"{ep.result} steps={ep.steps} oracle={ep.oracle_steps} "
                  f"| SR so far {ep.num_success}/{ep.num_total}", flush=True)
        return jsonify(s)

    @app.post("/next")
    def nxt():
        s = ep.next_episode()
        if not ep.all_done:
            print(f"-> [{ep.position + 1}/{len(playlist)}] find a [{ep.target}] "
                  f"in {os.path.basename(ep.episode.scene_id)}", flush=True)
        else:
            print(f"Playlist complete: {ep.num_total} episodes, "
                  f"SR {ep.num_success / ep.num_total * 100:.2f}% -> {out_dir}",
                  flush=True)
        return jsonify(s)

    print(f"\n  open  http://localhost:{args.port}   (VS Code forwards this port)\n",
          flush=True)
    # threaded=False: every sim call must stay on one thread. habitat-sim binds
    # its EGL context to the thread that created it, so threaded=True would hand
    # sim calls to a different thread and break rendering -- a lock is not enough.
    #
    # The cost of one thread is that a single connection can wedge the whole
    # server: werkzeug's handler sets no socket timeout, so a client that opens a
    # TCP connection and never sends a request line leaves the server blocked in
    # readline() forever, with every later connection stuck unaccepted in the
    # backlog (symptom: browser spins, `ss` shows a growing Recv-Q on the listen
    # socket). VS Code's port-forwarder probes the port exactly that way. A
    # timeout makes those sockets expire instead of hanging: BaseHTTPRequestHandler
    # catches socket.timeout in handle_one_request and closes the connection.
    # It only bounds recv/send syscalls, not the sim step inside /act, so slow
    # episodes are unaffected.
    #
    # 10s rather than something longer because threaded=False means this is also
    # the worst-case stall the browser sees when a probe socket does wedge it.
    # Nothing legitimate sits idle mid-connection to be cut off: werkzeug leaves
    # protocol_version at HTTP/1.0 unless the server is threaded, so there is no
    # keep-alive and every connection is accept -> read -> respond -> close.
    WSGIRequestHandler.timeout = 10
    app.run(host="127.0.0.1", port=args.port, threaded=False, debug=False)


if __name__ == "__main__":
    main()
