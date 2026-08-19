"""
debug_ovon_dataset_load.py

Isolates ONLY the OVON dataset construction step (the thing that hangs right
after "Initializing dataset OVON-v1" prints in habitat_evaluation.py) -- no
roscore, no VLM/detector servers, no exploration_manager, no habitat.Env().
Just: compose the Hydra config, then build the dataset directly, with
per-scene-file timing printed as it goes (monkey-patched onto
PointNavDatasetV1._load_from_file rather than editing the installed package).

Confirmed sane data scale first (2026-08-04): val_seen/content/ is 36 files,
2.7MB total -- so if this hangs or is slow, it's NOT a data-volume problem,
it's something in the actual loading/parsing logic (or an environment issue
like the container's squashfuse mount) that this script should pinpoint.

Extended (2026-08-04) to also construct habitat.Env() -- i.e. actually build
the OVONSim-v0 simulator, not just the dataset -- after the single-episode
test died silently at "initializing sim OVONSim-v0" with no traceback. Root
cause was our own ovon/__init__.py trim (done to dodge an unneeded `import
clip`) accidentally dropping `from ovon.task import simulator`, which is what
registers OVONSim-v0 in the first place. Fixed, and this step is here so that
mistake (or ones like it -- e.g. missing sensors/measures) gets caught fast
via this script instead of burning 15+ min bringing up the full ROS stack.

Run via jobs/apptainer_run.sh (needs the full --nv/EGL bind set now that sim
construction is in scope, unlike the original dataset-only version of this
script):
  bash /scratch2/ml20/btripcon/jobs/apptainer_run.sh \
    /scratch2/ml20/btripcon/conda/envs/apexnav/bin/python debug_ovon_dataset_load.py

Authored by Claude (Anthropic Claude Opus 5) for Broden Tripcony.
"""
import time

t_start = time.time()


def log(msg):
    print(f"[{time.time() - t_start:7.2f}s] {msg}", flush=True)


log("Importing habitat / habitat_baselines (registers Hydra search-path plugin)...")
import habitat  # noqa: E402
import habitat_baselines  # noqa: E402,F401
log("Importing ovon (registers OVON-v1 dataset class -- this was the missing piece)...")
import ovon  # noqa: E402,F401
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1  # noqa: E402

log("Import done. Patching PointNavDatasetV1._load_from_file for per-file timing...")

_orig_load_from_file = PointNavDatasetV1._load_from_file


def _timed_load_from_file(self, fname, scenes_dir):
    t0 = time.time()
    log(f"  -> loading {fname} ...")
    _orig_load_from_file(self, fname, scenes_dir)
    log(
        f"     done in {time.time() - t0:.2f}s "
        f"(episodes so far: {len(self.episodes)})"
    )


PointNavDatasetV1._load_from_file = _timed_load_from_file

log("Composing Hydra config (habitat_eval_ovon)...")
from hydra import initialize, compose  # noqa: E402
from habitat.config.default import patch_config  # noqa: E402

with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="habitat_eval_ovon")
# habitat_evaluation.py calls this after compose() too (it's what fills in
# habitat.simulator.agents_order for single-agent setups) -- this debug
# script was composing raw and skipping it, hence the
# "Missing mandatory value: habitat.simulator.agents_order" crash in
# habitat.Env() below.
cfg = patch_config(cfg)

# OVON is open-vocabulary -- habitat-lab's default ObjectGoalSensor (defined in
# object_nav_task.py, pulled in by /habitat/task: objectnav) requires
# dataset.category_to_task_category_id, a closed-vocab concept OVONDatasetV1
# doesn't implement. ApexNav never reads this sensor's output (confirmed via
# grep -rn "objectgoal" ApexNav/*.py -- own VLM/LLM pipeline does detection),
# so drop it for OVON runs instead of faking a category mapping.
if cfg.habitat.dataset.type == "OVON-v1":
    from omegaconf import OmegaConf, open_dict
    was_readonly = OmegaConf.is_readonly(cfg)
    OmegaConf.set_readonly(cfg, False)
    with open_dict(cfg):
        cfg.habitat.task.lab_sensors.pop("objectgoal_sensor", None)
    OmegaConf.set_readonly(cfg, was_readonly)
log("Config composed.")

log(
    f"Dataset config: type={cfg.habitat.dataset.type}, split={cfg.habitat.dataset.split}, "
    f"data_path={cfg.habitat.dataset.data_path}"
)

log(
    "Constructing dataset via make_dataset() -- this is the exact call "
    "habitat.Env() makes internally, the one that hangs in the real run..."
)
from habitat.datasets import make_dataset  # noqa: E402

t0 = time.time()
dataset = make_dataset(cfg.habitat.dataset.type, config=cfg.habitat.dataset)
log(f"make_dataset() returned in {time.time() - t0:.2f}s, total episodes: {len(dataset.episodes)}")

log(
    "Constructing habitat.Env() -- this actually builds OVONSim-v0, the exact "
    "call that died silently (no traceback) in the single-episode test..."
)
t0 = time.time()
env = habitat.Env(config=cfg.habitat, dataset=dataset)
log(f"habitat.Env() constructed in {time.time() - t0:.2f}s")

log("Calling env.reset() to fully exercise one episode load...")
t0 = time.time()
obs = env.reset()
log(f"env.reset() returned in {time.time() - t0:.2f}s, obs keys: {list(obs.keys())}")

env.close()
log("SUCCESS -- dataset + sim construction + reset completed without hanging or crashing.")
