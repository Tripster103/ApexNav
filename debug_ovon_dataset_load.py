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

Run from the repo root, inside the container, on a GPU-node allocation:
  cd /scratch2/ml20/btripcon/ApexNav
  apptainer exec \
    --bind /scratch2/ml20:/scratch2/ml20 \
    --bind /fs04/scratch2/ml20:/fs04/scratch2/ml20 \
    /scratch2/ml20/btripcon/apexnav-ros.sif \
    /scratch2/ml20/btripcon/conda/envs/apexnav/bin/python \
    /scratch2/ml20/btripcon/ApexNav/debug_ovon_dataset_load.py

(Deliberately no --nv here -- this never touches habitat_sim/rendering, pure
JSON parsing, so the GPU driver binds aren't needed. If apptainer still wants
--nv for some import-time reason, add it back.)
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

with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="habitat_eval_ovon")
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

log("SUCCESS -- dataset construction completed without hanging.")
