# ApexNav Reproduction on M3 — Setup Reference

_Authored by Claude (Anthropic Claude Opus 5) for Broden Tripcony._

Goal: reproduce ApexNav's baseline SR/SPL results (HM3Dv1, HM3Dv2, MP3D) on Monash M3 before moving on to UniGoal or any cost-aware modifications.

Paper baseline numbers to check against (Table I, SR/SPL):
- HM3Dv1: 59.6 / 33.0
- HM3Dv2: 76.2 / 38.0
- MP3D: 39.2 / 17.8

Status: setup complete (steps 1-6 below). Full benchmark sbatch runs (step 7) are in progress — see `jobs/run_apexnav_benchmark.sh`.

## Directory layout (current snapshot, confirmed 2026-07-14)

```
~                                              (home — 20GB hard cap, config only: .bashrc, .condarc, .ssh/)

/projects/ml20/                                (SKIPPED — shared, was 97% full, no personal subdir; not worth it)

/scratch2/ml20/btripcon/                       (== /fs04/scratch2/ml20/btripcon/, autofs alias to the same data —
│                                                bind BOTH forms in Apptainer, since pip -e records the /fs04/ path)
├── ApexNav/                                   # git repo, origin = Tripster103/ApexNav fork
│   ├── build/, devel/                         # catkin output — lives IN the repo, not a separate catkin_ws
│   ├── config/                                # habitat_eval_{hm3dv1,hm3dv2,mp3d}.yaml, habitat_vel_control.yaml
│   ├── data/                                  # symlinks only -> ../../data/ and ../../data/model_weights/
│   ├── GroundingDINO/, yolov7/                # cloned external detector deps (README step 1.3)
│   ├── vlm/, llm/, habitat2ros/, basic_utils/, src/planner/   # ApexNav source
│   ├── habitat_evaluation.py                  # full-dataset sweep — the real benchmark entry point
│   ├── habitat_manual_control.py              # interactive debug tool only, not used for real runs
│   └── videos/, real_world_test_example/
├── apptainer/
│   ├── enter_container.sh                     # apptainer shell + all 9 NVIDIA/bind flags
│   ├── setup_env.sh                           # sourced after enter_container.sh: LD_LIBRARY_PATH, cd, ROS, GPU check
│   └── detect_gpu.sh                          # diagnostic only, see Step 4
├── apptainer-cache/, apptainer-tmp/            # APPTAINER_CACHEDIR / APPTAINER_TMPDIR (must be in ~/.bashrc, see below)
├── conda/envs/{apexnav,mp3d-dl}/               # apexnav = main env; mp3d-dl = throwaway py2.7 env for download_mp.py
├── data/
│   ├── scene_datasets/{hm3d, hm3d_v0.2->hm3d, mp3d}/   # licensed scene data
│   ├── datasets/objectnav/{hm3d,mp3d}/                 # episode zips, no license needed
│   └── model_weights/{groundingdino_swint_ogc.pth,mobile_sam.pt,yolov7-e6e.pt}
├── habitat-lab/                                # habitat-lab v0.3.1 + habitat-baselines + habitat-hitl, pip -e installed
├── jobs/                                       # sbatch scripts (run_apexnav_benchmark.sh)
├── logs/                                       # {gdino,sam,yolov7,blip2itm}.log + slurm-*.out from batch jobs
├── results/                                    # final SR/SPL outputs — sync small stuff to local Papers/results.xlsx
├── apexnav-ros.sif, apexnav-ros.tar            # the container image (built off-cluster, see Step 4)
└── download_mp.py                              # MP3D scene downloader (run once, py2.7)
```

`datasets/` (not `objectnav_datasets/`) is Habitat-lab's own convention, not a project choice — its config YAMLs hardcode this relative path, and `scene_datasets/` vs `datasets/` separates the 3D environments (reused across any task type) from task-specific episode files. Don't rename either.

## Key constraints on M3

- **No root access.** Everything (ROS, OSQP, system libs) has to be user-space or containerized — the README's `sudo apt-get`/`sudo make install` steps don't apply.
- **Storage: skip `/projects` entirely**, everything lives on `/scratch2/ml20/btripcon/` (3TB, not backed up). Backup = git remote for the repo, periodic copy-down of small results to this local FYP folder.
- **GPU access requires Slurm.** For batch jobs: `--partition=gpu --gres=gpu:A40:1`, no `--qos` needed (confirmed via `mon_qos` — see Step 7). For interactive dev: [m3-desktop.erc.monash.edu](https://m3-desktop.erc.monash.edu/) → **`Terminal` tab** (not `Desktop`) → configure GPU/time → `Launch` → `Connect` once `RUNNING`. Never run `sbatch --qos=desktopq --partition=desktop` directly at the CLI (hangs waiting on stdin).
- **Apptainer, not Docker.** Rootless `apptainer pull` works fine on M3, but `apptainer build` from a `.def`/Dockerfile needs root and isn't available — any custom image has to be built off-cluster and copied over as a `.sif`.
- **Debugging tools (gdb, strace, VS Code debugging) are disabled on M3** as of 2026-05-15 — plan on print/log-based debugging.

## Setup steps (what actually worked)

1. **Scratch2 skeleton + repo clone.**
   ```bash
   export MPROJECT=ml20
   mkdir -p /scratch2/$MPROJECT/$USER/{conda,apptainer-cache,apptainer-tmp,logs,jobs,results}
   mkdir -p /scratch2/$MPROJECT/$USER/data/{scene_datasets/{hm3d,mp3d},datasets/objectnav/{hm3d,mp3d},model_weights}
   git clone git@github.com:Tripster103/ApexNav.git /scratch2/$MPROJECT/$USER/ApexNav
   ```
   `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` must be exported in `~/.bashrc` (not just set ad hoc), or Apptainer silently falls back to `~/.apptainer` and eats the home quota:
   ```bash
   echo "export APPTAINER_CACHEDIR=/scratch2/$MPROJECT/$USER/apptainer-cache" >> ~/.bashrc
   echo "export APPTAINER_TMPDIR=/scratch2/$MPROJECT/$USER/apptainer-tmp" >> ~/.bashrc
   source ~/.bashrc
   ```
   If home quota is already tight, `rm -rf ~/.cache ~/.vscode-server` is safe (both fully regenerate).

2. **Datasets.** HM3D and MP3D scenes both need license approval first (already done — token in `datasets/HM3D_API_Token.md` locally, MP3D TOS approved). Working download commands:
   ```bash
   # HM3D scene data — needs --auth-no-challenge, plain wget --user/--password silently saves an HTML login page instead
   cd /scratch2/$MPROJECT/$USER/data/scene_datasets/hm3d/val
   wget --auth-no-challenge --user=<token-id> --password=<token-secret> \
        -O hm3d-val-habitat-v0.2.tar https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar
   file hm3d-val-habitat-v0.2.tar   # must say "POSIX tar archive", not HTML

   # MP3D scene data — separate py2.7 env, --id "" (not the default) avoids an accidental full 1.3TB download
   mamba create -n mp3d-dl python=2.7 -y && mamba activate mp3d-dl
   python download_mp.py -o /scratch2/$MPROJECT/$USER/data/scene_datasets/mp3d/ --task_data habitat --id ""

   # Episode zips (HM3D-v1, HM3D-v2, MP3D) — no license needed, exact unzip recipe matters (HM3D zips
   # have a wrapper folder that must be flattened, MP3D's doesn't) — see ApexNav README Datasets section.
   ```
   Extract everything into the paths shown in the directory layout above.

3. **Conda environment.**
   ```bash
   module load miniforge3
   conda config --add pkgs_dirs /scratch2/$MPROJECT/$USER/conda/pkgs
   conda config --add envs_dirs /scratch2/$MPROJECT/$USER/conda/envs
   # do NOT run `conda init` -- breaks Strudel

   cd /scratch2/$MPROJECT/$USER/ApexNav
   mamba env create -f apexnav_environment.yaml && mamba activate apexnav
   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
   python -c "import torch; print(torch.cuda.is_available())"   # confirm True

   cd /scratch2/$MPROJECT/$USER
   git clone https://github.com/facebookresearch/habitat-lab.git
   cd habitat-lab && git checkout tags/v0.3.1
   pip install -e habitat-lab && pip install -e habitat-baselines

   pip install salesforce-lavis==1.0.2
   cd /scratch2/$MPROJECT/$USER/ApexNav && pip install -e .
   ```
   Numpy-related warnings during install are safe to ignore as long as `numpy==1.23.5`/`numba==0.60.0` end up installed.

4. **Container (ROS + C++ side) and Habitat rendering.**
   No root means ROS Noetic + all C++ deps (armadillo, ompl, OSQP, PCL, cv_bridge, etc.) had to be built into a Docker image off-cluster, then converted: `apptainer build image.sif docker-archive://image.tar`. Dockerfile lives at `apptainer/Dockerfile`.

   Habitat-sim's off-screen EGL renderer needs 7 individual NVIDIA driver library files bind-mounted (`--nv` alone doesn't provide them), plus separate binds for the scratch2/fs04 paths and `/usr/share/glvnd` — already handled by `apptainer/enter_container.sh`, no need to reconstruct this by hand. Day-to-day usage:
   ```bash
   srun --jobid=<id> --overlap --pty bash
   bash /scratch2/$MPROJECT/$USER/apptainer/enter_container.sh
   source /scratch2/$MPROJECT/$USER/apptainer/setup_env.sh
   ```
   Inside the container: `catkin_make -DPYTHON_EXECUTABLE=/scratch2/$MPROJECT/$USER/conda/envs/apexnav/bin/python` (needs `pip install "empy==3.3.4" --force-reinstall` first — Noetic's message generation breaks on modern `empy`). `roscore` needs to be running in a separate pane for the whole session (`ROS_MASTER_URI=http://localhost:11311` in both).

   **One critical non-obvious fact:** always pass `habitat.simulator.habitat_sim_v0.gpu_device_id=0` (the CUDA-logical index, always `0` on a single-GPU M3 job) — never the raw `/dev/nvidiaN` number that `apptainer/detect_gpu.sh` reports (that script is a diagnostic only, confirms the cgroup restriction, not a value to plug into Habitat's config).

5. **Bring up the four VLM/detector servers.** Commands per the ApexNav README ("Run VLMs Servers"), each backgrounded and logged separately:
   ```bash
   CONDA_PY=/scratch2/$MPROJECT/$USER/conda/envs/apexnav/bin/python
   export SSL_CERT_FILE=$($CONDA_PY -c "import certifi; print(certifi.where())")   # BLIP-2 needs this, stale cert store otherwise

   nohup $CONDA_PY -m vlm.detector.grounding_dino --port 12181 > logs/gdino.log  2>&1 &
   nohup $CONDA_PY -m vlm.itm.blip2itm            --port 12182 > logs/blip2.log  2>&1 &
   nohup $CONDA_PY -m vlm.segmentor.sam           --port 12183 > logs/sam.log    2>&1 &
   nohup $CONDA_PY -m vlm.detector.yolov7         --port 12184 > logs/yolov7.log 2>&1 &

   curl -s -o /dev/null -w "%{http_code}\n" --max-time 3 http://localhost:<port>/   # any status code = up
   ```
   **CONFIRMED WORKING 2026-07-14** — all four up and responding. **Known open issue, not blocking:** GroundingDINO logs `Failed to load custom C++ ops. Running on CPU mode Only!` — works, but slower than intended; worth fixing before full-sweep timing matters. LLM step is optional/skippable — repo ships pre-generated LLM outputs in `llm/answers/`.

6. **Single-episode end-to-end test — CONFIRMED WORKING 2026-07-14.**
   ```bash
   PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH $CONDA_PY habitat_evaluation.py \
     --dataset hm3dv2 test_epi_num=0 habitat.simulator.habitat_sim_v0.gpu_device_id=0
   ```
   Confirmed the whole stack (Habitat, ROS exploration planner, all four detector servers) talks to itself correctly. Result varied slightly across repeated runs of the same episode — expected: no explicit seeding anywhere in `habitat_evaluation.py`, and the action loop is real-time ROS pub/sub, not lockstep — see `ApexNav_Model_Components.md`, "Evaluation protocol & result variance".

7. **Full benchmark runs — in progress.** `jobs/run_apexnav_benchmark.sh` runs one dataset's entire val split per `sbatch` submission (unattended: brings up roscore + all four servers + the ROS exploration planner, waits for the servers, then runs the full sweep). Submit once per dataset, `--job-name` on the command line (not a static `#SBATCH` directive — see the script's header comment for why):
   ```bash
   sbatch --job-name=an-hm3dv1 jobs/run_apexnav_benchmark.sh hm3dv1
   sbatch --job-name=an-hm3dv2 jobs/run_apexnav_benchmark.sh hm3dv2
   sbatch --job-name=an-mp3d   jobs/run_apexnav_benchmark.sh mp3d
   ```
   `gpu` partition uses the `normal` QoS by default (confirmed via `mon_qos`): `MaxWall 7-00:00:00`, `MaxTRESPU gres/gpu=4` — all three can run concurrently. If a job hits the wall before finishing, resubmitting the same command resumes from `continue.txt` rather than restarting the dataset.

   **GPU type:** requests `L40S` (`--gres=gpu:L40S:1`), not `A40` — `show_cluster` showed A40s at ~82% cluster-wide utilization (only 2-4 schedulable free slots), while L40S had multiple idle/partially-free nodes on the same `gpu` partition. Same VRAM tier (48GB), newer architecture, purely a scheduling-speed choice.

   **`--exclusive` is required.** The four detector/VLM servers bind fixed ports (12181-12184), and Apptainer on M3 doesn't isolate network namespaces by default — if two of these jobs land on the same physical node (real risk once a node has multiple free GPUs), the second job's servers fail with "Address already in use". `--exclusive` forces one node per job, trading away the node's other spare GPUs for guaranteed correctness. Confirmed this collision actually happened once (`blip2-58305839.log`: `Address already in use`) before the fix was added.

   **Missing `exploration_manager` node fixed 2026-07-14.** The first real attempt at this script (job `58305933`) hung forever on `Waiting for ROS to get odometry...` after loading the Habitat env and spawning the agent. Root cause: the script only started `roscore` (the ROS *master*, i.e. just a naming/registry service) but never launched the actual planner node. Per the ApexNav README's Usage section, the real sequence after the VLM servers is:
   ```bash
   source ./devel/setup.bash && roslaunch exploration_manager exploration.launch # ApexNav main algorithm
   ```
   (there's also `roslaunch exploration_manager rviz.launch` for visualization — intentionally skipped, no display in a headless batch job). Now launched backgrounded via `nohup`, same pattern as the four VLM servers, with a fixed 15s sleep before `habitat_evaluation.py` starts (not an active readiness poll — worth revisiting if timing ever proves flaky). This step was present in the manual single-episode test (step 6 above, run interactively) but didn't make it into the first version of this automated script.

8. **Compare results against paper Table I.** Confirmed 2026-07-14 by reading `habitat_evaluation.py` + `config/habitat_eval_hm3dv1.yaml` directly (previous version of this doc incorrectly assumed a top-level `results/` folder — that was never where the script actually writes). Real output path is per-dataset, under the `ApexNav` repo itself, not `results/`:
   ```
   ApexNav/videos/test_hm3dv1_{split}/record.txt      # per-episode running average, updated every episode
   ApexNav/videos/test_hm3dv1_{split}/continue.txt    # cumulative totals + resume state, updated every episode
   ```
   (`{split}` = `val`; hm3dv2/mp3d presumably follow the same `test_<dataset>_{split}` naming, not yet confirmed from their own config files). Both files update after *every* episode, not just at the end, so interim progress is checkable any time during a multi-day run — final SR/SPL = the totals in `continue.txt` once `num_total` reaches the full episode count (2000/1000/2195). `need_video: false` by default, so no video files are generated (matches our runs — just metrics, no extra overhead). Copy the small `.txt` files down to `Papers/results.xlsx` locally once each sweep finishes (the real backup, since scratch2 isn't backed up). Checkpoint before failure-mode analysis and cost-aware extensions (Phase 2).
