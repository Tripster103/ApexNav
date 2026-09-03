# tools/

Scripts written for this fork. Everything in here is ours; everything still in
the ApexNav root (`habitat_evaluation.py`, `habitat_manual_control.py`,
`habitat_vel_control.py`, `params.py`, `LICENSE`) is upstream's. That separation
is the point of this folder — it used to take a `git log --diff-filter=A` to tell
the two apart.

All of these run inside the apptainer container and from any working directory —
they resolve `config/` and the repo root from `__file__`, not from `cwd`:

```bash
bash containers/apexnav/apptainer_run.sh python <script> [args]
```

## `human/` — the human baseline pipeline

Play ObjectNav episodes by hand and score them exactly like the agent, to
establish a human reference on the same episodes. See `commands.md` for the full
workflow.

| script | what it does |
|---|---|
| `make_playlist.py` | Sample episodes **from the dataset**. Stratified round-robin over scenes, then shuffled into play order. `--n` defaults to 50. Writes `<repo>/playlists/<dataset>_random<N>_seed<S>.json`. |
| `human_play.py` | Play a playlist by hand in a browser. Writes `record.txt`/`continue.txt` in the same format the sweep writes, so results drop straight into `analyze_failures.py`. Needs a GPU node. Launch via `jobs/run_human_play.sh`, never by hand. |
| `find_episode_index.py` | Map `scene + episode_id` <-> `test_epi_num` (the 0-based iterator index). Its `build_iterator()` is what `make_playlist.py` imports, and it is the **only** correct way to compute these indices — `group_by_scene` defaults True, so iterator order is scene-clustered, not dataset file order. Also useful for `run_single_episode_test.sh --episode N`. |

The dataset-side selector here is distinct from
`basic_utils/record_episode/make_subset.py`, which carves a subset out of an
existing *agent run's* `continue.txt` and so can only select episodes some agent
already played. Both emit the same `selection.indices_0based` key, so
`human_play.py --playlist` takes either.

## `ovon/` — OVON setup and debugging

| script | what it does |
|---|---|
| `derive_camera_params.py` | Derives camera FOV/angle args from the dataset config. Called automatically by `run_apexnav_benchmark.sh` and `run_single_episode_test.sh` for `--dataset ovon`; a no-op for the LoCoBot datasets. |
| `debug_ovon_dataset_load.py` | One-off probe for the 2026-08-04 OVON dataset-load investigation. Times `PointNavDatasetV1._load_from_file` per scene. |

## Not moved

`basic_utils/failure_check/analyze_failures.py` and
`basic_utils/record_episode/make_subset.py` are also ours, but they stay where
they are: they sit alongside the upstream modules they extend
(`failure_check.py`, `read_record.py`/`write_record.py`) and are imported by
relative package path.

Authored by Claude (Anthropic Claude Opus 5) for Broden Tripcony.
