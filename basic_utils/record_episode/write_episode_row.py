import csv
import os

# One row per finished episode, next to record.txt / continue.txt. Those two only
# keep 2-dp running totals, so per-episode t (count_steps) and t* (oracle steps)
# were otherwise lost -- and they are what an exact StepSPL re-score with a
# different t* needs: step_spl = success * t* / max(t, t*, 1).
#
# episode_no is the 1-based No.N that record.txt uses for the same episode.
# Append-only: a resumed run keeps adding rows, and an episode re-run after a
# crash between this write and the next can appear twice -- readers should keep
# the LAST row per episode_no.
EPISODE_FIELDS = [
    "episode_no",
    "scene_id",
    "episode_id",
    "category",
    "result",
    "success",
    "steps",
    "oracle_steps",
    "oracle_ok",
    "spl",
    "soft_spl",
    "step_spl",
    "soft_step_spl",
    "start_distance_to_goal",
    "distance_to_goal",
]


def write_episode_row(file_path, row):
    """Append one episode's row to the per-episode CSV, writing the header first
    if the file does not exist yet. `row` must have exactly EPISODE_FIELDS."""
    new_file = not os.path.exists(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
