# Dataset access — HM3D (v0.1/v0.2) and MP3D

Both datasets are gated behind their own license agreements. **The data itself cannot be
redistributed** — Matterport's EULA explicitly prohibits any third party distributing a
substantial portion of the data without its own signed agreement covering the recipient
(*"You shall not, and shall not authorize any third party to use or distribute the
Matterport Dataset for any non-academic purpose"*). Each person who needs the data has to
go through the steps below themselves, using their own institutional email. This doc exists
so that process is fast the second time, not so the data can be skipped.

## HM3D (used for both v1 and v2 — same underlying scans, different semantic annotation versions)

1. Request access: https://matterport.com/habitat-matterport-3d-research-dataset — free for
   academic, non-commercial research.
2. Once approved, generate a Matterport API token: go to
   https://my.matterport.com/settings/account/devtools and generate a token. The token **ID**
   is your username, the token **secret** is your password for the download script below.
   Write the secret down immediately — Matterport will not show it again.
3. Download via Habitat's own utility (inside the apexnav conda env):
   ```bash
   python -m habitat_sim.utils.datasets_download \
     --username <api-token-id> --password <api-token-secret> \
     --uids hm3d_minival_v0.2
   ```
   Swap `hm3d_minival_v0.2` for whichever split/version uid is actually needed (full
   train/val, not just minival). **Run
   `python -m habitat_sim.utils.datasets_download --help` first to list every valid uid** —
   the exact v0.1-vs-v0.2 uid strings weren't pinned down with full confidence writing this,
   and Habitat's uid list does change between releases, so check it live rather than trust a
   hardcoded string here.
4. By default, downloading train/val pulls in HM3D-Semantics **v0.2** annotations
   automatically. For v0.1 specifically, look for the corresponding `_v0.1` uid variant in
   the `--help` output, or the separate HM3D-Semantics v0.1 entry if train/val alone doesn't
   pull it in.
5. Sanity-check the resulting folder structure matches what ApexNav's config expects (per
   habitat-sim's own docs): a scene folder per environment containing `<id>.basis.glb`,
   `<id>.basis.navmesh`, `<id>.semantic.glb`, `<id>.semantic.txt`, plus
   `hm3d_annotated_basis.scene_dataset_config.json` at the dataset root.

## MP3D (Matterport3D)

1. Sign the Terms of Use agreement form using an institutional email, and send it to
   `matterport3d@googlegroups.com`. They reply with a `download_mp.py` script — this is a
   manual, human-reviewed step, expect it to take some time, not instant.
2. Requires **Python 2.7** to run the download script (not the apexnav conda env's Python).
3. Download the Habitat-formatted subset only, not the full Matterport3D release:
   ```bash
   python download_mp.py --task habitat -o path/to/download/
   ```
4. Then separately download the scene dataset config file and place it at the root of the
   MP3D data folder:
   ```bash
   wget http://dl.fbaipublicfiles.com/habitat/mp3d/config_v1/mp3d.scene_dataset_config.json
   ```

## What's fine to share directly (not license-gated)

The download scripts/commands above, this doc, and ApexNav's own config files
(`config/habitat_eval_mp3d.yaml`, etc.) are all just code/config — no restriction on sharing
those. It's specifically the scene `.glb`/`.navmesh`/`.semantic.*` files themselves that
require each person's own signed access.
