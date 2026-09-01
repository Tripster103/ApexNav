# Dataset access — HM3D (v0.1/v0.2) and MP3D

Both datasets are gated behind their own license agreements. **The data itself cannot be
redistributed** — the Matterport EULA both HM3D and MP3D operate under explicitly prohibits
any third party distributing a substantial portion of the data without a mutually executed
agreement covering the recipient, and if you do redistribute, you're expected to collect the
same personal info (name/email/institution) from your recipients and pass it on to Matterport.
Each person who needs the data has to go through the steps below themselves, using their own
institutional email — that's not optional. This doc exists so the process is fast the second
time, not so the data can be skipped.

**Security note:** an API credential used for a previous HM3D download exists in this
project's local `datasets/HM3D_API_Token.md` — that file is not, and must never be, committed
to this or any repo. Generate your own token per the steps below; don't reuse or share
someone else's.

## Key license terms (Matterport EULA for Academic Use of Model Data)

- Non-commercial academic use only.
- **Attribution required**: any publication (paper, thesis) containing the data or results
  derived from it must include this Agreement or a link to it.
- No attempting to de-anonymise/identify the address or owner of any scanned location.
- No using any embedded Matterport software for anything beyond accessing the data, no
  reverse-engineering it.
- Can't use the data to build a competing product to Matterport's own 3D Showcase.
- Data provided as-is, no warranty; Matterport's liability capped at $50; you indemnify them
  for your use.
- Access requires giving Matterport your name, email, and academic institution.
- Governed by California law (Santa Clara County courts), standard US export-control
  restrictions apply.
- Signature page needs: requester's name/email, PI's name/email, affiliation, signature+date.

Full text: [Matterport End User License Agreement for Academic Use of Model Data](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data)

## HM3D (covers both v1 and v2 — same underlying scans, different semantic annotation versions)

1. Request access: https://matterport.com/habitat-matterport-3d-research-dataset — agree to
   the EULA above.
2. Once approved, generate your own Matterport API token: go to
   https://my.matterport.com/settings/account/devtools and generate a token. The token **ID**
   is your username, the token **secret** is your password. Write the secret down immediately
   — Matterport won't show it again.
3. **Confirmed working method** (used previously for this project) — direct download via the
   Matterport API:
   ```bash
   wget --user=<your-token-id> --password=<your-token-secret> \
        -O hm3d-val-habitat-v0.2.tar \
        https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar
   ```
   Swap the filename/URL for whichever split/version you need — check
   https://api.matterport.com/resources/habitat/ or Habitat's own docs for the full list of
   available archive names (train/val, v0.1/v0.2, minival, etc.) at the time of download,
   since exact naming can change between releases.
4. Alternative: Habitat's own download utility also works, if preferred —
   `python -m habitat_sim.utils.datasets_download --username <id> --password <secret> --uids hm3d_minival_v0.2`
   (run `--help` first to list every currently valid uid, including v0.1-specific ones).
5. Sanity-check the resulting folder structure: a folder per scene containing
   `<id>.basis.glb`, `<id>.basis.navmesh`, `<id>.semantic.glb`, `<id>.semantic.txt`, plus
   `hm3d_annotated_basis.scene_dataset_config.json` at the dataset root.

## MP3D (Matterport3D)

1. Sign the Terms of Use agreement (same EULA as above) using your institutional email, and
   send it to `matterport3d@googlegroups.com`. This is a manual, human-reviewed step — expect
   it to take some time, not instant. They reply with a `download_mp.py` script.
2. **Requires Python 2** to run the download script (not the apexnav conda env's Python 3.9
   — use a separate `python2` or system Python 2 install).
3. Download the Habitat-formatted subset only, not the full 1.3TB Matterport3D release:
   ```bash
   python download_mp.py --task habitat -o path/to/download/
   ```
4. Download the scene dataset config file and place it at the root of the MP3D data folder:
   ```bash
   wget http://dl.fbaipublicfiles.com/habitat/mp3d/config_v1/mp3d.scene_dataset_config.json
   ```

## What's fine to share directly (not license-gated)

The download scripts/commands above, this doc, and ApexNav's own config files
(`config/habitat_eval_mp3d.yaml` etc.) are all just code/config — no restriction on sharing
those. It's specifically the scene `.glb`/`.navmesh`/`.semantic.*` files, and any live API
credential, that must never be shared or committed.
