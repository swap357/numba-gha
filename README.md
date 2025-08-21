## gha_runner.py

Concise orchestrator for llvmlite and numba GitHub Actions builds (conda + wheels) using the `gh` CLI. It can trigger workflows across platforms, wait for completion, and download artifacts.

### Prerequisites
- GitHub CLI installed and authenticated
  - EITHER set a token: `export GH_TOKEN=<token>`
  - OR login once: `gh auth login`
- Python 3.8+

### Before you run (quick checklist)
- Open `gha_runner.py` and confirm constants match your setup:
```python
LLVMLITE_REPO = "swap357/llvmlite"
NUMBA_REPO    = "swap357/numba"
DEFAULT_BRANCH = "main"
DEFAULT_PLATFORMS = ["osx-64", "osx-arm64", "win-64", "linux-aarch64", "linux-64"]
```
- Ensure the target repos/branches contain these workflows:
  - llvmlite: `llvmdev_build.yml`, `llvmlite_conda_builder.yml`, `llvmlite_<platform>_wheel_builder.yml`
  - numba: `numba_<platform>_conda_builder.yml` (non-Windows), `numba_<platform>_builder.yml` (Windows conda), `numba_<platform>_wheel_builder.yml`

### Typical flows
- Everything (llvmdev, llvmlite conda+wheels, numba conda+wheels, then downloads):
```bash
python3 gha_runner.py \
  --llvmlite-branch <llvmlite_ref> \
  --numba-branch <numba_ref> \
  --steps all
```

- Conda-only:
```bash
python3 gha_runner.py \
  --llvmlite-branch <llvmlite_ref> \
  --numba-branch <numba_ref> \
  --steps llvmdev,llvmlite_conda,numba_conda,download_llvmlite_conda,download_numba_conda
```

- Include wheels too:
```bash
python3 gha_runner.py \
  --llvmlite-branch <llvmlite_ref> \
  --numba-branch <numba_ref> \
  --steps llvmdev,llvmlite_conda,llvmlite_wheels,numba_conda,numba_wheels,\
download_llvmlite_conda,download_llvmlite_wheels,download_numba_conda,download_numba_wheels
```

- Preview only:
```bash
python3 gha_runner.py --dry-run --steps all
```

### Step order and chaining
When included, steps run in this order:
1. `llvmdev`
2. `llvmlite_conda`
3. `llvmlite_wheels` (all platforms)
4. `numba_conda` (all platforms)
5. `numba_wheels` (all platforms)
6. `download_*` steps

Chaining of upstream artifacts:
- `llvmlite_conda` uses `llvmdev_run_id` (from `llvmdev` if successful)
- `llvmlite_wheels` use `llvmdev_run_id`
- `numba_conda` uses `llvmlite_run_id` (from `llvmlite_conda`)
- `numba_wheels` use per-platform `llvmlite_wheel_runid` (from `llvmlite_wheels`)

Notes:
- If a prerequisite run is not available/successful in state, inputs are omitted and the remote workflow must resolve artifacts (e.g., latest successful).

### Resume and reuse
- You can press Ctrl+C while watching; on the next run, the script resumes from `.ci_state.json` and reuses in-progress/successful runs.
- Seed state with an existing run:
```bash
python3 gha_runner.py --reuse-run llvmdev:123456789 --steps llvmdev, llvmlite_conda
```

### State file format (`.ci_state.json`) and manual edits
- The script stores minimal state per step key.
- Fields per step:
  - `run_id` (int)
  - `completed` (bool)
  - `conclusion` (string; e.g. `success`, `failure`, or empty when unknown)
  - `repo` (string; owner/repo)
  - `branch` (string; ref used when dispatching)

Example:
```json
{
  "llvmdev": {
    "run_id": 17118934013,
    "completed": false,
    "conclusion": "",
    "repo": "swap357/llvmlite",
    "branch": "main"
  },
  "llvmlite_wheels_linux-64": {
    "run_id": 17120001234,
    "completed": true,
    "conclusion": "success",
    "repo": "swap357/llvmlite",
    "branch": "main"
  },
  "numba_conda_osx-64": {
    "run_id": 17120009999,
    "completed": true,
    "conclusion": "failure",
    "repo": "swap357/numba",
    "branch": "main"
  }
}
```

Manual reuse via file edit:
- Identify the correct step key:
  - `llvmdev`
  - `llvmlite_conda`
  - `llvmlite_wheels_<platform>`
  - `numba_conda_<platform>`
  - `numba_wheels_<platform>`
- Put the known GitHub Actions run ID under `run_id`.
- Keep `repo` and `branch` aligned with your current invocation; mismatches are ignored for reuse.
- If you want the script to treat it as already successful (skip dispatch/watch): set `completed` to `true` and `conclusion` to `success`.
- Save the file and rerun the script with your desired `--steps`.

### Downloads only
If runs are already successful in state:
```bash
python3 gha_runner.py --steps download_llvmlite_conda,download_llvmlite_wheels,download_numba_conda,download_numba_wheels
```
Artifacts go under `artifacts/`.

### Tips
- Use `--state-path` to isolate experiments; `--reset-state` to clear.
- Start with `--dry-run` to verify the intended `gh` commands.

### Options and defaults
- Required options: none (sensible defaults are provided). You typically only set branches and steps.
- Common flags:
  - `-L, --llvmlite-branch` (default: `main`)
  - `-N, --numba-branch` (default: `main`)
  - `-s, --steps` (default: `all`)
  - `-p, --platforms` (default: `osx-64,osx-arm64,win-64,linux-aarch64,linux-64`)
  - `-f, --state-path` (default: `.ci_state.json` in repo root)
  - `-o, --artifacts-dir` (default: `artifacts/`)
  - `-d, --dry-run` to preview without executing
  - `-r, --reuse-run STEP:RUN_ID` to seed state
  - `-R, --reset-state` to clear state and exit

