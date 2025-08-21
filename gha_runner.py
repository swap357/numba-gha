#!/usr/bin/env python3
"""
gha_runner.py

Minimal, readable orchestrator for CI builds (conda + wheels) for llvmlite & numba.

Usage example:
  export GH_TOKEN=<your_token>
  python3 gha_runner.py \
    --llvmlite-branch pr-1240-llvmlite \
    --numba-branch   pr-1240-numba \
    --steps llvmdev,llvmlite_conda,llvmlite_wheels,numba_conda,numba_wheels,download_llvmlite_conda,download_llvmlite_wheels,download_numba_conda,download_numba_wheels \
    --reuse-run llvmdev:12345

Highlights:
- Class-based design (GhClient, StateStore, GHARunner) for testability.
- Single dispatch/wait/download pipeline with tiny, declarative Step config.
- Atomic state writes, jittered retries, robust run selection after dispatch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ------------------------------- Constants --------------------------------- #

LLVMLITE_REPO = "swap357/llvmlite"
NUMBA_REPO    = "swap357/numba"
DEFAULT_BRANCH = "main"
DEFAULT_PLATFORMS = ["osx-64", "osx-arm64", "win-64", "linux-aarch64", "linux-64"]

# ------------------------------- Utilities --------------------------------- #

def iso8601_to_aware(ts: str) -> datetime:
    """GitHub returns 'Z' suffix; normalize for fromisoformat."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

# ------------------------------- gh client --------------------------------- #

class GhClient:
    """Thin wrapper over `gh` with retries, host awareness, and helpers."""
    def __init__(self, dry_run: bool = False, host: Optional[str] = None, log_cmds: bool = True):
        self.dry_run = dry_run
        self.host = host or os.getenv("GH_HOST", "github.com")
        self.log_cmds = log_cmds

    # ---- Low-level runners ---- #

    def _run(self, args: Sequence[str]) -> None:
        if self.dry_run:
            self._print_cmd(args)
            return
        self._retry(lambda: subprocess.run(args, check=True), args)

    def _check_output(self, args: Sequence[str]) -> str:
        if self.dry_run:
            self._print_cmd(args)
            return ""
        out = []
        def _call():
            out.append(subprocess.check_output(args, text=True))
        self._retry(_call, args)
        return out[-1] if out else ""

    def _retry(self, fn, args: Sequence[str], attempts: int = 4) -> None:
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                fn()
                return
            except (subprocess.CalledProcessError, OSError) as exc:
                last_exc = exc
                base = min(2 ** attempt, 8)
                pause = random.uniform(0.0, base)
                logging.warning("Command failed (%s). Retrying in %.2fs: %s",
                                type(exc).__name__, pause, shlex.join(args))
                time.sleep(pause)
        assert last_exc is not None
        raise last_exc

    def _print_cmd(self, args: Sequence[str]) -> None:
        if self.log_cmds:
            print(shlex.join(args))

    # ---- Auth & Host ---- #

    def is_authenticated(self) -> bool:
        try:
            subprocess.run(["gh", "auth", "status"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (subprocess.CalledProcessError, OSError):
            return False

    # ---- High-level helpers ---- #

    def dispatch_workflow(self, repo: str, workflow: str, ref: str, inputs: Mapping[str, str]) -> Optional[int]:
        """Trigger a workflow. Returns run_id if stdout contains it; else None (fallback to polling)."""
        args = ["gh", "workflow", "run", workflow, "--repo", repo, "--ref", ref]
        for k, v in inputs.items():
            args += ["-f", f"{k}={v}"]
        out = self._check_output(args)
        m = re.search(r"/actions/runs/(\d+)", out or "")
        return int(m.group(1)) if m else None

    def list_runs(self, repo: str, workflow: str, branch: str, limit: int = 50) -> List[Dict[str, Any]]:
        args = [
            "gh", "run", "list",
            "--repo", repo,
            "--workflow", workflow,
            "--branch", branch,
            "--limit", str(limit),
            "--json", "databaseId,headBranch,workflowName,status,createdAt"
        ]
        out = self._check_output(args)
        try:
            return json.loads(out or "[]")
        except json.JSONDecodeError:
            return []

    def watch_run(self, repo: str, run_id: int) -> None:
        self._run(["gh", "run", "watch", str(run_id), "--repo", repo])

    def get_run_conclusion(self, repo: str, run_id: int) -> str:
        out = self._check_output([
            "gh", "run", "view", str(run_id),
            "--repo", repo,
            "--json", "conclusion"
        ])
        data = json.loads(out or "{}")
        return data.get("conclusion", "")

    def download_artifacts(self, repo: str, run_id: int, dest: Path) -> None:
        args = ["gh", "run", "download", str(run_id), "--repo", repo, "--dir", str(dest)]
        self._run(args)

# ------------------------------- State IO ---------------------------------- #

@dataclass
class StateEntry:
    run_id: Optional[int] = None
    completed: bool = False
    conclusion: str = ""
    repo: Optional[str] = None
    branch: Optional[str] = None

@dataclass
class StateStore:
    path: Path
    data: Dict[str, StateEntry] = field(default_factory=dict)

    def load(self) -> None:
        if self.path.exists():
            logging.info("Reading state from %s", self.path)
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.data = {k: StateEntry(**v) for k, v in raw.items()}
        else:
            logging.info("No state file at %s; starting clean", self.path)

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({k: vars(v) for k, v in self.data.items()}, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        logging.info("Saved state to %s", self.path)

    def get(self, key: str) -> StateEntry:
        return self.data.setdefault(key, StateEntry())

    def set(self, key: str, entry: StateEntry) -> None:
        self.data[key] = entry
        self.save()

# ------------------------------- Steps ------------------------------------- #

@dataclass(frozen=True)
class Step:
    """Declarative description for a single CI step (may expand per-platform)."""
    key: str
    repo: str
    workflow: str             # may contain "{platform}"
    needs_platforms: bool = False
    downloads: bool = False   # if True, this is a download step (no dispatch)
    inputs_from: Optional[str] = None  # read run_id from another step's state and pass as input
    input_name: Optional[str] = None   # the workflow input name to set with the upstream run_id
    branch_ref: str = DEFAULT_BRANCH

# ------------------------------- Runner ------------------------------------ #

class GHARunner:
    def __init__(
        self,
        gh: GhClient,
        state: StateStore,
        platforms: Sequence[str],
        artifacts_dir: Path,
        continue_on_failure: bool = False,
        dry_run: bool = False,
    ):
        self.gh = gh
        self.state = state
        self.platforms = list(platforms)
        self.artifacts_dir = artifacts_dir
        self.continue_on_failure = continue_on_failure
        self.dry_run = dry_run

    # ---- Core lifecycle ---- #

    def run(self, steps: Iterable[Step]) -> int:
        failures: List[str] = []
        for step in steps:
            try:
                if step.downloads:
                    if self.dry_run:
                        # Skip download steps entirely in dry-run
                        continue
                    self._download(step)
                else:
                    self._dispatch_and_wait(step)
            except SystemExit as e:
                # preserve exit code but optionally continue
                failures.append(f"{step.key} (exit={e.code})")
                if not self.continue_on_failure:
                    return int(e.code or 1)
            except KeyboardInterrupt:
                logging.info("Interrupted; aborting.")
                return 130
            except Exception as exc:
                logging.exception("Unhandled error in step %s: %s", step.key, exc)
                if not self.continue_on_failure:
                    return 1
                failures.append(f"{step.key} (exception)")
        if failures:
            logging.error("Some steps failed: %s", ", ".join(failures))
            return 1
        return 0

    # ---- Actions ---- #

    def _dispatch_and_wait(self, step: Step) -> None:
        # Prepare workflow inputs (optional upstream run_id)
        inputs: Dict[str, str] = {}
        if step.inputs_from and step.input_name:
            upstream = self.state.get(step.inputs_from)
            if upstream.run_id and upstream.conclusion == "success":
                inputs[step.input_name] = str(upstream.run_id)
                logging.info("Using upstream %s run %s for %s", step.inputs_from, upstream.run_id, step.key)

        if step.needs_platforms:
            for plat in self.platforms:
                self._dispatch_and_wait_one(step, plat, inputs)
        else:
            self._dispatch_and_wait_one(step, None, inputs)

    def _dispatch_and_wait_one(self, step: Step, platform: Optional[str], base_inputs: Mapping[str, str]) -> None:
        key = step.key if not platform else f"{step.key}_{platform}"
        entry = self.state.get(key)
        branch = step.branch_ref

        # Respect repo/branch scoping if present in state; missing fields are treated as compatible
        scope_ok = (not entry.repo or entry.repo == step.repo) and (not entry.branch or entry.branch == branch)

        if entry.run_id and entry.completed and entry.conclusion == "success" and scope_ok:
            logging.info("Reusing successful %s run %s", key, entry.run_id)
            return

        inputs = dict(base_inputs)
        # If this step consumes an upstream run_id and we are fanned-out per-platform,
        # try to source the platform-aligned upstream entry (e.g., llvmlite_wheels_{platform}).
        if platform and step.inputs_from and step.input_name:
            upstream_key = f"{step.inputs_from}_{platform}"
            upstream_entry = self.state.get(upstream_key)
            if upstream_entry.run_id and upstream_entry.conclusion == "success":
                inputs[step.input_name] = str(upstream_entry.run_id)

        # Resolve workflow filename directly using provided platform
        workflow_template = step.workflow
        workflow = (
            workflow_template.format(platform=platform)
            if platform
            else workflow_template
        )

        # Dry-run: print the dispatch command and return without polling/watching
        if self.dry_run:
            # Ensure the dispatch command is shown
            _ = self.gh.dispatch_workflow(step.repo, workflow, branch, inputs)
            return

        # Dispatch (or reuse in-progress)
        if entry.run_id and not entry.completed and scope_ok:
            run_id = entry.run_id
            logging.info("Resuming in-progress %s run %s", key, run_id)
        else:
            pre = datetime.now(timezone.utc)
            run_id = self.gh.dispatch_workflow(step.repo, workflow, branch, inputs) or \
                     self._find_new_run_id(step.repo, workflow, branch, since=pre)
            self.state.set(key, StateEntry(run_id=run_id, completed=False, conclusion="", repo=step.repo, branch=branch))
            logging.info("Recorded new %s run %s", key, run_id)

        # Wait
        try:
            self.gh.watch_run(step.repo, int(run_id))
        except KeyboardInterrupt:
            logging.info("Interrupted during watch of %s; aborting.", key)
            sys.exit(130)

        concl = self.gh.get_run_conclusion(step.repo, int(run_id))
        if concl != "success":
            logging.error("Run %s for %s ended '%s'. See: https://%s/%s/actions/runs/%s",
                          run_id, key, concl, self.gh.host, step.repo, run_id)
            self.state.set(key, StateEntry(run_id=run_id, completed=True, conclusion=concl, repo=step.repo, branch=branch))
            sys.exit(1)

        self.state.set(key, StateEntry(run_id=run_id, completed=True, conclusion=concl, repo=step.repo, branch=branch))

    def _download(self, step: Step) -> None:
        """Download artifacts for either a single step key or fan-out per platform."""
        keys = [step.key] if not step.needs_platforms else [f"{step.key}_{p}" for p in self.platforms]
        for key in keys:
            entry = self.state.get(key)
            if not entry.run_id or entry.conclusion != "success":
                logging.error("Cannot download %s: no successful run recorded.", key)
                if not self.continue_on_failure:
                    sys.exit(1)
                continue
            dest = self.artifacts_dir / key
            dest.mkdir(parents=True, exist_ok=True)
            expected_repo = self._repo_for_key(step, key)
            if entry.repo and entry.repo != expected_repo:
                logging.error("Recorded repo %s for %s does not match expected %s; skipping download.", entry.repo, key, expected_repo)
                if not self.continue_on_failure:
                    sys.exit(1)
                continue
            self.gh.download_artifacts(expected_repo, int(entry.run_id), dest)
            # best-effort check
            if not any(dest.iterdir()):
                logging.warning("No artifacts found for %s (run %s).", key, entry.run_id)

    # ---- Helpers ---- #

    def _find_new_run_id(self, repo: str, workflow: str, branch: str, since: datetime, timeout_s: int = 60) -> int:
        """Deterministically locate the run created after we dispatched."""
        statuses = {"queued", "in_progress", "waiting"}
        end = time.time() + timeout_s
        best: Optional[int] = None

        while time.time() < end:
            runs = self.gh.list_runs(repo, workflow, branch)
            candidates: List[tuple[datetime, int]] = []
            for r in runs:
                try:
                    created = iso8601_to_aware(r.get("createdAt", ""))
                except Exception:
                    continue
                if r.get("status") in statuses and r.get("headBranch") == branch and created >= since:
                    candidates.append((created, int(r.get("databaseId"))))
            if candidates:
                candidates.sort(key=lambda x: x[0])  # earliest created after dispatch
                best = candidates[0][1]
                break
            time.sleep(2)

        if best is None:
            logging.error("Unable to locate new run for %s on %s.", workflow, repo)
            sys.exit(1)
        return best

    @staticmethod
    def _repo_for_key(step: Step, key: str) -> str:
        # Download steps re-use the step.repo directly
        return step.repo


# ------------------------------- CLI --------------------------------------- #

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage CI workflows for llvmlite & numba (conda and wheels)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Requires GH_TOKEN environment variable or prior 'gh auth login'."
    )
    parser.add_argument("-L", "--llvmlite-branch", default=DEFAULT_BRANCH, metavar="REF",
                        help="Ref for llvmlite workflows")
    parser.add_argument("-N", "--numba-branch",    default=DEFAULT_BRANCH, metavar="REF",
                        help="Ref for numba workflows")
    parser.add_argument("-s", "--steps", default="all", metavar="STEPS",
                        help="Comma-separated steps (llvmdev,llvmlite_conda,llvmlite_wheels,numba_conda,numba_wheels,download_llvmdev,download_llvmlite_conda,download_llvmlite_wheels,download_numba_conda,download_numba_wheels,all)")
    parser.add_argument("-p", "--platforms", default=",".join(DEFAULT_PLATFORMS), metavar="PLATS",
                        help="Comma-separated platforms")
    parser.add_argument("-f", "--state-path", default=str((Path(__file__).parent / ".ci_state.json").resolve()),
                        metavar="FILE")
    parser.add_argument("-o", "--artifacts-dir", default=str((Path("artifacts")).resolve()), metavar="DIR")
    parser.add_argument("-r", "--reuse-run", action="append", metavar="STEP:RUN_ID",
                        help="STEP:RUN_ID to seed state")
    parser.add_argument("-R", "--reset-state", action="store_true", help="Delete state and exit")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Print gh commands instead of executing")
    parser.add_argument("-c", "--continue-on-failure", action="store_true", help="Keep going after a failure")
    parser.add_argument("-v", "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "ERROR"],
                        help="Logging level")
    return parser.parse_args(argv)


def resolve_requested_steps(spec: str, all_steps: Mapping[str, Step]) -> List[Step]:
    raw = [s.strip() for s in spec.split(",") if s.strip()]
    if not raw or "all" in raw:
        return list(all_steps.values())
    unknown = [s for s in raw if s not in all_steps]
    if unknown:
        logging.warning("Ignoring unknown steps: %s. Valid: %s", ", ".join(unknown), ", ".join(all_steps))
    return [all_steps[s] for s in raw if s in all_steps]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    # Resolve paths and flags
    state_path = Path(args.state_path).expanduser().resolve()
    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]

    # Pre-flight: reset?
    if args.reset_state:
        if state_path.exists():
            try:
                state_path.unlink()
                logging.info("Removed state file: %s", state_path)
            except OSError as exc:
                logging.error("Failed to remove state file: %s", exc)
                return 1
        else:
            logging.info("No state file found at %s; nothing to reset", state_path)
        return 0

    gh = GhClient(dry_run=args.dry_run)
    if not args.dry_run and not os.getenv("GH_TOKEN") and not gh.is_authenticated():
        logging.error("Authentication required: set GH_TOKEN or run 'gh auth login'.")
        return 1
    logging.info("Using GitHub host: %s", gh.host)

    # Build declarative step set
    steps: Dict[str, Step] = {
        # Build llvmdev on llvmlite repo
        "llvmdev": Step(
            key="llvmdev", repo=LLVMLITE_REPO, workflow="llvmdev_build.yml",
            needs_platforms=False, downloads=False, branch_ref=args.llvmlite_branch
        ),
        # Build llvmlite conda (optionally consumes llvmdev run_id)
        "llvmlite_conda": Step(
            key="llvmlite_conda", repo=LLVMLITE_REPO, workflow="llvmlite_conda_builder.yml",
            needs_platforms=False, downloads=False, inputs_from="llvmdev", input_name="llvmdev_run_id",
            branch_ref=args.llvmlite_branch
        ),
        # Build llvmlite wheels (per-platform; optionally consumes llvmdev run_id)
        "llvmlite_wheels": Step(
            key="llvmlite_wheels", repo=LLVMLITE_REPO, workflow="llvmlite_{platform}_wheel_builder.yml",
            needs_platforms=True, downloads=False, inputs_from="llvmdev", input_name="llvmdev_run_id",
            branch_ref=args.llvmlite_branch
        ),
        # Build numba conda (per-platform; consumes llvmlite_conda run_id)
        "numba_conda": Step(
            key="numba_conda", repo=NUMBA_REPO, workflow="numba_{platform}_conda_builder.yml",
            needs_platforms=True, downloads=False, inputs_from="llvmlite_conda", input_name="llvmlite_run_id",
            branch_ref=args.numba_branch
        ),
        # Build numba wheels (per-platform)
        "numba_wheels": Step(
            key="numba_wheels", repo=NUMBA_REPO, workflow="numba_{platform}_wheel_builder.yml",
            needs_platforms=True, downloads=False, inputs_from="llvmlite_wheels", input_name="llvmlite_wheel_runid",
            branch_ref=args.numba_branch
        ),
        # Downloads
        "download_llvmdev": Step(
            key="llvmdev", repo=LLVMLITE_REPO, workflow="", downloads=True
        ),
        "download_llvmlite_conda": Step(
            key="llvmlite_conda", repo=LLVMLITE_REPO, workflow="", downloads=True
        ),
        "download_llvmlite_wheels": Step(
            key="llvmlite_wheels", repo=LLVMLITE_REPO, workflow="", downloads=True, needs_platforms=True
        ),
        "download_numba_conda": Step(
            key="numba_conda", repo=NUMBA_REPO, workflow="", downloads=True, needs_platforms=True
        ),
        "download_numba_wheels": Step(
            key="numba_wheels", repo=NUMBA_REPO, workflow="", downloads=True, needs_platforms=True
        ),
    }

    requested = resolve_requested_steps(args.steps, steps)

    # State
    store = StateStore(path=state_path)
    store.load()

    # Seed manual runs
    if args.reuse_run:
        for seed in args.reuse_run:
            try:
                seed_step, seed_id = seed.split(":")
                seed_id_i = int(seed_id)
                # Try to fetch actual conclusion to normalize
                repo = LLVMLITE_REPO if seed_step.startswith("llvmlite") or seed_step == "llvmdev" else NUMBA_REPO
                concl = gh.get_run_conclusion(repo, seed_id_i) if not args.dry_run else ""
                # Record scope with current branch for the relevant repo
                branch = args.llvmlite_branch if repo == LLVMLITE_REPO else args.numba_branch
                store.set(seed_step, StateEntry(run_id=seed_id_i, completed=bool(concl), conclusion=concl, repo=repo, branch=branch))
                logging.info("Seeded %s with run %s (conclusion=%s)", seed_step, seed_id, concl or "unknown")
            except ValueError:
                logging.error("Invalid --reuse-run format: %s (expected STEP:RUN_ID)", seed)
                return 1

    # Run
    runner = GHARunner(
        gh=gh,
        state=store,
        platforms=platforms,
        artifacts_dir=artifacts_dir,
        continue_on_failure=args.continue_on_failure,
        dry_run=args.dry_run,
    )
    return runner.run(requested)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.info("Interrupted by user; exiting.")
        sys.exit(130)
