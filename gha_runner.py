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

LLVMLITE_REPO = "numba/llvmlite"
NUMBA_REPO    = "numba/numba"
DEFAULT_BRANCH = "main"
DEFAULT_PLATFORMS = ["win-64", "osx-64", "linux-64", "linux-aarch64", "osx-arm64"]

# ------------------------------- Utilities --------------------------------- #

def iso8601_to_aware(ts: str) -> datetime:
    """GitHub returns 'Z' suffix; normalize for fromisoformat."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

# ------------------------------- gh client --------------------------------- #

class GhClient:
    """Thin wrapper over `gh` with retries, host awareness, and helpers."""
    def __init__(self, host: Optional[str] = None, log_cmds: bool = True):
        self.host = host or os.getenv("GH_HOST", "github.com")
        self.log_cmds = log_cmds

    # ---- Low-level runners ---- #

    def _run(self, args: Sequence[str]) -> None:
        # Always log the gh command about to run
        self._print_cmd(args)
        self._retry(lambda: subprocess.run(args, check=True), args)

    def _check_output(self, args: Sequence[str]) -> str:
        # Always log the gh command about to run
        self._print_cmd(args)
        out: List[str] = []
        def _call() -> None:
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
            logging.info(shlex.join(args))

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
            "--json", "databaseId,headBranch,workflowName,status,conclusion,createdAt"
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

    def get_pr_head_branch(self, repo: str, pr_number: int) -> str:
        out = self._check_output([
            "gh", "pr", "view", str(pr_number),
            "--repo", repo,
            "--json", "headRefName"
        ])
        data = json.loads(out or "{}")
        branch = data.get("headRefName")
        if not branch:
            logging.error("Unable to resolve headRefName for PR #%s on %s", pr_number, repo)
            sys.exit(1)
        return branch

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
    ):
        self.gh = gh
        self.state = state
        self.platforms = list(platforms)
        self.artifacts_dir = artifacts_dir
        self.continue_on_failure = continue_on_failure
        self.from_pr: Optional[int] = None
        # Planning mode: when set, we only generate gh commands (no execution)
        self.plan_only: bool = False
        # Per-platform overrides for llvmlite wheel run IDs used by numba_wheels
        # e.g., {"linux-64": "17123456789", "osx-64": "17123456788"}
        self.override_wheel_run_ids_by_platform: Dict[str, str] = {}
        # Manual overrides for upstream run IDs keyed by upstream step key
        # e.g., {"llvmlite_conda": "17246956402"}
        self.override_input_run_ids: Dict[str, str] = {}
        # Dispatch-only mode: dispatch workflows but do not wait for completion
        self.dispatch_only: bool = False
        self.dispatch_delay_seconds: float = 10.0

    # ---- Core lifecycle ---- #

    def run(self, steps: Iterable[Step]) -> int:
        failures: List[str] = []
        for step in steps:
            try:
                if step.downloads:
                    if self.dry_run:
                        # Skip download steps entirely in dry-run
                        continue
                    self._download(step, from_pr=getattr(self, "from_pr", None))
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
            # Highest precedence: explicit override from CLI
            override_run_id = self.override_input_run_ids.get(step.inputs_from)
            if override_run_id:
                inputs[step.input_name] = override_run_id
                logging.info("Using override %s=%s for %s", step.input_name, override_run_id, step.key)
            else:
                upstream = self.state.get(step.inputs_from)
                if upstream.run_id and upstream.conclusion == "success":
                    inputs[step.input_name] = str(upstream.run_id)
                    logging.info("Using upstream %s run %s for %s", step.inputs_from, upstream.run_id, step.key)

        if step.needs_platforms:
            # Determine platform dispatch order
            platforms_order = list(self.platforms)
            if self.dispatch_only:
                preferred = ["win-64", "osx-64", "linux-64", "linux-aarch64", "osx-arm64"]
                ordered = [p for p in preferred if p in platforms_order]
                remaining = [p for p in platforms_order if p not in ordered]
                platforms_order = ordered + remaining
            for plat in platforms_order:
                self._dispatch_and_wait_one(step, plat, inputs)
                if self.dispatch_only:
                    time.sleep(self.dispatch_delay_seconds)
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
            # Highest precedence: per-platform override for llvmlite wheels → numba_wheels
            if step.inputs_from == "llvmlite_wheels":
                override_plat_run = self.override_wheel_run_ids_by_platform.get(platform)
                if override_plat_run:
                    inputs[step.input_name] = override_plat_run
                    logging.info("Using override %s=%s for %s (%s)", step.input_name, override_plat_run, step.key, platform)
                else:
                    upstream_key = f"{step.inputs_from}_{platform}"
                    upstream_entry = self.state.get(upstream_key)
                    if upstream_entry.run_id and upstream_entry.conclusion == "success":
                        inputs[step.input_name] = str(upstream_entry.run_id)
            else:
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

        # Plan-only: print the dispatch command and return without polling/watching
        if self.plan_only:
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

        # In dispatch-only mode, do not wait
        if not self.dispatch_only:
            try:
                self.gh.watch_run(step.repo, int(run_id))
            except KeyboardInterrupt:
                logging.info("Interrupted during watch of %s; aborting.", key)
                sys.exit(130)

        if not self.dispatch_only:
            concl = self.gh.get_run_conclusion(step.repo, int(run_id))
            if concl != "success":
                logging.error("Run %s for %s ended '%s'. See: https://%s/%s/actions/runs/%s",
                              run_id, key, concl, self.gh.host, step.repo, run_id)
                self.state.set(key, StateEntry(run_id=run_id, completed=True, conclusion=concl, repo=step.repo, branch=branch))
                sys.exit(1)

            self.state.set(key, StateEntry(run_id=run_id, completed=True, conclusion=concl, repo=step.repo, branch=branch))

    # ---- Planning helpers ---- #
    def build_commands_for_steps(self, steps: Iterable[Step], llvmlite_pr: Optional[int] = None) -> List[str]:
        """Return a list of gh CLI command strings that would be executed.

        - For dispatch steps, we include `gh workflow run ... -f key=value` inputs.
        - For download steps, we include `gh run download ... --dir ...` commands.
        - If llvmlite_pr is provided, we will attempt to resolve upstream run ids
          for inputs by inspecting the PR head branch for relevant workflows.
        """
        original_log_flag = self.gh.log_cmds
        cmds: List[str] = []

        try:
            # Capture printed commands by temporarily intercepting logging
            # Instead, build the commands directly to avoid parsing logs
            for step in steps:
                if step.downloads:
                    # Enumerate keys (single or per platform)
                    keys = [step.key] if not step.needs_platforms else [f"{step.key}_{p}" for p in self.platforms]
                    for key in keys:
                        # Determine run id: prefer state; else from --from-pr logic
                        entry = self.state.get(key)
                        expected_repo = step.repo
                        run_id: Optional[int] = None
                        if entry.run_id and entry.conclusion == "success":
                            run_id = entry.run_id
                        elif self.from_pr is not None:
                            # Infer workflow filename
                            platform = key.split("_")[-1] if "_" in key and step.needs_platforms else None
                            if step.key == "llvmlite_wheels" and platform:
                                workflow = f"llvmlite_{platform}_wheel_builder.yml"
                            elif step.key == "llvmlite_conda":
                                workflow = "llvmlite_conda_builder.yml"
                            elif step.key == "llvmdev":
                                workflow = "llvmdev_build.yml"
                            elif step.key == "numba_conda" and platform:
                                workflow = f"numba_{platform}_conda_builder.yml"
                            elif step.key == "numba_wheels" and platform:
                                workflow = f"numba_{platform}_wheel_builder.yml"
                            else:
                                workflow = ""

                            if workflow:
                                pr_branch = self.gh.get_pr_head_branch(expected_repo, int(self.from_pr))
                                runs = self.gh.list_runs(expected_repo, workflow, pr_branch)
                                sorted_runs = sorted(runs, key=lambda r: r.get("createdAt", ""), reverse=True)
                                for r in sorted_runs:
                                    if r.get("conclusion") == "success":
                                        run_id = int(r.get("databaseId"))
                                        break
                                if run_id is None:
                                    for r in sorted_runs:
                                        if r.get("status") == "completed":
                                            run_id = int(r.get("databaseId"))
                                            break
                        if run_id is None:
                            # Leave a placeholder so users see intent
                            run_id_str = "<RUN_ID>"
                        else:
                            run_id_str = str(run_id)
                        dest = str((self.artifacts_dir / key).resolve())
                        cmd = shlex.join(["gh", "run", "download", run_id_str, "--repo", expected_repo, "--dir", dest])
                        cmds.append(cmd)
                else:
                    # Dispatch steps: we may need to gather inputs
                    base_inputs: Dict[str, str] = {}
                    if step.inputs_from and step.input_name:
                        # 1) Explicit override takes precedence
                        override_run = self.override_input_run_ids.get(step.inputs_from)
                        if override_run:
                            base_inputs[step.input_name] = override_run
                        else:
                            # 2) Try state
                            upstream = self.state.get(step.inputs_from)
                            if upstream.run_id and upstream.conclusion == "success":
                                base_inputs[step.input_name] = str(upstream.run_id)
                            # 3) Optionally resolve from llvmlite PR (non-platform inputs only)
                            elif llvmlite_pr is not None and step.inputs_from in {"llvmdev", "llvmlite_conda"}:
                                wf = "llvmdev_build.yml" if step.inputs_from == "llvmdev" else "llvmlite_conda_builder.yml"
                                pr_branch = self.gh.get_pr_head_branch(LLVMLITE_REPO, int(llvmlite_pr))
                                runs = self.gh.list_runs(LLVMLITE_REPO, wf, pr_branch)
                                sorted_runs = sorted(runs, key=lambda r: r.get("createdAt", ""), reverse=True)
                                resolved: Optional[int] = None
                                for r in sorted_runs:
                                    if r.get("conclusion") == "success":
                                        resolved = int(r.get("databaseId"))
                                        break
                                if resolved is None:
                                    for r in sorted_runs:
                                        if r.get("status") == "completed":
                                            resolved = int(r.get("databaseId"))
                                            break
                                if resolved is not None:
                                    base_inputs[step.input_name] = str(resolved)
                        # For llvmlite_wheels, per-platform handling happens below

                    targets = self.platforms if step.needs_platforms else [None]
                    for plat in targets:
                        inputs = dict(base_inputs)
                        if plat and step.inputs_from == "llvmlite_wheels" and step.input_name and step.downloads is False:
                            # Precedence: per-platform override → state → llvmlite PR resolution
                            override_plat = self.override_wheel_run_ids_by_platform.get(plat)
                            if override_plat:
                                inputs[step.input_name] = override_plat
                            else:
                                upstream_key = f"llvmlite_wheels_{plat}"
                                upstream_entry = self.state.get(upstream_key)
                                if upstream_entry.run_id and upstream_entry.conclusion == "success":
                                    inputs[step.input_name] = str(upstream_entry.run_id)
                                elif llvmlite_pr is not None:
                                    wf = f"llvmlite_{plat}_wheel_builder.yml"
                                    pr_branch = self.gh.get_pr_head_branch(LLVMLITE_REPO, int(llvmlite_pr))
                                    runs = self.gh.list_runs(LLVMLITE_REPO, wf, pr_branch)
                                    sorted_runs = sorted(runs, key=lambda r: r.get("createdAt", ""), reverse=True)
                                    for r in sorted_runs:
                                        if r.get("conclusion") == "success":
                                            inputs[step.input_name] = str(int(r.get("databaseId")))
                                            break

                        workflow_template = step.workflow
                        workflow = workflow_template.format(platform=plat) if plat else workflow_template
                        args: List[str] = ["gh", "workflow", "run", workflow, "--repo", step.repo, "--ref", step.branch_ref]
                        for k, v in inputs.items():
                            args += ["-f", f"{k}={v}"]
                        cmds.append(shlex.join(args))

        finally:
            self.gh.log_cmds = original_log_flag

        return cmds

    def _download(self, step: Step, from_pr: Optional[int] = None) -> None:
        """Download artifacts for either a single step key or fan-out per platform."""
        keys = [step.key] if not step.needs_platforms else [f"{step.key}_{p}" for p in self.platforms]
        for key in keys:
            entry = self.state.get(key)
            if (not entry.run_id or entry.conclusion != "success") and from_pr is None:
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
            # Resolve run_id from PR if requested and missing/unsuccessful
            run_id_to_download = entry.run_id if entry.run_id and entry.conclusion == "success" else None
            if from_pr is not None and run_id_to_download is None:
                # Derive workflow filename for download steps (step.workflow is empty by design)
                platform = key.split("_")[-1] if "_" in key and step.needs_platforms else None
                if step.key == "llvmlite_wheels" and platform:
                    workflow = f"llvmlite_{platform}_wheel_builder.yml"
                elif step.key == "llvmlite_conda":
                    workflow = "llvmlite_conda_builder.yml"
                elif step.key == "llvmdev":
                    workflow = "llvmdev_build.yml"
                elif step.key == "numba_conda" and platform:
                    workflow = f"numba_{platform}_conda_builder.yml"
                elif step.key == "numba_wheels" and platform:
                    workflow = f"numba_{platform}_wheel_builder.yml"
                else:
                    workflow = ""

                if not workflow:
                    logging.error("Cannot infer workflow filename for %s (platform=%s)", key, platform or "-")
                    if not self.continue_on_failure:
                        sys.exit(1)
                    continue

                # Find PR head branch and select latest success
                pr_branch = self.gh.get_pr_head_branch(expected_repo, int(from_pr))
                runs = self.gh.list_runs(expected_repo, workflow, pr_branch)
                # prefer latest success else latest completed
                def _sort_key(r: Dict[str, Any]) -> str:
                    return r.get("createdAt", "")
                sorted_runs = sorted(runs, key=_sort_key, reverse=True)
                for r in sorted_runs:
                    if r.get("conclusion") == "success":
                        run_id_to_download = int(r.get("databaseId"))
                        break
                if run_id_to_download is None:
                    for r in sorted_runs:
                        if r.get("status") == "completed":
                            run_id_to_download = int(r.get("databaseId"))
                            break
            if run_id_to_download is None:
                logging.error("No suitable run found for %s", key)
                if not self.continue_on_failure:
                    sys.exit(1)
                continue
            self.gh.download_artifacts(expected_repo, int(run_id_to_download), dest)
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
                ts = r.get("createdAt", "")
                if not ts:
                    continue
                try:
                    created = iso8601_to_aware(ts)
                except ValueError:
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
    parser.add_argument("-pr", "--from-pr", type=int, metavar="PR",
                        help="Resolve and use the latest successful runs from this PR's head branch for download_* steps")
    parser.add_argument("--print-commands", action="store_true",
                        help="Print the gh CLI commands that would be executed for the requested steps, then exit")
    # Deprecated/advanced flags (hidden to keep CLI concise)
    parser.add_argument("--llvmlite-pr", type=int, metavar="PR", help=argparse.SUPPRESS)
    parser.add_argument("--llvmlite-conda-run-id", type=int, metavar="RUN_ID", help=argparse.SUPPRESS)
    parser.add_argument("--llvmlite-wheel-run-ids", metavar="PLAT=RUN_ID[,PLAT=RUN_ID]", help=argparse.SUPPRESS)
    parser.add_argument("-u", "--use-last-run", action="store_true",
                        help="Resolve latest successful run IDs from the target branch (or PR head with -pr) and use them (writes to state for downloads; injects inputs for dependent steps)")
    # Back-compat alias (hidden)
    parser.add_argument("--auto-populate", action="store_true", dest="use_last_run", help=argparse.SUPPRESS)
    parser.add_argument("-R", "--reset-state", action="store_true", help="Delete state and exit")
    parser.add_argument("-c", "--continue-on-failure", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-v", "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARN", "ERROR"],
                        help="Logging level")
    parser.add_argument("--dispatch-all", action="store_true",
                        help="Dispatch workflows without waiting for completion (stacks per-platform with a small delay)")
    parser.add_argument("--dispatch-delay-seconds", type=float, default=2.0,
                        help="Delay between dispatches in dispatch-all mode")
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

    gh = GhClient()
    if not (args.print_commands) and not os.getenv("GH_TOKEN") and not gh.is_authenticated():
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

    # Run or plan-only output
    runner = GHARunner(
        gh=gh,
        state=store,
        platforms=platforms,
        artifacts_dir=artifacts_dir,
        continue_on_failure=args.continue_on_failure,
    )
    runner.dispatch_only = bool(args.dispatch_all)
    runner.dispatch_delay_seconds = float(args.dispatch_delay_seconds)
    # Wire --from-pr for download steps (store for later use)
    if args.from_pr is not None:
        # Validate only if user also requested download_* steps population
        non_download = [s.key for s in requested if not s.downloads]
        if non_download and args.use_last_run:
            logging.error("--from-pr can only be used with download_* steps when --auto-populate is set. Non-download: %s", ", ".join(non_download))
            return 1
        runner.from_pr = int(args.from_pr)

    # Use last runs: resolve latest successful run ids using PR head branch if provided, else branch refs
    if args.use_last_run:
        for step in requested:
            # Populate download_* step run_ids directly into state
            if step.downloads:
                keys = [step.key] if not step.needs_platforms else [f"{step.key}_{p}" for p in platforms]
                for key in keys:
                    entry = store.get(key)
                    if entry.run_id and entry.conclusion == "success":
                        continue
                    expected_repo = step.repo
                    platform = key.split("_")[-1] if "_" in key and step.needs_platforms else None
                    if step.key == "llvmlite_wheels" and platform:
                        workflow = f"llvmlite_{platform}_wheel_builder.yml"
                    elif step.key == "llvmlite_conda":
                        workflow = "llvmlite_conda_builder.yml"
                    elif step.key == "llvmdev":
                        workflow = "llvmdev_build.yml"
                    elif step.key == "numba_conda" and platform:
                        workflow = f"numba_{platform}_conda_builder.yml"
                    elif step.key == "numba_wheels" and platform:
                        workflow = f"numba_{platform}_wheel_builder.yml"
                    else:
                        workflow = ""
                    if not workflow:
                        continue
                    # Choose branch: PR head branch if provided, else the configured ref for the repo
                    if args.from_pr is not None:
                        branch_ref = gh.get_pr_head_branch(expected_repo, int(args.from_pr))
                    else:
                        branch_ref = args.llvmlite_branch if expected_repo == LLVMLITE_REPO else args.numba_branch
                    runs = gh.list_runs(expected_repo, workflow, branch_ref)
                    sorted_runs = sorted(runs, key=lambda r: r.get("createdAt", ""), reverse=True)
                    resolved: Optional[int] = None
                    for r in sorted_runs:
                        if r.get("conclusion") == "success":
                            resolved = int(r.get("databaseId"))
                            break
                    if resolved is None:
                        for r in sorted_runs:
                            if r.get("status") == "completed":
                                resolved = int(r.get("databaseId"))
                                break
                    if resolved is not None:
                        store.set(key, StateEntry(run_id=resolved, completed=True, conclusion="success", repo=expected_repo, branch=branch_ref))
            # Populate upstream inputs for dispatch steps (e.g., numba_conda and numba_wheels)
            else:
                if step.key == "numba_conda":
                    # Resolve llvmlite_conda run id on llvmlite branch/PR
                    branch_ref = gh.get_pr_head_branch(LLVMLITE_REPO, int(args.from_pr)) if args.from_pr is not None else args.llvmlite_branch
                    wf = "llvmlite_conda_builder.yml"
                    runs = gh.list_runs(LLVMLITE_REPO, wf, branch_ref)
                    sorted_runs = sorted(runs, key=lambda r: r.get("createdAt", ""), reverse=True)
                    resolved: Optional[int] = None
                    for r in sorted_runs:
                        if r.get("conclusion") == "success":
                            resolved = int(r.get("databaseId"))
                            break
                    if resolved is None:
                        for r in sorted_runs:
                            if r.get("status") == "completed":
                                resolved = int(r.get("databaseId"))
                                break
                    if resolved is not None:
                        runner.override_input_run_ids["llvmlite_conda"] = str(resolved)
                elif step.key == "numba_wheels":
                    # Resolve llvmlite wheel run ids per platform
                    branch_ref = gh.get_pr_head_branch(LLVMLITE_REPO, int(args.from_pr)) if args.from_pr is not None else args.llvmlite_branch
                    for plat in platforms:
                        wf = f"llvmlite_{plat}_wheel_builder.yml"
                        runs = gh.list_runs(LLVMLITE_REPO, wf, branch_ref)
                        sorted_runs = sorted(runs, key=lambda r: r.get("createdAt", ""), reverse=True)
                        resolved: Optional[int] = None
                        for r in sorted_runs:
                            if r.get("conclusion") == "success":
                                resolved = int(r.get("databaseId"))
                                break
                        if resolved is None:
                            for r in sorted_runs:
                                if r.get("status") == "completed":
                                    resolved = int(r.get("databaseId"))
                                    break
                        if resolved is not None:
                            runner.override_wheel_run_ids_by_platform[plat] = str(resolved)
    # Wire explicit override for llvmlite conda run id
    if args.llvmlite_conda_run_id is not None:
        runner.override_input_run_ids["llvmlite_conda"] = str(int(args.llvmlite_conda_run_id))
    if args.print_commands:
        runner.plan_only = True
        cmds = runner.build_commands_for_steps(requested, llvmlite_pr=args.llvmlite_pr)
        for c in cmds:
            print(c)
        return 0
    # Parse per-platform wheel run id overrides
    if args.llvmlite_wheel_run_ids:
        try:
            for item in args.llvmlite_wheel_run_ids.split(","):
                if not item.strip():
                    continue
                plat, runid = item.split("=", 1)
                runner.override_wheel_run_ids_by_platform[plat.strip()] = runid.strip()
        except ValueError:
            logging.error("Invalid --llvmlite-wheel-run-ids format. Expected PLAT=RUN_ID[,PLAT=RUN_ID]")
            return 1
    return runner.run(requested)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.info("Interrupted by user; exiting.")
        sys.exit(130)
