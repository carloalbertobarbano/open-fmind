"""fMRIPrep run for one subject, through Singularity or Apptainer.

fMRIPrep runs without FreeSurfer surface reconstruction and writes the
preprocessed BOLD in MNI152NLin2009cAsym at 2 mm. The command, container image
and exit code are appended to {fmriprep_dir}/logs/preprocessing.jsonl, from
which the sidecars of the later steps take the fMRIPrep command line.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .download import raw_bold_files
from .parcellate import PREPROC_BOLD_SUFFIX
from .provenance import load_preproc_record

OUTPUT_SPACES = "MNI152NLin2009cAsym:res-2"
LICENSE_IN_CONTAINER = "/opt/freesurfer/license.txt"
TEMPLATEFLOW_IN_CONTAINER = "/templateflow"

# Flags passed to fMRIPrep besides the paths and resources.
FMRIPREP_FLAGS = [
    "--fs-no-reconall",
    f"--fs-license-file {LICENSE_IN_CONTAINER}",
    f"--output-spaces {OUTPUT_SPACES}",
    "--skip-bids-validation",
    "--notrack",
    "-v",
]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Runs on disk
# ---------------------------------------------------------------------------
def fmriprep_bold_files(fmriprep_dir: Path, dataset_id: str, subject: str) -> list[Path]:
    sub_dir = fmriprep_dir / dataset_id / f"sub-{subject}"
    return [
        p for pattern in (f"func/*{PREPROC_BOLD_SUFFIX}", f"ses-*/func/*{PREPROC_BOLD_SUFFIX}")
        for p in sub_dir.glob(pattern)
    ]


def fmriprep_done(fmriprep_dir: Path, dataset_id: str, subject: str) -> bool:
    """True if fMRIPrep finished for this subject.

    That is a success record in logs/preprocessing.jsonl and at least one
    preprocessed BOLD. A run interrupted midway leaves outputs but no success
    record, and is run again.
    """
    record = load_preproc_record(fmriprep_dir, dataset_id, subject)
    return bool(record and record.get("status") == "success"
                and fmriprep_bold_files(fmriprep_dir, dataset_id, subject))


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@dataclass
class FmriprepConfig:
    image: Path
    fs_license: Path
    templateflow_home: Path
    nprocs: int
    mem_mb: int


def fmriprep_version(image: Path) -> str | None:
    """Version from an image named like fmriprep-25.2.3.sif, if it is."""
    m = re.search(r"fmriprep[-_](\d+\.\d+\.\d+)", image.name)
    return m.group(1) if m else None


def build_command(runtime: str, config: FmriprepConfig, bids_dir: Path, out_dir: Path,
                  work_dir: Path, subject: str) -> list[str]:
    return [
        runtime, "run", "--cleanenv",
        "-B", f"{bids_dir}:{bids_dir}:ro",
        "-B", f"{out_dir}:{out_dir}",
        "-B", f"{work_dir}:{work_dir}",
        "-B", f"{config.templateflow_home}:{TEMPLATEFLOW_IN_CONTAINER}",
        "-B", f"{config.fs_license}:{LICENSE_IN_CONTAINER}:ro",
        str(config.image),
        str(bids_dir), str(out_dir), "participant",
        "--participant-label", subject,
        *[part for flag in FMRIPREP_FLAGS for part in flag.split()],
        "--nprocs", str(config.nprocs),
        "--mem-mb", str(config.mem_mb),
        "--work-dir", str(work_dir),
    ]


def container_env(runtime: str) -> dict:
    """Host environment plus the variables passed into the container.

    --cleanenv clears the host environment inside the container; variables
    prefixed with SINGULARITYENV_ / APPTAINERENV_ are passed in without the
    prefix.
    """
    env = dict(os.environ)
    prefix = "APPTAINERENV_" if runtime == "apptainer" else "SINGULARITYENV_"
    env[f"{prefix}TEMPLATEFLOW_HOME"] = TEMPLATEFLOW_IN_CONTAINER
    env[f"{prefix}MPLCONFIGDIR"] = "/tmp/matplotlib-cache"
    return env


def check_config(config: FmriprepConfig) -> None:
    if not config.image.is_file():
        raise RuntimeError(f"fMRIPrep image not found: {config.image}")
    if not config.fs_license.is_file():
        raise RuntimeError(f"FreeSurfer license not found: {config.fs_license}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_fmriprep(
    dataset_id: str,
    subject: str,
    raw_root: Path,
    fmriprep_dir: Path,
    work_root: Path,
    config: FmriprepConfig,
    dry_run: bool = False,
) -> None:
    """Run fMRIPrep on one subject into {fmriprep_dir}/{dataset_id}/.

    The work directory is removed when fMRIPrep succeeds and kept for
    debugging when it fails.
    """
    bids_dir = (raw_root / dataset_id).resolve()
    out_dir = fmriprep_dir / dataset_id
    work_dir = work_root / dataset_id / f"sub-{subject}"
    found = shutil.which("singularity") or shutil.which("apptainer")
    runtime = Path(found).name if found else "singularity"
    cmd = build_command(runtime, config, bids_dir, out_dir, work_dir, subject)
    log(f"  output:   {out_dir}\n  command:  {' '.join(cmd)}")
    if dry_run:
        return
    if not found:
        raise RuntimeError("Neither singularity nor apptainer was found on PATH.")
    if not raw_bold_files(bids_dir, subject):
        raise RuntimeError(f"No raw BOLD for sub-{subject} under {bids_dir}")
    check_config(config)
    env = container_env(runtime)

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    config.templateflow_home.mkdir(parents=True, exist_ok=True)

    # The raw data must be readable from inside the container.
    check = subprocess.run(
        [runtime, "exec", "-B", f"{bids_dir}:{bids_dir}:ro", str(config.image),
         "ls", str(bids_dir / f"sub-{subject}")],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(f"{bids_dir} is not readable inside the container: "
                           f"{check.stderr.strip()[:300]}")

    started = datetime.now(timezone.utc)
    exit_code = subprocess.run(cmd, env=env).returncode
    finished = datetime.now(timezone.utc)

    log_dir = fmriprep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset_id": dataset_id,
        "subject": subject,
        "status": "success" if exit_code == 0 else "failed",
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "exit_code": exit_code,
        "command": " ".join(cmd),
        "container_image": str(config.image),
        "fmriprep_version": fmriprep_version(config.image),
        "output_spaces": OUTPUT_SPACES,
        "slurm_job_id": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
    }
    with (log_dir / "preprocessing.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    if exit_code != 0:
        raise RuntimeError(f"fMRIPrep failed (exit {exit_code}); work directory kept: {work_dir}")
    shutil.rmtree(work_dir, ignore_errors=True)
    try:
        work_dir.parent.rmdir()  # only when no other subject of the dataset uses it
    except OSError:
        pass
    if not fmriprep_bold_files(fmriprep_dir, dataset_id, subject):
        raise RuntimeError("fMRIPrep finished but produced no res-2 preprocessed BOLD.")
