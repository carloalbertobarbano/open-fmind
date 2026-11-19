"""BIDS provenance for the open-fmind outputs.

Assembles the ``GeneratedBy`` chain that traces every output back to the raw
OpenNeuro data and to the commands that produced it:

    openneuro-py download -> fMRIPrep -> open-fmind parcellate -> open-fmind denoise
    (-> open-fmind connectivity)

Each stage writes its own record; later stages read the earlier records off
disk and carry the whole chain into their sidecars. When a record is missing,
the entry is written with null fields and a ``Note`` saying what could not be
determined.
"""
from __future__ import annotations

import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CODE_URL = "https://github.com/carloalbertobarbano/open-fmind"

OPENNEURO_DATASET_URL = "https://openneuro.org/datasets/{dataset_id}"

# Names used in dataset_description.json "DatasetLinks", and therefore the
# prefixes of the "bids:<link>:<path>" URIs written into "Sources".
LINK_RAW = "raw"
LINK_FMRIPREP = "fmriprep"
LINK_PARCELLATED = "parcellated"


# ---------------------------------------------------------------------------
# Environment identity
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def git_revision(repo_dir: Path = PACKAGE_DIR) -> dict | None:
    """Commit SHA of the checkout holding this package and whether it was dirty.

    None when the package is not installed from a git checkout.
    """
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    return {"CommitSHA": sha, "Dirty": bool(status)}


@lru_cache(maxsize=32)
def package_version(name: str) -> str | None:
    """An installed package's version, or None if not installed."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------
# BIDS URIs
# ---------------------------------------------------------------------------
def bids_uri(link: str, relative_path: str | Path) -> str:
    """BIDS URI ("bids:<link>:<path>") for a Sources entry."""
    return f"bids:{link}:{Path(relative_path).as_posix()}"


def relative_to_dataset(path: Path, dataset_root: Path) -> str:
    """Path of `path` relative to its dataset root, as a POSIX string.

    Falls back to the bare filename when `path` lies outside `dataset_root`.
    """
    try:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _read_jsonl(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return records


# ---------------------------------------------------------------------------
# Stage 1: download
# ---------------------------------------------------------------------------
def load_download_record(log_dir: Path, dataset_id: str,
                         subject: str | None = None) -> dict | None:
    """Last successful download record of the dataset in {log_dir}/download*.jsonl.

    A record of `subject` is preferred over one of another subject.
    """
    records = [
        r for r in _read_jsonl(sorted(log_dir.glob("download*.jsonl")))
        if r.get("dataset_id") == dataset_id and r.get("status") == "success"
    ]
    own = [r for r in records if subject is not None and r.get("subject") == subject]
    return (own or records or [None])[-1]


def download_generated_by(dataset_id: str, log_dir: Path,
                          subject: str | None = None) -> dict:
    """GeneratedBy entry for the openneuro-py download of `dataset_id`."""
    entry: dict = {
        "Name": "openneuro-py",
        "Version": None,
        "CodeURL": "https://github.com/hoechenberger/openneuro-py",
        "Description": f"Download of OpenNeuro {dataset_id}.",
        "SourceURL": OPENNEURO_DATASET_URL.format(dataset_id=dataset_id),
        "SnapshotTag": None,
        "Parameters": {},
    }

    record = load_download_record(log_dir, dataset_id, subject)
    if record is None:
        entry["Note"] = f"No successful download record found in {log_dir}/download*.jsonl."
        return entry

    entry["Version"] = record.get("openneuro_py_version")
    entry["SnapshotTag"] = record.get("snapshot_tag")
    entry["Parameters"] = {
        "include": record.get("include"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
    }
    if record.get("git_revision"):
        entry["GitRevision"] = record["git_revision"]
    return entry


# ---------------------------------------------------------------------------
# Stage 2: fMRIPrep
# ---------------------------------------------------------------------------
def read_fmriprep_description(fmriprep_dir: Path, dataset_id: str) -> dict | None:
    """fMRIPrep's own GeneratedBy[0] entry, or None if unreadable."""
    desc_path = fmriprep_dir / dataset_id / "dataset_description.json"
    try:
        data = json.loads(desc_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    generated_by = data.get("GeneratedBy")
    if isinstance(generated_by, list) and generated_by:
        first = generated_by[0]
        if isinstance(first, dict) and first.get("Version"):
            return first
    return None


def load_preproc_record(fmriprep_dir: Path, dataset_id: str, subject: str) -> dict | None:
    """Last record of the subject in {fmriprep_dir}/logs/preprocessing*.jsonl."""
    records = [
        r for r in _read_jsonl(sorted((fmriprep_dir / "logs").glob("preprocessing*.jsonl")))
        if r.get("dataset_id") == dataset_id and r.get("subject") == subject
    ]
    return records[-1] if records else None


def fmriprep_generated_by(
    fmriprep_dir: Path,
    dataset_id: str,
    subject: str | None = None,
) -> dict:
    """GeneratedBy entry for the fMRIPrep run that produced this subject.

    Version, CodeURL and Container come from fMRIPrep's dataset_description.json.
    The command line comes from preprocessing.jsonl; when no command was
    recorded, it is rebuilt from the flags open-fmind passes and flagged as such.

    With `subject=None` the command line is omitted (dataset level, where
    fMRIPrep ran once per subject).
    """
    from .fmriprep import FMRIPREP_FLAGS, OUTPUT_SPACES

    entry: dict = {
        "Name": "fMRIPrep",
        "Version": None,
        "CodeURL": None,
        "Container": None,
        "CommandLine": None,
        "OutputSpaces": None,
    }
    notes: list[str] = []

    desc = read_fmriprep_description(fmriprep_dir, dataset_id)
    if desc:
        entry["Version"] = desc.get("Version")
        entry["CodeURL"] = desc.get("CodeURL")
        entry["Container"] = desc.get("Container")
    else:
        notes.append(
            f"fMRIPrep dataset_description.json not found under {fmriprep_dir}/"
            f"{dataset_id}; version and container are unknown."
        )

    if subject is None:
        entry.pop("CommandLine")
        notes.append(
            "fMRIPrep ran once per subject; the exact command for each run is "
            "recorded in that run's JSON sidecar."
        )
        entry["Note"] = " ".join(notes)
        return entry

    record = load_preproc_record(fmriprep_dir, dataset_id, subject)
    if record and record.get("command"):
        entry["CommandLine"] = record["command"]
        entry["OutputSpaces"] = record.get("output_spaces")
        if record.get("container_image"):
            container = entry["Container"] or {"Type": "singularity"}
            container.setdefault("Type", "singularity")
            container["URI"] = record["container_image"]
            container.setdefault("Tag", Path(record["container_image"]).name)
            entry["Container"] = container
        if record.get("slurm_job_id"):
            entry["SLURMJobID"] = record["slurm_job_id"]
    else:
        entry["CommandLine"] = (
            "fmriprep {bids_dir} {output_dir} participant "
            f"--participant-label {subject} " + " ".join(FMRIPREP_FLAGS)
        )
        entry["OutputSpaces"] = OUTPUT_SPACES
        notes.append(
            "No command was recorded for this subject in logs/preprocessing.jsonl. "
            "CommandLine is rebuilt from the flags open-fmind passes to fMRIPrep "
            "and is not a verbatim record."
        )

    if notes:
        entry["Note"] = " ".join(notes)
    return entry


# ---------------------------------------------------------------------------
# Stages 3 and 4: open-fmind
# ---------------------------------------------------------------------------
def script_generated_by(
    name: str,
    description: str,
    parameters: dict | None = None,
) -> dict:
    """GeneratedBy entry for a step of this package, run from this process."""
    entry: dict = {
        "Name": name,
        "Version": package_version("open-fmind"),
        "CodeURL": CODE_URL,
        "Description": description,
        "CommandLine": " ".join(sys.argv) if sys.argv else None,
        "Parameters": parameters or {},
    }
    rev = git_revision()
    if rev:
        entry["GitRevision"] = rev
    return entry
