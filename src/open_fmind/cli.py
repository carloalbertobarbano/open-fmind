"""Command line: run the open-fmind pipeline on one OpenNeuro subject.

Example:
    open-fmind ds000030 10159 --atlas aicha glasser --data-root /data/open-fmind \\
        --fmriprep-image fmriprep-25.2.3.sif --fs-license license.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .atlases import resolve_atlas_names
from .denoise import (
    DEFAULT_DESC, DEFAULT_FD_THRESHOLD, DEFAULT_HIGH_PASS, DEFAULT_LOW_PASS,
    DEFAULT_MIN_TIMEPOINTS, DenoiseConfig,
)
from .connectivity import CONNECTIVITY_KINDS
from .fmriprep import FmriprepConfig
from .pipeline import STEPS, run

DEFAULT_NPROCS = 16
DEFAULT_MEM_MB = 32000

# Directories written to, and their location under --data-root.
WRITE_DIRS = {
    "raw_root": "openneuro",
    "fmriprep_dir": "fmriprep",
    "work_dir": "work",
    "quasiraw_dir": "quasiraw/parcellated",
    "preproc_dir": "preproc/parcellated",
    "quasiraw_fc_dir": "quasiraw/functional-connectivity",
    "preproc_fc_dir": "preproc/functional-connectivity",
    "log_dir": "logs",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="open-fmind",
        description="Download, preprocess (fMRIPrep), parcellate and denoise one "
                    "OpenNeuro subject, and compute its connectivity matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dataset_id", help="OpenNeuro dataset ID, e.g. ds000030.")
    p.add_argument("subject", help="Subject label, with or without the sub- prefix.")
    p.add_argument(
        "--atlas", nargs="+", required=True, metavar="NAME",
        help="Atlas name(s) to use, or 'all'.",
    )
    p.add_argument(
        "--steps", nargs="+", choices=STEPS, default=STEPS,
        help="Steps to run. Download and fMRIPrep are skipped anyway when "
             "their outputs exist.",
    )
    p.add_argument(
        "--data-root", type=Path, required=True, metavar="DIR",
        help="Root of everything this command writes: "
             + ", ".join(f"{v}/" for v in WRITE_DIRS.values()) + ".",
    )
    p.add_argument(
        "--snapshot-tag", default=None, metavar="TAG",
        help="OpenNeuro snapshot to download (default: the version already on "
             "disk for the dataset, then the one in the download log, then the latest).",
    )
    p.add_argument("--max-retries", type=int, default=7, metavar="N",
                   help="Per-file download retries.")
    p.add_argument(
        "--no-overwrite", dest="overwrite", action="store_false",
        help="Keep existing parcellated and denoised outputs instead of regenerating them.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print what each step would do without running it.")

    fp = p.add_argument_group("fMRIPrep")
    fp.add_argument(
        "--fmriprep-image", type=Path, default=os.environ.get("FMRIPREP_IMAGE"),
        metavar="SIF", help="Singularity/Apptainer image of fMRIPrep 25.2.3 "
                            "(default: $FMRIPREP_IMAGE).",
    )
    fp.add_argument(
        "--fs-license", type=Path, default=os.environ.get("FS_LICENSE"), metavar="FILE",
        help="FreeSurfer license file (default: $FS_LICENSE).",
    )
    fp.add_argument(
        "--templateflow-home", type=Path, metavar="DIR",
        default=os.environ.get("TEMPLATEFLOW_HOME", Path.home() / ".cache" / "templateflow"),
        help="TemplateFlow cache mounted in the container (default: $TEMPLATEFLOW_HOME).",
    )
    fp.add_argument(
        "--nprocs", type=int, metavar="N",
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", DEFAULT_NPROCS)),
        help="CPUs for fMRIPrep (default: $SLURM_CPUS_PER_TASK if set).",
    )
    fp.add_argument("--mem-mb", type=int, default=DEFAULT_MEM_MB, metavar="MB",
                    help="Memory for fMRIPrep.")

    conf = p.add_argument_group("denoising: confound regression")
    conf.add_argument("--no-motion", dest="motion", action="store_false",
                      help="Drop the 24-parameter motion model from the regressors.")
    conf.add_argument("--no-wm-csf", dest="wm_csf", action="store_false",
                      help="Drop the mean white-matter and CSF signals from the regressors.")
    conf.add_argument("--gsr", action="store_true", help="Add global signal regression.")
    conf.add_argument("--acompcor", default=0, type=int, metavar="N",
                      help="Add the first N aCompCor components.")
    conf.add_argument("--cosine", action="store_true",
                      help="Add fMRIPrep's cosine drift regressors.")
    conf.add_argument("--confound-columns", nargs="+", default=None, metavar="COL",
                      help="Explicit confound column list; overrides the flags above.")

    tmp = p.add_argument_group("denoising: temporal processing")
    tmp.add_argument("--fd-threshold", default=DEFAULT_FD_THRESHOLD, type=float, metavar="MM",
                     help="Scrub frames with framewise displacement above this (mm). "
                          "Pass 'inf' to disable scrubbing.")
    tmp.add_argument("--min-timepoints", default=DEFAULT_MIN_TIMEPOINTS, type=int, metavar="N",
                     help="Skip runs with fewer than N frames left after scrubbing.")
    tmp.add_argument("--no-detrend", dest="detrend", action="store_false",
                     help="Disable linear detrending.")
    tmp.add_argument("--standardize", default="zscore_sample",
                     choices=["zscore_sample", "psc", "off"],
                     help="Per-ROI standardization of the output signal.")
    tmp.add_argument("--high-pass", default=DEFAULT_HIGH_PASS, type=float, metavar="HZ",
                     help="High-pass filter cutoff in Hz.")
    tmp.add_argument("--low-pass", default=DEFAULT_LOW_PASS, type=float, metavar="HZ",
                     help="Low-pass filter cutoff in Hz.")
    tmp.add_argument("--no-bandpass", action="store_true",
                     help="Disable band-pass filtering.")
    tmp.add_argument("--desc", default=DEFAULT_DESC, metavar="LABEL",
                     help="Value of the desc- entity of the denoised files.")

    fc = p.add_argument_group("connectivity")
    fc.add_argument("--connectivity-kind", choices=CONNECTIVITY_KINDS, default="correlation",
                    help="Connectivity measure (nilearn ConnectivityMeasure).")
    fc.add_argument("--connectivity-standardize", action="store_true",
                    help="Z-score each timeseries before computing the matrix.")
    return p


def check_paths(args: argparse.Namespace) -> None:
    """Refuse to write anywhere but under --data-root."""
    for name in WRITE_DIRS:
        path = getattr(args, name)
        if not path.is_relative_to(args.data_root):
            raise SystemExit(f"ERROR: {path} is outside --data-root {args.data_root}.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = build_parser()
    args = p.parse_args(argv)
    args.subject = args.subject.removeprefix("sub-")
    try:
        args.atlas = resolve_atlas_names(args.atlas)
    except ValueError as e:
        p.error(str(e))
    if "fmriprep" in args.steps and not args.dry_run:
        if args.fmriprep_image is None:
            p.error("--fmriprep-image (or $FMRIPREP_IMAGE) is required to run fMRIPrep.")
        if args.fs_license is None:
            p.error("--fs-license (or $FS_LICENSE) is required to run fMRIPrep.")

    args.data_root = args.data_root.resolve()
    for name, rel in WRITE_DIRS.items():
        setattr(args, name, (args.data_root / rel).resolve())
    check_paths(args)

    args.fmriprep = FmriprepConfig(
        image=Path(args.fmriprep_image or "fmriprep.sif").resolve(),
        fs_license=Path(args.fs_license or "license.txt").resolve(),
        templateflow_home=Path(args.templateflow_home).expanduser().resolve(),
        nprocs=args.nprocs,
        mem_mb=args.mem_mb,
    )
    args.denoise = DenoiseConfig(
        desc=args.desc,
        motion=args.motion,
        wm_csf=args.wm_csf,
        gsr=args.gsr,
        acompcor=args.acompcor,
        cosine=args.cosine,
        confound_columns=args.confound_columns,
        fd_threshold=args.fd_threshold,
        min_timepoints=args.min_timepoints,
        detrend=args.detrend,
        standardize=args.standardize,
        high_pass=None if args.no_bandpass else args.high_pass,
        low_pass=None if args.no_bandpass else args.low_pass,
    )
    args.pinned_tag = args.snapshot_tag
    args.ran_fmriprep = False
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
