"""The five steps for one subject, from OpenNeuro to connectivity matrices.

  1. download      func/ and anat/ of the subject from OpenNeuro into {raw_root}/{dataset}/
  2. fmriprep      fMRIPrep on the subject into {fmriprep_dir}/{dataset}/
  3. parcellate    raw ROI timeseries for each atlas into {quasiraw_dir}
  4. denoise       denoised ROI timeseries for each atlas into {preproc_dir}
  5. connectivity  one connectivity matrix per run and atlas, from the raw and
                   from the denoised timeseries

Download and fMRIPrep are skipped when their outputs are on disk. The later
steps regenerate the requested atlases unless overwrite is off.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import provenance
from .connectivity import compute_connectivity
from .denoise import denoise_subject
from .download import (
    _read_json, _tag_from_generated_by, bolds_without_tr, download_subject,
    fill_missing_sidecars, first_sidecar, raw_bold_files, resolve_snapshot_tag,
    snapshot_tags_on_disk,
)
from .fmriprep import fmriprep_done, run_fmriprep
from .parcellate import parcellate_subject

STEPS = ["download", "fmriprep", "parcellate", "denoise", "connectivity"]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def tags_on_disk(args: argparse.Namespace) -> dict[str, str]:
    return snapshot_tags_on_disk(
        args.dataset_id, args.raw_root, args.fmriprep_dir,
        {"quasiraw": args.quasiraw_dir, "preproc": args.preproc_dir},
    )


def pinned_tag(args: argparse.Namespace) -> str | None:
    """Snapshot of the dataset this subject is (or will be) downloaded at."""
    if args.pinned_tag is None:
        args.pinned_tag = resolve_snapshot_tag(
            args.dataset_id, args.snapshot_tag, tags_on_disk(args), args.log_dir,
        )
    return args.pinned_tag


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def step_download(args: argparse.Namespace) -> None:
    download_subject(
        args.dataset_id, args.subject, pinned_tag(args), args.raw_root, args.log_dir,
        max_retries=args.max_retries, dry_run=args.dry_run,
    )


def step_fmriprep(args: argparse.Namespace) -> None:
    if bolds_without_tr(args.raw_root / args.dataset_id, args.subject):
        # fMRIPrep needs the RepetitionTime of every run.
        fill_missing_sidecars(args.dataset_id, args.subject, pinned_tag(args),
                              args.raw_root, dry_run=args.dry_run)
    run_fmriprep(args.dataset_id, args.subject, args.raw_root, args.fmriprep_dir,
                 args.work_dir, args.fmriprep, dry_run=args.dry_run)
    args.ran_fmriprep = True


def download_entry(args: argparse.Namespace) -> dict:
    """openneuro-py GeneratedBy entry for the sidecars of this subject.

    From the download record of the subject (or, failing that, of the
    dataset) in {log_dir}/download.jsonl; else from an existing sidecar of the
    dataset; else built from the snapshot pinned for the download, with a Note.
    """
    entry = provenance.download_generated_by(args.dataset_id, args.log_dir, args.subject)
    if entry.get("SnapshotTag") is not None:
        return entry
    sidecar = first_sidecar(args.quasiraw_dir / args.dataset_id, args.subject)
    if sidecar is not None and _tag_from_generated_by(_read_json(sidecar)):
        for e in _read_json(sidecar).get("GeneratedBy") or []:
            if isinstance(e, dict) and e.get("Name") == "openneuro-py":
                log(f"  download provenance from {sidecar.name}")
                return e
    tag = args.pinned_tag
    if tag is None:
        tags = set(tags_on_disk(args).values())
        tag = tags.pop() if len(tags) == 1 else None
    if tag is not None:
        entry["SnapshotTag"] = tag
        entry["Note"] = (
            "No download record with a snapshot tag was found; SnapshotTag is the "
            "version of the dataset on disk (or --snapshot-tag)."
        )
    return entry


def atlases_on_disk(args: argparse.Namespace) -> set[str]:
    sub_dir = args.quasiraw_dir / args.dataset_id / f"sub-{args.subject}"
    names = set()
    for p in sub_dir.glob("**/func/*_atlas-*_timeseries.nii.gz"):
        atlas = p.name.partition("_atlas-")[2].removesuffix("_timeseries.nii.gz")
        names.add(atlas.split("_res-")[0])
    return names


def step_parcellate(args: argparse.Namespace) -> None:
    if args.ran_fmriprep:
        # A new fMRIPrep run gives new confounds, shared by every atlas of a
        # run, so atlases left alone now pair an old timeseries with new confounds.
        stale = sorted(atlases_on_disk(args) - set(args.atlas))
        if stale:
            log("  warning:  fMRIPrep was run again but these atlases on disk are "
                f"not regenerated: {' '.join(stale)}. Consider --atlas all.")
    log(f"  atlases:  {' '.join(args.atlas)}\n  output:   {args.quasiraw_dir}")
    if args.dry_run:
        return
    parcellate_subject(
        fmriprep_dir=args.fmriprep_dir,
        raw_root=args.raw_root,
        out_dir=args.quasiraw_dir,
        dataset_id=args.dataset_id,
        subject=args.subject,
        atlases=args.atlas,
        download_entry=download_entry(args),
        overwrite=args.overwrite,
    )
    sub_dir = args.quasiraw_dir / args.dataset_id / f"sub-{args.subject}"
    n_done = sum(
        len(list(sub_dir.glob(f"func/*_atlas-{a}_timeseries.nii.gz")))
        + len(list(sub_dir.glob(f"ses-*/func/*_atlas-{a}_timeseries.nii.gz")))
        for a in args.atlas
    )
    if n_done == 0:
        raise RuntimeError("parcellation wrote no timeseries for this subject.")


def step_denoise(args: argparse.Namespace) -> None:
    # Pointing both trees at the subject directory keeps the relative layout
    # ({dataset}/sub-*/[ses-*/]func/) of the output identical to the input.
    rel = Path(args.dataset_id) / f"sub-{args.subject}"
    input_dir = args.quasiraw_dir / rel
    output_dir = args.preproc_dir / rel
    log(f"  atlases:  {' '.join(args.atlas)}\n  output:   {output_dir}")
    if args.dry_run:
        return
    if not input_dir.exists():
        raise RuntimeError(f"No parcellated timeseries under {input_dir}")
    for atlas in args.atlas:
        denoise_subject(input_dir, output_dir, atlas, args.denoise, overwrite=args.overwrite)


def step_connectivity(args: argparse.Namespace) -> None:
    trees = [
        (args.quasiraw_dir, args.quasiraw_fc_dir, None),
        (args.preproc_dir, args.preproc_fc_dir, args.denoise.desc),
    ]
    log(f"  measure:  {args.connectivity_kind} (standardize: {args.connectivity_standardize})")
    for src, out, desc in trees:
        log(f"  output:   {out}")
        if args.dry_run:
            continue
        for atlas in args.atlas:
            compute_connectivity(
                src, out, atlas, kind=args.connectivity_kind,
                standardize=args.connectivity_standardize, desc=desc,
                dataset_ids=[args.dataset_id], subjects=[args.subject],
                overwrite=args.overwrite,
            )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    label = f"{args.dataset_id} sub-{args.subject}"
    log(f"=== {label} | atlases: {' '.join(args.atlas)} ===")

    have_fmriprep = fmriprep_done(args.fmriprep_dir, args.dataset_id, args.subject)
    have_raw = bool(raw_bold_files(args.raw_root / args.dataset_id, args.subject))
    plan = [
        ("download", step_download, have_raw or have_fmriprep),
        ("fmriprep", step_fmriprep, have_fmriprep),
        ("parcellate", step_parcellate, False),
        ("denoise", step_denoise, False),
        ("connectivity", step_connectivity, False),
    ]
    for name, func, done in plan:
        if name not in args.steps:
            continue
        if done:
            log(f"[{name}] already on disk, skipping")
            continue
        log(f"[{name}]")
        try:
            func(args)
        except RuntimeError as e:
            log(f"ERROR in step '{name}' for {label}: {e}")
            return 1

    log(f"=== {label} done ===")
    return 0
