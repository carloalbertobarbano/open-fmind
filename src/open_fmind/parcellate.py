"""Raw parcellation of fMRIPrep outputs.

No denoising is applied here: no confound regression, scrubbing, detrending,
filtering or standardization. Each run is only projected onto the atlas
(NiftiLabelsMasker, or NiftiMapsMasker for probabilistic atlases), so the
output stays frame-aligned with the fMRIPrep confounds TSV, which is copied
next to it. Denoising is applied afterwards (see open_fmind.denoise); every
temporal denoising step is linear and commutes with parcellation.

Output, under {out_dir}/{dataset}/sub-{sub}/[ses-{ses}/]func/:
    sub-{sub}[_ses-{ses}]_task-{task}[_run-{run}]_atlas-{atlas}_timeseries.nii.gz
    sub-{sub}[_ses-{ses}]_task-{task}[_run-{run}]_atlas-{atlas}_timeseries.json
    sub-{sub}[_ses-{ses}]_task-{task}[_run-{run}]_desc-confounds_timeseries.tsv
NIfTI shape (n_rois, 1, 1, n_timepoints), TR in pixdim[4]. The JSON sidecar
holds RepetitionTime, Atlas, ROILabels, ConfoundsFile and the GeneratedBy
provenance chain.
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import provenance
from .atlases import load_atlas

# The only preprocessed BOLD this pipeline consumes.
PREPROC_BOLD_SUFFIX = "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"

PARCELLATE_NAME = "open-fmind parcellate"

# BIDS entities carried over, in the order the spec recommends.
OUTPUT_ENTITY_ORDER = ["sub", "ses", "task", "acq", "ce", "dir", "rec", "run", "echo"]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BoldRun:
    bold_path: Path
    confounds_path: Path | None
    json_path: Path | None
    dataset_id: str
    subject: str        # without sub- prefix
    session: str | None
    task: str | None
    run: str | None
    res: str | None = None


# ---------------------------------------------------------------------------
# BIDS entity parsing
# ---------------------------------------------------------------------------
def _parse_entities(stem: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    for part in stem.split("_"):
        if "-" in part:
            key, _, value = part.partition("-")
            entities[key] = value
    return entities


def _find_companion(bold_path: Path, suffix: str) -> Path | None:
    """Find a companion BIDS file matching the BOLD path but with a different suffix."""
    func_dir = bold_path.parent
    name = bold_path.name
    # Strip everything from _space onwards to get the BIDS entities stem.
    # Companion files usually share the same entities (sub, ses, task, run, acq, etc.)
    # but don't carry the spatial normalization entities (space, res).
    if "_space-" in name:
        base = name.partition("_space-")[0]
    else:
        # Fallback if no space entity
        base = name.split("_desc-")[0]

    # Prefer the exact companion (e.g. sub-01_task-rest_desc-confounds_timeseries.tsv).
    exact = func_dir / f"{base}{suffix}"
    if exact.exists():
        return exact
    # Otherwise allow extra entities (e.g. _part-mag), but only after a complete
    # entity: "{base}_*", never "{base}*", or run-1 would match run-10's files.
    candidates = sorted(func_dir.glob(f"{base}_*{suffix}"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def find_bold_runs(fmriprep_dir: Path, dataset_id: str, subject: str) -> list[BoldRun]:
    """Preprocessed BOLD runs of one subject under {fmriprep_dir}/{dataset_id}/."""
    sub_dir = fmriprep_dir / dataset_id / f"sub-{subject}"
    found: list[BoldRun] = []
    for pattern in (f"func/*{PREPROC_BOLD_SUFFIX}", f"ses-*/func/*{PREPROC_BOLD_SUFFIX}"):
        for bold_path in sorted(sub_dir.glob(pattern)):
            entities = _parse_entities(bold_path.name.replace(".nii.gz", ""))

            confounds_path = _find_companion(bold_path, "_desc-confounds_timeseries.tsv")
            # The BOLD sidecar shares the BOLD file's exact name.
            json_path = bold_path.with_name(bold_path.name.replace(".nii.gz", ".json"))
            if not json_path.exists():
                json_path = _find_companion(bold_path, "_desc-preproc_bold.json")

            found.append(BoldRun(
                bold_path=bold_path,
                confounds_path=confounds_path,
                json_path=json_path,
                dataset_id=dataset_id,
                subject=entities.get("sub", ""),
                session=entities.get("ses"),
                task=entities.get("task"),
                run=entities.get("run"),
                res=entities.get("res"),
            ))
    return found


# ---------------------------------------------------------------------------
# TR extraction
# ---------------------------------------------------------------------------
def get_tr(run: BoldRun) -> float:
    """Return RepetitionTime in seconds from the JSON sidecar."""
    if run.json_path and run.json_path.exists():
        try:
            data = json.loads(run.json_path.read_text())
            tr = data.get("RepetitionTime")
            if tr and float(tr) > 0:
                return float(tr)
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    # Fallback: read NIfTI header pixdim[4]
    try:
        import nibabel as nib
        img = nib.load(str(run.bold_path))
        tr = float(img.header.get_zooms()[3])
        if tr > 0:
            return tr
    except Exception:
        pass

    raise ValueError(f"Cannot determine TR for {run.bold_path}")


# ---------------------------------------------------------------------------
# Confounds copy
# ---------------------------------------------------------------------------
def copy_confounds(run: BoldRun, out_dir: Path, overwrite: bool = False) -> str | None:
    """Copy the fMRIPrep confounds TSV (and JSON sidecar) next to the output.

    The released timeseries are raw and frame-aligned with the confounds TSV,
    so shipping the TSV is what lets end users scrub/denoise later. Idempotent
    across atlases: the copy is shared by every atlas's output for this run and
    is skipped when already present (unless overwrite).

    Returns the basename of the copied TSV, or None when the run has no
    confounds file.
    """
    if run.confounds_path is None or not run.confounds_path.exists():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_tsv = out_dir / run.confounds_path.name
    if overwrite or not dest_tsv.exists():
        shutil.copy2(run.confounds_path, dest_tsv)
    confounds_json = _find_companion(run.bold_path, "_desc-confounds_timeseries.json")
    if confounds_json is not None and confounds_json.exists():
        dest_json = out_dir / confounds_json.name
        if overwrite or not dest_json.exists():
            shutil.copy2(confounds_json, dest_json)
    return dest_tsv.name


# ---------------------------------------------------------------------------
# BIDS NIfTI output helpers
# ---------------------------------------------------------------------------
def entity_stem_from_filename(name: str) -> str:
    """Join whichever OUTPUT_ENTITY_ORDER entities are present in a filename.

    Works from a bare filename or path string rather than a live BoldRun, for
    callers that only have a logged path (e.g. a JSONL bold_path field) and no
    BoldRun object to hand to bids_entity_stem.
    """
    stem = Path(name).name.replace(".nii.gz", "")
    entities = _parse_entities(stem)
    return "_".join(
        f"{key}-{entities[key]}" for key in OUTPUT_ENTITY_ORDER if key in entities
    )


def bids_entity_stem(run: BoldRun) -> str:
    """Entity prefix shared by every atlas's output for one run.

    Re-extracted from the fMRIPrep filename so that distinguishing tags (acq,
    echo, dir, …) survive into the output. This is the piece a parcellated
    filename and its fMRIPrep source have in common.
    """
    return entity_stem_from_filename(str(run.bold_path))


def bids_output_paths(
    run: BoldRun,
    atlas_name: str,
    bids_dir: Path,
) -> tuple[Path, Path]:
    """Return (nii_path, json_path) for the BIDS derivative output of a run."""
    entities = _parse_entities(run.bold_path.name.replace(".nii.gz", ""))

    parts = [bids_entity_stem(run), f"atlas-{atlas_name}"]

    # Handle resolution: reflect in the filename if it deviates from the 2mm standard.
    if "res" in entities and entities["res"] != "2":
        parts.append(f"res-{entities['res']}")

    stem = "_".join(parts) + "_timeseries"

    dir_parts: list[str] = [run.dataset_id, f"sub-{run.subject}"]
    if run.session:
        dir_parts.append(f"ses-{run.session}")
    dir_parts.append("func")
    out_dir = bids_dir.joinpath(*dir_parts)

    return out_dir / f"{stem}.nii.gz", out_dir / f"{stem}.json"


def fmriprep_dataset_root(fmriprep_dir: Path, dataset_id: str) -> Path:
    """Root of one dataset inside the fMRIPrep tree (multi- or single-dataset layout)."""
    candidate = fmriprep_dir / dataset_id
    return candidate if candidate.is_dir() else fmriprep_dir


def build_generated_by(
    run: BoldRun,
    atlas_name: str,
    fmriprep_dir: Path,
    download_entry: dict,
) -> list[dict]:
    """The openneuro-py -> fMRIPrep -> parcellation chain for one run."""
    return [
        download_entry,
        provenance.fmriprep_generated_by(fmriprep_dir, run.dataset_id, run.subject),
        provenance.script_generated_by(
            name=PARCELLATE_NAME,
            description=(
                f"Raw parcellation of the preprocessed BOLD with the {atlas_name} "
                f"atlas. No denoising applied: no confound regression, scrubbing, "
                f"detrending, filtering or standardization."
            ),
            parameters={
                "atlas": atlas_name,
                "confound_regression": False,
                "scrubbing": False,
                "detrend": False,
                "standardize": False,
                "high_pass": None,
                "low_pass": None,
                "nilearn_version": provenance.package_version("nilearn"),
            },
        ),
    ]


def build_run_sidecar(
    run: BoldRun,
    roi_labels: list[str],
    tr: float,
    atlas_name: str,
    info: dict,
    generated_by: list[dict],
    fmriprep_dir: Path,
) -> dict:
    """The sidecar for one parcellated run.

    Self-contained by design: everything needed to trace this file back to the
    raw OpenNeuro data lives here, because the fMRIPrep and raw BIDS trees it
    derives from are routinely deleted once processing is done.

    The signal is raw: the sidecar says so explicitly (false/null fields below)
    so no reader can mistake it for a denoised release. ConfoundsFile names the
    fMRIPrep confounds TSV copied next to this output, frame-aligned with the
    timeseries, from which any denoising can be applied downstream.
    """
    entities = _parse_entities(run.bold_path.name.replace(".nii.gz", ""))

    # Sources: every input this run consumed, as BIDS URIs into the fMRIPrep tree.
    ds_root = fmriprep_dataset_root(fmriprep_dir, run.dataset_id)
    sources = [
        provenance.bids_uri(
            provenance.LINK_FMRIPREP, provenance.relative_to_dataset(p, ds_root)
        )
        for p in (run.bold_path, run.confounds_path, run.json_path)
        if p is not None
    ]

    return {
        "RepetitionTime": tr,
        "SamplingFrequency": round(1.0 / tr, 6),
        "Atlas": atlas_name,
        "ROILabels": roi_labels,
        # The space/resolution the timeseries were sampled in. Not recoverable
        # from the output filename, which drops the space- entity.
        "Space": entities.get("space"),
        "Resolution": entities.get("res"),
        "Sources": sources,
        # Basename of the fMRIPrep confounds TSV copied into this directory
        # (null when the run had none).
        "ConfoundsFile": info.get("confounds_file"),
        # Raw signal: nothing was regressed, removed, filtered or scaled.
        "ConfoundRegressionApplied": False,
        "ConfoundRegressors": [],
        "Standardize": False,
        "Detrend": False,
        "HighPassFilter": None,
        "LowPassFilter": None,
        "FramewiseDisplacementThreshold": None,
        "NVolumesTotal": info.get("n_volumes_total"),
        "NVolumesRemoved": 0,
        "GeneratedBy": generated_by,
    }


def save_run_to_bids(
    nii_path: Path,
    json_path: Path,
    timeseries: np.ndarray,
    roi_labels: list[str],
    run: BoldRun,
    tr: float,
    atlas_name: str,
    info: dict,
    generated_by: list[dict],
    fmriprep_dir: Path,
) -> None:
    """Save parcellated timeseries as a BIDS derivative NIfTI + JSON sidecar."""
    import nibabel as nib

    nii_path.parent.mkdir(parents=True, exist_ok=True)

    # Shape: (n_rois, 1, 1, n_timepoints)
    data = timeseries.T[:, np.newaxis, np.newaxis, :].astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0, tr))
    nib.save(img, str(nii_path))

    sidecar = build_run_sidecar(
        run=run,
        roi_labels=roi_labels,
        tr=tr,
        atlas_name=atlas_name,
        info=info,
        generated_by=generated_by,
        fmriprep_dir=fmriprep_dir,
    )
    json_path.write_text(json.dumps(sidecar, indent=2))


def write_bids_dataset_description(bids_dir: Path) -> None:
    """Write the derivative root's dataset_description.json.

    Atlas-free: the root spans every dataset and every atlas. Atlas identity
    belongs on the run sidecar.
    """
    bids_dir.mkdir(parents=True, exist_ok=True)
    desc = {
        "Name": "parcellated-timeseries",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            provenance.script_generated_by(
                name=PARCELLATE_NAME,
                description=(
                    "Raw parcellated BOLD timeseries derived from fMRIPrep "
                    "outputs. No confound regression, scrubbing, detrending, "
                    "filtering or standardization applied. Each run's fMRIPrep "
                    "confounds TSV is copied alongside its timeseries so "
                    "denoising can be applied downstream (see "
                    "open-fmind denoise). Per-run provenance, including the "
                    "fMRIPrep and OpenNeuro download commands, is recorded in "
                    "each run's JSON sidecar."
                ),
            )
        ],
    }
    (bids_dir / "dataset_description.json").write_text(json.dumps(desc, indent=2))


def write_dataset_provenance(
    bids_dir: Path,
    dataset_id: str,
    fmriprep_dir: Path,
    raw_bids_root: Path | None,
    generated_by: list[dict],
) -> None:
    """Write a per-dataset dataset_description.json under bids_dir/{dataset_id}.

    Mirrors fMRIPrep's own layout (it writes one per dataset too) and carries the
    DatasetLinks that make the "bids:fmriprep:..." URIs in each run's Sources
    resolvable, plus the OpenNeuro accession this dataset came from.

    """
    ds_dir = bids_dir / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    links = {
        provenance.LINK_FMRIPREP: str(fmriprep_dataset_root(fmriprep_dir, dataset_id)),
    }
    if raw_bids_root is not None:
        links[provenance.LINK_RAW] = str(raw_bids_root / dataset_id)

    download = next(
        (g for g in generated_by if g.get("Name") == "openneuro-py"), {}
    )
    desc = {
        "Name": f"parcellated-timeseries/{dataset_id}",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "DatasetLinks": links,
        "SourceDatasets": [
            {
                "URL": download.get(
                    "SourceURL",
                    provenance.OPENNEURO_DATASET_URL.format(dataset_id=dataset_id),
                ),
                "Version": download.get("SnapshotTag"),
            }
        ],
        "GeneratedBy": generated_by,
    }
    (ds_dir / "dataset_description.json").write_text(json.dumps(desc, indent=2))


# ---------------------------------------------------------------------------
# Process a single run
# ---------------------------------------------------------------------------
def process_run(
    run: BoldRun,
    masker_class,
    atlas_img,
    roi_labels: list[str],
    atlas_name: str,
    tr_override: float | None,
    masker_kwargs: dict | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Raw parcellation of one BOLD run. Returns (timeseries, info_dict).

    Deliberately applies no denoising — no confounds, no sample_mask, no
    detrend/filter/standardize — so the output stays frame-aligned with the
    fMRIPrep confounds TSV shipped next to it.
    """
    info: dict = {
        "bold_path": str(run.bold_path),
        "skip_reason": None,
    }

    # Get TR
    try:
        tr = tr_override if tr_override is not None else get_tr(run)
    except ValueError as e:
        info["skip_reason"] = f"TR unknown: {e}"
        return None, info
    info["tr"] = tr

    import nibabel as nib
    info["n_volumes_total"] = nib.load(str(run.bold_path)).shape[-1]

    if run.confounds_path is None or not run.confounds_path.exists():
        print(
            f"  WARNING: No confounds file for {run.bold_path.name}. "
            "The output will have no confounds TSV to denoise from.",
            file=sys.stderr,
        )

    # Build masker: pure spatial projection onto the atlas.
    masker = masker_class(
        atlas_img,
        standardize=False,
        detrend=False,
        low_pass=None,
        high_pass=None,
        t_r=tr,
        memory=None,
        verbose=0,
        **(masker_kwargs or {}),
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = masker.fit_transform(str(run.bold_path))
    except Exception as e:
        info["skip_reason"] = f"Masker error: {e}"
        return None, info

    # The masker must emit exactly one column per named ROI. A mismatch means
    # the atlas image and its label list have silently gone out of sync (e.g.
    # overlapping labels after independent resampling producing spurious
    # extra columns) — fail loudly instead of writing a corrupted timeseries.
    if ts.shape[1] != len(roi_labels):
        info["skip_reason"] = (
            f"Masker produced {ts.shape[1]} ROI column(s) but atlas {atlas_name!r} "
            f"has {len(roi_labels)} label(s) — atlas/label mismatch"
        )
        return None, info

    return ts, info


# ---------------------------------------------------------------------------
# One subject
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parcellate_subject(
    fmriprep_dir: Path,
    raw_root: Path,
    out_dir: Path,
    dataset_id: str,
    subject: str,
    atlases: list[str],
    download_entry: dict,
    overwrite: bool = True,
) -> int:
    """Parcellate every preprocessed run of one subject with each atlas.

    Writes under {out_dir}/{dataset_id}/sub-{subject}/. The dataset_description.json
    files of out_dir and of the dataset are written only when absent, so other
    subjects' records are left alone. Runs that fail are reported and skipped.
    Returns the number of timeseries written.
    """
    runs = find_bold_runs(fmriprep_dir, dataset_id, subject)
    if not runs:
        raise RuntimeError(
            f"No preprocessed BOLD (*{PREPROC_BOLD_SUFFIX}) under "
            f"{fmriprep_dir / dataset_id / ('sub-' + subject)}"
        )

    if not (out_dir / "dataset_description.json").exists():
        write_bids_dataset_description(out_dir)
    if not (out_dir / dataset_id / "dataset_description.json").exists():
        write_dataset_provenance(
            bids_dir=out_dir,
            dataset_id=dataset_id,
            fmriprep_dir=fmriprep_dir,
            raw_bids_root=raw_root,
            generated_by=[
                download_entry,
                provenance.fmriprep_generated_by(fmriprep_dir, dataset_id),
                provenance.script_generated_by(
                    name=PARCELLATE_NAME,
                    description=(
                        "Parcellated BOLD timeseries. Per-run denoising details "
                        "and the exact fMRIPrep command are in each run's sidecar."
                    ),
                ),
            ],
        )

    n_written = 0
    for atlas_name in atlases:
        try:
            masker_class, atlas_img, roi_labels, masker_kwargs = load_atlas(atlas_name)
        except Exception as e:
            _log(f"  ERROR loading atlas {atlas_name!r}: {e}")
            continue
        for run in runs:
            nii_path, json_path = bids_output_paths(run, atlas_name, out_dir)
            if not overwrite and nii_path.exists():
                continue
            ts, info = process_run(
                run=run,
                masker_class=masker_class,
                atlas_img=atlas_img,
                roi_labels=roi_labels,
                atlas_name=atlas_name,
                tr_override=None,
                masker_kwargs=masker_kwargs,
            )
            if ts is None:
                _log(f"  skip ({info.get('skip_reason', 'error')}): {nii_path.name}")
                continue
            try:
                # The confounds TSV next to the raw timeseries is what the
                # denoising step reads.
                info["confounds_file"] = copy_confounds(
                    run, nii_path.parent, overwrite=overwrite,
                )
                save_run_to_bids(
                    nii_path=nii_path,
                    json_path=json_path,
                    timeseries=ts,
                    roi_labels=roi_labels,
                    run=run,
                    tr=info["tr"],
                    atlas_name=atlas_name,
                    info=info,
                    generated_by=build_generated_by(
                        run=run,
                        atlas_name=atlas_name,
                        fmriprep_dir=fmriprep_dir,
                        download_entry=download_entry,
                    ),
                    fmriprep_dir=fmriprep_dir,
                )
            except Exception as e:
                _log(f"  skip (save error: {e}): {nii_path.name}")
                continue
            n_written += 1
            _log(f"  saved: {nii_path.name}")
    return n_written
