"""Functional connectivity matrices from parcellated timeseries.

Computes one ROI x ROI matrix per run with nilearn's ConnectivityMeasure, from
the raw timeseries (quasiraw/parcellated) or the denoised ones
(preproc/parcellated, desc-preproc).

Output, one file per run, atlas and measure:
    {output_dir}/{dataset}/sub-{sub}/[ses-{ses}/]func/
        sub-{sub}[_ses-{ses}]_task-{task}[_run-{run}]_atlas-{atlas}[_desc-{desc}]_measure-{kind}_standardize-{bool}_connectivity.nii.gz
        (+ .json sidecar: Atlas, Measure, Standardize, ROILabels, GeneratedBy)
NIfTI shape (n_rois, n_rois, 1, 1), float32.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import provenance
from .parcellate import OUTPUT_ENTITY_ORDER, _parse_entities

CONNECTIVITY_NAME = "open-fmind connectivity"
# Measures computed from each run alone.
CONNECTIVITY_KINDS = ["correlation", "partial_correlation", "covariance", "precision"]


def _timeseries_suffix(desc: str | None = None) -> str:
    """Filename tail after the atlas label, e.g. '_desc-preproc_timeseries.nii.gz'."""
    return (f"_desc-{desc}" if desc else "") + "_timeseries.nii.gz"


def _timeseries_globs(atlas: str, desc: str | None = None) -> list[str]:
    # With desc=None only undescribed (raw) timeseries match a concrete atlas;
    # desc selects a derived variant such as the denoised desc-preproc.
    tail = f"*_atlas-{atlas}{_timeseries_suffix(desc)}"
    return [f"*/sub-*/func/{tail}", f"*/sub-*/ses-*/func/{tail}"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ParcellatedRun:
    nii_path: Path
    json_path: Path | None
    dataset_id: str
    subject: str
    session: str | None
    task: str | None
    run: str | None
    atlas: str
    roi_labels: list[str]
    # Full parsed entity set (sub/ses/task/acq/ce/dir/rec/run/echo/...) from the
    # source filename. bids_output_paths builds the output name from this
    # rather than the narrow subject/session/task/run fields above, so runs
    # that share those four but differ in e.g. acq don't collide.
    entities: dict[str, str] = field(default_factory=dict)
    # The upstream sidecar in full, so the provenance chain (and the denoising
    # settings the timeseries were built with) can be carried into our own.
    source_sidecar: dict = field(default_factory=dict)




# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def find_parcellated_runs(
    parcellated_dir: Path,
    atlas: str,
    dataset_ids: list[str] | None = None,
    desc: str | None = None,
    subjects: list[str] | None = None,
) -> list[ParcellatedRun]:
    """Parcellated timeseries under {parcellated_dir}/{dataset}/sub-*/."""
    found: list[ParcellatedRun] = []
    for glob_pattern in _timeseries_globs(atlas, desc):
        for nii_path in sorted(parcellated_dir.glob(glob_pattern)):
            dataset_id = nii_path.relative_to(parcellated_dir).parts[0]
            if dataset_ids and dataset_id not in dataset_ids:
                continue

            entities = _parse_entities(nii_path.name.replace(".nii.gz", ""))
            if subjects and entities.get("sub") not in subjects:
                continue

            # JSON sidecar shares the same stem as the NIfTI
            json_path = nii_path.with_suffix("").with_suffix(".json")
            if not json_path.exists():
                json_path = None

            roi_labels: list[str] = []
            source_sidecar: dict = {}
            if json_path is not None:
                try:
                    source_sidecar = json.loads(json_path.read_text())
                    roi_labels = source_sidecar.get("ROILabels", [])
                except (json.JSONDecodeError, OSError):
                    pass

            found.append(ParcellatedRun(
                nii_path=nii_path,
                json_path=json_path,
                dataset_id=dataset_id,
                subject=entities.get("sub", ""),
                session=entities.get("ses"),
                task=entities.get("task"),
                run=entities.get("run"),
                # Not entities.get("atlas"): atlas names containing "_" (e.g.
                # yan100_yeo7) are cut at the "_" by _parse_entities. The glob
                # pattern already selects files of exactly this atlas.
                atlas=atlas,
                roi_labels=roi_labels,
                entities=entities,
                source_sidecar=source_sidecar,
            ))
    return found


# ---------------------------------------------------------------------------
# Timeseries loading
# ---------------------------------------------------------------------------
def load_timeseries(run: ParcellatedRun) -> np.ndarray:
    """Load parcellated timeseries. Returns (n_timepoints, n_rois) float32 array."""
    import nibabel as nib

    img = nib.load(str(run.nii_path))
    data = img.get_fdata(dtype=np.float32)
    # Timeseries are stored as (n_rois, 1, 1, n_timepoints)
    # Squeeze spatial dims → (n_rois, n_timepoints), then transpose to (n_timepoints, n_rois)
    data = data.squeeze()
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return data.T


# ---------------------------------------------------------------------------
# Connectivity computation
# ---------------------------------------------------------------------------
def compute_connectivity_matrix(
    timeseries: np.ndarray,
    kind: str,
    standardize: bool,
) -> np.ndarray:
    """Compute a connectivity matrix for a single run.

    Returns (n_rois, n_rois) float32 array.
    """
    from nilearn.connectome import ConnectivityMeasure

    measure = ConnectivityMeasure(kind=kind, standardize=standardize)
    # float64: raw (non-standardized) timeseries can reach ~1e9 in scanner
    # units, and the Ledoit-Wolf covariance raises values to the 4th power,
    # which overflows float32 into NaN ("Input contains NaN").
    mat = measure.fit_transform([timeseries.astype(np.float64)])[0]
    return mat.astype(np.float32)


# ---------------------------------------------------------------------------
# BIDS NIfTI output helpers
# ---------------------------------------------------------------------------
def bids_output_paths(
    run: ParcellatedRun,
    kind: str,
    standardize: bool,
    bids_dir: Path,
) -> tuple[Path, Path]:
    """Return (nii_path, json_path) for the BIDS derivative output of a run.

    The entity portion of the stem is built from run.entities over
    OUTPUT_ENTITY_ORDER (not just sub/ses/task/run), so runs that share those
    four but differ in e.g. acq get distinct output filenames instead of
    silently overwriting each other.
    """
    entity_stem = "_".join(
        f"{key}-{run.entities[key]}" for key in OUTPUT_ENTITY_ORDER if key in run.entities
    )
    parts = [entity_stem] if entity_stem else [f"sub-{run.subject}"]
    parts.append(f"atlas-{run.atlas}")
    # Carry the input's desc- (e.g. desc-preproc) so matrices from different
    # timeseries variants never share a filename.
    if "desc" in run.entities:
        parts.append(f"desc-{run.entities['desc']}")
    parts.append(f"measure-{kind}")
    parts.append(f"standardize-{str(standardize).lower()}")
    stem = "_".join(parts) + "_connectivity"

    dir_parts: list[str] = [run.dataset_id, f"sub-{run.subject}"]
    if run.session:
        dir_parts.append(f"ses-{run.session}")
    dir_parts.append("func")
    out_dir = bids_dir.joinpath(*dir_parts)

    return out_dir / f"{stem}.nii.gz", out_dir / f"{stem}.json"



def save_to_bids(
    nii_path: Path,
    json_path: Path,
    connectivity: np.ndarray,
    run: ParcellatedRun,
    kind: str,
    standardize: bool,
    parcellated_dir: Path,
) -> None:
    import nibabel as nib

    nii_path.parent.mkdir(parents=True, exist_ok=True)

    # Shape (n_rois, n_rois, 1, 1), like the timeseries' (n_rois, 1, 1, n_timepoints)
    data = connectivity[:, :, np.newaxis, np.newaxis].astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, str(nii_path))

    # Extend the upstream chain rather than starting a new one, so a connectivity
    # matrix is traceable to raw OpenNeuro data by the same mechanism as the
    # timeseries it was computed from.
    generated_by = list(run.source_sidecar.get("GeneratedBy", []))
    generated_by.append(
        provenance.script_generated_by(
            name=CONNECTIVITY_NAME,
            description=(
                f"{kind} connectivity matrix computed from the parcellated timeseries."
            ),
            parameters={
                "atlas": run.atlas,
                "desc": run.entities.get("desc"),
                "kind": kind,
                "standardize": standardize,
                "nilearn_version": provenance.package_version("nilearn"),
            },
        )
    )

    ds_root = parcellated_dir / run.dataset_id
    if not ds_root.is_dir():
        ds_root = parcellated_dir

    sidecar = {
        "Atlas": run.atlas,
        "Measure": kind,
        "Standardize": standardize,
        "ROILabels": run.roi_labels,
        "Sources": [
            provenance.bids_uri(
                provenance.LINK_PARCELLATED,
                provenance.relative_to_dataset(run.nii_path, ds_root),
            )
        ],
        # Carried from the timeseries: the connectivity matrix inherits whatever
        # denoising produced them, and that cannot be reconstructed from here.
        "RepetitionTime": run.source_sidecar.get("RepetitionTime"),
        "ConfoundRegressors": run.source_sidecar.get("ConfoundRegressors"),
        "NVolumesRemoved": run.source_sidecar.get("NVolumesRemoved"),
        "HighPassFilter": run.source_sidecar.get("HighPassFilter"),
        "LowPassFilter": run.source_sidecar.get("LowPassFilter"),
        "FramewiseDisplacementThreshold": run.source_sidecar.get(
            "FramewiseDisplacementThreshold"
        ),
        "GeneratedBy": generated_by,
    }
    json_path.write_text(json.dumps(sidecar, indent=2))


def write_bids_dataset_description(bids_dir: Path, parcellated_dir: Path) -> None:
    """Write the connectivity derivative root's dataset_description.json.

    """
    bids_dir.mkdir(parents=True, exist_ok=True)
    desc = {
        "Name": "functional-connectivity",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "DatasetLinks": {provenance.LINK_PARCELLATED: str(parcellated_dir)},
        "GeneratedBy": [
            provenance.script_generated_by(
                name=CONNECTIVITY_NAME,
                description=(
                    "Connectivity matrices derived from parcellated BOLD timeseries. "
                    "The full chain back to the raw OpenNeuro data is recorded in "
                    "each run's JSON sidecar."
                ),
            )
        ],
    }
    (bids_dir / "dataset_description.json").write_text(json.dumps(desc, indent=2))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def compute_connectivity(
    parcellated_dir: Path,
    output_dir: Path,
    atlas: str,
    kind: str = "correlation",
    standardize: bool = False,
    desc: str | None = None,
    dataset_ids: list[str] | None = None,
    subjects: list[str] | None = None,
    overwrite: bool = True,
) -> int:
    """Connectivity matrices of every selected run of one atlas.

    The dataset_description.json of output_dir is written only when absent.
    Returns the number of matrices written.
    """
    if kind not in CONNECTIVITY_KINDS:
        raise ValueError(f"Unknown connectivity measure {kind!r}; choose from {CONNECTIVITY_KINDS}.")
    runs = find_parcellated_runs(parcellated_dir, atlas, dataset_ids, desc, subjects)
    if runs and not (output_dir / "dataset_description.json").exists():
        write_bids_dataset_description(output_dir, parcellated_dir)
    n_written = 0
    for run in runs:
        nii_path, json_path = bids_output_paths(run, kind, standardize, output_dir)
        if not overwrite and nii_path.exists():
            continue
        try:
            mat = compute_connectivity_matrix(load_timeseries(run), kind, standardize)
            save_to_bids(nii_path, json_path, mat, run, kind, standardize, parcellated_dir)
        except Exception as e:
            _log(f"  skip ({e}): {nii_path.name}")
            continue
        n_written += 1
        _log(f"  saved: {nii_path.name}")
    return n_written
