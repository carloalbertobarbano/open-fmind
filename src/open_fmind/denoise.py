"""Denoising of the raw parcellated timeseries.

The parcellated timeseries are raw and frame-aligned with the fMRIPrep
confounds TSV copied next to each run. Denoising is applied directly on the
parcellated signal: every step is linear in time, so the result equals
denoising before parcellation (it is the nilearn.signal.clean call that
nilearn's maskers run internally).

Steps, each configurable through DenoiseConfig:
  1. Confound regression   24-parameter motion model + WM + CSF signals
                           (optionally GSR, aCompCor, cosine drift, or an
                           explicit column list) from the confounds TSV.
  2. Scrubbing             frames with framewise displacement above
                           fd_threshold are censored before filtering.
  3. Detrending            linear trend removed.
  4. Band-pass filtering   0.01 to 0.1 Hz Butterworth.
  5. Standardization       per-ROI z-scoring.

Output: sub-..._atlas-{atlas}_desc-preproc_timeseries.nii.gz and a JSON
sidecar recording what was applied. Runs with fewer than min_timepoints
frames left after scrubbing are skipped.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import provenance

DENOISE_NAME = "open-fmind denoise"

# ---------------------------------------------------------------------------
# Defaults (those of the released data)
# ---------------------------------------------------------------------------
DEFAULT_HIGH_PASS = 0.01    # Hz
DEFAULT_LOW_PASS = 0.1      # Hz
DEFAULT_FD_THRESHOLD = 0.5  # mm
DEFAULT_MIN_TIMEPOINTS = 50
DEFAULT_DESC = "preproc"


@dataclass
class DenoiseConfig:
    """Denoising options. The defaults are the ones of the released data."""
    desc: str = DEFAULT_DESC
    # Confound regression
    motion: bool = True
    wm_csf: bool = True
    gsr: bool = False
    acompcor: int = 0
    cosine: bool = False
    confound_columns: list[str] | None = None
    # Temporal processing
    fd_threshold: float = DEFAULT_FD_THRESHOLD
    min_timepoints: int = DEFAULT_MIN_TIMEPOINTS
    detrend: bool = True
    standardize: str = "zscore_sample"
    high_pass: float | None = DEFAULT_HIGH_PASS
    low_pass: float | None = DEFAULT_LOW_PASS
    t_r: float | None = None


TIMESERIES_SUFFIX = "_timeseries.nii.gz"
CONFOUNDS_SUFFIX = "_desc-confounds_timeseries.tsv"

# 24-parameter motion model column names in the fMRIPrep confounds TSV.
MOTION_24P = [
    "trans_x",                    "trans_x_derivative1",
    "trans_x_power2",             "trans_x_derivative1_power2",
    "trans_y",                    "trans_y_derivative1",
    "trans_y_power2",             "trans_y_derivative1_power2",
    "trans_z",                    "trans_z_derivative1",
    "trans_z_power2",             "trans_z_derivative1_power2",
    "rot_x",                      "rot_x_derivative1",
    "rot_x_power2",               "rot_x_derivative1_power2",
    "rot_y",                      "rot_y_derivative1",
    "rot_y_power2",               "rot_y_derivative1_power2",
    "rot_z",                      "rot_z_derivative1",
    "rot_z_power2",               "rot_z_derivative1_power2",
]
TISSUE_SIGNALS = ["white_matter", "csf"]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def find_timeseries(input_dir: Path, atlas: str | None = None) -> list[Path]:
    """Find raw parcellated timeseries NIfTIs under input_dir.

    Files already carrying a _desc- entity are outputs of this script (or
    other derivatives) and are excluded, so re-running never chains denoising
    on top of denoising.
    """
    # With a named atlas, match it exactly: a trailing wildcard would also pick
    # up atlases sharing the prefix (schaefer400 -> schaefer400tians3). The
    # raw parcellated name ends in _atlas-<a>_timeseries, or _atlas-<a>_res-<r>_
    # timeseries for non-2mm runs.
    if atlas:
        found = set(input_dir.rglob(f"*_atlas-{atlas}{TIMESERIES_SUFFIX}"))
        found |= set(input_dir.rglob(f"*_atlas-{atlas}_res-*{TIMESERIES_SUFFIX}"))
        return sorted(p for p in found if "_desc-" not in p.name)
    pattern = f"*_atlas-*{TIMESERIES_SUFFIX}"
    found = [
        p for p in sorted(input_dir.rglob(pattern))
        if "_desc-" not in p.name
    ]
    return found


def entity_prefix(ts_path: Path) -> str:
    """Entity stem before the _atlas- entity (sub-..._task-..._run-...)."""
    name = ts_path.name
    if "_atlas-" not in name:
        raise ValueError(f"No _atlas- entity in {name}")
    return name.partition("_atlas-")[0]


def find_confounds(ts_path: Path) -> Path | None:
    """Locate the confounds TSV shipped next to a timeseries file."""
    prefix = entity_prefix(ts_path)
    exact = ts_path.parent / f"{prefix}{CONFOUNDS_SUFFIX}"
    if exact.exists():
        return exact
    # Extra entities (e.g. _part-mag) are allowed only after a complete entity:
    # "{prefix}_*", never "{prefix}*", or run-1 would pick up run-10's confounds.
    candidates = sorted(ts_path.parent.glob(f"{prefix}_*{CONFOUNDS_SUFFIX}"))
    return candidates[0] if candidates else None


def load_sidecar(ts_path: Path) -> dict:
    json_path = Path(str(ts_path).replace(".nii.gz", ".json"))
    if json_path.exists():
        try:
            return json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: unreadable sidecar {json_path}: {e}", file=sys.stderr)
    return {}


def output_paths(ts_path: Path, desc: str, input_dir: Path,
                 output_dir: Path | None) -> tuple[Path, Path]:
    """(nii, json) output paths, inserting the desc entity before _timeseries."""
    out_name = ts_path.name.replace(
        TIMESERIES_SUFFIX, f"_desc-{desc}{TIMESERIES_SUFFIX}"
    )
    if output_dir is None:
        out_dir = ts_path.parent
    else:
        out_dir = output_dir / ts_path.parent.relative_to(input_dir)
    nii_path = out_dir / out_name
    json_path = Path(str(nii_path).replace(".nii.gz", ".json"))
    return nii_path, json_path


# ---------------------------------------------------------------------------
# Confound selection
# ---------------------------------------------------------------------------
def select_confound_columns(df: pd.DataFrame, args: DenoiseConfig) -> list[str]:
    """The confound columns requested by the CLI, in a stable order."""
    if args.confound_columns:
        return list(args.confound_columns)
    wanted: list[str] = []
    if args.motion:
        wanted += MOTION_24P
    if args.wm_csf:
        wanted += TISSUE_SIGNALS
    if args.gsr:
        wanted.append("global_signal")
    if args.acompcor:
        acomp = sorted(
            (c for c in df.columns if c.startswith("a_comp_cor_")),
            key=lambda c: int(c.rsplit("_", 1)[-1]),
        )
        wanted += acomp[: args.acompcor]
    if args.cosine:
        wanted += sorted(c for c in df.columns if c.startswith("cosine"))
    return wanted


def build_confound_matrix(
    df: pd.DataFrame, wanted: list[str],
) -> tuple[np.ndarray | None, list[str], list[str]]:
    """Returns (matrix, available, missing). Missing columns are dropped with
    a warning."""
    available = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print(
            f"  WARNING: Missing confound columns (will be skipped): {sorted(missing)}",
            file=sys.stderr,
        )
    if not available:
        return None, available, missing
    sub_df = df[available].copy()
    # First-row NaN in derivative columns → 0; any other NaN → 0.
    sub_df.iloc[0] = sub_df.iloc[0].fillna(0.0)
    sub_df = sub_df.fillna(0.0)
    return sub_df.values.astype(np.float64), available, missing


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------
def process_timeseries(
    ts_path: Path,
    args: DenoiseConfig,
) -> tuple[np.ndarray | None, dict]:
    """Denoise one run. Returns (cleaned (T_kept, n_rois), info dict)."""
    import nibabel as nib
    from nilearn import signal as nilearn_signal

    info: dict = {"skip_reason": None}

    img = nib.load(str(ts_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 4 or data.shape[1] != 1 or data.shape[2] != 1:
        info["skip_reason"] = f"Unexpected NIfTI shape {data.shape}"
        return None, info
    ts = data[:, 0, 0, :].T  # (T, n_rois)
    n_vols = ts.shape[0]
    info["n_volumes_total"] = n_vols

    sidecar = load_sidecar(ts_path)
    tr = args.t_r if args.t_r is not None else sidecar.get("RepetitionTime")
    if (args.high_pass is not None or args.low_pass is not None) and not tr:
        # TR from the NIfTI pixdim[4] as a fallback (parcellate_bold encodes it)
        tr = float(img.header.get_zooms()[3]) or None
    if (args.high_pass is not None or args.low_pass is not None) and not tr:
        info["skip_reason"] = "TR unknown (needed for filtering); pass --t-r"
        return None, info
    info["tr"] = tr
    info["sidecar"] = sidecar

    # --- Confounds ---
    confounds_path = find_confounds(ts_path)
    matrix = None
    available: list[str] = []
    missing: list[str] = []
    fd = None
    if confounds_path is None:
        print(
            f"  WARNING: No confounds TSV next to {ts_path.name}; "
            "proceeding without confound regression or scrubbing.",
            file=sys.stderr,
        )
    else:
        try:
            df = pd.read_csv(confounds_path, sep="\t", na_values="n/a")
        except Exception as e:
            info["skip_reason"] = f"Could not read confounds {confounds_path}: {e}"
            return None, info
        if len(df) != n_vols:
            info["skip_reason"] = (
                f"Confounds TSV has {len(df)} rows but timeseries has "
                f"{n_vols} frames — not frame-aligned (was the signal already "
                "scrubbed?)"
            )
            return None, info
        wanted = select_confound_columns(df, args)
        matrix, available, missing = build_confound_matrix(df, wanted)
        if "framewise_displacement" in df.columns:
            fd = df["framewise_displacement"].fillna(0.0).values
    info["confounds_file"] = confounds_path.name if confounds_path else None
    info["confound_regressors"] = available
    info["missing_confound_regressors"] = missing

    # --- Scrubbing mask ---
    sample_mask = None
    n_removed = 0
    if fd is not None and np.isfinite(args.fd_threshold):
        keep = fd <= args.fd_threshold
        n_removed = int(n_vols - keep.sum())
        if n_removed:
            sample_mask = np.where(keep)[0]
        info["mean_fd"] = round(float(fd.mean()), 6)
        info["max_fd"] = round(float(fd.max()), 6)
    info["n_volumes_removed"] = n_removed

    n_kept = n_vols - n_removed
    if n_kept < args.min_timepoints:
        info["skip_reason"] = (
            f"Only {n_kept} volumes remain after scrubbing "
            f"(minimum: {args.min_timepoints})"
        )
        return None, info

    # --- Clean: the exact call nilearn's maskers run internally ---
    standardize = False if args.standardize == "off" else args.standardize
    cleaned = nilearn_signal.clean(
        ts,
        confounds=matrix,
        sample_mask=sample_mask,
        detrend=args.detrend,
        standardize=standardize,
        low_pass=args.low_pass,
        high_pass=args.high_pass,
        t_r=tr,
    )
    return cleaned, info


def save_output(
    nii_path: Path,
    json_path: Path,
    cleaned: np.ndarray,
    info: dict,
    args: DenoiseConfig,
) -> None:
    """Write the denoised NIfTI + a sidecar recording what was applied."""
    import nibabel as nib

    nii_path.parent.mkdir(parents=True, exist_ok=True)
    tr = info.get("tr") or 1.0
    data = cleaned.T[:, np.newaxis, np.newaxis, :].astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0, tr))
    nib.save(img, str(nii_path))

    sidecar = dict(info.get("sidecar") or {})
    regressors = info.get("confound_regressors", [])
    sidecar.update({
        "ConfoundRegressionApplied": bool(regressors),
        "ConfoundRegressors": regressors,
        "NConfoundRegressors": len(regressors),
        "MissingConfoundRegressors": info.get("missing_confound_regressors", []),
        "ConfoundsFile": info.get("confounds_file"),
        "Standardize": args.standardize if args.standardize != "off" else False,
        "Detrend": args.detrend,
        "HighPassFilter": args.high_pass,
        "LowPassFilter": args.low_pass,
        "FramewiseDisplacementThreshold": (
            None if not np.isfinite(args.fd_threshold) else args.fd_threshold
        ),
        "MeanFramewiseDisplacement": info.get("mean_fd"),
        "MaxFramewiseDisplacement": info.get("max_fd"),
        "NVolumesTotal": info.get("n_volumes_total"),
        "NVolumesRemoved": info.get("n_volumes_removed", 0),
        "MinTimepointsAfterScrub": args.min_timepoints,
    })
    generated_by = list(sidecar.get("GeneratedBy") or [])
    entry = provenance.script_generated_by(
        name=DENOISE_NAME,
        description=(
            "Denoising applied directly on the raw parcellated timeseries: "
            "confound regression, scrubbing, detrending, filtering and "
            "standardization as recorded in this sidecar."
        ),
        parameters={
            "confound_regressors": regressors,
            "fd_threshold": (
                None if not np.isfinite(args.fd_threshold) else args.fd_threshold
            ),
            "min_timepoints": args.min_timepoints,
            "detrend": args.detrend,
            "standardize": args.standardize,
            "high_pass": args.high_pass,
            "low_pass": args.low_pass,
        },
    )
    entry["Date"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    generated_by.append(entry)
    sidecar["GeneratedBy"] = generated_by
    json_path.write_text(json.dumps(sidecar, indent=2))


# ---------------------------------------------------------------------------
# One subject
# ---------------------------------------------------------------------------
def denoise_subject(
    input_dir: Path,
    output_dir: Path,
    atlas: str,
    config: DenoiseConfig,
    overwrite: bool = True,
) -> int:
    """Denoise every raw timeseries of one atlas found under input_dir.

    Outputs mirror the layout of input_dir under output_dir. Runs that fail or
    keep too few frames after scrubbing are reported and skipped. Returns the
    number of timeseries written.
    """
    n_written = 0
    for ts_path in find_timeseries(input_dir, atlas):
        nii_out, json_out = output_paths(ts_path, config.desc, input_dir, output_dir)
        if nii_out.exists() and not overwrite:
            continue
        try:
            cleaned, info = process_timeseries(ts_path, config)
        except Exception as e:
            cleaned, info = None, {"skip_reason": f"Error: {e}"}
        if cleaned is None:
            print(f"  skip ({info.get('skip_reason')}): {ts_path.name}", file=sys.stderr)
            continue
        try:
            save_output(nii_out, json_out, cleaned, info, config)
        except Exception as e:
            print(f"  skip (save error: {e}): {ts_path.name}", file=sys.stderr)
            continue
        n_written += 1
        print(f"  saved: {nii_out.name}", file=sys.stderr, flush=True)
    return n_written
