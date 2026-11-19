"""A tiny fMRIPrep output tree and a toy atlas, so the parcellation and
denoising steps run offline."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from open_fmind import atlases
from open_fmind.denoise import MOTION_24P, TISSUE_SIGNALS
from open_fmind.parcellate import PREPROC_BOLD_SUFFIX

DATASET = "ds000001"
SUBJECT = "01"
N_VOLS = 120
TR = 2.0
SHAPE = (6, 6, 6)


def _labels_img() -> nib.Nifti1Image:
    data = np.zeros(SHAPE, dtype=np.int16)
    data[:3] = 1
    data[3:] = 2
    return nib.Nifti1Image(data, np.eye(4))


@pytest.fixture
def toy_atlas(monkeypatch):
    from nilearn.maskers import NiftiLabelsMasker

    monkeypatch.setitem(
        atlases.ATLAS_REGISTRY, "toy",
        lambda: (NiftiLabelsMasker, _labels_img(), ["left", "right"], {}),
    )
    return "toy"


@pytest.fixture
def fmriprep_tree(tmp_path: Path) -> Path:
    """{tmp}/fmriprep/{DATASET}/sub-01/func with one run, confounds and sidecar."""
    rng = np.random.default_rng(0)
    root = tmp_path / "fmriprep"
    func = root / DATASET / f"sub-{SUBJECT}" / "func"
    func.mkdir(parents=True)
    (root / DATASET / "dataset_description.json").write_text(json.dumps({
        "Name": "fMRIPrep - fMRI PREProcessing workflow",
        "GeneratedBy": [{"Name": "fMRIPrep", "Version": "25.2.3",
                         "CodeURL": "https://github.com/nipreps/fmriprep"}],
    }))

    stem = f"sub-{SUBJECT}_task-rest_run-1"
    bold = rng.normal(100, 1, size=SHAPE + (N_VOLS,)).astype(np.float32)
    img = nib.Nifti1Image(bold, np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0, TR))
    nib.save(img, func / f"{stem}{PREPROC_BOLD_SUFFIX}")
    (func / f"{stem}{PREPROC_BOLD_SUFFIX}".replace(".nii.gz", ".json")).write_text(
        json.dumps({"RepetitionTime": TR}))

    cols = {c: rng.normal(size=N_VOLS) for c in MOTION_24P + TISSUE_SIGNALS}
    fd = np.clip(np.abs(rng.normal(0.2, 0.1, size=N_VOLS)), 0, 0.45)
    fd[[10, 50, 51]] = 1.0
    fd[0] = np.nan
    cols["framewise_displacement"] = fd
    cols["trans_x_derivative1"][0] = np.nan
    pd.DataFrame(cols).to_csv(func / f"{stem}_desc-confounds_timeseries.tsv",
                              sep="\t", index=False, na_rep="n/a")
    return root
