from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from conftest import DATASET, N_VOLS, SUBJECT, TR
from open_fmind import cli, pipeline
from open_fmind.denoise import (
    MOTION_24P, TISSUE_SIGNALS, DenoiseConfig, denoise_subject,
)
from open_fmind.download import _tag_from_doi, bolds_without_tr, snapshot_tags_on_disk
from open_fmind.fmriprep import FMRIPREP_FLAGS, FmriprepConfig, build_command, fmriprep_done
from open_fmind.parcellate import (
    BoldRun, bids_output_paths, find_bold_runs, parcellate_subject,
)

DOWNLOAD_ENTRY = {"Name": "openneuro-py", "SnapshotTag": "1.0.0"}


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def test_output_names_keep_entities(tmp_path):
    bold = Path(f"/x/sub-01_ses-a_task-rest_acq-mb_run-2_echo-1_part-mag"
                f"_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    run = BoldRun(bold, None, None, DATASET, "01", "a", "rest", "2", "2")
    nii, js = bids_output_paths(run, "aicha", tmp_path)
    assert nii == (tmp_path / DATASET / "sub-01" / "ses-a" / "func"
                   / "sub-01_ses-a_task-rest_acq-mb_run-2_echo-1_atlas-aicha_timeseries.nii.gz")
    assert js.name.endswith("_atlas-aicha_timeseries.json")


def test_find_bold_runs(fmriprep_tree):
    runs = find_bold_runs(fmriprep_tree, DATASET, SUBJECT)
    assert len(runs) == 1
    run = runs[0]
    assert (run.subject, run.task, run.run, run.session) == (SUBJECT, "rest", "1", None)
    assert run.confounds_path.name == f"sub-{SUBJECT}_task-rest_run-1_desc-confounds_timeseries.tsv"
    assert find_bold_runs(fmriprep_tree, DATASET, "02") == []


# ---------------------------------------------------------------------------
# Parcellation and denoising
# ---------------------------------------------------------------------------
def test_parcellate_then_denoise(tmp_path, fmriprep_tree, toy_atlas):
    out = tmp_path / "quasiraw"
    n = parcellate_subject(fmriprep_tree, tmp_path / "openneuro", out, DATASET, SUBJECT,
                           [toy_atlas], DOWNLOAD_ENTRY)
    assert n == 1
    func = out / DATASET / f"sub-{SUBJECT}" / "func"
    ts_path = func / f"sub-{SUBJECT}_task-rest_run-1_atlas-toy_timeseries.nii.gz"
    img = nib.load(ts_path)
    assert img.shape == (2, 1, 1, N_VOLS)
    assert img.header.get_zooms()[3] == TR
    assert (func / f"sub-{SUBJECT}_task-rest_run-1_desc-confounds_timeseries.tsv").exists()

    # Raw parcellation: the mean of the voxels of each region.
    bold = nib.load(next((fmriprep_tree / DATASET).rglob("*_bold.nii.gz"))).get_fdata()
    expected = np.stack([bold[:3].mean(axis=(0, 1, 2)), bold[3:].mean(axis=(0, 1, 2))])
    np.testing.assert_allclose(img.get_fdata()[:, 0, 0, :], expected, rtol=1e-5)

    sidecar = json.loads(ts_path.with_name(ts_path.name.replace(".nii.gz", ".json")).read_text())
    assert sidecar["ROILabels"] == ["left", "right"]
    assert [g["Name"] for g in sidecar["GeneratedBy"]] == [
        "openneuro-py", "fMRIPrep", "open-fmind parcellate"]
    assert sidecar["GeneratedBy"][1]["Version"] == "25.2.3"
    assert json.loads((out / DATASET / "dataset_description.json").read_text())[
        "SourceDatasets"][0]["Version"] == "1.0.0"

    # Denoising equals one nilearn.signal.clean call on the raw timeseries.
    from nilearn import signal

    config = DenoiseConfig()
    preproc = tmp_path / "preproc"
    assert denoise_subject(out, preproc, toy_atlas, config) == 1
    den_path = next(preproc.rglob("*_atlas-toy_desc-preproc_timeseries.nii.gz"))
    assert den_path.parent.relative_to(preproc) == func.relative_to(out)
    cleaned = nib.load(den_path).get_fdata()[:, 0, 0, :].T

    df = pd.read_csv(func / f"sub-{SUBJECT}_task-rest_run-1_desc-confounds_timeseries.tsv",
                     sep="\t", na_values="n/a")
    confounds = df[MOTION_24P + TISSUE_SIGNALS].fillna(0.0).values
    fd = df["framewise_displacement"].fillna(0.0).values
    keep = np.where(fd <= 0.5)[0]
    ref = signal.clean(
        img.get_fdata()[:, 0, 0, :].T.astype(np.float32), confounds=confounds,
        sample_mask=keep, detrend=True, standardize="zscore_sample",
        low_pass=0.1, high_pass=0.01, t_r=TR,
    )
    assert cleaned.shape == (N_VOLS - 3, 2)
    np.testing.assert_allclose(cleaned, ref, rtol=1e-5, atol=1e-5)

    den_sidecar = json.loads(den_path.with_name(den_path.name.replace(".nii.gz", ".json")).read_text())
    assert den_sidecar["NVolumesRemoved"] == 3
    assert den_sidecar["FramewiseDisplacementThreshold"] == 0.5
    assert den_sidecar["GeneratedBy"][-1]["Name"] == "open-fmind denoise"


def test_denoise_skips_short_runs(tmp_path, fmriprep_tree, toy_atlas):
    out = tmp_path / "quasiraw"
    parcellate_subject(fmriprep_tree, tmp_path / "openneuro", out, DATASET, SUBJECT,
                       [toy_atlas], DOWNLOAD_ENTRY)
    config = DenoiseConfig(min_timepoints=N_VOLS)
    assert denoise_subject(out, tmp_path / "preproc", toy_atlas, config) == 0


def test_denoise_defaults():
    c = DenoiseConfig()
    assert (c.motion, c.wm_csf, c.gsr, c.acompcor) == (True, True, False, 0)
    assert (c.fd_threshold, c.min_timepoints, c.high_pass, c.low_pass) == (0.5, 50, 0.01, 0.1)
    assert (c.detrend, c.standardize, c.desc) == (True, "zscore_sample", "preproc")


# ---------------------------------------------------------------------------
# Download and fMRIPrep helpers
# ---------------------------------------------------------------------------
def test_tag_from_doi():
    assert _tag_from_doi("10.18112/openneuro.ds000001.v1.0.0", "ds000001") == "1.0.0"
    assert _tag_from_doi("doi:10.18112/openneuro.ds000001.v2.0.1", "ds000001") == "2.0.1"
    assert _tag_from_doi("10.18112/openneuro.ds000002.v1.0.0", "ds000001") is None
    assert _tag_from_doi(None, "ds000001") is None


def test_snapshot_tags_on_disk(tmp_path):
    raw = tmp_path / "openneuro" / DATASET
    raw.mkdir(parents=True)
    (raw / "dataset_description.json").write_text(
        json.dumps({"DatasetDOI": f"10.18112/openneuro.{DATASET}.v1.2.0"}))
    tags = snapshot_tags_on_disk(DATASET, tmp_path / "openneuro", tmp_path / "fmriprep",
                                 {"quasiraw": tmp_path / "quasiraw"})
    assert tags == {"raw": "1.2.0"}


def test_bolds_without_tr(tmp_path):
    ds = tmp_path / DATASET
    func = ds / "sub-01" / "func"
    func.mkdir(parents=True)
    (func / "sub-01_task-rest_bold.nii.gz").touch()
    (func / "sub-01_task-motor_bold.nii.gz").touch()
    (ds / "task-rest_bold.json").write_text(json.dumps({"RepetitionTime": 2.0}))
    assert [p.name for p in bolds_without_tr(ds, "01")] == ["sub-01_task-motor_bold.nii.gz"]


def test_fmriprep_command(tmp_path):
    config = FmriprepConfig(Path("/img/fmriprep-25.2.3.sif"), Path("/lic.txt"),
                            Path("/tf"), nprocs=8, mem_mb=16000)
    cmd = build_command("apptainer", config, Path("/raw/ds"), Path("/out/ds"),
                        Path("/work/ds/sub-01"), "01")
    joined = " ".join(cmd)
    assert cmd[:3] == ["apptainer", "run", "--cleanenv"]
    assert "/img/fmriprep-25.2.3.sif /raw/ds /out/ds participant --participant-label 01" in joined
    for flag in FMRIPREP_FLAGS:
        assert flag in joined
    assert "--nprocs 8 --mem-mb 16000 --work-dir /work/ds/sub-01" in joined


def test_fmriprep_done(tmp_path, fmriprep_tree):
    assert not fmriprep_done(fmriprep_tree, DATASET, SUBJECT)
    log = fmriprep_tree / "logs" / "preprocessing.jsonl"
    log.parent.mkdir()
    log.write_text(json.dumps({"dataset_id": DATASET, "subject": SUBJECT,
                               "status": "success"}) + "\n")
    assert fmriprep_done(fmriprep_tree, DATASET, SUBJECT)
    with log.open("a") as f:
        f.write(json.dumps({"dataset_id": DATASET, "subject": SUBJECT,
                            "status": "failed"}) + "\n")
    assert not fmriprep_done(fmriprep_tree, DATASET, SUBJECT)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_requires_fmriprep_inputs(tmp_path, monkeypatch):
    monkeypatch.delenv("FMRIPREP_IMAGE", raising=False)
    monkeypatch.delenv("FS_LICENSE", raising=False)
    with pytest.raises(SystemExit):
        cli.main([DATASET, SUBJECT, "--atlas", "aicha", "--data-root", str(tmp_path)])


def test_cli_unknown_atlas(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args([DATASET, SUBJECT, "--atlas", "nope", "--data-root", str(tmp_path),
                        "--steps", "parcellate"])


def test_cli_paths(tmp_path):
    args = cli.parse_args([DATASET, "sub-01", "--atlas", "aicha", "glasser",
                           "--data-root", str(tmp_path), "--steps", "parcellate", "denoise",
                           "--no-bandpass", "--gsr"])
    root = tmp_path.resolve()
    assert args.subject == "01"
    assert args.fmriprep_dir == root / "fmriprep"
    assert args.quasiraw_dir == root / "quasiraw" / "parcellated"
    assert (args.denoise.high_pass, args.denoise.low_pass, args.denoise.gsr) == (None, None, True)


def test_cli_parcellate_and_denoise(tmp_path, fmriprep_tree, toy_atlas):
    """The two last steps through the CLI, on an fMRIPrep tree under --data-root."""
    root = fmriprep_tree.parent
    log = root / "logs"
    log.mkdir()
    (log / "download.jsonl").write_text(json.dumps({
        "dataset_id": DATASET, "subject": SUBJECT, "status": "success",
        "snapshot_tag": "1.0.0", "openneuro_py_version": "2026.7.1"}) + "\n")
    rc = cli.main([DATASET, SUBJECT, "--atlas", toy_atlas, "--data-root", str(root),
                   "--steps", "parcellate", "denoise"])
    assert rc == 0
    den = next((root / "preproc" / "parcellated").rglob("*_desc-preproc_timeseries.json"))
    chain = json.loads(den.read_text())["GeneratedBy"]
    assert [g["Name"] for g in chain] == [
        "openneuro-py", "fMRIPrep", "open-fmind parcellate", "open-fmind denoise"]
    assert chain[0]["SnapshotTag"] == "1.0.0"
    assert "rebuilt from the flags" in chain[1]["Note"]


def test_dry_run_writes_nothing(tmp_path, toy_atlas):
    rc = cli.main([DATASET, SUBJECT, "--atlas", toy_atlas, "--data-root", str(tmp_path),
                   "--steps", "parcellate", "denoise", "--dry-run"])
    assert rc == 0
    assert list(tmp_path.iterdir()) == []


def test_pipeline_skips_finished_fmriprep(tmp_path, fmriprep_tree, monkeypatch, toy_atlas):
    root = fmriprep_tree.parent
    (fmriprep_tree / "logs").mkdir()
    (fmriprep_tree / "logs" / "preprocessing.jsonl").write_text(json.dumps(
        {"dataset_id": DATASET, "subject": SUBJECT, "status": "success",
         "command": "singularity run ..."}) + "\n")
    called = []
    monkeypatch.setattr(pipeline, "step_download", lambda a: called.append("download"))
    monkeypatch.setattr(pipeline, "step_fmriprep", lambda a: called.append("fmriprep"))
    rc = cli.main([DATASET, SUBJECT, "--atlas", toy_atlas, "--data-root", str(root),
                   "--fmriprep-image", "x.sif", "--fs-license", "x.txt"])
    assert rc == 0
    assert called == []


def test_download_through_file_tree(tmp_path, monkeypatch):
    """Fallback download: the subject is moved in, the DOI is recorded and logged."""
    from open_fmind import download

    def no_listing(*a, **k):
        raise RuntimeError("No func/ files in the flat listing.")

    def fake_tree(dataset_id, tag, subject, target, retries):
        func = target / f"sub-{subject}" / "func"
        func.mkdir(parents=True)
        (func / f"sub-{subject}_task-rest_bold.nii.gz").write_bytes(b"x" * 10)
        (target / "dataset_description.json").write_text(json.dumps({"Name": "toy"}))
        return 2

    monkeypatch.setattr(download, "subject_include_patterns", no_listing)
    monkeypatch.setattr(download, "tree_download_subject", fake_tree)
    raw, logs = tmp_path / "openneuro", tmp_path / "logs"
    download.download_subject(DATASET, SUBJECT, "1.0.0", raw, logs)

    assert (raw / DATASET / f"sub-{SUBJECT}" / "func"
            / f"sub-{SUBJECT}_task-rest_bold.nii.gz").exists()
    assert not (raw / ".staging" / f"{DATASET}_sub-{SUBJECT}").exists()
    desc = json.loads((raw / DATASET / "dataset_description.json").read_text())
    assert desc["DatasetDOI"] == f"10.18112/openneuro.{DATASET}.v1.0.0"
    record = json.loads((logs / "download.jsonl").read_text())
    assert (record["subject"], record["snapshot_tag"], record["size_bytes"],
            record["files_downloaded"]) == (SUBJECT, "1.0.0", 10, 1)


def test_connectivity_step(tmp_path, fmriprep_tree, toy_atlas):
    """Parcellation, denoising and connectivity through the CLI, as in the release:
    matrices from the raw and from the denoised timeseries."""
    from nilearn.connectome import ConnectivityMeasure

    root = fmriprep_tree.parent
    rc = cli.main([DATASET, SUBJECT, "--atlas", toy_atlas, "--data-root", str(root),
                   "--steps", "parcellate", "denoise", "connectivity"])
    assert rc == 0
    stem = f"sub-{SUBJECT}_task-rest_run-1_atlas-toy"
    for tree, desc in (("quasiraw", ""), ("preproc", "_desc-preproc")):
        func = root / tree / "functional-connectivity" / DATASET / f"sub-{SUBJECT}" / "func"
        nii = func / f"{stem}{desc}_measure-correlation_standardize-false_connectivity.nii.gz"
        mat = nib.load(nii).get_fdata()[:, :, 0, 0]
        ts_path = next((root / tree / "parcellated").rglob(f"{stem}{desc}_timeseries.nii.gz"))
        ts = nib.load(ts_path).get_fdata()[:, 0, 0, :]
        ref = ConnectivityMeasure(kind="correlation", standardize=False).fit_transform([ts.T])[0]
        np.testing.assert_allclose(mat, ref, atol=1e-5)
        chain = json.loads(nii.with_name(nii.name.replace(".nii.gz", ".json")).read_text())
        assert [g["Name"] for g in chain["GeneratedBy"]][-1] == "open-fmind connectivity"
        assert (func.parents[2] / "dataset_description.json").exists()


def test_connectivity_rejects_tangent(tmp_path):
    with pytest.raises(SystemExit):
        cli.parse_args([DATASET, SUBJECT, "--atlas", "aicha", "--data-root", str(tmp_path),
                        "--steps", "connectivity", "--connectivity-kind", "tangent"])
