# Open-fMIND

Code to build Open-fMIND, a dataset of parcellated fMRI timeseries from public [OpenNeuro](https://openneuro.org/) datasets.

For one subject of an OpenNeuro dataset, `open-fmind` downloads the raw data, preprocesses it with fMRIPrep 25.2.3, averages the BOLD signal within the regions of one or more brain atlases, and denoises the resulting timeseries.

## Requirements

- Linux, Python >= 3.12
- [Singularity](https://sylabs.io/singularity/) or [Apptainer](https://apptainer.org/), with an fMRIPrep 25.2.3 image:
  ```bash
  singularity build fmriprep-25.2.3.sif docker://nipreps/fmriprep:25.2.3
  ```
- A [FreeSurfer license](https://surfer.nmr.mgh.harvard.edu/registration.html) (free), which fMRIPrep requires even without surface reconstruction
- Internet access for the download step, for TemplateFlow templates fetched by fMRIPrep, and for the atlases on first use

## Installation

```bash
git clone https://github.com/carloalbertobarbano/open-fmind
cd open-fmind
uv sync
```

or, with pip, `pip install .` in a virtual environment.

## Usage

```bash
uv run open-fmind ds000030 10159 \
    --atlas aicha glasser \
    --data-root /path/to/open-fmind \
    --fmriprep-image /path/to/fmriprep-25.2.3.sif \
    --fs-license /path/to/license.txt
```

`--fmriprep-image` and `--fs-license` can also be given through the `FMRIPREP_IMAGE` and `FS_LICENSE` environment variables.

The command runs five steps in order:

| Step | What it does | Output (under `--data-root`) |
|---|---|---|
| `download` | Downloads the subject's `func/` and `anat/` files from OpenNeuro, pinned to one snapshot of the dataset | `openneuro/` |
| `fmriprep` | Runs fMRIPrep (MNI152NLin2009cAsym at 2 mm, no FreeSurfer surface reconstruction) | `fmriprep/` |
| `parcellate` | Averages the preprocessed BOLD within each atlas region, without any denoising | `quasiraw/parcellated/` |
| `denoise` | Confound regression, scrubbing and band-pass filtering of the parcellated timeseries | `preproc/parcellated/` |
| `connectivity` | One correlation matrix per run and atlas, from the raw and from the denoised timeseries | `quasiraw/functional-connectivity/`, `preproc/functional-connectivity/` |

The download and fMRIPrep steps are skipped when their outputs are already on disk. The later steps regenerate the requested atlases each time, unless `--no-overwrite` is given. Download and fMRIPrep records are appended to `logs/download.jsonl` and `fmriprep/logs/preprocessing.jsonl`; the command writes nothing outside `--data-root`, except the atlas cache (see below) and the TemplateFlow cache (`--templateflow-home`, default `~/.cache/templateflow`).

Useful options:

- `--steps parcellate denoise` runs a subset of the steps.
- `--dry-run` prints what each step would do.
- `--snapshot-tag TAG` sets the OpenNeuro snapshot to download. By default, a subject is downloaded at the version of the dataset already on disk, so all subjects of a dataset come from the same snapshot; the latest snapshot is used for a new dataset.
- `--atlas all` uses every atlas listed below.
- `--nprocs` and `--mem-mb` set the resources of fMRIPrep (default: `$SLURM_CPUS_PER_TASK` CPUs, or 16, and 32 GB).

To process many subjects, run one `open-fmind` command per subject, for example as the tasks of a SLURM job array. Several subjects of the same dataset can be processed at the same time.

## Denoising

The denoising step regresses out the 24 motion parameters (6 parameters, their derivatives and the squares of both) and the mean white-matter and CSF signals, removes frames with framewise displacement above 0.5 mm, removes the linear trend, applies a 0.01 to 0.1 Hz band-pass filter, and z-scores each region. All of it is a single call to `nilearn.signal.clean`, with the frames kept after scrubbing given as `sample_mask`. Runs with fewer than 50 frames left after scrubbing have no denoised file.

All of these are options of the command (`--gsr`, `--acompcor N`, `--fd-threshold`, `--no-bandpass`, `--standardize`, ...; see `open-fmind --help`). Since the parcellated timeseries are kept raw, with the fMRIPrep confounds file next to them, other denoising strategies can be applied afterwards without running fMRIPrep again.

## Connectivity

The connectivity step computes one matrix per run and atlas with nilearn's `ConnectivityMeasure`, once from the raw timeseries and once from the denoised ones. The measure is the correlation by default; `--connectivity-kind` selects `partial_correlation`, `covariance` or `precision` instead, and `--connectivity-standardize` z-scores the timeseries first. Each of these measures is computed from one run alone. Group-level measures such as the tangent space projection need all subjects together and are not part of this command.

## Atlases

| Family | Names |
|---|---|
| Schaefer 2018, 7 networks | `schaefer100`, `schaefer200`, `schaefer300`, `schaefer400`, `schaefer500`, `schaefer600`, `schaefer800`, `schaefer1000` |
| Schaefer 2018, 17 networks | `schaefer10017n`, `schaefer20017n`, `schaefer40017n`, `schaefer100017n` |
| Schaefer 400 + Tian subcortex | `schaefer400tians2`, `schaefer400tians3` |
| Tian subcortex | `tians1`, `tians2`, `tians3` |
| Yan 2023 homotopic | `yan{100,200,400,800}_{yeo7,yeo17,kong17}` |
| DiFuMo | `difumo64`, `difumo128`, `difumo256`, `difumo512`, `difumo1024` |
| BASC 2015 | `basc064`, `basc122`, `basc197`, `basc325`, `basc444` |
| Smith 2009 | `smith10`, `smith20`, `smith70` |
| Yeo 2011 | `yeo7`, `yeo17` |
| Anatomical | `aicha`, `glasser`, `aal`, `destrieux`, `harvardoxfordcort`, `harvardoxfordsub`, `pauli` |
| Other | `msdl` |

Atlases are downloaded on first use, through nilearn's fetchers (stored in nilearn's data directory, `~/nilearn_data` or `$NILEARN_DATA`) or into `~/.cache/open-fmind/atlases`. On machines without internet access, run the parcellation once on a machine that has it, so that the caches are filled. The AICHA, Glasser and AAL downloads skip SSL certificate verification, because the certificate chains of their hosts are incomplete on some HPC systems.

## Output format

Outputs follow BIDS derivative naming, one file per run and atlas:

```
quasiraw/parcellated/{dataset}/sub-{sub}/[ses-{ses}/]func/
    sub-{sub}_[ses-{ses}_]task-{task}_[run-{run}_]atlas-{atlas}_timeseries.nii.gz
    sub-{sub}_[ses-{ses}_]task-{task}_[run-{run}_]atlas-{atlas}_timeseries.json
    sub-{sub}_[ses-{ses}_]task-{task}_[run-{run}_]desc-confounds_timeseries.tsv
preproc/parcellated/{dataset}/sub-{sub}/[ses-{ses}/]func/
    sub-{sub}_[ses-{ses}_]task-{task}_[run-{run}_]atlas-{atlas}_desc-preproc_timeseries.nii.gz
    sub-{sub}_[ses-{ses}_]task-{task}_[run-{run}_]atlas-{atlas}_desc-preproc_timeseries.json
```

```
quasiraw/functional-connectivity/{dataset}/sub-{sub}/[ses-{ses}/]func/
    sub-{sub}_[ses-{ses}_]task-{task}_[run-{run}_]atlas-{atlas}_measure-correlation_standardize-false_connectivity.nii.gz
preproc/functional-connectivity/{dataset}/sub-{sub}/[ses-{ses}/]func/
    sub-{sub}_[ses-{ses}_]task-{task}_[run-{run}_]atlas-{atlas}_desc-preproc_measure-correlation_standardize-false_connectivity.nii.gz
```

Timeseries are float32 NIfTI images of shape `(n_rois, 1, 1, n_timepoints)`, with the repetition time in the header, and connectivity matrices float32 images of shape `(n_rois, n_rois, 1, 1)`. The JSON sidecar of each file gives the region labels, the repetition time, what was applied to the signal, and a `GeneratedBy` chain recording the OpenNeuro snapshot, the fMRIPrep command and the Open-fMIND version that produced it.

## License

MIT, see [LICENSE](LICENSE).
