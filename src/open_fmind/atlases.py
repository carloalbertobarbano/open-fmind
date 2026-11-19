"""Brain atlases used for parcellation.

Each entry of ATLAS_REGISTRY returns (masker_class, atlas_img, roi_labels,
masker_kwargs). Atlases come from nilearn's fetchers (cached in nilearn's data
directory) or are downloaded once into ~/.cache/open-fmind/atlases.
"""
from __future__ import annotations

import contextlib
import io
import os
import ssl
import sys
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

_ATLAS_CACHE_DIR = Path.home() / ".cache" / "open-fmind" / "atlases"
_TIAN_BASE_URL = (
    "https://github.com/yetianmed/subcortex/raw/master/"
    "Group-Parcellation/3T/Subcortex-Only/"
)
_AICHA_URL_V2 = "https://www.gin.cnrs.fr/wp-content/uploads/AICHA_v2.tar.zip"
# Glasser HCP MMP 1.0 from Glasser et al. 2016 (figshare #3501911, MNI2009a space)
_GLASSER_NII_URL = "https://ndownloader.figshare.com/files/5594363"
_GLASSER_LBL_URL = "https://ndownloader.figshare.com/files/5534027"
# Yeo 2011 networks. nilearn defaults to an ftp:// URL that times out on many
# HPC/compute nodes; the same host also serves the file over HTTPS.
_YEO_MNI152_URL = (
    "https://surfer.nmr.mgh.harvard.edu/pub/data/Yeo_JNeurophysiol11_MNI152.zip"
)
# Yan 2023 homotopic parcellations (ThomasYeoLab/CBIG), MNI / FSL MNI152 space
_YAN_BASE_URL = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Yan2023_homotopic/parcellations/MNI/"
)

# ---------------------------------------------------------------------------
# Atlas loading helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _no_ssl_verify():
    """Temporarily disable SSL certificate verification.

    Needed on some HPC systems where the CA certificate chain is incomplete.
    Patches both stdlib ssl and requests.adapters.HTTPAdapter.send (used by
    nilearn's downloader). Only applied for specific GIN downloads.
    """
    old_ssl = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    old_send = None
    try:
        import urllib3
        import requests.adapters
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        old_send = requests.adapters.HTTPAdapter.send

        def _unverified_send(self, request, **kwargs):
            kwargs["verify"] = False
            return old_send(self, request, **kwargs)

        requests.adapters.HTTPAdapter.send = _unverified_send
    except ImportError:
        pass
    try:
        yield
    finally:
        ssl._create_default_https_context = old_ssl
        if old_send is not None:
            import requests.adapters
            requests.adapters.HTTPAdapter.send = old_send


def _strip_background(labels: list) -> list[str]:
    """Remove a leading 'Background' or empty label returned by some atlases."""
    return [str(lb) for lb in labels if str(lb) not in ("Background", "")]


def _load_schaefer(n_rois: int, yeo_networks: int = 7) -> tuple:
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.maskers import NiftiLabelsMasker
    atlas = fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=yeo_networks, resolution_mm=2)
    labels = _strip_background(atlas.labels)
    return NiftiLabelsMasker, atlas.maps, labels, {}


def _load_difumo(dimension: int) -> tuple:
    from nilearn.datasets import fetch_atlas_difumo
    from nilearn.maskers import NiftiMapsMasker
    atlas = fetch_atlas_difumo(dimension=dimension, resolution_mm=2)
    labels = atlas.labels["difumo_names"].tolist()
    return NiftiMapsMasker, atlas.maps, labels, {}


def _load_aal() -> tuple:
    from nilearn.datasets import fetch_atlas_aal
    from nilearn.maskers import NiftiLabelsMasker
    with _no_ssl_verify():
        atlas = fetch_atlas_aal()
    labels = _strip_background(atlas.labels)
    return NiftiLabelsMasker, atlas.maps, labels, {}


def _load_harvardoxford(atlas_type: str) -> tuple:
    """atlas_type: 'cort' or 'sub'."""
    from nilearn.datasets import fetch_atlas_harvard_oxford
    from nilearn.maskers import NiftiLabelsMasker
    name = "cort-maxprob-thr25-2mm" if atlas_type == "cort" else "sub-maxprob-thr25-2mm"
    atlas = fetch_atlas_harvard_oxford(name)
    labels = _strip_background(atlas.labels)
    return NiftiLabelsMasker, atlas.maps, labels, {}


def _load_destrieux() -> tuple:
    from nilearn.datasets import fetch_atlas_destrieux_2009
    from nilearn.maskers import NiftiLabelsMasker
    atlas = fetch_atlas_destrieux_2009()
    raw = list(atlas.labels)
    # Labels can be list of strings or list of (index, name) tuples
    if raw and isinstance(raw[0], (list, tuple)):
        labels = _strip_background([item[1] for item in raw])
    else:
        labels = _strip_background(raw)
    return NiftiLabelsMasker, atlas.maps, labels, {}


def _load_yeo(networks: int) -> tuple:
    """networks: 7 or 17.  Uses the MNI152 'thick' liberal mask (1mm, resampled by masker)."""
    from nilearn.datasets import fetch_atlas_yeo_2011
    from nilearn.maskers import NiftiLabelsMasker
    # Use the HTTPS mirror instead of nilearn's default ftp:// URL, which times
    # out on many networks. Same archive/host, just a reachable protocol.
    atlas = fetch_atlas_yeo_2011(n_networks=networks, url=_YEO_MNI152_URL)
    img = atlas.maps
    if networks == 7:
        labels = [
            "Visual", "Somatomotor", "DorsalAttention",
            "VentralAttention", "Limbic", "Frontoparietal", "Default",
        ]
    else:
        labels = [
            "VisPeri", "VisCent",
            "SomMot_A", "SomMot_B",
            "DorsAttn_Post", "DorsAttn_FEF",
            "SalVentAttn_TempOccPar", "SalVentAttn_FrOper",
            "Limbic_TempPole", "Limbic_OFC",
            "Cont_Par", "Cont_PFC", "Cont_Temp",
            "Default_Par", "Default_PFC", "Default_PFCv", "Default_Temp",
        ]
    return NiftiLabelsMasker, img, labels, {}


def _load_msdl() -> tuple:
    from nilearn.datasets import fetch_atlas_msdl
    from nilearn.maskers import NiftiMapsMasker
    atlas = fetch_atlas_msdl()
    labels = list(atlas.labels)
    return NiftiMapsMasker, atlas.maps, labels, {}


def _load_basc(n_clusters: int) -> tuple:
    """n_clusters: 64, 122, 197, 325, or 444."""
    from nilearn.datasets import fetch_atlas_basc_multiscale_2015
    from nilearn.maskers import NiftiLabelsMasker
    atlas = fetch_atlas_basc_multiscale_2015(resolution=n_clusters, version="sym")
    labels = [f"BASC{n_clusters:03d}_{i:03d}" for i in range(1, n_clusters + 1)]
    return NiftiLabelsMasker, atlas.maps, labels, {}


def _load_smith(n_components: int) -> tuple:
    """n_components: 10, 20, or 70 (resting-state ICA networks)."""
    from nilearn.datasets import fetch_atlas_smith_2009
    from nilearn.maskers import NiftiMapsMasker
    atlas = fetch_atlas_smith_2009(dimension=n_components)
    labels = [f"Smith{n_components:02d}_IC{i:02d}" for i in range(1, n_components + 1)]
    return NiftiMapsMasker, atlas.maps, labels, {}


def _load_pauli() -> tuple:
    from nilearn.datasets import fetch_atlas_pauli_2017
    from nilearn.maskers import NiftiLabelsMasker
    atlas = fetch_atlas_pauli_2017(atlas_type="deterministic")
    labels = _strip_background(atlas.labels)
    return NiftiLabelsMasker, atlas.maps, labels, {}


def _download_file(url: str, dest: Path, verify_ssl: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"  Downloading {dest.name} ...", file=sys.stderr)
        ctx = ssl.create_default_context() if verify_ssl else ssl._create_unverified_context()
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        with opener.open(url) as resp, open(dest, "wb") as f:
            f.write(resp.read())


def _load_tian(scale: int) -> tuple:
    """Tian subcortex atlas (S1–S4) in MNI152NLin2009cAsym 2mm space."""
    from nilearn.maskers import NiftiLabelsMasker
    cache = _ATLAS_CACHE_DIR
    nii_name = f"Tian_Subcortex_S{scale}_3T_2009cAsym.nii.gz"
    lbl_name = f"Tian_Subcortex_S{scale}_3T_label.txt"
    nii_path = cache / nii_name
    lbl_path = cache / lbl_name
    _download_file(_TIAN_BASE_URL + nii_name, nii_path)
    _download_file(_TIAN_BASE_URL + lbl_name, lbl_path)
    labels = lbl_path.read_text().strip().splitlines()
    return NiftiLabelsMasker, str(nii_path), labels, {}


def _yan_read_labels(lbl_path: Path) -> list[str]:
    """Parse a Yan2023 fsleyes LUT: 'index r g b name' per line; name is column 5."""
    labels = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            labels.append(parts[4])
    return labels


def _load_yan(n_rois: int, network: str) -> tuple:
    """Yan 2023 homotopic atlas. network: 'yeo7', 'yeo17', or 'kong17'."""
    from nilearn.maskers import NiftiLabelsMasker
    net_tags = {
        "yeo7":   "Yeo2011_7Networks",
        "yeo17":  "Yeo2011_17Networks",
        "kong17": "Kong2022_17Networks",
    }
    if network not in net_tags:
        raise ValueError(f"Unknown Yan network mapping {network!r}")
    tag = net_tags[network]
    nii_name = f"{n_rois}Parcels_{tag}_FSLMNI152_2mm.nii.gz"
    lbl_name = f"{n_rois}Parcels_{tag}_LUT.txt"
    nii_path = _ATLAS_CACHE_DIR / nii_name
    lbl_path = _ATLAS_CACHE_DIR / lbl_name
    _download_file(f"{_YAN_BASE_URL}{network}/{nii_name}", nii_path)
    _download_file(f"{_YAN_BASE_URL}{network}/fsleyes_lut/{lbl_name}", lbl_path)
    labels = _yan_read_labels(lbl_path)
    return NiftiLabelsMasker, str(nii_path), labels, {}


class _MultiLabelsMasker:
    """N independent NiftiLabelsMaskers, one per label image, whose outputs are
    concatenated column-wise on fit_transform.

    Used to combine atlases (e.g. Schaefer cortex + Tian subcortex) instead of
    merging their label images into one atlas on a shared grid: two label
    images that are each independently nearest-neighbor-resampled onto a
    common grid are not guaranteed to stay mutually exclusive, and a merge
    that just adds the (shifted) label arrays turns any voxel where both
    happen to be nonzero into a spurious label — one that matches none of the
    combined atlas's names, but that NiftiLabelsMasker (which emits one output
    column per unique nonzero label) still turns into a real, uncounted-for
    output column. Fitting each image with its own masker sidesteps this
    entirely, since neither masker ever sees the other's labels.
    """

    def __init__(self, imgs, **masker_kwargs):
        from nilearn.maskers import NiftiLabelsMasker
        self._maskers = [NiftiLabelsMasker(img, **masker_kwargs) for img in imgs]

    def fit_transform(self, bold_path):
        import numpy as np
        return np.concatenate(
            [masker.fit_transform(bold_path) for masker in self._maskers], axis=1,
        )


def _load_schaefer_tian(n_rois: int, yeo_networks: int, tian_scale: int) -> tuple:
    """Schaefer cortical + Tian subcortical combined atlas.

    See _MultiLabelsMasker for why this fits each atlas with its own masker
    rather than merging them into one label image first.
    """
    _, schaefer_img, schaefer_labels, _ = _load_schaefer(n_rois, yeo_networks)
    _, tian_img, tian_labels, _ = _load_tian(tian_scale)
    combined_labels = schaefer_labels + tian_labels
    return _MultiLabelsMasker, (schaefer_img, tian_img), combined_labels, {}


def _extract_aicha(archive: Path, dest: Path) -> None:
    """Extract the AICHA archive (zip-of-tar or plain zip/tar) to dest."""
    try:
        with zipfile.ZipFile(archive) as zf:
            tar_members = [n for n in zf.namelist() if n.endswith((".tar", ".tar.gz"))]
            if tar_members:
                tar_bytes = zf.read(tar_members[0])
                with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
                    tf.extractall(dest, filter="data")
            else:
                zf.extractall(dest)
    except zipfile.BadZipFile:
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")


def _save_atomic(img, dest: Path) -> None:
    """Write a NIfTI through a temporary file in the same directory, so parallel
    jobs building the same cached image never read a half-written file."""
    tmp = dest.with_name(f".{dest.name.removesuffix('.nii.gz')}.{os.getpid()}.nii.gz")
    img.to_filename(str(tmp))
    os.replace(tmp, dest)


def _aicha_read_labels(table: Path) -> dict[int, str]:
    """{label value: region name} from an AICHA *_vol3.txt table.

    The table has one row per region; 'color' is the label value in the image
    and 'nom_c' the region name (e.g. G_Frontal_Sup-1-L).
    """
    t = pd.read_csv(table, sep="\t")
    return {int(v): str(n) for v, n in zip(t["color"], t["nom_c"])}


def _load_aicha() -> tuple:
    """AICHA v2 atlas, 384 regions (192 per hemisphere), downloaded from GIN.

    Uses AICHA1mm.nii, where each hemisphere has its own labels. The archive
    also ships AICHAmc*.nii.gz, a version where left and right homologues share
    one label (192 regions), which must not be used here. The 1 mm image stores
    labels as float with NaN outside the brain, so it is rewritten once as an
    integer image with 0 as background.
    """
    import nibabel as nib
    import numpy as np
    from nilearn.maskers import NiftiLabelsMasker
    cache = _ATLAS_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    src = next(iter(cache.rglob("AICHA1mm.nii")), None)
    table = next(iter(cache.rglob("AICHA1mm_vol3.txt")), None)
    if src is None or table is None:
        archive = cache / "AICHA_v2.tar.zip"
        # GIN website requires SSL verification bypass on some HPC systems
        _download_file(_AICHA_URL_V2, archive, verify_ssl=False)
        _extract_aicha(archive, cache)
        src = next(iter(cache.rglob("AICHA1mm.nii")), None)
        table = next(iter(cache.rglob("AICHA1mm_vol3.txt")), None)
    if src is None or table is None:
        raise RuntimeError(
            f"AICHA1mm.nii or AICHA1mm_vol3.txt not found after extraction in {cache}."
        )

    nii_int = cache / "AICHA1mm_lr384_int.nii.gz"
    if not nii_int.exists():
        raw = nib.load(str(src))
        data = np.round(np.nan_to_num(np.asarray(raw.dataobj, dtype=np.float64))).astype(np.int16)
        img = nib.Nifti1Image(data, raw.affine)
        img.header.set_data_dtype(np.int16)
        _save_atomic(img, nii_int)

    names = _aicha_read_labels(table)
    values = np.unique(np.asarray(nib.load(str(nii_int)).dataobj))
    values = [int(v) for v in values if v != 0]
    if values != sorted(names) or len(values) != 384:
        raise RuntimeError(
            f"AICHA image has {len(values)} labels, table {len(names)}; expected 384."
        )
    # NiftiLabelsMasker returns one column per label value in increasing order.
    labels = [names[v] for v in values]
    return NiftiLabelsMasker, str(nii_int), labels, {}


def _glasser_parse_labels(lbl_path: Path) -> list[str]:
    """Parse the Glasser label .txt file (index<tab>name per line)."""
    labels = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            labels.append(parts[1])
        elif parts:
            labels.append(parts[0])
    return labels


def _glasser_split_hemispheres(data, affine):
    """Give right-hemisphere voxels of the bilateral Glasser volume their own labels.

    The figshare volume uses labels 1-180 for both hemispheres. Voxels with
    x > 0 (RAS) keep their region but get label + 180, so the result has
    1-180 on the left and 181-360 on the right. Voxels exactly on the midline
    (x = 0) go to the side where the same region has more voxels among their
    26 neighbours; ties are set to background.
    """
    import nibabel as nib
    import numpy as np
    labels = np.round(np.nan_to_num(np.asarray(data, dtype=np.float64))).astype(np.int32)
    ijk = np.indices(labels.shape).reshape(3, -1).T
    x = nib.affines.apply_affine(affine, ijk)[:, 0].reshape(labels.shape)
    out = labels.copy()
    out[(x > 0) & (labels > 0)] += 180

    padded = np.pad(labels, 1)
    padded_x = np.pad(x, 1, constant_values=np.nan)
    for i, j, k in np.argwhere((x == 0) & (labels > 0)):
        lab = labels[i, j, k]
        block = padded[i:i + 3, j:j + 3, k:k + 3]
        block_x = padded_x[i:i + 3, j:j + 3, k:k + 3]
        n_left = np.sum((block == lab) & (block_x < 0))
        n_right = np.sum((block == lab) & (block_x > 0))
        if n_left > n_right:
            out[i, j, k] = lab
        elif n_right > n_left:
            out[i, j, k] = lab + 180
        else:
            out[i, j, k] = 0
    return out


def _load_glasser() -> tuple:
    """Glasser HCP MMP 1.0 volumetric atlas, 360 regions (180 per hemisphere).

    Source: Glasser et al. 2016, figshare #3501911 (MNI152 ICBM 2009a nlin).
    That volume uses labels 1-180 for both hemispheres, so it is split by
    hemisphere (see _glasser_split_hemispheres): labels 1-180 are the left
    regions (L_* names from the label file) and 181-360 the right ones (R_*).
    """
    import nibabel as nib
    import numpy as np
    from nilearn.maskers import NiftiLabelsMasker
    cache = _ATLAS_CACHE_DIR
    nii_raw = cache / "HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz"
    nii_lr = cache / "HCP-MMP1_on_MNI152_ICBM2009a_nlin_lr360_int.nii.gz"
    lbl_path = cache / "HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt"
    _download_file(_GLASSER_NII_URL, nii_raw, verify_ssl=False)
    _download_file(_GLASSER_LBL_URL, lbl_path, verify_ssl=False)
    if not nii_lr.exists():
        raw = nib.load(str(nii_raw))
        data = _glasser_split_hemispheres(raw.dataobj, raw.affine).astype(np.int16)
        img = nib.Nifti1Image(data, raw.affine)
        img.header.set_data_dtype(np.int16)
        _save_atomic(img, nii_lr)

    left = _glasser_parse_labels(lbl_path)
    if len(left) != 180 or not all(n.startswith("L_") for n in left):
        raise RuntimeError(f"Unexpected Glasser label file {lbl_path}: {len(left)} names.")
    labels = left + ["R_" + n[2:] for n in left]
    values = np.unique(np.asarray(nib.load(str(nii_lr)).dataobj))
    if [int(v) for v in values if v != 0] != list(range(1, 361)):
        raise RuntimeError("Glasser image does not have labels 1-360 after the hemisphere split.")
    return NiftiLabelsMasker, str(nii_lr), labels, {}


ATLAS_REGISTRY: dict[str, Callable[[], tuple]] = {
    # Schaefer 2018 — 7 Yeo networks
    "schaefer100":  lambda: _load_schaefer(100, 7),
    "schaefer200":  lambda: _load_schaefer(200, 7),
    "schaefer300":  lambda: _load_schaefer(300, 7),
    "schaefer400":  lambda: _load_schaefer(400, 7),
    "schaefer500":  lambda: _load_schaefer(500, 7),
    "schaefer600":  lambda: _load_schaefer(600, 7),
    "schaefer800":  lambda: _load_schaefer(800, 7),
    "schaefer1000": lambda: _load_schaefer(1000, 7),
    # Schaefer 2018 — 17 Yeo networks
    "schaefer10017n":  lambda: _load_schaefer(100, 17),
    "schaefer20017n":  lambda: _load_schaefer(200, 17),
    "schaefer40017n":  lambda: _load_schaefer(400, 17),
    "schaefer100017n": lambda: _load_schaefer(1000, 17),
    # DiFuMo probabilistic ICA atlases
    "difumo64":   lambda: _load_difumo(64),
    "difumo128":  lambda: _load_difumo(128),
    "difumo256":  lambda: _load_difumo(256),
    "difumo512":  lambda: _load_difumo(512),
    "difumo1024": lambda: _load_difumo(1024),
    # Anatomical
    "aal":       lambda: _load_aal(),
    "destrieux": lambda: _load_destrieux(),
    # Harvard-Oxford
    "harvardoxfordcort": lambda: _load_harvardoxford("cort"),
    "harvardoxfordsub":  lambda: _load_harvardoxford("sub"),
    # Yeo 2011 functional networks
    "yeo7":  lambda: _load_yeo(7),
    "yeo17": lambda: _load_yeo(17),
    # Probabilistic / ICA
    "msdl":    lambda: _load_msdl(),
    "smith10": lambda: _load_smith(10),
    "smith20": lambda: _load_smith(20),
    "smith70": lambda: _load_smith(70),
    # BASC multiscale 2015
    "basc064": lambda: _load_basc(64),
    "basc122": lambda: _load_basc(122),
    "basc197": lambda: _load_basc(197),
    "basc325": lambda: _load_basc(325),
    "basc444": lambda: _load_basc(444),
    # Subcortical
    "pauli":  lambda: _load_pauli(),
    # Tian subcortex (web download — yetianmed/subcortex)
    "tians1": lambda: _load_tian(1),   # 16 regions
    "tians2": lambda: _load_tian(2),   # 32 regions
    "tians3": lambda: _load_tian(3),   # 50 regions
    # Schaefer cortex + Tian subcortex combined
    "schaefer400tians2": lambda: _load_schaefer_tian(400, 7, 2),  # 432 regions — BrainJEPA
    "schaefer400tians3": lambda: _load_schaefer_tian(400, 7, 3),  # 450 regions
    # Homotopic atlases (web download)
    "aicha":   lambda: _load_aicha(),    # 384 homotopic regions, AICHA v2
    "glasser": lambda: _load_glasser(),  # 360 cortical regions, HCP MMP 1.0
    # Yan 2023 homotopic (web download — ThomasYeoLab/CBIG)
    "yan100_yeo7":   lambda: _load_yan(100, "yeo7"),
    "yan200_yeo7":   lambda: _load_yan(200, "yeo7"),
    "yan400_yeo7":   lambda: _load_yan(400, "yeo7"),
    "yan800_yeo7":   lambda: _load_yan(800, "yeo7"),
    "yan100_yeo17":  lambda: _load_yan(100, "yeo17"),
    "yan200_yeo17":  lambda: _load_yan(200, "yeo17"),
    "yan400_yeo17":  lambda: _load_yan(400, "yeo17"),
    "yan800_yeo17":  lambda: _load_yan(800, "yeo17"),
    "yan100_kong17": lambda: _load_yan(100, "kong17"),
    "yan200_kong17": lambda: _load_yan(200, "kong17"),
    "yan400_kong17": lambda: _load_yan(400, "kong17"),
    "yan800_kong17": lambda: _load_yan(800, "kong17"),
}


# ---------------------------------------------------------------------------
# Atlas loading
# ---------------------------------------------------------------------------
def load_atlas(atlas_name: str) -> tuple:
    """Return (masker_class, atlas_img, roi_labels, masker_kwargs).

    Pass atlas_name='all' to iterate over all atlases via ATLAS_REGISTRY.
    """
    if atlas_name not in ATLAS_REGISTRY:
        valid = "\n  ".join(sorted(ATLAS_REGISTRY))
        raise ValueError(
            f"Unknown atlas {atlas_name!r}. Valid choices:\n  {valid}"
        )
    return ATLAS_REGISTRY[atlas_name]()


def resolve_atlas_names(atlas_args: list[str] | None) -> list[str]:
    """Expand a --atlas nargs="+" value into concrete ATLAS_REGISTRY names.

    None, an empty list, or a list containing 'all' means every atlas.
    Otherwise the given names are used as-is (order preserved, so a caller can
    group e.g. only the still-incomplete atlases into one job) after checking
    they're all known. Raises ValueError listing the unknown ones.
    """
    if not atlas_args or "all" in atlas_args:
        return sorted(ATLAS_REGISTRY)
    unknown = [a for a in atlas_args if a not in ATLAS_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown atlas name(s): {unknown}. "
            f"Valid choices:\n  {sorted(ATLAS_REGISTRY)}"
        )
    return list(atlas_args)

