"""Download of one subject from OpenNeuro.

The subject's func/ and anat/ folders and the JSON files it inherits metadata
from are fetched with openneuro-py, pinned to one snapshot of the dataset. When
openneuro-py cannot do it (some snapshots list only part of their files, and
OpenNeuro sometimes answers HTTP 500 on single files), the files are taken
from the snapshot's file tree through the GraphQL API instead.
"""
from __future__ import annotations

import fcntl
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import provenance

OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"

# Not read by fMRIPrep or by the later steps; OpenNeuro sometimes answers HTTP
# 500 on them, which fails the whole download.
DOWNLOAD_EXCLUDE = ["**/*_events.tsv"]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# openneuro-py network patch
# ---------------------------------------------------------------------------
_PATCHED = False


def patch_niquests() -> None:
    """Make openneuro-py work behind HTTP proxies and with GET-only S3 URLs.

    openneuro-py (>=2026.7) creates niquests sessions with happy_eyeballs=True,
    which fails behind an HTTP forward proxy without working IPv6 instead of
    falling back to IPv4. It also probes every file with a HEAD request and
    gives up on a 403, which some S3 presigned URLs (signed for GET only)
    return although the GET succeeds. Such responses are passed on as 200; the
    file size is still checked by openneuro-py.
    """
    global _PATCHED
    if _PATCHED:
        return
    import niquests

    class _NoHappyEyeballsSession(niquests.Session):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs["happy_eyeballs"] = False
            super().__init__(*args, **kwargs)

    class _NoHappyEyeballsAsyncSession(niquests.AsyncSession):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs["happy_eyeballs"] = False
            super().__init__(*args, **kwargs)

        async def head(self, url: str, **kwargs: object):
            response = await super().head(url, **kwargs)
            if response.status_code == 403:
                log("  warning:  HEAD returned 403 (GET-only S3 signature); "
                    f"downloading anyway: {url.split('?', 1)[0]}")
                response.status_code = 200
            return response

    niquests.Session = _NoHappyEyeballsSession
    niquests.AsyncSession = _NoHappyEyeballsAsyncSession
    _PATCHED = True


# ---------------------------------------------------------------------------
# OpenNeuro GraphQL
# ---------------------------------------------------------------------------
def graphql(query: str, retries: int = 5) -> dict:
    req = urllib.request.Request(
        OPENNEURO_GRAPHQL, data=json.dumps({"query": query}).encode(),
        headers={"content-type": "application/json", "user-agent": "open-fmind"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            if data.get("errors"):
                raise OSError(str(data["errors"])[:300])
            return data["data"]
        except OSError:
            if attempt == retries:
                raise
            time.sleep(min(60, 2 ** attempt))


def latest_snapshot_tag(dataset_id: str) -> str | None:
    """The dataset's latest snapshot tag, or None if it cannot be retrieved."""
    try:
        data = graphql(f'{{dataset(id:"{dataset_id}"){{latestSnapshot{{tag}}}}}}')
    except OSError as e:
        log(f"  warning:  could not get the latest snapshot of {dataset_id}: {e}")
        return None
    return (((data or {}).get("dataset") or {}).get("latestSnapshot") or {}).get("tag")


# ---------------------------------------------------------------------------
# Raw files on disk
# ---------------------------------------------------------------------------
def raw_bold_files(raw_dataset: Path, subject: str) -> list[Path]:
    sub_dir = raw_dataset / f"sub-{subject}"
    return [
        p for pattern in ("func/*_bold.nii*", "ses-*/func/*_bold.nii*")
        for p in sub_dir.glob(pattern)
    ]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Snapshot to download
# ---------------------------------------------------------------------------
def _tag_from_doi(doi: str | None, dataset_id: str) -> str | None:
    """'10.18112/openneuro.ds000001.v1.0.0' (or doi:/https://doi.org/ forms) -> '1.0.0'."""
    if not doi:
        return None
    marker = f"openneuro.{dataset_id}.v"
    _, found, tag = doi.partition(marker)
    if not found:
        return None
    return tag.strip() or None


def _tag_from_generated_by(meta: dict) -> str | None:
    for entry in meta.get("GeneratedBy") or []:
        if isinstance(entry, dict) and entry.get("Name") == "openneuro-py":
            return entry.get("SnapshotTag")
    return None


def first_sidecar(dataset_dir: Path, avoid_subject: str | None = None) -> Path | None:
    """One run sidecar of the dataset, without walking the whole tree.

    A sidecar of another subject is preferred over one of `avoid_subject`.
    """
    fallback = None
    for sub_dir in dataset_dir.glob("sub-*"):
        found = None
        for pattern in ("func/*_timeseries.json", "ses-*/func/*_timeseries.json"):
            found = next(sub_dir.glob(pattern), None)
            if found is not None:
                break
        if found is None:
            continue
        if sub_dir.name != f"sub-{avoid_subject}":
            return found
        fallback = found
    return fallback


def snapshot_tags_on_disk(
    dataset_id: str, raw_root: Path, fmriprep_dir: Path, parcellated_dirs: dict[str, Path],
) -> dict[str, str]:
    """Snapshot tag recorded for the dataset in each output tree, {source: tag}.

    - raw:      DatasetDOI of the downloaded dataset_description.json
    - fmriprep: SourceDatasets DOI, which fMRIPrep copies from the raw DatasetDOI
    - parcellated trees: SourceDatasets Version of the dataset's
      dataset_description.json, and the openneuro-py SnapshotTag in the
      GeneratedBy chain of a run sidecar
    """
    ds = dataset_id
    tags: dict[str, str] = {}

    raw = _read_json(raw_root / ds / "dataset_description.json")
    if tag := _tag_from_doi(raw.get("DatasetDOI"), ds):
        tags["raw"] = tag

    desc = _read_json(fmriprep_dir / ds / "dataset_description.json")
    for src in desc.get("SourceDatasets") or []:
        if tag := _tag_from_doi(src.get("DOI") or src.get("URL"), ds):
            tags["fmriprep"] = tag
            break

    for name, tree in parcellated_dirs.items():
        desc_path = tree / ds / "dataset_description.json"
        desc = _read_json(desc_path)
        for src in desc.get("SourceDatasets") or []:
            if src.get("Version"):
                tags[f"{name} {desc_path.name}"] = src["Version"]
                break
        sidecar = first_sidecar(tree / ds)
        if sidecar is not None and (tag := _tag_from_generated_by(_read_json(sidecar))):
            tags[f"{name} {sidecar.name}"] = tag
    return tags


def resolve_snapshot_tag(
    dataset_id: str,
    snapshot_tag: str | None,
    on_disk: dict[str, str],
    log_dir: Path,
) -> str | None:
    """Snapshot to download, so the subject matches the rest of the dataset.

    In order: `snapshot_tag` (--snapshot-tag); the tag recorded in the output
    trees (which must agree); the download log; the latest snapshot. Matching
    the raw version on disk is also what lets a subject be added to a dataset
    directory downloaded earlier.
    """
    for source, tag in on_disk.items():
        log(f"  on disk:  {tag:<10} ({source})")

    if snapshot_tag:
        if on_disk and set(on_disk.values()) != {snapshot_tag}:
            log(f"  warning:  --snapshot-tag {snapshot_tag} differs from the version on disk")
        return snapshot_tag
    if len(set(on_disk.values())) > 1:
        raise RuntimeError(
            f"{dataset_id} is recorded at different snapshots on disk "
            f"({sorted(set(on_disk.values()))}); pass --snapshot-tag to choose one."
        )
    if on_disk:
        return next(iter(on_disk.values()))
    record = provenance.load_download_record(log_dir, dataset_id) or {}
    if tag := record.get("snapshot_tag"):
        log(f"  logged:   {tag}")
        return tag
    tag = latest_snapshot_tag(dataset_id)
    log(f"  latest:   {tag} (no version on disk or in the logs)")
    return tag


# ---------------------------------------------------------------------------
# Download through openneuro-py
# ---------------------------------------------------------------------------
def subject_include_patterns(dataset_id: str, subject: str, tag: str | None) -> list[str]:
    """openneuro-py include patterns for one subject.

    The subject's func/ and anat/ folders, its subject-level JSON files, and the
    root-level JSON files (dataset_description, task sidecars inherited by every
    run). openneuro-py aborts if any pattern matches nothing, so patterns are
    checked against the snapshot's file list first.
    """
    from openneuro import _glob as openneuro_glob
    from openneuro._download import _get_download_metadata

    candidates = [
        f"/sub-{subject}/**/func/**",
        f"/sub-{subject}/**/anat/**",
        f"/sub-{subject}/*.json",
        "/*.json",
    ]
    metadata = _get_download_metadata(
        dataset_id=dataset_id, tag=tag, max_retries=3, retry_backoff=0.5,
    )
    filenames = [f.filename for f in metadata.files]
    matches = openneuro_glob.glob_filter(filenames, candidates)
    if not matches[candidates[0]]:
        raise RuntimeError(
            f"No func/ files for sub-{subject} in {dataset_id}"
            f"{' @ ' + tag if tag else ''} on OpenNeuro."
        )
    return [p for p in candidates if matches[p]]


# ---------------------------------------------------------------------------
# Download through the snapshot file tree
# ---------------------------------------------------------------------------
def snapshot_tree_files(dataset_id: str, tag: str, subject: str) -> list[tuple[str, int, str]]:
    """(path, size, url) of the root JSON files and the subject's func/ and
    anat/ files in a snapshot, walking its file tree through the GraphQL API.

    The tree lists every file of the snapshot with a URL pinned to that
    snapshot's version of it, also when the flat listing openneuro-py uses
    is incomplete.
    """
    def ls(tree: str | None) -> list[dict]:
        arg = f'(tree:"{tree}")' if tree else ""
        q = (f'{{snapshot(datasetId:"{dataset_id}",tag:"{tag}")'
             f'{{files{arg}{{id filename directory size urls}}}}}}')
        return graphql(q)["snapshot"]["files"]

    def url(f: dict) -> str:
        if not f.get("urls"):
            raise RuntimeError(f"no download URL for {f['filename']} in {dataset_id} @ {tag}")
        return f["urls"][0]

    out = []
    top = ls(None)
    for f in top:
        if not f["directory"] and f["filename"].endswith(".json"):
            out.append((f["filename"], f["size"], url(f)))
    sub = [f for f in top if f["directory"] and f["filename"] == f"sub-{subject}"]
    if not sub:
        raise RuntimeError(f"sub-{subject} not in {dataset_id} @ {tag}")
    # Subject- and session-level JSON files carry metadata inherited by the runs
    # (ds002322 keeps the RepetitionTime of each subject there).
    keep = re.compile(rf"^sub-{re.escape(subject)}/(ses-[^/]+/)?((func|anat)/[^/]+|[^/]+\.json)$")

    def walk(tree: str, prefix: str) -> None:
        for f in ls(tree):
            rel = prefix + f["filename"]
            if f["directory"]:
                walk(f["id"], rel + "/")
            elif keep.match(rel) and not rel.endswith("_events.tsv"):
                out.append((rel, f["size"], url(f)))

    walk(sub[0]["id"], f"sub-{subject}/")
    return out


_ANNEX_KEY = re.compile(r"(MD5E|SHA256E)-s(\d+)--([0-9a-f]+)")


def _fetch(url: str, tmp: Path, digest=None) -> None:
    with urllib.request.urlopen(url, timeout=300) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            if digest is not None:
                digest.update(chunk)


def tree_download_subject(dataset_id: str, tag: str, subject: str, target: Path,
                          retries: int) -> int:
    """Download snapshot_tree_files into target/, checking each size.

    OpenNeuro's object endpoint sometimes cuts transfers after a few MB and
    ignores range requests. After a few failed attempts, a file whose snapshot
    URL carries its checksum (git-annex key) is taken from the public S3 bucket
    instead and accepted only if the checksum matches, so its content is the
    snapshot's.
    """
    files = snapshot_tree_files(dataset_id, tag, subject)
    attempts = max(retries, 6)
    n = 0
    for rel, size, url in files:
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.part")
        key = _ANNEX_KEY.search(url)
        for attempt in range(attempts + 1):
            from_s3 = attempt >= 3 and key is not None
            try:
                digest = None
                if from_s3:
                    digest = hashlib.md5() if key.group(1) == "MD5E" else hashlib.sha256()
                    _fetch(f"https://s3.amazonaws.com/openneuro.org/{dataset_id}/{rel}", tmp, digest)
                    if digest.hexdigest() != key.group(3):
                        raise OSError(f"{rel}: S3 copy does not match the snapshot checksum")
                else:
                    _fetch(url, tmp)
                if tmp.stat().st_size != size:
                    raise OSError(f"{rel}: got {tmp.stat().st_size} bytes, expected {size}")
                os.replace(tmp, dest)
                n += 1
                break
            except (OSError, http.client.HTTPException) as e:  # IncompleteRead is no OSError
                tmp.unlink(missing_ok=True)
                if attempt == attempts:
                    raise RuntimeError(f"download of {rel} failed: {e}") from e
                time.sleep(min(30, 2 ** attempt))
    return n


# ---------------------------------------------------------------------------
# Download step
# ---------------------------------------------------------------------------
@contextmanager
def dataset_lock(raw_root: Path, dataset_id: str):
    """Exclusive lock per dataset, so parallel jobs downloading subjects of the
    same dataset never write its top-level files at the same time."""
    lock_dir = raw_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / f"{dataset_id}.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _write_record(log_dir: Path, record: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "download.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def download_subject(
    dataset_id: str,
    subject: str,
    tag: str | None,
    raw_root: Path,
    log_dir: Path,
    max_retries: int = 7,
    dry_run: bool = False,
) -> None:
    """Download one subject into {raw_root}/{dataset_id}/ and log it.

    The files are downloaded into a staging directory of this subject, then
    moved in: other subjects of the dataset may be running fMRIPrep on the
    dataset directory at the same time, and openneuro-py deletes and re-fetches
    the root files of the directory it downloads into. So the root files are
    written once, atomically, and the subject folder appears with a single
    rename.
    """
    import openneuro

    patch_niquests()
    dest = raw_root / dataset_id
    try:
        include = subject_include_patterns(dataset_id, subject, tag)
    except RuntimeError as e:
        # Some snapshots list only part of their files in the flat listing
        # (ds002345 1.1.4, ds002837 2.0.0 list no func/ or anat/). Their file
        # tree is complete, so download through it.
        if tag is None:
            raise
        log(f"  note:     {e} Downloading through the snapshot file tree.")
        include = None
    if include is not None:
        log(f"  snapshot: {tag}\n  include:  {include}\n  exclude:  {DOWNLOAD_EXCLUDE}"
            f"\n  into:     {dest}")
    else:
        log(f"  snapshot: {tag} (file tree, sub-{subject})\n  into:     {dest}")
    if dry_run:
        return

    started = datetime.now(timezone.utc)
    staging = raw_root / ".staging" / f"{dataset_id}_sub-{subject}"
    if staging.exists():
        shutil.rmtree(staging)  # leftover of an interrupted download of this subject
    staging.mkdir(parents=True)
    if include is None:
        n = tree_download_subject(dataset_id, tag, subject, staging, max_retries)
        log(f"  downloaded {n} file(s) through the snapshot file tree")
    else:
        try:
            openneuro.download(
                dataset=dataset_id, target_dir=staging, include=include,
                exclude=DOWNLOAD_EXCLUDE, tag=tag, max_retries=max_retries,
            )
        except Exception as e:  # openneuro-py raises RuntimeError, OSError, ...
            if tag is None:
                raise
            # OpenNeuro sometimes answers HTTP 500 on files openneuro-py always
            # fetches (the README) or fails HEAD requests. The file tree gives
            # direct URLs for only the files needed here.
            log(f"  note:     openneuro-py failed ({str(e).splitlines()[0][:120]}). "
                "Downloading through the snapshot file tree.")
            shutil.rmtree(staging)
            staging.mkdir(parents=True)
            include = None
            n = tree_download_subject(dataset_id, tag, subject, staging, max_retries)
            log(f"  downloaded {n} file(s) through the snapshot file tree")
    if not raw_bold_files(staging, subject):
        raise RuntimeError(f"Download finished but no BOLD found under {staging}/sub-{subject}")

    sub_staged = staging / f"sub-{subject}"
    files = [p for p in sub_staged.rglob("*") if p.is_file()]
    size_bytes = sum(p.stat().st_size for p in files)
    with dataset_lock(raw_root, dataset_id):
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted(staging.iterdir()):
            if not f.is_file() or (dest / f.name).exists():
                continue
            data = f.read_bytes()
            if f.name == "dataset_description.json":
                # Some datasets are published without DatasetDOI; record the
                # snapshot there.
                desc = json.loads(data)
                if "DatasetDOI" not in desc and tag:
                    desc["DatasetDOI"] = f"10.18112/openneuro.{dataset_id}.v{tag}"
                    data = json.dumps(desc).encode()
            tmp = dest / f".{f.name}.{os.getpid()}"
            tmp.write_bytes(data)
            os.replace(tmp, dest / f.name)
        sub_dest = dest / f"sub-{subject}"
        if sub_dest.exists():
            # Has no BOLD (else this step is skipped): an incomplete earlier download.
            shutil.rmtree(sub_dest)
        os.replace(sub_staged, sub_dest)
    shutil.rmtree(staging)

    _write_record(log_dir, {
        "dataset_id": dataset_id,
        "subject": subject,
        "status": "success",
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "size_bytes": size_bytes,
        "error": None,
        "snapshot_tag": tag,
        "include": include if include is not None else "snapshot file tree",
        "exclude": DOWNLOAD_EXCLUDE,
        "files_downloaded": len(files),
        "openneuro_py_version": provenance.package_version("openneuro-py"),
        "dataset_url": provenance.OPENNEURO_DATASET_URL.format(dataset_id=dataset_id),
        "git_revision": provenance.git_revision(),
    })


# ---------------------------------------------------------------------------
# Missing RepetitionTime
# ---------------------------------------------------------------------------
_ENTITY = re.compile(r"([A-Za-z0-9]+)-([A-Za-z0-9]+)")


def bolds_without_tr(raw_dataset: Path, subject: str) -> list[Path]:
    """Raw BOLD files with no RepetitionTime in their inherited BIDS metadata."""
    sub_dir = raw_dataset / f"sub-{subject}"
    out = []
    for bold in raw_bold_files(raw_dataset, subject):
        entities = set(_ENTITY.findall(bold.name.split("_bold")[0]))
        meta: dict = {}
        for folder in (raw_dataset, sub_dir, bold.parent.parent, bold.parent):
            for side in sorted(folder.glob("*_bold.json")):
                if set(_ENTITY.findall(side.name.split("_bold")[0])) <= entities:
                    meta.update(_read_json(side))
        if "RepetitionTime" not in meta:
            out.append(bold)
    return out


def fill_missing_sidecars(dataset_id: str, subject: str, tag: str | None,
                          raw_root: Path, dry_run: bool = False) -> None:
    """Download the JSON sidecars of a subject whose raw BOLD has no RepetitionTime.

    A download through openneuro-py can miss files its listing lacks (ds002322,
    ds002837). Files already on disk are never replaced.
    """
    raw_dataset = raw_root / dataset_id
    missing = bolds_without_tr(raw_dataset, subject)
    if not missing:
        return
    log(f"  sidecars: {len(missing)} BOLD file(s) without RepetitionTime; "
        f"fetching the missing JSON files of the snapshot {tag}")
    if dry_run or tag is None:
        return
    n = 0
    for rel, size, url in snapshot_tree_files(dataset_id, tag, subject):
        dest = raw_dataset / rel
        if not rel.endswith(".json") or dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.part")
        for attempt in range(4):
            try:
                _fetch(url, tmp)
                if tmp.stat().st_size != size:
                    raise OSError(f"{rel}: got {tmp.stat().st_size} bytes, expected {size}")
                os.replace(tmp, dest)
                n += 1
                break
            except (OSError, http.client.HTTPException) as e:
                tmp.unlink(missing_ok=True)
                if attempt == 3:
                    raise RuntimeError(f"download of {rel} failed: {e}") from e
                time.sleep(2 ** attempt)
    left = bolds_without_tr(raw_dataset, subject)
    log(f"  sidecars: added {n} file(s); {len(left)} BOLD file(s) still without RepetitionTime")
