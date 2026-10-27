"""Bounded local ZIP inspection/extraction; never downloads or decodes imagery.

Inspect an acquired archive before choosing entries. Extraction requires exact
entry names and a new destination. Nested archives remain opaque files.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile

MAX_ARCHIVE_BYTES = 130_000_000
MAX_EXPANDED_BYTES = 512 * 1024**2
MAX_ENTRIES = 10_000


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _safe_name(info):
    name = info.orig_filename
    parts = PurePosixPath(name).parts
    if (not parts or name.startswith("/") or "\\" in name or ":" in name
            or "\x00" in info.orig_filename or any(ord(c) < 32 for c in name)):
        raise ValueError(f"Unsafe archive path: {name!r}")
    if any(p in (".", "..") or p.endswith((" ", ".")) or
           re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", p)
           for p in name.rstrip("/").split("/")):
        raise ValueError(f"Nonportable archive path: {name!r}")
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"Links/special entries are unsupported: {name}")
    if info.flag_bits & 1:
        raise ValueError("Encrypted entries are unsupported")
    return name.rstrip("/")


def inspect_zip(path, *, max_archive_bytes=MAX_ARCHIVE_BYTES,
                max_expanded_bytes=MAX_EXPANDED_BYTES, max_entries=MAX_ENTRIES,
                expected_md5=None):
    """Hash and validate metadata only; sizes are checked again while extracting.

    Reject the whole archive when its declared expansion exceeds the cap, even
    for partial extraction. ZIP metadata is not proof of raster compatibility.
    """
    for name, value in (("archive cap", max_archive_bytes), ("expansion cap", max_expanded_bytes),
                        ("entry cap", max_entries)):
        _positive(value, name)
    path = Path(path)
    size = path.stat().st_size
    if size > max_archive_bytes:
        raise ValueError("Archive byte cap exceeded")
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with path.open("rb") as src:
        count = 0
        while chunk := src.read(64 * 1024):
            count += len(chunk)
            if count > max_archive_bytes:
                raise ValueError("Archive grew beyond byte cap")
            sha.update(chunk)
            md5.update(chunk)
    if expected_md5 is not None and md5.hexdigest() != expected_md5.lower():
        raise ValueError("Published MD5 mismatch")
    records, names, total = [], set(), 0
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > max_entries:
            raise ValueError("Archive entry cap exceeded")
        for info in infos:
            name = _safe_name(info)
            key = name.casefold()
            if key in names:
                raise ValueError("Duplicate/case-colliding archive paths")
            names.add(key)
            total += info.file_size
            if total > max_expanded_bytes:
                raise ValueError("Declared expansion cap exceeded")
            records.append({"name": info.filename, "bytes": info.file_size,
                            "compressed_bytes": info.compress_size, "directory": info.is_dir()})
        files = {r["name"].casefold() for r in records if not r["directory"]}
        for record in records:
            if any(str(p).casefold() in files for p in PurePosixPath(record["name"]).parents
                   if str(p) != "."):
                raise ValueError("File/directory prefix collision")
    return {"archive_bytes": count, "sha256": sha.hexdigest(), "md5": md5.hexdigest(),
            "declared_expanded_bytes": total, "entries": records,
            "scope": "Archive metadata only; no imagery or scientific acceptance"}


def extract_selected(path, destination, names, *, max_expanded_bytes=MAX_EXPANDED_BYTES,
                     reserve_bytes=1024**3, **inspection_options):
    """Extract exact selected files to an exclusive staging directory, then publish.

    Single-process local workflow; concurrent replacement of source/destination
    is unsupported. CRC errors and quota violations remove partial staged output.
    """
    _positive(reserve_bytes, "free disk reserve")
    report = inspect_zip(path, max_expanded_bytes=max_expanded_bytes, **inspection_options)
    entries = {r["name"]: r for r in report["entries"] if not r["directory"]}
    names = list(names)
    if not names or len(set(names)) != len(names) or any(n not in entries for n in names):
        raise ValueError("Select distinct exact file names from the inspected archive")
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    parent = destination.parent.resolve(strict=True)
    required = sum(entries[n]["bytes"] for n in names)
    if shutil.disk_usage(parent).free < required + reserve_bytes:
        raise ValueError("Insufficient free disk for selected entries plus reserve")
    staging = Path(tempfile.mkdtemp(prefix=".rssr-intake-", dir=parent))
    try:
        written, outputs = 0, []
        with zipfile.ZipFile(path) as archive:
            for name in names:
                target = staging.joinpath(*PurePosixPath(name).parts)
                if not target.resolve().is_relative_to(staging.resolve()):
                    raise ValueError("Extraction escaped staging directory")
                target.parent.mkdir(parents=True, exist_ok=True)
                digest, entry_bytes = hashlib.sha256(), 0
                with archive.open(name) as src, target.open("xb") as out:
                    while chunk := src.read(64 * 1024):
                        written += len(chunk)
                        entry_bytes += len(chunk)
                        if written > max_expanded_bytes or entry_bytes > entries[name]["bytes"]:
                            raise ValueError("Actual extraction byte cap exceeded")
                        digest.update(chunk)
                        out.write(chunk)
                if entry_bytes != entries[name]["bytes"]:
                    raise ValueError("Extracted size differs from ZIP metadata")
                outputs.append({"name": name, "bytes": entry_bytes, "sha256": digest.hexdigest()})
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
        return {"archive_sha256": report["sha256"], "extracted_bytes": written, "files": outputs}
    finally:
        if staging.exists() and staging.resolve().parent == parent:
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-md5")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--entry", action="append", default=[])
    args = parser.parse_args()
    if args.destination:
        report = extract_selected(args.archive, args.destination, args.entry, expected_md5=args.expected_md5)
    else:
        if args.entry:
            parser.error("--entry requires --destination")
        report = inspect_zip(args.archive, expected_md5=args.expected_md5)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
