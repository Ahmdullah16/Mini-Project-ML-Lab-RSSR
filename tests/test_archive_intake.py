import hashlib
import stat
import zipfile

import pytest

from rssr.archive_intake import extract_selected, inspect_zip


def make_zip(tmp_path, entries):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return path


def test_inspect_and_extract_selected_without_overwrite(tmp_path):
    path = make_zip(tmp_path, [("pair/lr.txt", b"LR"), ("pair/hr.txt", b"HR")])
    report = inspect_zip(path)
    assert report["declared_expanded_bytes"] == 4
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    out = tmp_path / "selected"
    extracted = extract_selected(path, out, ["pair/lr.txt"], reserve_bytes=1)
    assert extracted["extracted_bytes"] == 2
    assert (out / "pair/lr.txt").read_bytes() == b"LR"
    assert not (out / "pair/hr.txt").exists()
    with pytest.raises(FileExistsError):
        extract_selected(path, out, ["pair/lr.txt"], reserve_bytes=1)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/drive", "a\\b",
                                  "a/../b", "CON", "a/NUL.txt", "file.", "file "])
def test_reject_nonportable_paths(tmp_path, name):
    # Preserve the raw stored name: ZipInfo normalizes Windows separators.
    info = zipfile.ZipInfo(name)
    info.filename = name
    path = make_zip(tmp_path, [(info, b"x")])
    with pytest.raises(ValueError):
        inspect_zip(path)


@pytest.mark.parametrize("entries", [[("a", b"a"), ("A", b"b")],
                                    [("a", b"a"), ("a/b", b"b")]])
def test_reject_collisions(tmp_path, entries):
    with pytest.raises(ValueError):
        inspect_zip(make_zip(tmp_path, entries))


def test_reject_symlink(tmp_path):
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError):
        inspect_zip(make_zip(tmp_path, [(link, b"../outside")]))


@pytest.mark.parametrize("options", [{"max_archive_bytes": 1}, {"max_expanded_bytes": 1},
                                     {"max_entries": 1}, {"expected_md5": "0"*32}])
def test_reject_quota_and_checksum(tmp_path, options):
    path = make_zip(tmp_path, [("a", b"abc"), ("b", b"def")])
    with pytest.raises(ValueError):
        inspect_zip(path, **options)


def test_failed_extraction_cleans_staging(tmp_path, monkeypatch):
    path = make_zip(tmp_path, [("a", b"abc")])
    def fail(*args, **kwargs):
        raise zipfile.BadZipFile("simulated CRC failure")
    monkeypatch.setattr(zipfile.ZipFile, "open", fail)
    with pytest.raises(zipfile.BadZipFile):
        extract_selected(path, tmp_path / "out", ["a"], reserve_bytes=1)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".rssr-intake-*"))


@pytest.mark.parametrize("names", [[], ["missing"], ["a", "a"]])
def test_requires_exact_distinct_selection(tmp_path, names):
    path = make_zip(tmp_path, [("a", b"abc")])
    with pytest.raises(ValueError):
        extract_selected(path, tmp_path / "out", names, reserve_bytes=1)
