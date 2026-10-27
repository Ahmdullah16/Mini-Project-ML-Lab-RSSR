import hashlib
import io

import pytest

from rssr.archive_transfer import transfer_archive


class Response(io.BytesIO):
    status = 200
    def __init__(self, body, headers=None):
        super().__init__(body)
        self.headers = {} if headers is None else headers


def test_one_attempt_pinned_bytes_and_hash(tmp_path):
    calls = []
    def opener(req, timeout):
        calls.append(req.full_url)
        return Response(b"abc", {"Content-Length": "3"})
    result = transfer_archive("https://example.org/archive.zip", tmp_path / "a.zip",
        expected_bytes=3, expected_md5=hashlib.md5(b"abc").hexdigest(), reserve_bytes=1, opener=opener)
    assert result["application_body_bytes"] == 3
    assert result["attempts"] == len(calls) == 1
    assert (tmp_path / "a.zip").read_bytes() == b"abc"


@pytest.mark.parametrize("body,headers", [(b"ab", {}), (b"xyz", {}),
    (b"abc", {"Content-Length":"4"}), (b"abc", {"Content-Encoding":"gzip"})])
def test_failed_transfer_no_retry_no_partial_artifacts(tmp_path, body, headers):
    calls = []
    def opener(*args, **kwargs):
        calls.append(1)
        return Response(body, headers)
    with pytest.raises(ValueError):
        transfer_archive("https://example.org/a", tmp_path / "a.zip", expected_bytes=3,
            expected_md5=hashlib.md5(b"abc").hexdigest(), reserve_bytes=1, opener=opener)
    assert len(calls) == 1
    assert not list(tmp_path.iterdir())


def test_cap_rejected_before_network(tmp_path):
    def opener(*args, **kwargs):
        pytest.fail("Network must not be invoked")
    with pytest.raises(ValueError, match="cap"):
        transfer_archive("https://example.org/a", tmp_path / "a", expected_bytes=4,
            max_bytes=3, expected_md5="0"*32, opener=opener)


def test_existing_output_preserved(tmp_path):
    target = tmp_path / "a"
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        transfer_archive("https://example.org/a", target, expected_bytes=3, expected_md5="0"*32)
    assert target.read_bytes() == b"keep"
