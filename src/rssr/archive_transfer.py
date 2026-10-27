"""Single-attempt pinned archive transfer. No download occurs at import.

Call only after resource approval. Counts application response bytes, not TLS,
HTTP overhead or bytes buffered by the OS. Never resumes or retries implicitly.
"""
import hashlib
from pathlib import Path
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request

from .archive_intake import MAX_ARCHIVE_BYTES, _positive


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirect refused; verify and pin the final public URL")


def transfer_archive(url, destination, *, expected_bytes, expected_md5,
                     max_bytes=MAX_ARCHIVE_BYTES, reserve_bytes=1024**3,
                     timeout_seconds=45, budget_seconds=600, opener=None):
    """Transfer exactly the provider-declared bytes and verify published MD5.

    Source identity and licenses require separate review. MD5 checks accidental
    corruption against provider metadata; SHA256 is recorded for later pinning.
    Deadline is checked between blocking reads (each read has socket timeout).
    """
    for name, value in (("expected_bytes", expected_bytes), ("max_bytes", max_bytes),
                        ("reserve_bytes", reserve_bytes), ("timeout_seconds", timeout_seconds),
                        ("budget_seconds", budget_seconds)):
        _positive(value, name)
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("A public HTTPS URL without embedded credentials is required")
    if expected_bytes > max_bytes:
        raise ValueError("Provider size exceeds approved byte cap")
    if not re.fullmatch(r"[0-9a-fA-F]{32}", expected_md5):
        raise ValueError("Provider MD5 required")
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    parent = destination.parent.resolve(strict=True)
    if shutil.disk_usage(parent).free < expected_bytes + reserve_bytes:
        raise ValueError("Insufficient disk reserve")
    open_url = opener or urllib.request.build_opener(_NoRedirect()).open
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    started = time.monotonic()
    temporary = None
    try:
        with open_url(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise ValueError("Expected a complete HTTP200 response")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("Encoded transfer not supported")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != expected_bytes:
                raise ValueError("Content-Length differs from pinned provider size")
            sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
            received = 0
            with tempfile.NamedTemporaryFile(prefix=".rssr-transfer-", dir=parent, delete=False) as out:
                temporary = Path(out.name)
                while received < expected_bytes:
                    if time.monotonic() - started > budget_seconds:
                        raise TimeoutError("Transfer time budget exceeded")
                    chunk = response.read(min(65536, expected_bytes-received))
                    if not chunk:
                        raise ValueError("Truncated archive response")
                    received += len(chunk)
                    if received > expected_bytes:
                        raise ValueError("Response reader exceeded requested byte bound")
                    out.write(chunk)
                    sha.update(chunk)
                    md5.update(chunk)
            if md5.hexdigest() != expected_md5.lower():
                raise ValueError("Provider MD5 mismatch")
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
        return {"application_body_bytes": received, "sha256": sha.hexdigest(),
                "md5": md5.hexdigest(), "attempts": 1,
                "seconds": time.monotonic()-started,
                "accounting_scope": "Application reads only; not total network wire bytes"}
    finally:
        if temporary is not None and temporary.exists() and temporary.resolve().parent == parent:
            temporary.unlink()
