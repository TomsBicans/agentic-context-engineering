from __future__ import annotations

from pathlib import Path


def write_payload(path: Path, payload: bytes, compress: bool) -> Path:
    if compress:
        import gzip

        target = path.with_suffix(path.suffix + ".gz")
        with gzip.open(target, "wb") as file_obj:
            file_obj.write(payload)
        return target

    path.write_bytes(payload)
    return path
