#!/usr/bin/env python3
"""生成轻量项目清单；默认跳过 canonical 大型 MPH 的逐字节哈希。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "MANIFEST.json"
SKIP_PARTS = {".pytest_cache", "__pycache__", ".venv"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == TARGET or SKIP_PARTS.intersection(path.parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        item = {"path": relative, "bytes": path.stat().st_size}
        if path.stat().st_size <= 100 * 1024 * 1024:
            item["sha256"] = digest(path)
        else:
            item["sha256"] = None
            item["note"] = "large file; use SHA256SUMS.txt in standalone delivery"
        files.append(item)
    payload = {
        "project": "loudspeakerTimeFEM",
        "generated": "2026-08-01",
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
