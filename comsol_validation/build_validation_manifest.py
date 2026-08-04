#!/usr/bin/env python3
"""Build a byte-level manifest for the retained COMSOL validation evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SCRATCH = Path("/mnt/d/loudspeakerFEM_comsol_validation")


def describe(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> int:
    target = HERE / "raw/validation_manifest.json"
    retained = [
        PROJECT / "loudspeaker_driver_transient.mph",
        SCRATCH / "baseline/input.mph",
        SCRATCH / "baseline/solved.mph",
        SCRATCH / "tight_time/solved.mph",
        SCRATCH / "refined_mesh/solved.mph",
    ]
    evidence = sorted(path for path in (HERE / "raw").rglob("*") if path != target)
    manifest = {
        "comsol_version": "6.3.0.290",
        "execution_host": "Windows native COMSOL batch; WSL used only for orchestration and external analysis",
        "large_models": [describe(path) for path in retained if path.is_file()],
        "workspace_evidence": [describe(path) for path in evidence if path.is_file()],
    }
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
