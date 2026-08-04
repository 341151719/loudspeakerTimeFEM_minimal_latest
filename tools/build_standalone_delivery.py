#!/usr/bin/env python3
"""Build the standalone D-drive delivery with the frequency-mainline layout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
FREQUENCY_PROJECT = PROJECT.parent / "loudspeakerFEM"
WORKSPACE = FREQUENCY_PROJECT
FREQUENCY = FREQUENCY_PROJECT / "00_MAINLINE/loudspeakerFEM_current_20260717"
COMSOL = Path("/mnt/d/loudspeakerFEM_comsol_validation")
EXCLUDED_NAMES = {".venv", ".pytest_cache", "__pycache__"}

FULLFIELDS = {
    100: WORKSPACE / "20_ANALYSIS/runs/validation6/checkpoints/100Hz/acoustic_100Hz.vtu",
    600: WORKSPACE / "20_ANALYSIS/runs/validation6/checkpoints/600Hz/acoustic_600Hz.vtu",
    630: WORKSPACE / "20_ANALYSIS/runs/validation6/checkpoints/630Hz/acoustic_630Hz.vtu",
    1000: WORKSPACE / "20_ANALYSIS/runs/validation6/checkpoints/1000Hz/acoustic_1000Hz.vtu",
    6300: WORKSPACE / "20_ANALYSIS/runs/validation6/checkpoints/6300Hz/acoustic_6300Hz.vtu",
    8000: WORKSPACE / "20_ANALYSIS/runs/stage34_directivity_15k/native_refined2_500Hz/checkpoints/8000Hz/acoustic_8000Hz.vtu",
    10000: WORKSPACE / "20_ANALYSIS/runs/stage34_directivity_15k/native_refined2_500Hz/checkpoints/10000Hz/acoustic_10000Hz.vtu",
    12000: WORKSPACE / "20_ANALYSIS/runs/stage34_directivity_15k/native_refined2_500Hz/checkpoints/12000Hz/acoustic_12000Hz.vtu",
    13500: WORKSPACE / "20_ANALYSIS/runs/stage34_directivity_15k/native_refined2_500Hz/checkpoints/13500Hz/acoustic_13500Hz.vtu",
    15000: WORKSPACE / "20_ANALYSIS/runs/stage34_directivity_15k/native_refined2_500Hz/checkpoints/15000Hz/acoustic_15000Hz.vtu",
}


def ignored(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in EXCLUDED_NAMES or name.endswith(".pyc") or name.endswith(".pyo")
        or name.endswith(".previous")
    }


def copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, target, ignore=ignored)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="/mnt/d/loudspeakerTimeFEM")
    args = parser.parse_args()
    target = Path(args.target).resolve()
    if target.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖: {target}")
    if not COMSOL.is_dir():
        raise FileNotFoundError(COMSOL)

    copy_tree(PROJECT, target)

    bundled_frequency = target / "inputs/frequency_mainline"
    if not bundled_frequency.is_dir():
        copy_tree(FREQUENCY, bundled_frequency)

    fullfield_dir = target / "inputs/reference_fields/frequency_fullfield"
    for frequency, source in FULLFIELDS.items():
        copy_file(source, fullfield_dir / f"acoustic_{frequency}Hz.vtu")

    export = target / "comsol_exports"
    for name in (
        "loudspeaker_driver_transient.mph",
        "loudspeaker_driver_transient.java",
        "models.aco.loudspeaker_driver_transient.pdf",
    ):
        copy_file(PROJECT / name, export / "input" / name)

    java_dir = export / "java_validation"
    java_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted((PROJECT / "comsol_validation").glob("*.java")):
        copy_file(source, java_dir / source.name)
    for source in sorted((COMSOL / "tools").glob("*.class")):
        copy_file(source, java_dir / source.name)

    for case in ("baseline", "tight_time", "refined_mesh"):
        source = COMSOL / case
        case_target = export / "solved_cases" / case
        copy_file(source / "solved.mph", case_target / "solved.mph")
        copy_file(source / "solve.log", case_target / "solve.log")
        settings = source / "solved.mph.settings.txt"
        if settings.is_file():
            copy_file(settings, case_target / settings.name)
        copy_tree(source / "raw_v2", case_target / "raw_v2")

    copy_file(
        COMSOL / "refined_mesh/interface_displacement_traces_v2.csv",
        export / "interface/interface_displacement_traces_v2.csv",
    )
    copy_file(
        COMSOL / "refined_mesh/native_interface_quadrature_v2.csv",
        export / "interface/native_interface_quadrature_v2.csv",
    )
    for name in ("boundary_catalog.csv",):
        copy_file(COMSOL / "refined_mesh" / name, export / "interface" / name)
    for name in ("transient_mesh.mphtxt", "native_interface_quadrature.csv"):
        copy_file(COMSOL / "native_mesh" / name, export / "native_mesh" / name)

    build_info = {
        "project": "loudspeakerTimeFEM",
        "build_date": "2026-07-28",
        "source_project": str(PROJECT),
        "frequency_mainline": str(FREQUENCY),
        "comsol_source": str(COMSOL),
        "canonical_solved_cases": ["baseline", "tight_time", "refined_mesh"],
        "excluded": [
            "Python caches and virtual environments",
            "*.previous duplicate runs",
            "COMSOL tools/*_export.mph automatic duplicate saves",
            "COMSOL temporary and recovery directories",
        ],
    }
    (target / "PACKAGE_BUILD_INFO.json").write_text(
        json.dumps(build_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    subprocess.run(
        ["python3", str(target / "tools/build_project_manifest.py")],
        cwd=target,
        check=True,
    )

    inventory_path = target / "PACKAGE_INVENTORY.csv"
    checksum_path = target / "SHA256SUMS.txt"
    records: list[tuple[str, int, str]] = []
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path in {inventory_path, checksum_path}:
            continue
        records.append((path.relative_to(target).as_posix(), path.stat().st_size, sha256(path)))
    with inventory_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "bytes", "sha256"])
        writer.writerows(records)
    inventory_hash = sha256(inventory_path)
    with checksum_path.open("w", encoding="utf-8", newline="\n") as stream:
        for relative, _size, digest in records:
            stream.write(f"{digest}  {relative}\n")
        stream.write(f"{inventory_hash}  PACKAGE_INVENTORY.csv\n")

    total = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
    print(json.dumps({"target": str(target), "bytes": total, "files": len(records) + 2}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
