from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    parent = data.get("extends")
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        base, _ = load_config(parent_path)

        def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
            result = dict(left)
            for key, value in right.items():
                if key == "extends":
                    continue
                if isinstance(value, dict) and isinstance(result.get(key), dict):
                    result[key] = merge(result[key], value)
                else:
                    result[key] = value
            return result

        data = merge(base, data)
    required = ("base_mainline", "mesh", "mphtxt", "magnetostatic_vtu", "air", "drive", "time")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"配置缺少字段: {', '.join(missing)}")
    return data, config_path


def reference_dependency_findings(config: dict[str, Any]) -> list[str]:
    """Return reference-derived inputs that disqualify a native production run."""
    findings: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{path}.{key}" if path else str(key)
                if key == "reference_identified" and item is True:
                    findings.append(f"{child}=true")
                walk(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str):
            normalized = value.replace("\\", "/").lower()
            if "comsol_validation/" in normalized:
                findings.append(f"{path} points under comsol_validation")
            if path.endswith("coefficient_source") and "comsol" in normalized:
                findings.append(f"{path} contains COMSOL")

    walk(config, "")
    return sorted(set(findings))


def assert_native_production_config(config: dict[str, Any]) -> None:
    findings = reference_dependency_findings(config)
    nonlinear = config.get("nonlinear", {})
    if bool(nonlinear.get("tensor_coenergy_enabled", False)) or "coenergy_tensor" in str(
        nonlinear.get("law", "")
    ):
        findings.append("nonlinear.tensor_coenergy_enabled is diagnostic-only")
    if findings:
        raise ValueError(
            "native production rejects reference-identified inputs: "
            + "; ".join(findings)
        )


def resolve_base_mainline(config: dict[str, Any], config_path: Path) -> Path:
    project = config_path.parent.parent
    configured = Path(config["base_mainline"])
    # A normal checkout is self-contained.  Prefer its bundled, validated
    # frequency-domain source so moving the directory cannot silently switch
    # the model to a neighbouring workspace.
    candidates: list[Path] = [project / "inputs/frequency_mainline"]
    candidates.append(configured if configured.is_absolute() else project / configured)
    environment = os.environ.get("LOUDSPEAKER_FREQUENCY_MAINLINE")
    if environment:
        candidates.append(Path(environment))
    checked: list[str] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(str(candidate))
        if (candidate / "best_model").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("找不到频域生产主线；已检查: " + "；".join(checked))
