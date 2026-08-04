#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from loudspeaker_time_fem.config import load_config, reference_dependency_findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.configs.glob("*.json")):
        config, _ = load_config(path)
        findings = reference_dependency_findings(config)
        rows.append(
            {
                "config": path.as_posix(),
                "role": "reference_diagnostic" if findings else "native",
                "findings": findings,
            }
        )
    report = {"status": "completed", "configs": rows}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
