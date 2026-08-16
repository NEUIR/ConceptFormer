#!/usr/bin/env python3
"""Write deterministic SHA-256 manifests for release directories."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for directory in sorted(p for p in args.root.iterdir() if p.is_dir()):
        files = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.name != "SHA256SUMS"
            and ".cache" not in path.relative_to(directory).parts
        )
        lines = [f"{digest(path)}  {path.relative_to(directory)}" for path in files]
        (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
