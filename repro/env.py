"""Capture the environment a run was produced in.

The paper's artifacts were produced on Windows 11 / CPython 3.14.5 / CPU only.
We re-run on the same OS family and a different CPython minor version, so the
exact interpreter and every pinned package version are recorded here and quoted
verbatim in the report's setup section.
"""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

from repro import config as cfg

PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "pillow",
    "pytest",
    "torch",
)


def capture() -> dict:
    upstream = cfg.UPSTREAM_DIR
    rev = None
    if (upstream / ".git").exists():
        import subprocess

        try:
            rev = subprocess.run(
                ["git", "-C", str(upstream), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False,
            ).stdout.strip() or None
        except OSError:
            rev = None

    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None

    return {
        "captured_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "upstream_url": cfg.UPSTREAM_URL,
        "upstream_head": rev,
        "packages": versions,
    }


def main() -> None:
    env = capture()
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.OUT_DIR / "environment.json"
    out.write_text(json.dumps(env, indent=2), encoding="utf-8")

    print(f"python      : {env['python']} ({env['python_implementation']})")
    print(f"platform    : {env['platform']}")
    print(f"cpu_count   : {env['cpu_count']}")
    print(f"upstream    : {env['upstream_head']}")
    for k, v in env["packages"].items():
        print(f"  {k:<14s}: {v}")
    print(f"wrote {cfg.rel(out)}")


if __name__ == "__main__":
    main()
