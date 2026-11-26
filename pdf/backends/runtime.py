from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def find_venv_python(
    python_exe: str | Path | None = None,
    *,
    env_var: str,
    default_venv: str,
) -> Path:
    candidates: list[Path] = []
    if python_exe is not None:
        candidates.append(Path(python_exe))
    env_py = os.getenv(env_var)
    if env_py:
        candidates.append(Path(env_py))
    default_dir = Path.home() / ".venvs" / default_venv
    candidates.append(default_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Isolated Python executable not found; set {env_var} or install in ~/.venvs/{default_venv}"
    )


def run_isolated_json(
    python_exe: Path,
    script_path: Path,
    make_args: Callable[[Path], Sequence[str]],
    *,
    env_overrides: Mapping[str, str] | None = None,
    metrics_prefix: str = "SUBPROCESS_METRICS=",
    label: str = "subprocess",
    timeout_sec: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "output.json"
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [str(python_exe), "-B", str(script_path), *make_args(out_path)],
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{label} subprocess ({Path(script_path).name}) timed out after {timeout_sec}s"
            ) from exc
        wall_sec = round(time.perf_counter() - t0, 4)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout)[-2000:]
            raise RuntimeError(f"{label} subprocess failed (exit {proc.returncode}):\n{tail}")
        data = json.loads(out_path.read_text(encoding="utf-8"))

    metrics: dict[str, Any] = {"wall_sec": wall_sec}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(metrics_prefix):
            metrics.update(json.loads(line.removeprefix(metrics_prefix)))
            break
    combined_logs = f"{proc.stdout}\n{proc.stderr}"
    return data, metrics, combined_logs
