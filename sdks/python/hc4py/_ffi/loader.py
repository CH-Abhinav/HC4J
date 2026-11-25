# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Native shared-library resolver and process-wide loader for ``hc4x-ffi``."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import threading
from pathlib import Path

from .bindings import bind_all

_LOAD_LOCK = threading.Lock()
_LOADED_LIB: ctypes.CDLL | None = None
_LOADED_PATH: Path | None = None


def _platform_lib_names() -> list[str]:
    system = platform.system().lower()
    if system == "windows":
        return ["hc4j.dll"]
    if system == "darwin":
        return ["libhc4j.dylib"]
    return ["libhc4j.so"]


def _candidate_paths() -> list[Path]:
    names = _platform_lib_names()
    candidates: list[Path] = []

    for env_var in ("HC4X_NATIVE_PATH", "HC4J_NATIVE_PATH"):
        raw = os.environ.get(env_var, "").strip()
        if raw:
            p = Path(raw).expanduser()
            if p.is_file():
                candidates.append(p)
            elif p.is_dir():
                for name in names:
                    candidates.append(p / name)

    pkg_dir = Path(__file__).resolve().parent.parent
    for name in names:
        candidates.append(pkg_dir / name)
        candidates.append(pkg_dir / "lib" / name)

    # Walk ancestor directories to locate native build outputs in source checkouts.
    cur = Path(__file__).resolve().parent
    for _ in range(7):
        for rel_dir in (
            "native/target/release",
            "native/target/debug",
            "core/build/rust/release",
            "core/build/rust/debug",
        ):
            for name in names:
                candidates.append(cur / rel_dir / name)
        if cur.parent == cur:
            break
        cur = cur.parent

    found = ctypes.util.find_library("hc4j")
    if found:
        candidates.append(Path(found))

    return candidates


def get_library() -> ctypes.CDLL:
    """Loads and caches the ``hc4x-ffi`` shared library with all 107 FFI prototypes bound."""
    global _LOADED_LIB, _LOADED_PATH
    if _LOADED_LIB is not None:
        return _LOADED_LIB

    with _LOAD_LOCK:
        if _LOADED_LIB is not None:
            return _LOADED_LIB

        errors: list[str] = []
        for candidate in _candidate_paths():
            if not candidate.exists():
                continue
            try:
                lib = ctypes.CDLL(str(candidate))
                bind_all(lib)
                _LOADED_PATH = candidate.resolve()
                _LOADED_LIB = lib
                return lib
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")

        tried = ", ".join(str(p) for p in _candidate_paths()[:8])
        detail = f" ({'; '.join(errors)})" if errors else ""
        raise OSError(
            f"Unable to locate or load hc4x-ffi shared library ({_platform_lib_names()[0]}). "
            f"Set HC4X_NATIVE_PATH or build native/. Checked: {tried}{detail}"
        )


def loaded_library_path() -> Path:
    """Returns the filesystem path of the loaded ``hc4x-ffi`` shared library."""
    get_library()
    assert _LOADED_PATH is not None
    return _LOADED_PATH
