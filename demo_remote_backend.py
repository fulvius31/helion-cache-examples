"""Directory-backed and failing RemoteCacheBackends (case 5).

Instantiated with no args by Helion, self-configured from
``HELION_DEMO_REMOTE_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

from helion.autotuner.remote_cache import RemoteCacheBackend


class FileBackend(RemoteCacheBackend):
    """Directory-backed remote cache; root from HELION_DEMO_REMOTE_DIR."""

    def __init__(self) -> None:
        self.root = Path(os.environ["HELION_DEMO_REMOTE_DIR"])
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> str | None:
        p = self.root / f"{key}.json"
        return p.read_text() if p.exists() else None

    def put(self, key: str, data: str) -> None:
        (self.root / f"{key}.json").write_text(data)

    def list(self, max_results: int | None = None):
        files = sorted(
            self.root.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if max_results is not None:
            files = files[:max_results]
        return [p.read_text() for p in files]


class FailingBackend(RemoteCacheBackend):
    """Backend that raises on every operation (simulated outage)."""

    def get(self, key: str) -> str | None:
        raise ConnectionError("simulated outage")

    def put(self, key: str, data: str) -> None:
        raise ConnectionError("simulated outage")

    def list(self, max_results: int | None = None):
        raise ConnectionError("simulated outage")
