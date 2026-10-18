"""
Storage abstraction.

Everything is stored locally for now. If disk space runs out, add an
HFStorage (Hugging Face Hub) or S3Storage (Backblaze B2 / Cloudflare R2)
class; the rest of the code only uses put/get/exists and stays unchanged.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path


class Storage(ABC):
    """Interface: put a file, fetch a file, check existence."""

    @abstractmethod
    def put(self, local: Path, remote: str) -> None: ...

    @abstractmethod
    def get(self, remote: str, local: Path) -> Path: ...

    @abstractmethod
    def exists(self, remote: str) -> bool: ...


class LocalStorage(Storage):
    """Files live in a project folder; nothing leaves the machine."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, remote: str) -> Path:
        return self.root / remote

    def put(self, local: Path, remote: str) -> None:
        dst = self._p(remote)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if Path(local).resolve() != dst.resolve():
            shutil.copy2(local, dst)

    def get(self, remote: str, local: Path) -> Path:
        src = self._p(remote)
        if not src.exists():
            raise FileNotFoundError(f"file not found in storage: {remote}")
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != local.resolve():
            shutil.copy2(src, local)
        return local

    def exists(self, remote: str) -> bool:
        return self._p(remote).exists()


def get_storage(backend: str, root: Path) -> Storage:
    """Factory. Only the local backend is supported for now."""
    if backend == "local":
        return LocalStorage(root)
    raise NotImplementedError(
        f"backend '{backend}' is not implemented yet. "
        "Possible future options: hf (Hugging Face Hub), s3 (B2/R2). "
        "Keep storage_backend: local in the config."
    )
