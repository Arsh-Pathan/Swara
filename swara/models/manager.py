"""Model artifact manager: local caching, downloading, and SHA-256 verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request
import tempfile
import shutil

from swara.core.config import settings
from swara.models.registry import MODEL_REGISTRY, ModelSpec
from swara.utils.logger import get_logger

logger = get_logger("swara.models")


class ModelError(Exception):
    """Base exception for model artifact errors."""


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found in registry or local disk."""


class ModelChecksumError(ModelError):
    """Raised when downloaded model fails SHA-256 integrity verification."""


class ModelManager:
    """Manages local storage, downloading, and validation of ONNX models."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = Path(cache_dir or settings.models_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_spec(self, model_id: str) -> ModelSpec:
        """Retrieves specification for given model_id."""
        if model_id not in MODEL_REGISTRY:
            raise ModelNotFoundError(
                f"Unknown model '{model_id}'. Available models: {list(MODEL_REGISTRY.keys())}"
            )
        return MODEL_REGISTRY[model_id]

    def get_model_path(self, model_id: str) -> Path:
        """Returns the local filesystem path to a model artifact."""
        spec = self.get_spec(model_id)
        return self.cache_dir / spec.filename

    def is_downloaded(self, model_id: str, verify: bool = False) -> bool:
        """Checks if model artifact exists locally on disk."""
        path = self.get_model_path(model_id)
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if verify:
            return self.verify_checksum(model_id)
        return True

    def verify_checksum(self, model_id: str) -> bool:
        """Computes and compares SHA-256 hash of local model file against registry."""
        spec = self.get_spec(model_id)
        path = self.get_model_path(model_id)
        if not path.is_file():
            return False

        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        is_valid = digest.lower() == spec.sha256.lower()
        if not is_valid:
            logger.warning(
                f"Checksum mismatch for {model_id}! Expected {spec.sha256}, got {digest}"
            )
        return is_valid

    def download_model(self, model_id: str, force: bool = False) -> Path:
        """Downloads model artifact with progress reporting and checksum verification."""
        spec = self.get_spec(model_id)
        target_path = self.get_model_path(model_id)

        if not force and target_path.is_file() and self.verify_checksum(model_id):
            logger.info(f"Model '{model_id}' is already present and verified at {target_path}")
            return target_path

        logger.info(f"Downloading model '{model_id}' from {spec.url} ({spec.size_mb:.2f} MB)...")

        # Download to a temporary file first to prevent half-written files
        with tempfile.NamedTemporaryFile(delete=False, dir=self.cache_dir) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            req = urllib.request.Request(spec.url, headers={"User-Agent": "Swara-Downloader/1.0"})
            with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as f:
                shutil.copyfileobj(resp, f)

            # Verify checksum of downloaded temp file
            hasher = hashlib.sha256()
            with open(tmp_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            digest = hasher.hexdigest()

            if digest.lower() != spec.sha256.lower():
                raise ModelChecksumError(
                    f"Integrity check failed for {model_id}! Expected {spec.sha256}, got {digest}"
                )

            # Move temp file to final location atomically
            tmp_path.replace(target_path)
            logger.info(f"Successfully saved and verified {model_id} -> {target_path}")
            return target_path

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise ModelError(f"Failed to download model '{model_id}': {e}") from e

    def download_all(self, include_optional: bool = False, force: bool = False) -> Dict[str, Path]:
        """Downloads all default models (and optional ones if specified)."""
        paths: Dict[str, Path] = {}
        for mid, spec in MODEL_REGISTRY.items():
            if spec.is_default or include_optional:
                paths[mid] = self.download_model(mid, force=force)
        return paths

    def list_models(self) -> List[Dict[str, object]]:
        """Returns status of all registered models."""
        result = []
        for mid, spec in MODEL_REGISTRY.items():
            path = self.get_model_path(mid)
            exists = path.is_file()
            verified = self.verify_checksum(mid) if exists else False
            result.append(
                {
                    "model_id": mid,
                    "filename": spec.filename,
                    "architecture": spec.architecture,
                    "size_mb": round(spec.size_mb, 2),
                    "license": spec.license,
                    "is_downloaded": exists,
                    "is_verified": verified,
                    "is_default": spec.is_default,
                    "path": str(path) if exists else None,
                }
            )
        return result
