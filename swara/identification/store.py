"""Persistent storage for enrolled speaker profiles and embedding vectors."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple
import numpy as np

from swara.core.config import settings
from swara.core.types import EnrolledSpeaker
from swara.utils.logger import get_logger

logger = get_logger("swara.store")


class SpeakerStore:
    """Thread-safe and process-safe persistent store for speaker embeddings and metadata."""

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self.storage_dir = Path(storage_dir or settings.speaker_store_path)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "speakers_meta.json"
        self.embeddings_dir = self.storage_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self._speakers: Dict[str, EnrolledSpeaker] = {}
        self._load()

    def _load(self) -> None:
        """Loads metadata from JSON storage."""
        if self.metadata_file.is_file():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for sid, d in data.items():
                    self._speakers[sid] = EnrolledSpeaker(
                        speaker_id=d["speaker_id"],
                        name=d.get("name"),
                        created_at=d.get("created_at"),
                        updated_at=d.get("updated_at"),
                        sample_count=d.get("sample_count", 1),
                        total_audio_duration_s=d.get("total_audio_duration_s", 0.0),
                        embedding_dim=d.get("embedding_dim", 256),
                        model_version=d.get("model_version", "wespeaker-resnet34-lm"),
                        metadata=d.get("metadata", {}),
                    )
            except Exception as e:
                logger.error(f"Failed to read speaker metadata: {e}")

    def _save(self) -> None:
        """Atomically persists metadata to disk using a temporary file."""
        data = {sid: spk.to_dict() for sid, spk in self._speakers.items()}
        with tempfile.NamedTemporaryFile("w", delete=False, dir=self.storage_dir, encoding="utf-8") as tmp:
            json.dump(data, tmp, indent=2)
            tmp_path = Path(tmp.name)
        tmp_path.replace(self.metadata_file)

    def _get_embedding_path(self, speaker_id: str) -> Path:
        # Sanitize speaker_id to prevent path traversal
        clean_id = "".join(c for c in speaker_id if c.isalnum() or c in ("-", "_"))
        return self.embeddings_dir / f"{clean_id}.npy"

    def get_speaker(self, speaker_id: str) -> Optional[EnrolledSpeaker]:
        """Gets metadata for speaker."""
        return self._speakers.get(speaker_id)

    def get_embedding(self, speaker_id: str) -> Optional[np.ndarray]:
        """Loads embedding vector from disk."""
        path = self._get_embedding_path(speaker_id)
        if not path.is_file():
            return None
        try:
            emb = np.load(path).astype(np.float32)
            # Ensure unit norm
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb = emb / norm
            return emb
        except Exception as e:
            logger.error(f"Error loading embedding for {speaker_id}: {e}")
            return None

    def list_speakers(self) -> List[EnrolledSpeaker]:
        """Returns list of all enrolled speakers."""
        return list(self._speakers.values())

    def save_speaker(
        self,
        speaker: EnrolledSpeaker,
        embedding: np.ndarray,
        duration_s: float = 0.0,
    ) -> None:
        """Saves a new speaker or updates an existing speaker with running centroid average."""
        speaker_id = speaker.speaker_id
        path = self._get_embedding_path(speaker_id)

        # Normalize embedding to unit norm
        norm = np.linalg.norm(embedding)
        unit_emb = embedding / norm if norm > 1e-12 else embedding

        if speaker_id in self._speakers and path.is_file():
            # Update existing speaker by computing running average of embeddings
            existing = self._speakers[speaker_id]
            existing_emb = self.get_embedding(speaker_id)
            if existing_emb is not None:
                # Weighted average based on sample count
                n = existing.sample_count
                combined = (existing_emb * n + unit_emb) / (n + 1)
                comb_norm = np.linalg.norm(combined)
                unit_emb = combined / comb_norm if comb_norm > 1e-12 else combined
                existing.sample_count += 1
                existing.total_audio_duration_s += duration_s
                from datetime import datetime

                existing.updated_at = datetime.utcnow().isoformat()
                if speaker.name:
                    existing.name = speaker.name
                self._speakers[speaker_id] = existing
            else:
                self._speakers[speaker_id] = speaker
        else:
            self._speakers[speaker_id] = speaker

        # Atomically write numpy file
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=self.embeddings_dir) as tmp:
            np.save(tmp, unit_emb.astype(np.float32))
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)

        self._save()
        logger.info(f"Persisted speaker '{speaker_id}' (samples: {self._speakers[speaker_id].sample_count})")

    def delete_speaker(self, speaker_id: str) -> bool:
        """Deletes a speaker profile and their embedding file."""
        if speaker_id not in self._speakers:
            return False

        del self._speakers[speaker_id]
        path = self._get_embedding_path(speaker_id)
        if path.exists():
            path.unlink()

        self._save()
        logger.info(f"Deleted speaker '{speaker_id}'")
        return True

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Returns tuple of (speaker_ids, embeddings_matrix [N, D])."""
        speaker_ids = []
        embeddings = []
        for sid in self._speakers.keys():
            emb = self.get_embedding(sid)
            if emb is not None:
                speaker_ids.append(sid)
                embeddings.append(emb)

        if not embeddings:
            return [], np.zeros((0, 256), dtype=np.float32)

        return speaker_ids, np.vstack(embeddings).astype(np.float32)
