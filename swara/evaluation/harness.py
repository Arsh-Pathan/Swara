"""Evaluation harness for speaker verification and diarization benchmarks."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import numpy as np

from swara.core.audio import load_audio
from swara.engine import Swara
from swara.evaluation.metrics import compute_eer, compute_min_dcf
from swara.utils.logger import get_logger

logger = get_logger("swara.eval")


class EvaluationHarness:
    """Executes evaluation protocols on audio datasets and reports benchmark metrics."""

    def __init__(self, engine: Optional[Swara] = None) -> None:
        self.engine = engine or Swara()

    def evaluate_from_config(self, config_path: str | Path) -> Dict[str, Any]:
        """Loads evaluation protocol from YAML config and executes verification trials."""
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        trials = cfg.get("trials", [])
        if not trials:
            logger.warning("No evaluation trials found in configuration.")
            return {"eer_percent": 0.0, "total_trials": 0}

        target_scores: List[float] = []
        nontarget_scores: List[float] = []

        # Cache embeddings to avoid redundant extraction
        emb_cache: Dict[str, np.ndarray] = {}

        def get_emb(fpath: str) -> np.ndarray:
            if fpath not in emb_cache:
                emb_cache[fpath] = self.engine.extract_embedding(fpath)
            return emb_cache[fpath]

        for trial in trials:
            f1 = trial["audio_1"]
            f2 = trial["audio_2"]
            is_target = trial["is_same_speaker"]

            emb1 = get_emb(f1)
            emb2 = get_emb(f2)
            sim = float(np.dot(emb1, emb2))

            if is_target:
                target_scores.append(sim)
            else:
                nontarget_scores.append(sim)

        eer, opt_thresh = compute_eer(target_scores, nontarget_scores)
        min_dcf = compute_min_dcf(target_scores, nontarget_scores)

        return {
            "total_trials": len(trials),
            "target_trials": len(target_scores),
            "nontarget_trials": len(nontarget_scores),
            "eer_percent": round(eer, 2),
            "optimal_threshold": round(opt_thresh, 4),
            "min_dcf": round(min_dcf, 4),
            "mean_target_similarity": round(float(np.mean(target_scores)), 4) if target_scores else None,
            "mean_nontarget_similarity": round(float(np.mean(nontarget_scores)), 4) if nontarget_scores else None,
        }
