"""Model metadata, registry, and licensing specifications."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Specification of an approved, validated model artifact."""

    model_id: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    license: str
    license_url: str
    architecture: str
    description: str
    is_default: bool = False

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


# Official validated model registry
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "silero_vad": ModelSpec(
        model_id="silero_vad",
        filename="silero_vad.onnx",
        url="https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
        sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        size_bytes=2327914,  # ~2.22 MB
        license="MIT",
        license_url="https://github.com/snakers4/silero-vad/blob/master/LICENSE",
        architecture="Silero VAD v5 (Stateful LSTM)",
        description="Lightweight enterprise-grade voice activity detection model.",
        is_default=True,
    ),
    "wespeaker_resnet34": ModelSpec(
        model_id="wespeaker_resnet34",
        filename="wespeaker_resnet34.onnx",
        url="https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/main/onnx/model.onnx",
        sha256="3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb",
        size_bytes=26535567,  # ~25.31 MB
        license="Apache-2.0",
        license_url="https://github.com/wenet-e2e/wespeaker/blob/master/LICENSE",
        architecture="WeSpeaker ResNet-34 Large-Margin (256-dim embeddings)",
        description="High-accuracy deep speaker verification model fine-tuned on VoxCeleb.",
        is_default=True,
    ),
    "wespeaker_resnet34_int8": ModelSpec(
        model_id="wespeaker_resnet34_int8",
        filename="wespeaker_resnet34_int8.onnx",
        url="https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/main/onnx/model_int8.onnx",
        sha256="6416fae0599c33bdaf7188e99695edc5fafde190a102c511c881245a109de4c3",
        size_bytes=6685160,  # ~6.38 MB
        license="Apache-2.0",
        license_url="https://github.com/wenet-e2e/wespeaker/blob/master/LICENSE",
        architecture="WeSpeaker ResNet-34 LM INT8 Quantized",
        description="Ultra-compact 6.4MB quantized speaker embedding model for edge and low-memory environments.",
        is_default=False,
    ),
}
