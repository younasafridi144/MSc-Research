from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms

try:
    import timm
    from timm.models.swin_transformer import window_reverse
except ImportError as exc:  # pragma: no cover - shown to the GUI user at runtime
    raise RuntimeError("The 'timm' package is required to load Swin Transformer models.") from exc


DEFAULT_MODEL_NAME = "swin_base_patch4_window7_224"
DEFAULT_IMAGE_SIZE = 672
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelLoadInfo:
    device: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]


@dataclass(frozen=True)
class Prediction:
    value: float | None
    label: str | None
    probabilities: list[tuple[str, float]]
    raw_outputs: list[float]


@dataclass(frozen=True)
class TrainingMetadata:
    model_name: str | None = None
    target_name: str = "GSI"
    image_size: int | None = None
    y_mean: float | None = None
    y_std: float | None = None


def load_training_metadata(path: str | Path) -> TrainingMetadata:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    return TrainingMetadata(
        model_name=data.get("model_name"),
        target_name=data.get("target_col") or "GSI",
        image_size=int(data["img_size"]) if data.get("img_size") is not None else None,
        y_mean=float(data["y_mean"]) if data.get("y_mean") is not None else None,
        y_std=float(data["y_std"]) if data.get("y_std") is not None else None,
    )


def available_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but PyTorch cannot see a CUDA device.")
    return torch.device(requested)


def parse_float_tuple(value: str, fallback: Iterable[float]) -> tuple[float, float, float]:
    if not value.strip():
        return tuple(float(x) for x in fallback)  # type: ignore[return-value]
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected exactly three comma-separated numbers.")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Could not find model weights in this checkpoint. Expected a state dict or a "
        "dict containing one of: state_dict, model_state_dict, model, net, weights."
    )


def _load_checkpoint_file(checkpoint_path: Path) -> Any:
    if checkpoint_path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise RuntimeError("Install 'safetensors' to load .safetensors checkpoints.") from exc
        return load_file(str(checkpoint_path), device="cpu")

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _strip_known_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "net.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {_strip_known_prefixes(key): value for key, value in state_dict.items()}


def _jet_colormap(normalized: np.ndarray) -> np.ndarray:
    normalized = np.clip(normalized, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * normalized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * normalized - 1.0), 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


class GSIPredictor:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        output_size: int = 1,
        image_size: int = DEFAULT_IMAGE_SIZE,
        mean: tuple[float, float, float] = DEFAULT_MEAN,
        std: tuple[float, float, float] = DEFAULT_STD,
        y_mean: float | None = None,
        y_std: float | None = None,
        device: str = "auto",
    ) -> None:
        if output_size < 1:
            raise ValueError("Output size must be at least 1.")
        self.model_name = model_name
        self.output_size = output_size
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.y_mean = y_mean
        self.y_std = y_std
        self.device = available_device(device)
        self.model = timm.create_model(model_name, pretrained=False, num_classes=output_size)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def load_checkpoint(self, checkpoint_path: str | Path) -> ModelLoadInfo:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = _load_checkpoint_file(checkpoint_path)
        state_dict = _normalize_state_dict(_checkpoint_state_dict(checkpoint))
        model_state = self.model.state_dict()

        compatible: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        unexpected: list[str] = []
        for key, tensor in state_dict.items():
            if key not in model_state:
                unexpected.append(key)
                continue
            if tuple(model_state[key].shape) != tuple(tensor.shape):
                skipped.append(f"{key} {tuple(tensor.shape)} -> {tuple(model_state[key].shape)}")
                continue
            compatible[key] = tensor

        result = self.model.load_state_dict(compatible, strict=False)
        self.model.to(self.device)
        self.model.eval()

        combined_unexpected = tuple(unexpected) + tuple(result.unexpected_keys)
        return ModelLoadInfo(
            device=str(self.device),
            missing_keys=tuple(result.missing_keys),
            unexpected_keys=combined_unexpected,
            skipped_keys=tuple(skipped),
        )

    def predict_image(
        self,
        image_path: str | Path,
        task: str = "regression",
        class_names: list[str] | None = None,
    ) -> Prediction:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor).squeeze(0)

        prediction, _ = self._build_prediction(outputs, task, class_names)
        return prediction

    def predict_image_with_heatmap(
        self,
        image_path: str | Path,
        task: str = "regression",
        class_names: list[str] | None = None,
    ) -> tuple[Prediction, Image.Image]:
        image = Image.open(image_path).convert("RGB")
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        tensor_array = self.transform(resized).unsqueeze(0).to(self.device)
        tensor_array.requires_grad_(True)

        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def forward_hook(module: torch.nn.Module, _input: tuple[Any, ...], output: torch.Tensor) -> None:
            activations.append(output)

        def backward_hook(module: torch.nn.Module, grad_input: tuple[Any, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
            gradients.append(grad_output[0])

        target_blocks = [self.model.layers[2].blocks[-1], self.model.layers[3].blocks[-1]]
        hooks: list[torch.utils.hooks.RemovableHandle] = []
        for block in target_blocks:
            hooks.append(block.register_forward_hook(forward_hook))
            if hasattr(block, "register_full_backward_hook"):
                hooks.append(block.register_full_backward_hook(backward_hook))
            else:
                hooks.append(block.register_backward_hook(backward_hook))

        try:
            outputs = self.model(tensor_array)
            outputs_flat = outputs.squeeze(0)
            prediction, target_index = self._build_prediction(outputs_flat.detach(), task, class_names)
            target_score = outputs_flat[target_index]
            self.model.zero_grad(set_to_none=True)
            target_score.backward(retain_graph=False)
        finally:
            for hook in hooks:
                hook.remove()

        if not activations or not gradients:
            raise RuntimeError("Transformer relevancy hooks did not capture activations or gradients.")

        relevancy_maps: list[np.ndarray] = []
        for activation, gradient in zip(activations, reversed(gradients)):
            relevance_tensor = torch.relu(activation * gradient)
            relevance_tensor = relevance_tensor.mean(dim=-1).squeeze(0)

            if relevance_tensor.ndim == 2:
                rel_2d = relevance_tensor.cpu().detach().numpy()
            else:
                num_tokens = int(relevance_tensor.shape[0])
                spatial_dim = math.isqrt(num_tokens)
                if spatial_dim * spatial_dim != num_tokens:
                    raise RuntimeError(
                        f"Unexpected token count for relevancy reshape: {num_tokens}."
                    )
                rel_2d = relevance_tensor.reshape(spatial_dim, spatial_dim).cpu().detach().numpy()

            rel_resized = cv2.resize(rel_2d, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            relevancy_maps.append(rel_resized)

        aggregated_relevancy = np.sum(relevancy_maps, axis=0)
        aggregated_relevancy = (aggregated_relevancy - aggregated_relevancy.min()) / (
            aggregated_relevancy.max() - aggregated_relevancy.min() + 1e-8
        )

        original_rgb = np.asarray(image, dtype=np.float32) / 255.0
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * aggregated_relevancy), cv2.COLORMAP_JET)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        heatmap_colored_rgb = cv2.resize(
            heatmap_colored_rgb,
            (original_rgb.shape[1], original_rgb.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        overlay_float = 0.5 * original_rgb + 0.5 * heatmap_colored_rgb
        overlay_image = Image.fromarray(np.uint8(255 * np.clip(overlay_float, 0.0, 1.0)))

        return prediction, overlay_image

    def predict_image_with_saliency(
        self,
        image_path: str | Path,
        task: str = "regression",
        class_names: list[str] | None = None,
    ) -> tuple[Prediction, Image.Image]:
        image = Image.open(image_path).convert("RGB")
        resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        tensor_array = self.transform(resized).unsqueeze(0).to(self.device)
        tensor_array.requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        outputs = self.model(tensor_array)
        outputs_flat = outputs.squeeze(0)
        prediction, target_index = self._build_prediction(outputs_flat.detach(), task, class_names)
        target_score = outputs_flat[target_index]
        target_score.backward(retain_graph=False)

        if tensor_array.grad is None:
            raise RuntimeError("Saliency gradients not available; backward pass failed.")

        saliency_map = tensor_array.grad.abs().max(dim=1)[0].squeeze(0).cpu()
        saliency_map = saliency_map - saliency_map.min()
        denom = saliency_map.max() + 1e-8
        saliency_map = saliency_map / denom
        heatmap = np.uint8(255 * saliency_map.numpy())

        heatmap_rgb = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
        heatmap_rgb[..., 0] = heatmap
        heatmap_rgb[..., 1] = np.uint8(255 * (1.0 - saliency_map.numpy()))
        heatmap_image = Image.fromarray(heatmap_rgb, mode="RGB")
        heatmap_image = heatmap_image.resize(resized.size, Image.Resampling.BILINEAR)
        overlay = Image.blend(resized, heatmap_image, alpha=0.4)
        overlay = overlay.resize(image.size, Image.Resampling.BILINEAR)
        return prediction, overlay

    def _build_prediction(
        self,
        outputs: torch.Tensor,
        task: str,
        class_names: list[str] | None,
    ) -> tuple[Prediction, int]:
        outputs = outputs.detach().cpu().flatten()
        raw_outputs = [float(value) for value in outputs.tolist()]
        if task == "classification":
            probabilities_tensor = torch.softmax(outputs, dim=0)
            probabilities = [float(value) for value in probabilities_tensor.tolist()]
            labels = _labels_for_outputs(len(probabilities), class_names)
            ranked = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)
            target_index = int(probabilities_tensor.argmax().item())
            return (
                Prediction(
                    value=None,
                    label=ranked[0][0] if ranked else None,
                    probabilities=ranked,
                    raw_outputs=raw_outputs,
                ),
                target_index,
            )

        value = raw_outputs[0] if raw_outputs else None
        if value is not None and self.y_mean is not None and self.y_std is not None:
            value = value * self.y_std + self.y_mean
        return Prediction(value=value, label=None, probabilities=[], raw_outputs=raw_outputs), 0

def _labels_for_outputs(size: int, class_names: list[str] | None) -> list[str]:
    if class_names:
        names = [name.strip() for name in class_names if name.strip()]
        if len(names) >= size:
            return names[:size]
        return names + [f"Class {index}" for index in range(len(names), size)]
    return [f"Class {index}" for index in range(size)]
