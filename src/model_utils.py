import json
from pathlib import Path

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, list_repo_files
from loguru import logger

from config import Config

try:
    from safetensors.torch import load_file as load_safetensors_file
except ImportError:
    load_safetensors_file = None


def resolve_model_name(config: Config) -> str:
    if config.model.hf_repo_id:
        return f"hf_hub:{config.model.hf_repo_id}"
    return config.model.model


def clean_state_dict(state_dict):
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def extract_state_dict(checkpoint):
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    return clean_state_dict(state_dict)


def load_state_dict_file(checkpoint_path: Path):
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix == ".safetensors":
        if load_safetensors_file is None:
            raise ImportError("safetensors is required to load `.safetensors` checkpoints.")
        state_dict = load_safetensors_file(str(checkpoint_path))
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)
    return clean_state_dict(state_dict)


def hf_repo_has_custom_resnet_stem(config: Config):
    if not config.model.hf_repo_id or not config.model.model.startswith("resnet"):
        return False
    try:
        config_path = hf_hub_download(config.model.hf_repo_id, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            repo_config = json.load(f)
        input_size = repo_config.get("input_size") or repo_config.get("pretrained_cfg", {}).get("input_size")
        return input_size == [3, 32, 32]
    except Exception as exc:
        logger.warning(
            "Failed to inspect Hugging Face config for {}: {}",
            config.model.hf_repo_id,
            exc,
        )
        return False


def adapt_model_to_state_dict(model, state_dict):
    if hasattr(model, "conv1") and "conv1.weight" in state_dict:
        current_shape = tuple(model.conv1.weight.shape)
        target_shape = tuple(state_dict["conv1.weight"].shape)
        if current_shape != target_shape:
            out_channels, in_channels, kernel_h, kernel_w = target_shape
            model.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_h, kernel_w),
                stride=(1, 1) if kernel_h <= 3 else model.conv1.stride,
                padding=(kernel_h // 2, kernel_w // 2),
                bias=model.conv1.bias is not None,
            )
            if kernel_h <= 3 and hasattr(model, "maxpool"):
                model.maxpool = nn.Identity()

    if hasattr(model, "fc") and "fc.weight" in state_dict:
        out_features, in_features = state_dict["fc.weight"].shape
        if tuple(model.fc.weight.shape) != (out_features, in_features):
            model.fc = nn.Linear(in_features, out_features, bias=model.fc.bias is not None)

    return model


def create_model_from_config(config: Config, pretrained: bool | None = None):
    if pretrained is None:
        pretrained = config.model.pretrained

    # Some Hugging Face timm repos, such as CIFAR-100 ResNet checkpoints, keep
    # the standard `resnet18` architecture name in config.json but actually use
    # a CIFAR-style stem (3x3 conv + no maxpool). Build the base model locally
    # and adapt it to the downloaded weights instead of relying on timm's direct
    # hf_hub loading path in those cases.
    if pretrained and hf_repo_has_custom_resnet_stem(config):
        files = set(list_repo_files(config.model.hf_repo_id))
        preferred_files = ("model.safetensors", "pytorch_model.bin")
        checkpoint_file = next((name for name in preferred_files if name in files), None)
        if checkpoint_file is None:
            raise FileNotFoundError(
                f"No supported checkpoint file found in Hugging Face repo {config.model.hf_repo_id}"
            )
        checkpoint_path = Path(hf_hub_download(config.model.hf_repo_id, checkpoint_file))
        state_dict = load_state_dict_file(checkpoint_path)
        model = timm.create_model(
            config.model.model,
            pretrained=False,
            num_classes=config.data.num_class,
        )
        model = adapt_model_to_state_dict(model, state_dict)
        incompatible = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "Loaded pretrained weights from Hugging Face repo {} with {} missing keys and {} unexpected keys",
            config.model.hf_repo_id,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        return model

    return timm.create_model(
        resolve_model_name(config),
        pretrained=pretrained,
        num_classes=config.data.num_class,
    )
