from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoConfig


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    """
    Load the project's YAML configuration.

    This function is intentionally very small:
    every other Python file will receive the same configuration
    instead of independently defining model/training parameters.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.resolve()}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a YAML mapping/object.")

    return config


def get_dtype(name: str) -> torch.dtype:
    """
    Convert the dtype name in config.yaml into a PyTorch dtype.
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if name not in dtype_map:
        raise ValueError(
            f"Unsupported dtype '{name}'. "
            f"Choose one of: {', '.join(dtype_map)}"
        )

    return dtype_map[name]


def load_model_info(config: dict[str, Any]) -> dict[str, Any]:
    """
    Inspect the selected Hugging Face model and derive the
    architecture information required by the NLA pipeline.

    Nothing here is Qwen-specific.

    Returns:
        model_name
        model_type
        hidden_size
        num_hidden_layers
        extraction_layer
        ar_num_hidden_layers
    """
    model_cfg = config["model"]
    extraction_cfg = config["extraction"]

    model_name = model_cfg["base_model"]

    hf_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    # NLA is defined around decoder-only causal language models.
    if getattr(hf_config, "is_encoder_decoder", False):
        raise ValueError(
            f"{model_name} is an encoder-decoder model. "
            "This NLA implementation requires a decoder-only causal LM."
        )

    hidden_size = getattr(hf_config, "hidden_size", None)
    num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)

    if hidden_size is None:
        raise ValueError(
            f"Could not determine hidden_size for {model_name}."
        )

    if num_hidden_layers is None:
        raise ValueError(
            f"Could not determine num_hidden_layers for {model_name}."
        )
    
    # Determine extraction layer K.
    explicit_layer = extraction_cfg.get("layer_index")
    strategy = extraction_cfg.get("strategy", "two_thirds")

    if explicit_layer is not None:
        extraction_layer = int(explicit_layer)

    elif strategy == "two_thirds":
        extraction_layer = round((2 * num_hidden_layers) / 3)

    else:
        raise ValueError(
            f"Unknown extraction strategy: {strategy}"
        )

    if not 0 <= extraction_layer < num_hidden_layers:
        raise ValueError(
            f"Extraction layer K={extraction_layer} is invalid for "
            f"a model with {num_hidden_layers} transformer layers."
        )

    # Authors' AR architecture:
    # K = extraction index
    # AR backbone = first K+1 transformer blocks.
    ar_num_hidden_layers = extraction_layer + 1

    return {
        "model_name": model_name,
        "model_type": getattr(hf_config, "model_type", None),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "extraction_layer": extraction_layer,
        "ar_num_hidden_layers": ar_num_hidden_layers,
    }