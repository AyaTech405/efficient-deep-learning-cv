"""Post-training optimization utilities.

The default optimization uses PyTorch dynamic post-training quantization for
Linear layers. This is deliberately conservative and does not require extra
runtime packages. For these CNNs, most convolutional computation remains in
floating point, so the expected size/latency improvement may be modest.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from .benchmark import measure_latency
from .data import create_dataloaders, set_seed
from .evaluate import evaluate_model
from .models import build_model, model_size_mb


def load_checkpoint(
    model_name: str,
    checkpoint: str,
    device: torch.device,
) -> nn.Module:
    """Load a floating-point model checkpoint."""
    model = build_model(model_name, pretrained=False).to(device)

    data = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    state_dict = data.get("model_state_dict", data)
    model.load_state_dict(state_dict)

    return model.eval()


def dynamic_quantize(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to supported Linear layers."""
    quantized = copy.deepcopy(model).cpu().eval()

    return torch.ao.quantization.quantize_dynamic(
        quantized,
        {nn.Linear},
        dtype=torch.qint8,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply conservative post-training quantization."
    )

    parser.add_argument(
        "--model",
        choices=["resnet18", "mobilenetv3-small"],
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="results/optimization.json")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        train_size=1,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Use CPU for both original and optimized models so that
    # the latency comparison is fair.
    cpu_device = torch.device("cpu")

    original = load_checkpoint(
        args.model,
        args.checkpoint,
        cpu_device,
    )

    original_metrics = evaluate_model(
        original,
        test_loader,
        cpu_device,
    )

    original_latency = measure_latency(
        original,
        cpu_device,
    )

    original_size = model_size_mb(original)

    optimized = dynamic_quantize(original)

    optimized_metrics = evaluate_model(
        optimized,
        test_loader,
        cpu_device,
    )

    optimized_latency = measure_latency(
        optimized,
        cpu_device,
    )

    optimized_size = model_size_mb(optimized)

    result = {
        "model": args.model,
        "method": "dynamic_int8_linear_quantization",
        "original": {
            **original_metrics,
            "model_size_mb": original_size,
            "inference_latency_ms": original_latency,
        },
        "optimized": {
            **optimized_metrics,
            "model_size_mb": optimized_size,
            "inference_latency_ms": optimized_latency,
        },
        "note": (
            "Dynamic quantization affects Linear layers. CNN convolutional "
            "layers remain floating point, so improvements may be limited."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()