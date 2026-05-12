"""
ONNX export utility for PyTorch models.

Exports FP32, PTQ INT8, and QAT INT8 models to ONNX format.
"""

import argparse
import os

import torch

from model import get_model


def export_to_onnx(model_path: str, onnx_path: str, num_classes: int = 10,
                    n_mfcc: int = 40, input_length: int = 98,
                    quantizable: bool = False, is_quantized: bool = False):
    """Export a PyTorch model to ONNX.

    Args:
        model_path: Path to the .pth checkpoint
        onnx_path: Output path for the .onnx file
        num_classes: Number of output classes
        n_mfcc: Number of MFCC coefficients
        input_length: Number of time frames in MFCC input
        quantizable: Whether model has QuantStub/DeQuantStub
        is_quantized: Whether model is already converted to INT8
    """
    model = get_model(num_classes=num_classes, quantizable=quantizable)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    if is_quantized:
        model = torch.ao.quantization.convert(model)

    # Create dummy input matching MFCC shape: (1, 1, n_mfcc, time_frames)
    dummy_input = torch.randn(1, 1, n_mfcc, input_length)

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=17,
        input_names=["mfcc_input"],
        output_names=["keyword_output"],
        dynamic_axes={
            "mfcc_input": {0: "batch_size"},
            "keyword_output": {0: "batch_size"},
        },
    )

    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Exported ONNX model to {onnx_path} ({onnx_size:.2f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .pth model file")
    parser.add_argument("--onnx_path", type=str, required=True,
                        help="Output path for .onnx file")
    parser.add_argument("--quantizable", action="store_true",
                        help="Model has QuantStub/DeQuantStub")
    parser.add_argument("--is_quantized", action="store_true",
                        help="Model is already INT8 quantized")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--n_mfcc", type=int, default=40)
    args = parser.parse_args()

    export_to_onnx(
        args.model_path, args.onnx_path,
        quantizable=args.quantizable,
        is_quantized=args.is_quantized,
    )


if __name__ == "__main__":
    main()