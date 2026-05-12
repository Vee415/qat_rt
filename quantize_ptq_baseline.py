"""Standalone PTQ (Post-Training Quantization) script with verbose output.

Applies static INT8 quantization with calibration to the FP32 baseline model,
evaluates accuracy, exports to ONNX, and compares model sizes.
"""
import sys
sys.path.insert(0, 'src')

import os
import time
import torch
import torch.ao.quantization as quant
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model


def evaluate_on_dataset(model, loader, device, desc="Evaluating"):
    """Evaluate model accuracy on a dataset."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


def main():
    print("=" * 70)
    print("Post-Training Quantization (PTQ) — Static INT8")
    print("=" * 70)

    # PTQ must run on CPU (quantized ops are CPU-only in PyTorch)
    device = torch.device("cpu")
    print(f"Device: {device} (quantization requires CPU)")

    # --- Step 1: Load FP32 baseline ---
    print("\n[1/5] Loading FP32 baseline model...")
    fp32_model = get_model(num_classes=10, quantizable=True)
    fp32_model.load_state_dict(
        torch.load("models/kws_fp32.pth", map_location="cpu", weights_only=True)
    )
    fp32_model.eval()
    print(f"  Loaded from: models/kws_fp32.pth")

    # --- Step 2: FP32 baseline accuracy ---
    print("\n[2/5] Evaluating FP32 baseline on CPU...")
    loaders = get_dataloaders('data/speech_commands', batch_size=256, num_workers=0)
    fp32_acc = evaluate_on_dataset(fp32_model, loaders["test"], device, desc="FP32 Test")
    print(f"  FP32 Test Accuracy: {fp32_acc:.4f} ({fp32_acc*100:.2f}%)")

    # --- Step 3: Prepare for PTQ ---
    print("\n[3/5] Preparing model for PTQ...")
    model_ptq = get_model(num_classes=10, quantizable=True)
    model_ptq.load_state_dict(
        torch.load("models/kws_fp32.pth", map_location="cpu", weights_only=True)
    )
    model_ptq.eval()

    # Fuse Conv+BN+ReLU before quantization (critical for accuracy)
    print("  Fusing Conv+BN+ReLU layers...")
    model_ptq.fuse_model()
    print("  Fused layers: Conv+BN+ReLU x3")

    # Set qconfig: x86 for symmetric quantization
    model_ptq.qconfig = quant.get_default_qconfig("x86")
    print(f"  QConfig: {model_ptq.qconfig}")

    # Insert observers (prepare)
    model_prepared = quant.prepare(model_ptq)
    print("  Observers inserted. Ready for calibration.")

    # --- Step 4: Calibrate with training data ---
    print("\n[4/5] Calibrating with training data (100 batches)...")
    calib_batches = 100
    model_prepared.eval()  # calibration runs in eval mode
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(loaders["train"], desc="Calibrating",
                                              total=calib_batches, ncols=100)):
            if i >= calib_batches:
                break
            model_prepared(inputs)

    # Convert to INT8
    print("\n  Converting to INT8...")
    model_quantized = quant.convert(model_prepared)
    print("  PTQ conversion complete.")

    # --- Step 5: Evaluate quantized model ---
    print("\n[5/5] Evaluating INT8 quantized model...")
    ptq_acc = evaluate_on_dataset(model_quantized, loaders["test"], device, desc="INT8 Test")
    print(f"  INT8 Test Accuracy: {ptq_acc:.4f} ({ptq_acc*100:.2f}%)")

    # --- Results Summary ---
    print(f"\n{'='*70}")
    print("PTQ Results Summary")
    print(f"{'='*70}")
    print(f"  FP32 Accuracy:  {fp32_acc*100:.2f}%")
    print(f"  INT8 Accuracy:  {ptq_acc*100:.2f}%")
    acc_drop = (fp32_acc - ptq_acc) * 100
    print(f"  Accuracy Drop:  {acc_drop:.2f}%")
    if acc_drop < 1.0:
        print("  -> Excellent: < 1% accuracy drop from INT8 quantization")
    elif acc_drop < 2.0:
        print("  -> Good: < 2% accuracy drop — acceptable for edge deployment")
    else:
        print(f"  -> Significant drop — consider QAT (Module 04-06) to recover accuracy")

    # --- Save quantized model ---
    os.makedirs("models", exist_ok=True)
    ptq_path = "models/kws_ptq_int8.pth"
    torch.save(model_quantized.state_dict(), ptq_path)

    # --- ONNX export ---
    print("\nExporting to ONNX...")
    # Get actual input shape from a sample
    sample_batch = next(iter(loaders["test"]))[0]
    input_shape = sample_batch.shape
    print(f"  Input shape: {input_shape}")
    dummy_input = torch.randn(1, 1, input_shape[2], input_shape[3])

    # Export FP32 ONNX — TensorRT will use this for its own INT8 quantization
    fp32_onnx_path = "models/kws_fp32.onnx"
    fp32_model.eval()
    torch.onnx.export(
        fp32_model,
        dummy_input,
        fp32_onnx_path,
        opset_version=17,
        input_names=["mfcc_input"],
        output_names=["keyword_output"],
        dynamic_axes={
            "mfcc_input": {0: "batch_size"},
            "keyword_output": {0: "batch_size"},
        },
    )
    print(f"  FP32 ONNX saved to: {fp32_onnx_path}")

    # Note: quantized PyTorch models don't export cleanly to ONNX
    # (quantized::batch_norm2d is unsupported). TensorRT handles INT8
    # quantization from the FP32 ONNX using its own calibration.

    # --- Size comparison ---
    fp32_pth_size = os.path.getsize("models/kws_fp32.pth") / (1024 * 1024)
    ptq_pth_size = os.path.getsize(ptq_path) / (1024 * 1024)
    fp32_onnx_size = os.path.getsize(fp32_onnx_path) / (1024 * 1024)

    print(f"\n{'='*70}")
    print("Model Size Comparison")
    print(f"{'='*70}")
    print(f"  FP32 .pth:  {fp32_pth_size:.2f} MB")
    print(f"  INT8 .pth:  {ptq_pth_size:.2f} MB  ({fp32_pth_size/ptq_pth_size:.1f}x smaller)")
    print(f"  FP32 .onnx: {fp32_onnx_size:.2f} MB")
    print(f"\nPTQ complete. Next: Module 04 (QAT Prepare) or Module 07 (TensorRT)")


if __name__ == "__main__":
    main()