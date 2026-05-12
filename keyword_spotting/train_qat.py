"""Standalone QAT (Quantization-Aware Training) script with verbose output.

Three-stage pipeline:
1. Prepare: load FP32, fuse Conv+BN+ReLU, insert fake quant nodes
2. Train: fine-tune with quantization noise, freeze BN at epoch N
3. Convert: fold BN, remove fake quant, produce INT8 model

Compares FP32 vs PTQ INT8 vs QAT INT8 accuracy.
"""
import sys
sys.path.insert(0, 'src')

import os
import time
import copy
import torch
import torch.ao.quantization as quant
import torch.nn as nn
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
    print("Quantization-Aware Training (QAT) — Full Pipeline")
    print("=" * 70)

    # QAT runs on CPU (fake quant ops work on CPU; actual INT8 ops are CPU-only)
    device = torch.device("cpu")
    print(f"Device: {device} (QAT fake-quant works on CPU)")

    # --- Load data ---
    print("\nLoading dataset...")
    loaders = get_dataloaders('data/speech_commands', batch_size=256, num_workers=0)

    # ================================================================
    # Stage 1: Prepare QAT
    # ================================================================
    print("\n" + "=" * 70)
    print("Stage 1: Prepare QAT — Fuse BN, Insert Fake Quant Nodes")
    print("=" * 70)

    model_qat = get_model(num_classes=10, quantizable=True)
    model_qat.load_state_dict(
        torch.load("models/kws_fp32.pth", map_location="cpu", weights_only=True)
    )
    model_qat.eval()

    # Fuse Conv+BN+ReLU (critical before prepare_qat)
    print("  Fusing Conv+BN+ReLU layers...")
    model_qat.fuse_model()
    print("  Fused: Conv+BN+ReLU x3")

    # Configure QAT observers
    # get_default_qat_qconfig uses FakeQuantize observers (learnable scale/zero_point)
    model_qat.qconfig = quant.get_default_qat_qconfig("x86")
    print(f"  QConfig: {model_qat.qconfig}")

    # Insert fake quant nodes (prepare_qat)
    model_qat = quant.prepare_qat(model_qat.train())
    print("  Fake quantization nodes inserted.")
    print(f"  Model has {sum(1 for n, m in model_qat.named_modules() if 'FakeQuantize' in type(m).__name__)} fake quant modules")

    # ================================================================
    # Stage 2: Train with Fake Quantization
    # ================================================================
    print("\n" + "=" * 70)
    print("Stage 2: QAT Training — Fine-tune with Quantization Noise")
    print("=" * 70)

    epochs = 15
    lr = 1e-4
    freeze_bn_epoch = 3
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_qat.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    best_state = None

    print(f"  Epochs: {epochs}, LR: {lr}, BN freeze at epoch: {freeze_bn_epoch}")
    print(f"  {'='*60}")

    for epoch in range(epochs):
        t0 = time.time()

        # Freeze BN after specified epoch
        if epoch == freeze_bn_epoch:
            print(f"\n  *** Freezing BatchNorm at epoch {epoch} ***")
            for module in model_qat.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    module.track_running_stats = False
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False

        # --- Train ---
        model_qat.train()
        # Keep BN in eval mode after freeze
        if epoch >= freeze_bn_epoch:
            for module in model_qat.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

        running_loss = 0.0
        correct, total = 0, 0
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{epochs} [QAT Train]",
                     leave=False, ncols=100)

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_qat(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{correct/total:.3f}")

        train_acc = correct / total
        train_loss = running_loss / total

        # --- Validate ---
        val_acc = evaluate_on_dataset(model_qat, loaders["val"], device,
                                       desc=f"Epoch {epoch+1} [QAT Val]")
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:2d}/{epochs} | "
              f"Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model_qat.state_dict())
            torch.save(best_state, "models/kws_qat_best.pth")
            print(f"    -> Saved best QAT model (val_acc={val_acc:.4f})")

    print(f"\n  Best QAT val accuracy: {best_val_acc:.4f}")

    # ================================================================
    # Stage 3: Convert to INT8
    # ================================================================
    print("\n" + "=" * 70)
    print("Stage 3: Convert QAT Model to INT8")
    print("=" * 70)

    # Load best QAT checkpoint and convert
    model_qat.load_state_dict(best_state)
    model_qat.eval()

    # Convert: folds BN into conv weights, removes fake quant, produces INT8
    model_int8 = quant.convert(model_qat)
    print("  QAT model converted to INT8.")

    # Evaluate INT8 model
    qat_int8_acc = evaluate_on_dataset(model_int8, loaders["test"], device,
                                        desc="QAT INT8 Test")
    print(f"  QAT INT8 Test Accuracy: {qat_int8_acc:.4f} ({qat_int8_acc*100:.2f}%)")

    # Save INT8 model
    qat_int8_path = "models/kws_qat_int8.pth"
    torch.save(model_int8.state_dict(), qat_int8_path)

    # ================================================================
    # Comparison with FP32 and PTQ
    # ================================================================
    print("\n" + "=" * 70)
    print("Full Comparison: FP32 vs PTQ INT8 vs QAT INT8")
    print("=" * 70)

    # FP32 baseline
    fp32_model = get_model(num_classes=10, quantizable=True)
    fp32_model.load_state_dict(
        torch.load("models/kws_fp32.pth", map_location="cpu", weights_only=True)
    )
    fp32_model.eval()
    fp32_acc = evaluate_on_dataset(fp32_model, loaders["test"], device,
                                    desc="FP32 Test")

    # PTQ INT8 (re-run for fair comparison)
    ptq_model = get_model(num_classes=10, quantizable=True)
    ptq_model.load_state_dict(
        torch.load("models/kws_fp32.pth", map_location="cpu", weights_only=True)
    )
    ptq_model.eval()
    ptq_model.fuse_model()
    ptq_model.qconfig = quant.get_default_qconfig("x86")
    ptq_prepared = quant.prepare(ptq_model)
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loaders["train"]):
            if i >= 100:
                break
            ptq_prepared(inputs)
    ptq_int8 = quant.convert(ptq_prepared)
    ptq_acc = evaluate_on_dataset(ptq_int8, loaders["test"], device,
                                   desc="PTQ INT8 Test")

    # Print results table
    print(f"\n  {'Model':<15} {'Accuracy':>10} {'vs FP32':>10}")
    print(f"  {'-'*35}")
    print(f"  {'FP32':<15} {fp32_acc*100:>8.2f}% {'(baseline)':>10}")
    print(f"  {'PTQ INT8':<15} {ptq_acc*100:>8.2f}% {(ptq_acc-fp32_acc)*100:>+9.2f}%")
    print(f"  {'QAT INT8':<15} {qat_int8_acc*100:>8.2f}% {(qat_int8_acc-fp32_acc)*100:>+9.2f}%")

    # Size comparison
    fp32_size = os.path.getsize("models/kws_fp32.pth") / (1024 * 1024)
    ptq_size = os.path.getsize("models/kws_ptq_int8.pth") / (1024 * 1024)
    qat_size = os.path.getsize(qat_int8_path) / (1024 * 1024)

    print(f"\n  {'Model':<15} {'Size':>10}")
    print(f"  {'-'*25}")
    print(f"  {'FP32 .pth':<15} {fp32_size:>8.2f} MB")
    print(f"  {'PTQ INT8 .pth':<15} {ptq_size:>8.2f} MB ({fp32_size/ptq_size:.1f}x smaller)")
    print(f"  {'QAT INT8 .pth':<15} {qat_size:>8.2f} MB ({fp32_size/qat_size:.1f}x smaller)")

    # Export FP32 ONNX if not already done
    if not os.path.exists("models/kws_fp32.onnx"):
        print("\n  Exporting FP32 ONNX...")
        sample = next(iter(loaders["test"]))[0]
        dummy_input = torch.randn(1, 1, sample.shape[2], sample.shape[3])
        torch.onnx.export(fp32_model, dummy_input, "models/kws_fp32.onnx",
                          opset_version=17,
                          input_names=["mfcc_input"],
                          output_names=["keyword_output"],
                          dynamic_axes={"mfcc_input": {0: "batch_size"},
                                        "keyword_output": {0: "batch_size"}})
        print(f"  Saved: models/kws_fp32.onnx")

    print(f"\nQAT training complete. Saved:")
    print(f"  - models/kws_qat_best.pth (best QAT checkpoint with fake quant)")
    print(f"  - models/kws_qat_int8.pth (converted INT8 model)")
    print(f"\nNext: Module 07 (TensorRT Engine Build)")


if __name__ == "__main__":
    main()