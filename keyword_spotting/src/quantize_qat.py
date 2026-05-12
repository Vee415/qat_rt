"""
Quantization-Aware Training (QAT) pipeline.

Three-stage process (corresponds to modules m04, m05, m06):
1. prepare_qat() — insert fake quant nodes, configure observers
2. train_with_fake_quant — fine-tune with quantization noise
3. convert — fold BN, convert to INT8, export
"""

import argparse
import copy
import os

import torch
import torch.ao.quantization as quant
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import get_model
from evaluate import evaluate_model


def prepare_qat_model(model_path: str, device: str = "cpu"):
    """Stage 1: Load FP32 checkpoint and prepare for QAT.

    Inserts fake quantization nodes and configures observers.
    """
    model = get_model(num_classes=10, quantizable=True)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Configure QAT: HistogramObserver for activations, MinMaxObserver for weights
    model.qconfig = quant.get_default_qat_qconfig("x86")
    print(f"QAT qconfig: {model.qconfig}")

    # Prepare: inserts fake quant nodes
    model_prepared = quant.prepare_qat(model.train())
    print("QAT model prepared. Fake quantization nodes inserted.")
    print(f"Model summary:\n{model_prepared}")

    return model_prepared


def train_qat(model_prepared, loaders, epochs: int = 15, lr: float = 1e-4,
               freeze_bn_epoch: int = 3, device: str = "cpu",
               save_path: str = "models/kws_qat_best.pth"):
    """Stage 2: Fine-tune with fake quantization noise.

    Key steps:
    - Start from FP32 checkpoint with fake quant active
    - Use lower learning rate (1e-4 typical)
    - Freeze BatchNorm after freeze_bn_epoch to stabilize quantization
    """
    model_prepared.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_prepared.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Freeze BN after specified epoch
        if epoch == freeze_bn_epoch:
            print(f"\n=== Freezing BatchNorm at epoch {epoch} ===")
            model_prepared.apply(nn.BatchNorm2d)
            # Apply BN freeze to all BN modules
            for module in model_prepared.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    module.track_running_stats = False
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False

        model_prepared.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Re-enable BN running stats tracking before freeze point
        if epoch < freeze_bn_epoch:
            for module in model_prepared.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train()
                    module.track_running_stats = True

        for inputs, labels in loaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_prepared(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = correct / total

        # Validate
        val_acc, _ = evaluate_model(model_prepared, loaders["val"], device)
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model_prepared.state_dict(), save_path)
            print(f"  Saved best QAT model (val_acc={val_acc:.4f})")

    print(f"\nBest QAT val accuracy: {best_val_acc:.4f}")
    return model_prepared


def convert_qat(model_path: str, data_dir: str, batch_size: int = 64,
                device: str = "cpu"):
    """Stage 3: Convert QAT model to INT8.

    Folds BatchNorm into conv weights and removes fake quant nodes.
    """
    # Load best QAT checkpoint
    model = get_model(num_classes=10, quantizable=True)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Convert: folds BN, removes fake quant, produces actual INT8 model
    model_quantized = quant.convert(model.eval())
    print("QAT model converted to INT8.")

    # Evaluate
    loaders = get_dataloaders(data_dir, batch_size=batch_size)
    accuracy, confusion = evaluate_model(model_quantized, loaders["test"], device)
    print(f"QAT INT8 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Save
    qat_path = os.path.join(os.path.dirname(model_path), "kws_qat_int8.pth")
    torch.save(model_quantized.state_dict(), qat_path)
    print(f"Saved QAT INT8 model to {qat_path}")

    return model_quantized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32_model", type=str, default="models/kws_fp32.pth",
                        help="Path to FP32 checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/speech_commands")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_bn_epoch", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--stage", type=str, choices=["prepare", "train", "convert", "all"],
                        default="all", help="Which QAT stage to run")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.stage in ("prepare", "all"):
        print("\n=== Stage 1: Prepare QAT ===")
        model_prepared = prepare_qat_model(args.fp32_model, device)

    if args.stage in ("train", "all"):
        print("\n=== Stage 2: Train with Fake Quant ===")
        loaders = get_dataloaders(args.data_dir, batch_size=args.batch_size)
        if args.stage == "all":
            pass  # model_prepared already set
        else:
            model_prepared = prepare_qat_model(args.fp32_model, device)

        qat_best_path = os.path.join(args.save_dir, "kws_qat_best.pth")
        train_qat(model_prepared, loaders, args.epochs, args.lr,
                  args.freeze_bn_epoch, device, qat_best_path)

    if args.stage in ("convert", "all"):
        print("\n=== Stage 3: Convert to INT8 ===")
        qat_best_path = os.path.join(args.save_dir, "kws_qat_best.pth")
        convert_qat(qat_best_path, args.data_dir, args.batch_size, device)


if __name__ == "__main__":
    main()