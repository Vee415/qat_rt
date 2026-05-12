"""
Model evaluation: accuracy, per-class accuracy, confusion matrix.
"""

import argparse
from collections import defaultdict

import torch
import torch.nn as nn

from dataset import get_dataloaders, TARGET_CLASSES
from model import get_model


def evaluate_model(model, loader, device):
    """Evaluate model and return accuracy + confusion matrix."""
    model.eval()
    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for true, pred in zip(labels.cpu().tolist(), predicted.cpu().tolist()):
                confusion[true][pred] += 1

    accuracy = correct / total
    return accuracy, confusion


def print_confusion_matrix(confusion, class_names=TARGET_CLASSES):
    """Print formatted confusion matrix."""
    n = len(class_names)
    header = f"{'':>8}" + "".join(f"{name[:5]:>6}" for name in class_names)
    print(header)
    for i in range(n):
        row = f"{class_names[i]:>8}"
        for j in range(n):
            count = confusion.get(i, {}).get(j, 0)
            row += f"{count:>6}"
        print(row)


def per_class_accuracy(confusion, class_names=TARGET_CLASSES):
    """Compute per-class accuracy from confusion matrix."""
    results = {}
    for cls_idx, cls_name in enumerate(class_names):
        total = sum(confusion.get(cls_idx, {}).values())
        correct = confusion.get(cls_idx, {}).get(cls_idx, 0)
        acc = correct / total if total > 0 else 0.0
        results[cls_name] = acc
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/kws_fp32.pth")
    parser.add_argument("--data_dir", type=str, default="data/speech_commands")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--quantized", action="store_true",
                        help="Set if model is a quantized model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_dataloaders(args.data_dir, batch_size=args.batch_size)

    model = get_model(num_classes=10, quantizable=args.quantized)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)

    if args.quantized:
        model = torch.ao.quantization.convert(model)

    accuracy, confusion = evaluate_model(model, loaders["test"], device)

    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nPer-Class Accuracy:")
    per_class = per_class_accuracy(confusion)
    for cls_name, cls_acc in per_class.items():
        print(f"  {cls_name:>8}: {cls_acc:.4f}")

    print("\nConfusion Matrix:")
    print_confusion_matrix(confusion)


if __name__ == "__main__":
    main()