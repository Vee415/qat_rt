"""
FP32 baseline training loop with MLflow logging.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataset import get_dataloaders
from model import get_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    loaders = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    model = get_model(num_classes=10, quantizable=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0

    try:
        import mlflow
        mlflow.set_experiment("kws_fp32_baseline")
        mlflow.start_run()
        mlflow.log_params({
            "model": "KWSNet",
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
        })
        use_mlflow = True
    except ImportError:
        print("MLflow not installed. Skipping experiment logging.")
        use_mlflow = False

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, loaders["val"], criterion, device
        )
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if use_mlflow:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    # Final test evaluation
    model.load_state_dict(torch.load(args.save_path, weights_only=True))
    test_loss, test_acc = evaluate(model, loaders["test"], criterion, device)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    if use_mlflow:
        mlflow.log_metric("test_acc", test_acc)
        mlflow.end_run()

    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/speech_commands",
                        help="Path to Speech Commands dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--save_path", type=str, default="models/kws_fp32.pth",
                        help="Path to save best model")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    train(args)