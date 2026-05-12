"""Train FP32 baseline model with verbose output."""
import sys
sys.path.insert(0, 'src')

import os
import time
import torch
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== FP32 Baseline Training ===")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    print(f"\nLoading dataset...")
    loaders = get_dataloaders('data/speech_commands', batch_size=1024, num_workers=0)

    # Model
    model = get_model(num_classes=10, quantizable=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: KWSNet | Params: {total_params:,}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0
    epochs = 30
    os.makedirs("models", exist_ok=True)

    print(f"\nTraining for {epochs} epochs, batch_size=1024, lr=1e-3")
    print(f"{'='*70}")

    for epoch in range(epochs):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{epochs} [Train]",
                     leave=False, ncols=100)

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{train_correct/train_total:.3f}")

        train_acc = train_correct / train_total
        train_loss = train_loss / train_total

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in loaders["val"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss / val_total
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} Loss: {val_loss:.4f} | "
              f"Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/kws_fp32.pth")
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    # --- Final Test ---
    print(f"\n{'='*70}")
    print("Evaluating on test set...")
    model.load_state_dict(torch.load("models/kws_fp32.pth", weights_only=True))
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loaders["test"], desc="Testing", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = test_correct / test_total
    print(f"\n*** Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%) ***")

    if test_acc >= 0.90:
        print("Target achieved: >90% accuracy!")
    else:
        print(f"Below 90% target. Consider training more epochs or tuning hyperparameters.")

    model_size = os.path.getsize("models/kws_fp32.pth") / (1024 * 1024)
    print(f"Model size: {model_size:.2f} MB")
    print("Saved to: models/kws_fp32.pth")


if __name__ == "__main__":
    main()