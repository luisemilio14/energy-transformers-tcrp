# ================================== Imports ================================= #
import torch
from torch.amp import autocast


# ==================== Main evaluation function - accuracy =================== #
def evaluate_acc(model, dataloader, device):
    """Evaluate accuracy on validation/test set.

    Args:
        model: PyTorch model
        dataloader: DataLoader for validation/test data
        device: Device to move data to
    """

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y = y.squeeze()  # Remove extra dimension if present

            # Use autocast context to match training dtype
            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(X)

            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_cross_entropy(model, dataloader, device):
    """Evaluate cross-entropy loss on validation/test set.

    Args:
        model: PyTorch model
        dataloader: DataLoader for validation/test data
        device: Device to move data to
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y = y.squeeze()  # Remove extra dimension if present

            # Use autocast context to match training dtype
            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(X)

            loss = torch.nn.functional.cross_entropy(logits, y)
            total_loss += loss.item()
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0


# Create a single function in your evaluate.py
def evaluate_metrics(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.squeeze()

            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(X)

            # 1. Calculate Loss
            loss = torch.nn.functional.cross_entropy(logits, y)
            total_loss += loss.item() * X.size(0)

            # 2. Calculate Accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

    return total_loss / len(dataloader), correct / total
