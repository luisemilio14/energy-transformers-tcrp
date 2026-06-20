# ================================== Imports ================================= #
import torch


# ==================== Main evaluation function - accuracy =================== #
def evaluate_acc(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y = y.squeeze()  # Remove extra dimension if present
            logits = model(X)
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_cross_entropy(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y = y.squeeze()  # Remove extra dimension if present
            logits = model(X)
            loss = torch.nn.functional.cross_entropy(logits, y)
            total_loss += loss.item()
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
