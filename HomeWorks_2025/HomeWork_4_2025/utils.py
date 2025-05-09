import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += loss_fn(outputs, targets).item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def plot_metrics(train_losses, test_losses, train_accs, test_accs, title_prefix):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f'{title_prefix} Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title_prefix.lower()}_loss.png')
    plt.close()

    plt.figure()
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f'{title_prefix} Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title_prefix.lower()}_accuracy.png')
    plt.close()


def plot_confusion_matrix(model, data_loader, device, title_prefix):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{title_prefix} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{title_prefix.lower()}_confusion_matrix.png')
    plt.close()


def plot_misclassified_examples(model, data_loader, device, class_names, title_prefix, max_examples=9):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            mis_idx = (preds != targets).nonzero(as_tuple=True)[0]
            for i in mis_idx:
                misclassified.append((inputs[i].cpu(), preds[i].item(), targets[i].item()))
                if len(misclassified) >= max_examples:
                    break
            if len(misclassified) >= max_examples:
                break

    plt.figure(figsize=(10, 4))
    for idx, (img, pred, true) in enumerate(misclassified):
        plt.subplot(1, max_examples, idx + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"P:{class_names[pred]}\nT:{class_names[true]}")
        plt.axis('off')
    plt.suptitle(f"{title_prefix} Misclassified Examples")
    plt.tight_layout()
    plt.savefig(f'{title_prefix.lower()}_misclassified.png')
    plt.close()
