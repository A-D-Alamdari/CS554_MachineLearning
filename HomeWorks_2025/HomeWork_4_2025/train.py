import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import argparse

from models.linear_model import LinearModel
from models.mlp_model import MLPModel
from models.cnn_model import CNNModel
from utils import train_one_epoch, evaluate, plot_metrics, plot_confusion_matrix, plot_misclassified_examples


def get_model(model_type):
    if model_type == 'linear':
        return LinearModel()
    elif model_type == 'mlp':
        return MLPModel()
    elif model_type == 'cnn':
        return CNNModel()
    else:
        raise ValueError("Model type must be 'linear', 'mlp', or 'cnn'.")


def main(model_type='linear', epochs=10, batch_size=64, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.ToTensor()
    train_data = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = get_model(model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)
        print(
            f"Epoch {epoch + 1}: Train Loss={tr_loss:.4f}, Test Loss={te_loss:.4f}, Train Acc={tr_acc:.4f}, Test Acc={te_acc:.4f}")

    plot_metrics(train_losses, test_losses, train_accs, test_accs, model_type.upper())
    plot_confusion_matrix(model, test_loader, device, model_type.upper())
    class_names = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']  # KMNIST labels
    plot_misclassified_examples(model, test_loader, device, class_names, model_type.upper())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['linear', 'mlp', 'cnn'], default='linear')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    main(args.model, args.epochs, args.batch_size, args.lr)
