# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from torch import nn, optim
# import argparse
#
# from models.linear_model import LinearModel
# from models.mlp_model import MLPModel
# from models.cnn_model import CNNModelV1
# from utils import train_one_epoch, evaluate, plot_metrics, plot_confusion_matrix, plot_misclassified_examples
#
#
# def get_model(model_type):
#     if model_type == 'linear':
#         return LinearModel()
#     elif model_type == 'mlp':
#         return MLPModel()
#     elif model_type == 'cnn':
#         return CNNModelV1()
#     else:
#         raise ValueError("Model type must be 'linear', 'mlp', or 'cnn'.")
#
#
# def main(model_type='linear', epochs=10, batch_size=64, lr=0.001):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     transform = transforms.ToTensor()
#     train_data = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
#     test_data = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=batch_size)
#
#     model = get_model(model_type).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     loss_fn = nn.CrossEntropyLoss()
#
#     train_losses, test_losses, train_accs, test_accs = [], [], [], []
#
#     for epoch in range(epochs):
#         tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
#         te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)
#         train_losses.append(tr_loss)
#         test_losses.append(te_loss)
#         train_accs.append(tr_acc)
#         test_accs.append(te_acc)
#         print(
#             f"Epoch {epoch + 1}: Train Loss={tr_loss:.4f}, Test Loss={te_loss:.4f}, Train Acc={tr_acc:.4f}, Test Acc={te_acc:.4f}")
#
#     plot_metrics(train_losses, test_losses, train_accs, test_accs, model_type.upper())
#     plot_confusion_matrix(model, test_loader, device, model_type.upper())
#     class_names = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']  # KMNIST labels
#     plot_misclassified_examples(model, test_loader, device, class_names, model_type.upper())
#
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', choices=['linear', 'mlp', 'cnn'], default='linear')
#     parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--lr', type=float, default=0.001)
#     args = parser.parse_args()
#     main(args.model, args.epochs, args.batch_size, args.lr)


# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from torch import nn, optim
#
# from models.linear_model import LinearModel
# from models.mlp_model import MLPModel
# from models.cnn_models import CNNModelV1, CNNModelV2, CNNModelV3, CNNModelV4
# from utils import train_one_epoch, evaluate, plot_metrics, plot_confusion_matrix, plot_misclassified_examples
#
#
# def get_model(model_type, cnn_variant):
#     if model_type == 'linear':
#         return LinearModel()
#     elif model_type == 'mlp':
#         return MLPModel()
#     elif model_type == 'cnn':
#         if cnn_variant == 'v1':
#             return CNNModelV1()
#         elif cnn_variant == 'v2':
#             return CNNModelV2()
#         elif cnn_variant == 'v3':
#             return CNNModelV3()
#         elif cnn_variant == 'v4':
#             return CNNModelV4()
#         else:
#             raise ValueError("Invalid cnn_variant. Choose from: v1, v2, v3, v4")
#     else:
#         raise ValueError("Model type must be 'linear', 'mlp', or 'cnn'.")
#
#
# def train_model(model_type, cnn_variant, train_loader, test_loader, device, epochs, lr):
#     model = get_model(model_type, cnn_variant).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     loss_fn = nn.CrossEntropyLoss()
#
#     train_losses, test_losses, train_accs, test_accs = [], [], [], []
#
#     for epoch in range(epochs):
#         tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
#         te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)
#         train_losses.append(tr_loss)
#         test_losses.append(te_loss)
#         train_accs.append(tr_acc)
#         test_accs.append(te_acc)
#         print(
#             f"[{model_type.upper()} {cnn_variant.upper() if cnn_variant else ''}] Epoch {epoch + 1}: Train Loss={tr_loss:.4f}, Test Loss={te_loss:.4f}, Train Acc={tr_acc:.4f}, Test Acc={te_acc:.4f}")
#
#     title_prefix = f"{model_type.upper()}_{cnn_variant.upper()}" if model_type == 'cnn' else model_type.upper()
#     plot_metrics(train_losses, test_losses, train_accs, test_accs, title_prefix)
#     plot_confusion_matrix(model, test_loader, device, title_prefix)
#     class_names = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']
#     plot_misclassified_examples(model, test_loader, device, class_names, title_prefix)
#
#
# def main(epochs=10, batch_size=64, lr=0.001):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     transform = transforms.ToTensor()
#     train_data = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
#     test_data = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=batch_size)
#
#     print("Training Linear Model...")
#     train_model('linear', None, train_loader, test_loader, device, epochs, lr)
#     print("-" * 80)
#     print()
#
#     print("Training MLP Model...")
#     train_model('mlp', None, train_loader, test_loader, device, epochs, lr)
#
#     print("-" * 80)
#     print()
#
#     for variant in ['v1', 'v2', 'v3', 'v4']:
#         print(f"Training CNN Model {variant.upper()}...")
#         train_model('cnn', variant, train_loader, test_loader, device, epochs, lr)
#
#         print("-" * 80)
#         print()
#
#
# if __name__ == "__main__":
#     main(epochs=10, batch_size=64, lr=0.001)


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import os

from models.linear_model import LinearModel
from models.mlp_model import MLPModel
from models.cnn_models import CNNModelV1, CNNModelV2, CNNModelV3, CNNModelV4
from utils import train_one_epoch, evaluate, plot_metrics, plot_confusion_matrix, plot_misclassified_examples, \
    plot_comparative_metrics

# Ensure results directory exists
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_model(model_type, cnn_variant):
    if model_type == 'linear':
        return LinearModel()
    elif model_type == 'mlp':
        return MLPModel()
    elif model_type == 'cnn':
        if cnn_variant == 'v1':
            return CNNModelV1()
        elif cnn_variant == 'v2':
            return CNNModelV2()
        elif cnn_variant == 'v3':
            return CNNModelV3()
        elif cnn_variant == 'v4':
            return CNNModelV4()
        else:
            raise ValueError("Invalid cnn_variant. Choose from: v1, v2, v3, v4")
    else:
        raise ValueError("Model type must be 'linear', 'mlp', or 'cnn'.")


def train_model(model_type, cnn_variant, train_loader, test_loader, device, epochs, lr):
    model = get_model(model_type, cnn_variant).to(device)
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
            f"[{model_type.upper()} {cnn_variant.upper() if cnn_variant else ''}] Epoch {epoch + 1}: Train Loss={tr_loss:.4f}, Test Loss={te_loss:.4f}, Train Acc={tr_acc:.4f}, Test Acc={te_acc:.4f}")

    title_prefix = f"{model_type.upper()}_{cnn_variant.upper()}" if model_type == 'cnn' else model_type.upper()
    plot_metrics(train_losses, test_losses, train_accs, test_accs, title_prefix)
    plot_confusion_matrix(model, test_loader, device, title_prefix)
    class_names = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']
    plot_misclassified_examples(model, test_loader, device, class_names, title_prefix)

    return {
        'train_loss': train_losses,
        'test_loss': test_losses,
        'train_acc': train_accs,
        'test_acc': test_accs
    }


def main(epochs=10, batch_size=64, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()
    train_data = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    all_results = {}

    print("Training Linear Model...")
    all_results['Linear'] = train_model('linear', None, train_loader, test_loader, device, epochs, lr)
    print("-" * 80)
    print()

    print("Training MLP Model...")
    all_results['MLP'] = train_model('mlp', None, train_loader, test_loader, device, epochs, lr)
    print("-" * 80)
    print()

    for variant in ['v1', 'v2', 'v3', 'v4']:
        name = f'CNN_{variant.upper()}'
        print(f"Training CNN Model {variant.upper()}...")
        all_results[name] = train_model('cnn', variant, train_loader, test_loader, device, epochs, lr)
        print("-" * 80)
        print()

    plot_comparative_metrics(all_results, metric_type='loss')
    plot_comparative_metrics(all_results, metric_type='accuracy')


if __name__ == "__main__":
    main(epochs=20, batch_size=64, lr=0.0005)
