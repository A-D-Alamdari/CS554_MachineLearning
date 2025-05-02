import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs("figures", exist_ok=True)


# ========== Utility Functions ==========

def plot(y_hat, X, x_train, Y, ax, network):
    ax.plot(X, y_hat, color="r", label="Prediction")
    ax.scatter(x_train, Y, color="b", label="Training Data")
    title = "SLP (0 hidden units)" if network.hidden == 0 else f"MLP ({network.hidden} hidden units)"
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("R")
    ax.legend()


def plot_loss_vs_iters(networks):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Combined MSE vs Epoch (All Models)")
    for i, ax in enumerate(ax.flat):
        ax.plot(range(networks[i].iters), networks[i].losses, label="Train", color="red")
        ax.plot(range(networks[i].iters), networks[i].val_losses, label="Test", color="blue")
        title = "SLP (0 hidden units)" if networks[i].hidden == 0 else f"MLP ({networks[i].hidden} hidden units)"
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("MSE")
        ax.legend()
    plt.tight_layout()
    plt.savefig("figures/all_loss_curves.png")
    plt.close()


def plot_individual_losses(networks):
    for net in networks:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(net.iters), net.losses, color='red', label='Train Loss')
        ax.plot(range(net.iters), net.val_losses, color='blue', label='Test Loss')
        title = "SLP (0 hidden units)" if net.hidden == 0 else f"MLP ({net.hidden} hidden units)"
        filename = f"figures/mse_slp.png" if net.hidden == 0 else f"figures/mse_mlp_{net.hidden}.png"
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("MSE")
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def plot_individual_model_fits(networks, lin, x_train, y_train):
    for net in networks:
        y_pred = net.forward(lin)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lin, y_pred, color="r", label="Prediction")
        ax.scatter(x_train, y_train, color="b", label="Training Data")
        title = "SLP (0 hidden units)" if net.hidden == 0 else f"MLP ({net.hidden} hidden units)"
        filename = f"figures/fit_slp.png" if net.hidden == 0 else f"figures/fit_mlp_{net.hidden}.png"
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("R")
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def create_fig():
    fig = plt.figure()
    ax = fig.subplots()
    return ax


# ========== Network Classes ==========

class SLP():
    def __init__(self, lr, iters):
        self.w0 = self.xavier_init(1, 1)
        self.b0 = np.zeros(1)
        self.hidden = 0
        self.lr = lr
        self.iters = iters
        self.minimum_loss = np.inf
        self.losses = []
        self.val_losses = []

    def xavier_init(self, n_input, n_output):
        return np.random.randn(n_input, n_output) * np.sqrt(1.0 / (n_input + n_output))

    def forward(self, x):
        self.a = np.dot(x, self.w0) + self.b0
        return self.a

    def backward(self, x, y):
        error = (self.a - y)
        dw = (1 / y.size) * np.dot(x.T, error)
        db = (1 / y.size) * np.sum(error)
        self.w0 -= self.lr * dw
        self.b0 -= self.lr * db

    def train(self, X, Y, val, val_y):
        print("-------Training Single Layer Perceptron------")
        for iter in range(self.iters):
            self.forward(X)
            self.backward(X, Y)
            iter_loss = self.loss(Y)
            self.losses.append(iter_loss)
            if iter_loss < self.minimum_loss:
                self.minimum_loss = iter_loss
            self.forward(val)
            val_loss = self.loss(val_y)
            self.val_losses.append(val_loss)
            if (iter + 1) % 100 == 0:
                print(f"Iteration {iter + 1}/{self.iters}, Training Loss: {iter_loss:.4f}")
        print()

    def loss(self, y):
        return np.mean((y - self.a) ** 2)


class MLP(SLP):
    def __init__(self, hidden, lr, iters):
        self.hidden = hidden
        self.w0 = self.xavier_init(1, self.hidden)
        self.b0 = np.zeros(self.hidden)
        self.w1 = self.xavier_init(self.hidden, 1)
        self.b1 = np.zeros(1)
        self.lr = lr
        self.iters = iters
        self.minimum_loss = np.inf
        self.losses = []
        self.val_losses = []

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, x):
        self.z0 = np.dot(x, self.w0) + self.b0
        self.a0 = self.tanh(self.z0)
        self.a1 = np.dot(self.a0, self.w1) + self.b1
        return self.a1

    def backward(self, x, y):
        error = self.a1 - y
        dw1 = (1 / y.size) * np.dot(self.a0.T, error)
        db1 = (1 / y.size) * np.sum(error)

        hidden_error = np.dot(error, self.w1.T) * self.tanh_derivative(self.z0)
        dw0 = (1 / y.size) * np.dot(x.T, hidden_error)
        db0 = (1 / y.size) * np.sum(hidden_error, axis=0)

        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w0 -= self.lr * dw0
        self.b0 -= self.lr * db0

    def train(self, X, Y, val, val_y):
        print(f"--------Training MLP with {self.hidden} hidden units--------")
        for iter in range(self.iters):
            self.forward(X)
            self.backward(X, Y)
            iter_loss = self.loss(Y)
            self.losses.append(iter_loss)
            if iter_loss < self.minimum_loss:
                self.minimum_loss = iter_loss
            self.forward(val)
            val_loss = self.loss(val_y)
            self.val_losses.append(val_loss)
            if (iter + 1) % 1000 == 0 or iter == 0:
                print(f"Iteration {iter + 1}/{self.iters}, Training Loss: {iter_loss:.4f}")
        print()

    def loss(self, y):
        return np.mean((y - self.a1) ** 2)


# ========== Execution Function ==========

def run():
    train_data = np.loadtxt("data/train.csv", delimiter=",", skiprows=1)
    test_data = np.loadtxt("data/test.csv", delimiter=",", skiprows=1)

    train_data = train_data[train_data[:, 0].argsort()]
    test_data = test_data[test_data[:, 0].argsort()]

    x_train, y_train = train_data[:, 0].reshape(-1, 1), train_data[:, 1].reshape(-1, 1)
    x_test, y_test = test_data[:, 0].reshape(-1, 1), test_data[:, 1].reshape(-1, 1)

    slp = SLP(lr=0.01, iters=500)
    slp.train(x_train, y_train, x_test, y_test)

    mlp_2 = MLP(hidden=2, lr=0.1, iters=10000)
    mlp_2.train(x_train, y_train, x_test, y_test)

    mlp_4 = MLP(hidden=4, lr=0.1, iters=10000)
    mlp_4.train(x_train, y_train, x_test, y_test)

    mlp_8 = MLP(hidden=8, lr=0.1, iters=10000)
    mlp_8.train(x_train, y_train, x_test, y_test)

    networks = [slp, mlp_2, mlp_4, mlp_8]

    lin = np.linspace(x_train.min(), x_train.max(), 1000).reshape(-1, 1)
    train_predictions = [network.forward(lin) for network in networks]
    train_losses = [net.minimum_loss for net in networks]

    test_losses = []
    for net in networks:
        net.forward(x_test)
        test_losses.append(net.loss(y_test))

    # plot_loss_vs_iters(networks)
    plot_individual_losses(networks)
    plot_individual_model_fits(networks, lin, x_train, y_train)

    fig = create_fig()
    fig.plot([net.hidden for net in networks], train_losses, label="Train Loss", color="blue")
    fig.plot([net.hidden for net in networks], test_losses, label="Test Loss", color="red")
    fig.set_title("Network Complexity vs Loss")
    fig.set_xlabel("Number of Hidden Units")
    fig.set_ylabel("Loss")
    fig.set_xticks([0, 2, 4, 8])
    fig.legend()
    plt.tight_layout()
    plt.savefig("figures/complexity_vs_loss.png")
    plt.close()


if __name__ == "__main__":
    run()
