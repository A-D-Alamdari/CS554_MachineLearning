import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import random_split

data_dir = 'data'

train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True)
test_dataset = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True)

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m = len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
batch_size = 256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Define the loss function
loss_fn = torch.nn.MSELoss()

# Define an optimizer (both for the encoder and the decoder!)
lr = 0.001

# Set the random seed for reproducible results
torch.manual_seed(0)


# Function to create a Convolutional Autoencoder with specified architecture
def create_autoencoder(encoded_space_dim, fc2_input_dim):
    class Encoder(nn.Module):

        def __init__(self, encoded_space_dim, fc2_input_dim):
            super().__init__()

            # # Convolutional section
            # self.encoder_cnn = nn.Sequential(
            #     nn.Conv2d(1, 8, 3, stride=2, padding=1),
            #     nn.ReLU(True),
            #     nn.Conv2d(8, 16, 3, stride=2, padding=1),
            #     nn.BatchNorm2d(16),
            #     nn.ReLU(True),
            #     nn.Conv2d(16, 32, 3, stride=2, padding=0),
            #     nn.ReLU(True)
            # )
            #
            # ### Flatten layer
            # self.flatten = nn.Flatten(start_dim=1)
            # ### Linear section
            # self.encoder_lin = nn.Sequential(
            #     nn.Linear(3 * 3 * 32, 128),
            #     nn.ReLU(True),
            #     nn.Linear(128, encoded_space_dim)
            # )
            self.encoder_cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 3, stride=2, padding=0),
                nn.ReLU(True)
            )

            self.flatten = nn.Flatten(start_dim=1)

            self.encoder_lin = nn.Sequential(
                nn.Linear(3 * 3 * 64, 256),
                nn.ReLU(True),
                nn.Linear(256, encoded_space_dim)
            )

        def forward(self, x):
            x = self.encoder_cnn(x)
            x = self.flatten(x)
            x = self.encoder_lin(x)
            return x

    class Decoder(nn.Module):

        def __init__(self, encoded_space_dim, fc2_input_dim):
            super().__init__()
            # self.decoder_lin = nn.Sequential(
            #     nn.Linear(encoded_space_dim, 128),
            #     nn.ReLU(True),
            #     nn.Linear(128, 3 * 3 * 32),
            #     nn.ReLU(True)
            # )
            #
            # self.unflatten = nn.Unflatten(dim=1,
            #                               unflattened_size=(32, 3, 3))
            #
            # self.decoder_conv = nn.Sequential(
            #     nn.ConvTranspose2d(32, 16, 3,
            #                        stride=2, output_padding=0),
            #     nn.BatchNorm2d(16),
            #     nn.ReLU(True),
            #     nn.ConvTranspose2d(16, 8, 3, stride=2,
            #                        padding=1, output_padding=1),
            #     nn.BatchNorm2d(8),
            #     nn.ReLU(True),
            #     nn.ConvTranspose2d(8, 1, 3, stride=2,
            #                        padding=1, output_padding=1)
            # )

            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, 256),
                nn.ReLU(True),
                nn.Linear(256, 3 * 3 * 64),
                nn.ReLU(True)
            )

            self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 3, 3))

            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
            )

        def forward(self, x):
            x = self.decoder_lin(x)
            x = self.unflatten(x)
            x = self.decoder_conv(x)
            x = torch.sigmoid(x)
            return x

    encoder = Encoder(encoded_space_dim, fc2_input_dim)
    decoder = Decoder(encoded_space_dim, fc2_input_dim)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)

    return encoder, decoder, optimizer


def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def plot_ae_outputs(encoder, decoder, n=10):
    plt.figure(figsize=(16, 4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(img))
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')
    plt.show()


# Function to train an autoencoder for a specified number of epochs
def train_autoencoder(encoder, decoder, device, train_loader, loss_fn, optimizer, num_epochs):
    diz_loss = {'train_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        train_loss = train_epoch(encoder, decoder, device, train_loader, loss_fn, optimizer)
        val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
    plot_ae_outputs(encoder, decoder, n=10)

    return diz_loss


# Function to display 7x7 input, 28x28 predicted output, and 28x28 desired output for each class
def display_samples(encoder, decoder, device, test_loader):
    plt.figure(figsize=(20, 30))
    targets = test_dataset.targets.numpy()
    classes = np.unique(targets)

    for i, class_label in enumerate(classes):
        class_idx = np.where(targets == class_label)[0][0]
        sample_image, _ = test_dataset[class_idx]
        sample_image = sample_image.unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_image = decoder(encoder(sample_image))

        plt.subplot(len(classes), 3, 3 * i + 1)
        plt.imshow(sample_image.cpu().squeeze().numpy(), cmap='gray')
        plt.title(f'Class {class_label} - 7x7 Input')
        plt.axis('off')

        plt.subplot(len(classes), 3, 3 * i + 2)
        plt.imshow(predicted_image.cpu().squeeze().numpy(), cmap='gray')
        plt.title(f'Class {class_label} - 28x28 Predicted Output')
        plt.axis('off')

        plt.subplot(len(classes), 3, 3 * i + 3)
        plt.imshow(test_dataset[class_idx][0].numpy().squeeze(), cmap='gray')
        plt.title(f'Class {class_label} - 28x28 Desired Output')
        plt.axis('off')

    plt.show()


data_dir = 'data'

train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True)
test_dataset = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True)

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m = len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
batch_size = 256

# Set the random seed for reproducible results
torch.manual_seed(0)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Initialize the autoencoder with different architectures
d_1 = 4
d_2 = 8
d_3 = 16

encoder_1, decoder_1, optimizer_1 = create_autoencoder(d_1, 128)
encoder_2, decoder_2, optimizer_2 = create_autoencoder(d_2, 128)
encoder_3, decoder_3, optimizer_3 = create_autoencoder(d_3, 128)

# Train each autoencoder and collect loss information
num_epochs = 30

print("Training Autoencoder 1:")
loss_1 = train_autoencoder(encoder_1, decoder_1, device, train_loader, loss_fn, optimizer_1, num_epochs)

print("Training Autoencoder 2:")
loss_2 = train_autoencoder(encoder_2, decoder_2, device, train_loader, loss_fn, optimizer_2, num_epochs)

print("Training Autoencoder 3:")
loss_3 = train_autoencoder(encoder_3, decoder_3, device, train_loader, loss_fn, optimizer_3, num_epochs)

# Display the training/test loss plots
plt.figure(figsize=(12, 6))
plt.plot(loss_1['train_loss'], label='Autoencoder 1 - Train Loss')
plt.plot(loss_1['val_loss'], label='Autoencoder 1 - Test Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training and Test Loss for Autoencoder Architecture 1')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(loss_2['train_loss'], label='Autoencoder 2 - Train Loss')
plt.plot(loss_2['val_loss'], label='Autoencoder 2 - Test Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training and Test Loss for Autoencoder Architecture 2')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(loss_3['train_loss'], label='Autoencoder 3 - Train Loss')
plt.plot(loss_3['val_loss'], label='Autoencoder 3 - Test Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training and Test Loss for Autoencoder Architecture 3')
plt.show()

# Display 7x7 input, 28x28 predicted output, and 28x28 desired output for each class
print("Displaying samples for Autoencoder 1:")
display_samples(encoder_1, decoder_1, device, test_loader)

print("Displaying samples for Autoencoder 2:")
display_samples(encoder_2, decoder_2, device, test_loader)

print("Displaying samples for Autoencoder 3:")
display_samples(encoder_3, decoder_3, device, test_loader)