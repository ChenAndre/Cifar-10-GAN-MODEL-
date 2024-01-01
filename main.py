import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import datasets
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchsummary import summary

print('Currently running PyTorch', torch.__version__)

device = 'cuda'

# CIFAR-10 specific parameters
image_channels = 3  # CIFAR-10 images are RGB
image_size = 32  # CIFAR-10 images are 32x32 pixels

# Batch size and noise dimension
batch_size = 128
noise_dim = 100

# Optimizer parameters
lr = 0.0002
beta1 = 0.5
beta2 = 0.99

# Number of epochs
epochs = 20

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

print('Total image examples in this training dataset are:', len(trainset))

# Discriminator architecture
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(image_channels, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Generator architecture
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, image_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Initialize models
D = Discriminator().to(device)
G = Generator(noise_dim).to(device)

# Apply weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

D.apply(weights_init)
G.apply(weights_init)

# Loss and optimizers
criterion = nn.BCELoss()
D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

# Helper functions for real and fake loss
def real_loss(D_out):
    labels = torch.ones_like(D_out)
    return criterion(D_out, labels)

def fake_loss(D_out):
    labels = torch.zeros_like(D_out)
    return criterion(D_out, labels)

# Training loop
for epoch in range(epochs):
    total_d_loss = 0.0
    total_g_loss = 0.0

    for real_images, _ in tqdm(trainloader):
        real_images = real_images.to(device)

        # Train Discriminator
        D_opt.zero_grad()
        D_real_loss = real_loss(D(real_images))

        z = torch.randn(batch_size, noise_dim, 1, 1).to(device)
        fake_images = G(z)
        D_fake_loss = fake_loss(D(fake_images.detach()))
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_opt.step()

        total_d_loss += D_loss.item()

        # Train Generator
        G_opt.zero_grad()
        G_loss = real_loss(D(fake_images))
        G_loss.backward()
        G_opt.step()

        total_g_loss += G_loss.item()

    avg_d_loss = total_d_loss / len(trainloader)
    avg_g_loss = total_g_loss / len(trainloader)

    print(f"Epoch: {epoch + 1}, D Loss: {avg_d_loss}, G Loss: {avg_g_loss}")

    # Generate and display image after each epoch
    z = torch.randn(1, noise_dim, 1, 1).to(device)
    with torch.no_grad():
        generated_image = G(z).detach().cpu()

    plt.imshow(np.transpose(generated_image[0], (1, 2, 0)))
    plt.title(f'Generated Image after Epoch {epoch+1}')
    plt.show()

# Test discriminator's prediction
prediction = D(generated_image).item()
print(f"Discriminator's prediction (closer to 1 means more 'real'): {prediction}")
