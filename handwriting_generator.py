import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, z_dim, text_emb_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + text_emb_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),  # Assuming image size is 28x28
            nn.Tanh()
        )

    def forward(self, z, text_emb):
        x = torch.cat((z, text_emb), dim=1)
        return self.fc(x).view(-1, 1, 28, 28)  # Reshape to image size (1, 28, 28)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, text_emb_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28*28 + text_emb_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, text_emb):
        x = torch.cat((img.view(img.size(0), -1), text_emb), dim=1)
        return self.fc(x)

# Define the dataset
class HandwritingDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image

# Hyperparameters
z_dim = 100
text_emb_dim = 50
lr = 0.0002
batch_size = 64
epochs = 50

# Data transformations and dataset loading
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Update the dataset path
image_dir = r"C:\Users\SHIVAKUMAR\Desktop\PythonPrograms\internship\data"
dataset = HandwritingDataset(image_dir=image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(z_dim, text_emb_dim).cuda()
discriminator = Discriminator(text_emb_dim).cuda()

# Loss and optimizers
adversarial_loss = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training loop
for epoch in range(epochs):
    for i, imgs in enumerate(dataloader):
        real_images = imgs.cuda()
        batch_size = real_images.size(0)
        
        # Create labels for real and fake images
        real_labels = torch.ones(batch_size, 1).cuda()
        fake_labels = torch.zeros(batch_size, 1).cuda()

        # Train the discriminator
        optimizer_d.zero_grad()
        
        # Real images
        real_output = discriminator(real_images, text_emb=torch.randn(batch_size, text_emb_dim).cuda())
        d_loss_real = adversarial_loss(real_output, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, z_dim).cuda()
        fake_images = generator(z, text_emb=torch.randn(batch_size, text_emb_dim).cuda())
        fake_output = discriminator(fake_images.detach(), text_emb=torch.randn(batch_size, text_emb_dim).cuda())
        d_loss_fake = adversarial_loss(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Train the generator
        optimizer_g.zero_grad()
        
        # Generator loss
        fake_output = discriminator(fake_images, text_emb=torch.randn(batch_size, text_emb_dim).cuda())
        g_loss = adversarial_loss(fake_output, real_labels)
        g_loss.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
