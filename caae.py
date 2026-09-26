import torch
import torch.nn as nn
import torch.optim as optim

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    def forward(self, x):
        return self.fc(x)

# Decoder (conditioned on MBTI label)
class Decoder(nn.Module):
    def __init__(self, latent_dim, label_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )
    def forward(self, z, labels):
        return self.fc(torch.cat([z, labels], dim=1))

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.fc(z)

# Classifier
class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes=16):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(latent_dim, num_classes)
    def forward(self, z):
        return self.fc(z)
