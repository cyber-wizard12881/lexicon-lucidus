"""Neural-network building blocks used by the MBTI training pipeline."""

import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """Compress a sentence embedding into a smaller latent representation."""

    def __init__(self, input_dim, latent_dim):
        """Create the encoder for the configured input and latent dimensions."""
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        """Return the latent representation for the input embeddings."""
        return self.fc(x)


class Decoder(nn.Module):
    """Reconstruct an embedding while conditioning on an MBTI label."""

    def __init__(self, latent_dim, label_dim, output_dim):
        """Create the decoder for latent vectors, labels, and output embeddings."""
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z, labels):
        """Reconstruct an embedding from a latent vector and one-hot label."""
        return self.fc(torch.cat([z, labels], dim=1))


class Discriminator(nn.Module):
    """Estimate whether a latent vector resembles a real latent sample."""

    def __init__(self, latent_dim):
        """Create the discriminator for the configured latent dimension."""
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        """Return the discriminator probability for each latent vector."""
        return self.fc(z)


class Classifier(nn.Module):
    """Map a latent representation to logits for the 16 MBTI classes."""

    def __init__(self, latent_dim, num_classes=16):
        """Create the classifier for the latent size and number of classes."""
        super(Classifier, self).__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, z):
        """Return one classification logit for each configured MBTI class."""
        return self.fc(z)
