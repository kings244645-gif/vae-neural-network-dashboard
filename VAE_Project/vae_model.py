import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder: 784 input pixels -> 128 features
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Mean and log-variance of the latent distribution
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        features = self.encoder(x)

        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)

        return features, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        z = mu + std * epsilon

        return std, epsilon, z

    def decode(self, z):
        reconstruction = self.decoder(z)
        return reconstruction.view(-1, 1, 28, 28)

    def forward(self, x):
        features, mu, logvar = self.encode(x)
        std, epsilon, z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return {
            "features": features,
            "mu": mu,
            "logvar": logvar,
            "std": std,
            "epsilon": epsilon,
            "z": z,
            "reconstruction": reconstruction
        }