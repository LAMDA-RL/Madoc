import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, num_input, num_hidden, num_output, device="cuda", dropout=False):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, num_output)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if self.dropout:
            x = self.relu(self.dropout_layer(self.fc1(x)))
            x = self.relu(self.dropout_layer(self.fc2(x)))
            x = self.relu(self.dropout_layer(self.fc3(x)))
        else:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
        output = 2 * torch.tanh(self.fc4(x))
        return output


class ConcatDiscriminator(nn.Module):

    def __init__(self, *args, ensemble_num=1, dim=1, **kwargs):
        super(ConcatDiscriminator, self).__init__()
        modules = []
        for _ in range(ensemble_num):
            modules.append(Discriminator(*args, **kwargs))
        self.model = nn.ModuleList(modules)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        outputs = []
        for module in self.model:
            output = module(flat_inputs, **kwargs)
            outputs.append(output)
        return torch.mean(torch.stack(outputs), dim=0)


class VAE(nn.Module):
    def __init__(self, agents_num, state_dim, action_dim, latent_dim, xi_dim):
        super(VAE, self).__init__()

        self.xi_dim = xi_dim
        self.encoder = nn.Sequential(
            nn.Linear(agents_num, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + state_dim + action_dim + xi_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = torch.chunk(x, 2, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, agent_id, obs, action, xi):
        mean, logvar = self.encode(agent_id)
        z = self.reparameterize(mean, logvar)
        xi = torch.repeat_interleave(xi, repeats=self.xi_dim, dim=1)
        x = torch.cat([z, obs, action, xi], dim=1)
        reconstructed_x = self.decode(x)
        return reconstructed_x, mean, logvar

    def calculate_loss(self, agent_id, obs, action, obs_next, xi):
        reconstructed_x, mean, logvar = self.forward(agent_id, obs, action, xi)
        reconstruction_loss = F.mse_loss(
            reconstructed_x, obs_next - obs, reduction='mean')
        kl_divergence = -0.0 * \
            torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return reconstruction_loss + kl_divergence
