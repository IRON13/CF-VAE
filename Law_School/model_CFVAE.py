import torch
import torch.nn as nn
import math

class VAE_model(nn.Module):
    def __init__(self, input_size, sensitive_latent_size, target_latent_size, model_name):
        super(VAE_model, self).__init__()

        self.input_size = input_size
        self.sensitive_latent_size = sensitive_latent_size
        self.target_latent_size = target_latent_size
        self.model_name = model_name

        self.encoder_model_a = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, sensitive_latent_size * 2)
        )

        self.encoder_model_x = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, target_latent_size * 2)

        )

        if model_name == 'VAE':  # Original version of VAE
            self.decoderA = nn.Sequential(
                nn.Linear(sensitive_latent_size, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
                nn.Sigmoid()
            )
            self.decoderX = nn.Sequential(
                nn.Linear(target_latent_size, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Sigmoid()
            )

    def encoder_a(self, input):
        out = self.encoder_model_a(input).view(-1, 2, self.sensitive_latent_size)
        mu, log_sigma = out[:, 0, :], out[:, 1, :]
        return mu, log_sigma

    def encoder_x(self, input):
        out = self.encoder_model_x(input).view(-1, 2, self.target_latent_size)
        mu, log_sigma = out[:, 0, :], out[:, 1, :]
        return mu, log_sigma


def reparam(mu, log_sigma):
    sigma = torch.exp(log_sigma * 0.5)
    error = torch.randn_like(sigma)
    return mu + sigma * error

def KL(mu, log_sigma):
    return 0.5 * torch.sum(torch.exp(log_sigma) + torch.pow(mu, 2) - 1. - log_sigma)
