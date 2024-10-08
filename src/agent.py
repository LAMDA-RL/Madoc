import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np


def extend_and_repeat(tensor, dim, repeat):
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)


def soft_target_update(network, target_network, soft_target_update_rate):
    target_network_params = {k: v for k,
                             v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            (1 - soft_target_update_rate) * target_network_params[k].data
            + soft_target_update_rate * v.data
        )


class ReparameterizedTanhGaussian(nn.Module):

    def __init__(self, log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, mean, log_std, deterministic=False):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        action_distribution = Normal(mean, std)
        action_prev_tanh = action_distribution.rsample()
        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = torch.tanh(action_prev_tanh)
        log_prob = action_distribution.log_prob(
            action_prev_tanh) - (2 * (np.log(2) - action_prev_tanh - torch.nn.functional.softplus(-2 * action_prev_tanh)))
        return action_sample, torch.sum(log_prob, dim=-1), action_prev_tanh


class TanhGaussianPolicy(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256',
                 log_std_multiplier=1.0, log_std_offset=-1.0):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.base_network = FullyConnectedNetwork(
            observation_dim, 2 * action_dim, arch)
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian()

    def forward(self, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        assert torch.isnan(observations).sum() == 0, print(observations)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(
            base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        assert torch.isnan(mean).sum() == 0, print(mean)
        assert torch.isnan(log_std).sum() == 0, print(log_std)
        return self.tanh_gaussian(mean, log_std, deterministic)


class BanditPolicy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(1, dim))
        self.log_std = nn.Parameter(torch.zeros(1, dim))
        self.prior_mean = nn.Parameter(
            torch.zeros(1, dim), requires_grad=False)
        self.tanh_gaussian = ReparameterizedTanhGaussian()

    def forward(self, deterministic=False, repeat=1):
        mean = torch.repeat_interleave(self.mean, repeats=repeat, dim=0)
        log_std = torch.repeat_interleave(
            self.log_std, repeats=repeat, dim=0)
        return self.tanh_gaussian(mean, log_std, deterministic)

    def update_prior(self, soft_target_update_rate):
        self.prior_mean.data = self.prior_mean.data * \
            (1 - soft_target_update_rate) + \
            self.mean.clone().data * soft_target_update_rate

    def KL_loss(self):
        return F.mse_loss(self.mean, self.prior_mean)


class SamplerPolicy(object):

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def __call__(self, observations, deterministic=False):
        with torch.no_grad():
            observations = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )
            actions, _, _ = self.policy(observations, deterministic)
            actions = actions.cpu().numpy()
        return actions


class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, arch='256-256'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        d = input_dim
        modules = []
        hidden_sizes = [int(h) for h in arch.split('-')]

        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size

        last_fc = nn.Linear(d, output_dim)
        nn.init.xavier_uniform_(last_fc.weight, gain=1e-2)
        nn.init.constant_(last_fc.bias, 0.0)

        modules.append(last_fc)

        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)


class FullyConnectedQFunction(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256'):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.network = FullyConnectedNetwork(
            observation_dim + action_dim, 1, arch)

    def forward(self, observations, actions):
        input_tensor = torch.cat([observations, actions], dim=-1)
        return torch.squeeze(self.network(input_tensor), dim=-1)


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )

    def forward(self):
        return self.constant


def get_para_dim(config_dict):
    dims = 0
    for _ in config_dict.values():
        dims += 1
    return dims


def get_designer_policies(groups):
    modules = []
    for group in groups:
        modules.append(BanditPolicy(len(group)))
    return nn.ModuleList(modules)


def get_designer_vnets(groups):
    vs = []
    for group in groups:
        vs.append(nn.Sequential(
            *[nn.Linear(len(group), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)]))
    return nn.ModuleList(vs)
