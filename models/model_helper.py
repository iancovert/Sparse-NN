import torch
import torch.nn as nn
import numpy as np


def activation_helper(activation, dim=None):
    '''Get activation function.'''
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'elu':
        act = nn.ELU()
    elif activation == 'softmax':
        act = nn.Softmax(dim=dim)
    elif activation is None:
        def act(x):
            return x
    else:
        raise ValueError('unsupported activation: {}'.format(activation))
    return act


def activation_grad(a, activation):
    '''Calculate gradient of activation function (from pre-activations).'''
    if isinstance(activation, nn.modules.activation.Sigmoid):
        x = torch.sigmoid(a)
        return x * (1 - x)
    elif isinstance(activation, nn.modules.activation.Tanh):
        x = torch.tanh(a)
        return 1 - x ** 2
    elif isinstance(activation, nn.modules.activation.ReLU):
        return (a > 0).float()
    elif isinstance(activation, nn.modules.activation.ELU):
        indicator = (a > 0).float()
        return torch.exp((1 - indicator) * a)
    elif activation is None:
        activation = None
    else:
        raise ValueError('unsupported activation: {}'.format(activation))


class ConcreteDropout(nn.Module):
    '''
    Module for applying learnable Concrete dropout.

    Args:
      input_size: number of features.
      p: initial dropout probability (float, np.ndarray, torch.Tensor).
      temperature: temperature for Concrete distribution.
      reference: reference value for applying dropout.
      penalty: penalty type, 'linear' or 'log'.
    '''
    def __init__(self,
                 input_size,
                 p,
                 temperature=0.1,
                 reference=0,
                 penalty='linear',
                 expectation_trick=False):
        super(ConcreteDropout, self).__init__()

        # Set up initial dropout rates.
        if isinstance(p, torch.Tensor):
            assert p.shape == (input_size,)
        elif isinstance(p, np.ndarray):
            assert p.shape == (input_size,)
            p = torch.tensor(p, dtype=torch.float32)
        elif isinstance(p, float) or isinstance(p, int):
            p = torch.ones(input_size, dtype=torch.float32) * p
        else:
            raise ValueError('unsupported type: {}'.format(p))

        # TODO consider implementing without sigmoid trick.
        p = torch.clamp(p, min=1e-6, max=1-1e-6)
        p_logit = torch.log(p / (1 - p)).view(1, -1).detach()
        self.logit = nn.Parameter(p_logit.requires_grad_(True))

        # Set up reference values.
        if isinstance(reference, np.ndarray):
            assert reference.shape == (input_size,)
            self.reference = torch.tensor(
                reference, dtype=torch.float32, requires_grad=False).view(1, -1)
        elif isinstance(reference, torch.Tensor):
            assert reference.shape == (input_size,)
            self.reference = reference.view(1, -1).requires_grad_(False)
        elif isinstance(reference, float) or isinstance(reference, int):
            self.reference = reference
        else:
            raise ValueError('unsupported type: {}'.format(type(reference)))

        self.temperature = temperature
        self.expectation_trick = expectation_trick

        assert penalty in ('linear', 'log')
        self.penalty_type = penalty

    def forward(self, x):
        if self.training:
            p = self.get_dropout_()

            # Sample from Concrete distribution.
            u = torch.rand(x.shape, device=x.device)
            z = torch.sigmoid(
                (torch.log(1 - p) - torch.log(p)
                 + torch.log(u) - torch.log(1 - u))
                / self.temperature)

            if self.expectation_trick:
                return ((x * z + self.reference * (1 - z)) / (1 - p)
                        - self.reference * p / (1 - p))
            else:
                return x * z + self.reference * (1 - z)
        else:
            if self.expectation_trick:
                return x
            else:
                p = self.get_dropout_()
                return x * (1 - p) + self.reference * p

    def penalty(self):
        p = self.get_dropout_()
        if self.penalty_type == 'linear':
            return torch.sum(1 - p)
        elif self.penalty_type == 'log':
            return - torch.sum(torch.log(p))

    def get_dropout_(self):
        self.project()
        return torch.sigmoid(self.logit)

    def get_dropout(self):
        return self.get_dropout_().detach()[0]

    def project(self):
        '''Ensure logits are not too large or too small.'''
        self.logit.data = torch.clamp(self.logit, min=-10.0, max=10.0)


class GaussianNoise(nn.Module):
    '''
    Module for applying learnable additive Gaussian noise.

    Args:
      input_size: number of features.
      std: initial standard deviation (float, np.ndarray, torch.Tensor).
    '''
    def __init__(self, input_size, std):
        super(GaussianNoise, self).__init__()

        # Set up initial standard deviations.
        if isinstance(std, torch.Tensor):
            assert std.shape == (input_size,)
        elif isinstance(std, np.ndarray):
            assert std.shape == (input_size,)
            std = torch.tensor(std, dtype=torch.float32)
        elif isinstance(std, float) or isinstance(std, int):
            std = torch.ones(input_size, dtype=torch.float32) * std

        std = torch.clamp(std, min=1e-6)
        log_std = torch.log(std).view(1, -1).detach()
        self.log = nn.Parameter(log_std.requires_grad_(True))

    def forward(self, x):
        if self.training:
            std = self.get_std_()
            return x + torch.randn(x.shape, device=x.device) * std
        else:
            return x

    def penalty(self):
        std = self.get_std_()
        return torch.sum(torch.log(1 + 1 / std ** 2))

    def get_std_(self):
        return torch.exp(self.log)

    def get_std(self):
        return self.get_std_().detach()[0]
