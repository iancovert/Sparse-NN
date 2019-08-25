import torch
import torch.nn as nn
import numpy as np
import models.model_helper as mh


class MLP(nn.Module):
    '''
    Multilayer perceptron (MLP).

    Args:
      input_size: input features.
      output_size: output dimensionality.
      hidden: number of hidden layers.
      activation: nonlinearity between hidden layers.
      output_activation: nonlinearity at output layer.
    '''
    def __init__(self,
                 input_size,
                 output_size,
                 hidden,
                 activation,
                 output_activation=None):
        super(MLP, self).__init__()

        # Fully connected layers.
        fc_layers = [nn.Linear(d_in, d_out) for d_in, d_out in
                     zip([input_size] + hidden, hidden + [output_size])]
        self.fc = nn.ModuleList(fc_layers)

        # Activation functions.
        self.activation = mh.activation_helper(activation)
        self.output_activation = mh.activation_helper(output_activation, dim=1)

    def forward(self, x):
        for i, fc in enumerate(self.fc):
            if i > 0:
                x = self.activation(x)
            x = fc(x)

        return self.output_activation(x)

    def shrink_inputs(self, inds):
        W, B = self.fc[0].weight, self.fc[0].bias
        hidden, input_size = int(W.shape[0]), int(W.shape[1])

        # Convert inds to torch.Tensor.
        if not isinstance(inds, torch.Tensor):
            if len(inds) < input_size:
                inds = np.array([i in inds for i in range(input_size)])
            elif isinstance(inds, list):
                inds = np.array(inds, dtype=bool)
            elif isinstance(inds, np.ndarray):
                inds = inds.astype(bool)
            inds = torch.tensor(inds)
        new_size = int(torch.sum(inds.int()))

        # Create new first layer.
        linear = nn.Linear(new_size, hidden).to(device=W.device)
        linear.weight.data = W[:, inds]
        linear.bias.data = B
        self.fc[0] = linear


class DropoutMLP(MLP):
    '''
    MLP with dropout at input layer.

    Args:
      input_size: input features.
      output_size: output dimensionality.
      hidden: number of hidden layers.
      activation: nonlinearity between hidden layers.
      p: dropout probability.
      output_activation: nonlinearity at output layer.
    '''
    def __init__(self,
                 input_size,
                 output_size,
                 hidden,
                 activation,
                 p,
                 output_activation=None):
        super(DropoutMLP, self).__init__(input_size, output_size, hidden,
                                         activation, output_activation)
        self.Dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = self.Dropout(x)
        return super(DropoutMLP, self).forward(x)


class BernoulliMLP(MLP):
    '''
    MLP with learnable Concrete dropout at input layer.

    Args:
      input_size: input features.
      output_size: output dimensionality.
      hidden: number of hidden layers.
      activation: nonlinearity between hidden layers.
      p: initial dropout probability.
      temperature: temperature for Concrete distribution.
      reference: reference value for applying dropout.
      penalty: penalty type for Concrete dropout.
      output_activation: nonlinearity at output layer.
    '''
    def __init__(self,
                 input_size,
                 output_size,
                 hidden,
                 activation,
                 p,
                 temperature=0.1,
                 reference=0,
                 penalty='linear',
                 output_activation=None):
        super(BernoulliMLP, self).__init__(input_size, output_size, hidden,
                                           activation, output_activation)
        self.ConcreteDropout = mh.ConcreteDropout(
            input_size, p, temperature, reference, penalty)

    def forward(self, x):
        x = self.ConcreteDropout(x)
        return super(BernoulliMLP, self).forward(x)

    def shrink_inputs(self, inds):
        # Convert inds to torch.Tensor.
        input_size = int(self.fc[0].weight.shape[1])
        if not isinstance(inds, torch.Tensor):
            if len(inds) < input_size:
                inds = np.array([i in inds for i in range(input_size)])
            elif isinstance(inds, list):
                inds = np.array(inds, dtype=bool)
            elif isinstance(inds, np.ndarray):
                inds = inds.astype(bool)
            inds = torch.tensor(inds)

        super(BernoulliMLP, self).shrink_inputs(inds)

        # Adjust ConcreteDropout module.
        reference = self.ConcreteDropout.reference
        if isinstance(reference, torch.Tensor):
            reference = self.ConcreteDropout.reference[0, inds]
        elif isinstance(reference, float) or isinstance(reference, int):
            pass

        self.ConcreteDropout = mh.ConcreteDropout(
            int(torch.sum(inds.int())),
            self.ConcreteDropout.get_dropout()[inds],
            self.ConcreteDropout.temperature,
            reference)

    def get_dropout(self):
        return self.ConcreteDropout.get_dropout()


class GaussianMLP(MLP):
    '''
    MLP with learnable additive Gaussian noise at input layer.

    Args:
      input_size: input features.
      output_size: output dimensionality.
      hidden: number of hidden layers.
      activation: nonlinearity between hidden layers.
      std: initial standard deviation.
      output_activation: nonlinearity at output layer.
    '''
    def __init__(self,
                 input_size,
                 output_size,
                 hidden,
                 activation,
                 std,
                 output_activation=None):
        super(GaussianMLP, self).__init__(input_size, output_size, hidden,
                                          activation, output_activation)
        self.GaussianNoise = mh.GaussianNoise(input_size, std)

    def forward(self, x):
        x = self.GaussianNoise(x)
        return super(GaussianMLP, self).forward(x)

    def shrink_inputs(self, inds):
        # Convert inds to torch.Tensor.
        input_size = int(self.fc[0].weight.shape[1])
        if not isinstance(inds, torch.Tensor):
            if len(inds) < input_size:
                inds = np.array([i in inds for i in range(input_size)])
            elif isinstance(inds, list):
                inds = np.array(inds, dtype=bool)
            elif isinstance(inds, np.ndarray):
                inds = inds.astype(bool)
            inds = torch.tensor(inds)

        super(GaussianMLP, self).shrink_inputs(inds)

        # Adjust GaussianNoise module.
        self.GaussianNoise = mh.GaussianNoise(
            int(torch.sum(inds.int())), self.GaussianNoise.get_std()[inds])

    def get_std(self):
        return self.GaussianNoise.get_std()
