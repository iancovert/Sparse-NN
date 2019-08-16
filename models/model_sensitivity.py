import torch
from models.train_models import MSELoss
from models.model_helper import activation_grad


def first_layer_sens(model):
    '''Calculates sensitivity through first layer weights.'''
    W = model.fc[0].weight
    with torch.no_grad():
        return torch.norm(W, dim=0)


def jacobian_sens(model, x):
    '''Calculates sensitivity through expectation of Jacobian norm.'''
    layers = len(model.fc)
    input_dim = x.shape[1]
    scores = []

    # Perform forward pass.
    grads = []
    with torch.no_grad():
        for i in range(layers - 1):
            a = model.fc[i](x)
            x = model.activation(a)
            grads.append(activation_grad(a, model.activation))

    # Perform backward pass.
    with torch.no_grad():
        for i in range(input_dim):
            da_dx = torch.unsqueeze(model.fc[0].weight[:, i], dim=0)
            for i in range(layers - 1):
                dz_dx = da_dx * grads[i]
                da_dx = torch.mm(dz_dx, model.fc[i + 1].weight.t())
            J = da_dx
            scores.append(torch.mean(torch.sum(J ** 2, dim=1)))

    return torch.stack(scores)


def jacobian_sens_dataloader(model, loader):
    '''Calculates sensitivity through expectation of Jacobian norm.'''
    N = 0
    scores = 0
    for x, _ in loader:
        n = x.shape[0]
        weight = N / (n + N)
        mse = jacobian_sens(model, x)
        scores = weight * scores + (1 - weight) * mse
        N += n
    return scores


def imputation_sens(model, x, y, loss_fn=MSELoss):
    '''Calculates sensitivity through loss when features are imputed.'''
    input_dim = x.shape[1]
    ones = torch.ones(1, input_dim, dtype=torch.float32, device=x.device)

    # Calculate MSE when each feature is imputed.
    mse_list = []
    with torch.no_grad():
        for i in range(input_dim):
            ones[0, i] = 0
            inputs = ones * x
            mse_list.append(loss_fn(model(inputs), y).detach())
            ones[0, i] = 1

    return torch.stack(mse_list)


def imputation_sens_dataloader(model, loader, loss_fn=MSELoss):
    '''Calculates sensitivity through loss when features are imputed.'''
    N = 0
    scores = 0
    for x, y in loader:
        n = x.shape[0]
        weight = N / (n + N)
        mse = imputation_sens(model, x, y, loss_fn)
        scores = weight * scores + (1 - weight) * mse
        N += n
    return scores


# TODO finish implementing this.
# def dropout_sens(model, loader, p, loss_fn=MSELoss):
#     if isinstance(p, float):
#         _, dim = loader.dataset.get_shape()
#         p = torch.ones(dim, dtype=torch.float32) * p
#         sampler = torch.distributions.Bernoulli.bernoulli
#     elif isinstance(p, torch.Tensor):
#         sampler = torch.distributions.Bernoulli.bernoulli

#     def dropout(x):
#         mask = sampler.sample().cuda(device=x.device)
#         return x * mask / (1 - p.view(1, -1))

#     return None
