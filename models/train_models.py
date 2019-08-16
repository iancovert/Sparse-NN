import torch
import torch.optim as optim
import numpy as np
from copy import deepcopy
from models.mlp import MLP, DropoutMLP, BernoulliMLP, GaussianMLP


def MSELoss(pred, target):
    '''Calculate MSE. Sum over output dimension, mean over batch.'''
    if isinstance(target, np.ndarray):
        return np.sum(np.mean((pred - target) ** 2, axis=0))
    elif isinstance(target, torch.Tensor):
        return torch.sum(torch.mean((pred - target) ** 2, dim=0))
    else:
        raise ValueError(
            'unsupported target data type: {}'.format(type(target)))


def Accuracy(pred, target):
    '''Calculate 0-1 accuracy.'''
    return (torch.argmax(pred, dim=1) == target).float().mean()


class AverageMeter(object):
    '''
    For tracking moving average of loss.

    Args:
      r: parameter for calcualting exponentially moving average.
    '''
    def __init__(self, r=0.1):
        self.r = r
        self.reset()

    def reset(self):
        self.loss = None

    def update(self, loss):
        if not self.loss:
            self.loss = loss.detach()
        else:
            self.loss = self.r * self.loss + (1 - self.r) * loss.detach()

    def get(self):
        return self.loss


def validate(model, loader, loss_fn):
    '''Calculate average loss using dataloader.'''
    mean_loss = 0
    N = 0
    for x, y in loader:
        n = x.shape[0]
        loss = loss_fn(model(x), y)
        weight = N / (N + n)
        mean_loss = weight * mean_loss + (1 - weight) * loss
        N += n
    return mean_loss


class MLPTrain:
    '''
    For training MLPs.

    Args:
      model: MLP.
    '''
    def __init__(self, model):
        assert isinstance(model, MLP) or isinstance(model, DropoutMLP)
        self.model = model
        self.train_loss_list = []
        self.val_loss_list = []

    def train(self,
              train_loader,
              val_loader,
              lr,
              mbsize,
              nepochs,
              lam,
              check_every,
              check_convergence=True,
              lookback=5,
              min_nepochs=0,
              loss_fn=MSELoss,
              eval_mode=True,
              verbose=False):
        '''
        Train model.

        Args:
          train_loader: training dataloader.
          val_loader: validation dataloader.
          lr: learning rate.
          mbsize: minibatch size.
          nepochs: number of epochs.
          lam: regularization strength.
          check_every: number of gradient steps between loss checks.
          check_convergence: whether to check for convergence.
          lookback: window for determining whether loss is still going down.
          min_nepochs: minimum number of epochs.
          loss_fn: loss function.
          eval_mode: whether to switch model to eval mode for validation set.
          verbose: verbosity.
        '''
        # Ensure training hasn't happened yet.
        if len(self.train_loss_list) != 0:
            raise ValueError('model has already been trained!')

        # Set parameters of train_loader.
        train_loader.batch_sampler.batch_size = mbsize
        train_loader.batch_sampler.sampler.num_samples = mbsize * check_every

        # For optimization.
        model = self.model
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_tracker = AverageMeter()
        min_criterion = np.inf

        done = False
        epoch = 0
        while epoch < nepochs and not done:
            for x, y in train_loader:
                # Calculate loss.
                loss = loss_fn(model(x), y)
                penalty = self.calculate_penalty(lam)
                loss += penalty
                loss_tracker.update(loss)

                # Take gradient step.
                loss.backward()
                optimizer.step()
                model.zero_grad()

            # Check progress.
            with torch.no_grad():
                # TODO consider printing training loss without penalty.
                train_loss = loss_tracker.get()
                loss_tracker.reset()
                if eval_mode:
                    model.eval()
                val_loss = validate(model, val_loader, loss_fn)
                model.train()
                penalty = self.calculate_penalty(lam)
                val_criterion = self.calculate_val_criterion(val_loss, penalty)

            self.record_progress(train_loss, val_loss, penalty)

            if verbose:
                self.print_progress(epoch, train_loss, val_loss, penalty)

            # Check convergence criterion.
            if val_criterion < min_criterion:
                min_criterion = val_criterion
                min_epoch = epoch
                best_model = deepcopy(model)
            elif check_convergence and (epoch > min_nepochs):
                if (epoch - min_epoch) > lookback:
                    done = True
            epoch += 1

        # Restore parameters of best model.
        for param, best_param in zip(model.parameters(),
                                     best_model.parameters()):
            param.data = best_param.data

    def calculate_penalty(self, lam):
        penalty = lam * sum(
            [torch.sum(param ** 2) for param in self.model.parameters()
             if len(param.shape) == 2])
        return penalty

    def calculate_val_criterion(self, val_loss, penalty):
        return val_loss

    def print_progress(self, epoch, train_loss, val_loss, penalty):
        print(('{}Epoch = {}{}').format('-' * 10, epoch, '-' * 10))
        print('Train loss = {:.4f}'.format(train_loss))
        print('Val loss = {:.4f}'.format(val_loss))
        if penalty > 0:
            print('Penalty = {:.4f}'.format(penalty))

    def record_progress(self, train_loss, val_loss, penalty):
        self.train_loss_list.append(train_loss)
        self.val_loss_list.append(val_loss)


class BernoulliMLPTrain(MLPTrain):
    '''
    For training BernoulliMLPs.

    Args:
      model: BernoulliMLP.
    '''
    def __init__(self, model):
        assert isinstance(model, BernoulliMLP)
        self.model = model
        self.train_loss_list = []
        self.val_loss_list = []
        self.penalty_list = []

    def calculate_penalty(self, lam):
        return lam * self.model.ConcreteDropout.penalty()

    def calculate_val_criterion(self, val_loss, penalty):
        return val_loss + penalty

    def print_progress(self, epoch, train_loss, val_loss, penalty):
        print(('{}Epoch = {}{}').format('-' * 10, epoch, '-' * 10))
        print('Train loss = {:.4f}'.format(train_loss))
        print('Val loss = {:.4f}'.format(val_loss))
        print('Val total loss = {:.4f}'.format(val_loss + penalty))
        print('Penalty = {:.4f}'.format(penalty))
        print('Mean p = {:.4f}'.format(
            torch.mean(self.model.ConcreteDropout.get_dropout())))

    def record_progress(self, train_loss, val_loss, penalty):
        self.train_loss_list.append(train_loss)
        self.val_loss_list.append(val_loss)
        self.penalty_list.append(penalty)


class GaussianMLPTrain(MLPTrain):
    '''
    For training GaussianMLPs.

    Args:
      model: GaussianMLP.
    '''
    def __init__(self, model):
        assert isinstance(model, GaussianMLP)
        self.model = model
        self.train_loss_list = []
        self.val_loss_list = []
        self.penalty_list = []

    def calculate_penalty(self, lam):
        return lam * self.model.GaussianNoise.penalty()

    def calculate_val_criterion(self, val_loss, penalty):
        return val_loss + penalty

    def print_progress(self, epoch, train_loss, val_loss, penalty):
        print(('{}Epoch = {}{}').format('-' * 10, epoch, '-' * 10))
        print('Train loss = {:.4f}'.format(train_loss))
        print('Val loss = {:.4f}'.format(val_loss))
        print('Val total loss = {:.4f}'.format(val_loss + penalty))
        print('Penalty = {:.4f}'.format(penalty))
        print('Mean std = {:.4f}'.format(
            torch.mean(self.model.GaussianNoise.get_std())))

    def record_progress(self, train_loss, val_loss, penalty):
        self.train_loss_list.append(train_loss)
        self.val_loss_list.append(val_loss)
        self.penalty_list.append(penalty)
