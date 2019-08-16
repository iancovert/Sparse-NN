import torch
import torch.nn as nn
import numpy as np
import models.model_sensitivity as sens
from models.train_models import validate, MSELoss
from models.train_models import MLPTrain, BernoulliMLPTrain, GaussianMLPTrain


def make_schedule(original_dim, k_list, ratio=0.5, intermediate_steps=False):
    '''
    Make schedule for RFE.

    Args:
      original_dim: original number of features.
      k_list: desired number of features.
      ratio: portion of features to eliminate at each iteration.
      intermediate_steps: whether to allow steps in between numbers in k_list.
        Allowing intermediate steps may increase variance between runs.

    Returns:
      schedule: number of features for each iteration of RFE.
    '''
    schedule = []
    k_list = list(np.sort(k_list)[::-1])
    current = original_dim

    if intermediate_steps:
        min_k = np.min(k_list)
        ind = 0
        while current > min_k:
            current = int(current * ratio)
            if current <= k_list[ind]:
                current = k_list[ind]
                ind += 1
            schedule.append(current)

    else:
        while current > 0:
            current = int(current * ratio)
            if current > k_list[0]:
                schedule.append(current)
            else:
                schedule = schedule + k_list
                break

    return schedule


class SPINN:
    '''
    Sparse input neural network (SPINN).

    Args:
      model: base model (MLP).
    '''
    def __init__(self, model):
        self.model = model
        self.train_loss_list = []
        self.val_loss_list = []
        self.subsets = []

    def train(self,
              schedule,
              train_loader,
              val_loader,
              lr,
              nepochs,
              mbsize,
              lam,
              check_every,
              check_convergence=True,
              lookback=5,
              min_nepochs=0,
              loss_fn=MSELoss,
              eval_mode=True,
              anneal=False,
              verbose=True):
        '''
        Train SPINN through RFE.

        Args:
          schedule: RFE schedule.
          train_loader: training dataloader.
          val_loader: validation dataloader.
          lr: learning rate.
          nepochs: number of epochs.
          mbsize: minibatch size.
          lam: regularization strength.
          check_every: number of gradient steps between loss checks.
          check_convergence: whether to check for convergence.
          lookback: window for determining whether loss is still going down.
          min_nepochs: minimum number of epochs.
          loss_fn: loss function.
          eval_mode: whether to switch model to eval mode for validation set.
          anneal: whether to anneal lam throughout RFE.
          verbose: verbosity.
        '''
        # Ensure training hasn't happened yet.
        assert len(self.train_loss_list) == 0

        _, original_dim = train_loader.dataset.get_shape()
        total_epochs = 0

        # Ensure schedule is valid.
        assert np.all(np.argsort(schedule)[::-1] == np.arange(len(schedule)))
        assert np.max(schedule) < original_dim
        assert np.min(schedule) > 0

        # For tracking selected features.
        included = torch.arange(0, original_dim)

        # Perform RFE.
        for k in schedule:
            done = False
            while not done:
                trainer = self.get_trainer()
                trainer.train(train_loader,
                              val_loader,
                              lam=lam,
                              lr=lr,
                              mbsize=mbsize,
                              nepochs=nepochs,
                              check_every=check_every,
                              check_convergence=check_convergence,
                              lookback=lookback,
                              min_nepochs=min_nepochs,
                              loss_fn=loss_fn,
                              eval_mode=eval_mode,
                              verbose=verbose)

                # For annealing lam.
                if anneal:
                    done, lam = self.done_training(trainer, lam)
                    if not done and verbose:
                        print('Annealing lam = {:.4f}'.format(lam))
                else:
                    done = True

                # Update loss variables.
                self.record_progress(trainer)
                total_epochs += len(trainer.train_loss_list)

            if verbose:
                print('Dropping to {} features at epoch = {}, iter = {}'
                      .format(k, total_epochs, total_epochs * check_every))

            # Get top k features.
            scores = self.rank_features(train_loader, loss_fn)
            sorted_scores = torch.argsort(scores, descending=True)
            top_scores = sorted_scores[:k].cpu().data.numpy()

            # Record this subset of features.
            inds = torch.tensor(np.array(
                [i in top_scores for i in
                 range(len(scores))], dtype=int)).byte()
            included = included[inds]
            included_np = included.cpu().data.numpy()
            self.subsets.append(included_np)

            # Modify network for warm start.
            self.model.shrink_inputs(inds)

            # Modify inputs.
            train_loader.dataset.set_inds(included_np)
            val_loader.dataset.set_inds(included_np)

        # Reset data loaders.
        train_loader.dataset.set_inds(None)
        val_loader.dataset.set_inds(None)

    def train_ranking(self,
                      num_features,
                      train_loader,
                      val_loader,
                      lr,
                      nepochs,
                      mbsize,
                      lam,
                      check_every,
                      check_convergence=True,
                      lookback=5,
                      min_nepochs=10,
                      loss_fn=MSELoss,
                      verbose=True):
        '''
        Train SPINN through feature ranking.

        Args:
          num_features: number of features to select.
          train_loader: training dataloader.
          val_loader: validation dataloader.
          lr: learning rate.
          nepochs: number of epochs.
          mbsize: minibatch size.
          lam: regularization strength.
          check_every: number of gradient steps between loss checks.
          check_convergence: whether to check for convergence.
          lookback: window for determining whether loss is still going down.
          min_nepochs: minimum number of epochs.
          loss_fn: loss function.
          verbose: verbosity.
        '''
        # Ensure training hasn't happened yet.
        assert len(self.train_loss_list) == 0

        _, original_dim = train_loader.dataset.get_shape()
        num_features = np.sort(num_features)

        # Perform ranking.
        trainer = self.get_trainer()
        trainer.train(train_loader,
                      val_loader,
                      lam=lam,
                      lr=lr,
                      mbsize=mbsize,
                      nepochs=nepochs,
                      check_every=check_every,
                      check_convergence=check_convergence,
                      lookback=lookback,
                      min_nepochs=min_nepochs,
                      loss_fn=loss_fn,
                      verbose=verbose)

        # Update loss variables.
        self.record_progress(trainer)

        # Record selected features.
        scores = self.rank_features(train_loader, loss_fn)
        sorted_scores = torch.argsort(scores, descending=True)
        self.subsets = [sorted_scores[:num].cpu().data.numpy()
                        for num in num_features]

    def rank_features(self, loader, loss_fn):
        raise NotImplementedError

    def get_trainer(self):
        return MLPTrain(self.model)

    def record_progress(self, trainer):
        self.train_loss_list.append(trainer.train_loss_list)
        self.val_loss_list.append(trainer.val_loss_list)

    def done_training(self, trainer, lam):
        return True, lam


class FirstLayerSPINN(SPINN):
    '''
    SPINN with feature ranking based on first layer weight matrix.

    Args:
      model: base model (MLP).
    '''
    def __init__(self, model):
        super(FirstLayerSPINN, self).__init__(model)

    def rank_features(self, loader, loss_fn):
        return sens.first_layer_sens(self.model)


class JacobianSPINN(SPINN):
    '''
    SPINN with feature ranking based on average norm of Jacobian matrix.

    Args:
      model: base model (MLP).
    '''
    def __init__(self, model):
        super(JacobianSPINN, self).__init__(model)

    def rank_features(self, loader, loss_fn):
        return sens.jacobian_sens_dataloader(self.model, loader)


class ImputationSPINN(SPINN):
    '''
    SPINN with feature ranking based on loss with imputed features.

    Args:
      model: base model (MLP).
    '''
    def __init__(self, model):
        super(ImputationSPINN, self).__init__(model)

    def rank_features(self, loader, loss_fn):
        return sens.imputation_sens_dataloader(self.model, loader, loss_fn)


class GaussianSPINN(SPINN):
    '''
    SPINN with feature ranking based on learnable Gaussian noise.

    Args:
      model: base model (GaussianMLP).
    '''
    def __init__(self, model):
        self.model = model
        self.train_loss_list = []
        self.val_loss_list = []
        self.penalty_list = []
        self.subsets = []

    def rank_features(self, loader, loss_fn):
        return 1 / self.model.get_std()

    def get_trainer(self):
        return GaussianMLPTrain(self.model)

    def record_progress(self, trainer):
        self.train_loss_list.append(trainer.train_loss_list)
        self.val_loss_list.append(trainer.val_loss_list)
        self.penalty_list.append(trainer.penalty_list)


class BernoulliSPINN(SPINN):
    '''
    SPINN with feature ranking based on learnable Concrete dropout.

    Args:
      model: base model (BernoulliMLP).
    '''
    def __init__(self, model):
        self.model = model
        self.train_loss_list = []
        self.val_loss_list = []
        self.penalty_list = []
        self.subsets = []

    def rank_features(self, loader, loss_fn):
        return 1 - self.model.get_dropout()

    def get_trainer(self):
        return BernoulliMLPTrain(self.model)

    def record_progress(self, trainer):
        self.train_loss_list.append(trainer.train_loss_list)
        self.val_loss_list.append(trainer.val_loss_list)
        self.penalty_list.append(trainer.penalty_list)

    def done_training(self, trainer, lam):
        '''Anneal lam if learned dropout rates are too low.'''
        if torch.mean(self.model.get_dropout()) < 0.2:
            return False, 2 * lam
        return True, lam


def train_model_sequence(trial_results,
                         model,
                         train_loader,
                         val_loader,
                         test_loader,
                         lr,
                         nepochs,
                         mbsize,
                         check_every,
                         check_convergence=True,
                         lookback=5,
                         loss_fn=MSELoss,
                         imputation=True,
                         verbose=False):
    '''
    Train sequence of debiased models after identifying feature subsets.

    Args:
      trial_results: list of dicts containing np.ndarrays of feature indices
        under 'inds' key.
      model: PyTorch MLP.
      train_loader: training dataloader.
      val_loader: validation dataloader.
      test_loader: test dataloader.
      lr: learning rate.
      nepochs: number of epochs.
      mbsize: minibatch size.
      check_every: number of gradient steps between loss checks.
      check_convergence: whether to check for convergence.
      lookback: window for determining whether loss is still going down.
      loss_fn: loss function.
      imputation: whether the task is self reconstruction. If True, input
        features are excluded from output.
      verbose: verbosity.
    '''
    # Go from largest to smallest number of inputs.
    order = np.argsort([len(subset['inds']) for subset in trial_results])[::-1]
    trial_results = np.array(trial_results)[order]

    # Ensure warm start is possible.
    for i in range(1, len(trial_results)):
        assert np.all([ind in trial_results[i - 1]['inds'] for ind in
                       trial_results[i]['inds']])

    _, original_dim = train_loader.dataset.get_shape()
    device = train_loader.dataset.device

    # For warm start.
    prev_model = None
    prev_inds = None

    for subset in trial_results:
        # Set up predictors and target.
        inds = subset['inds']
        num = len(inds)
        sparsity = np.array(
            [i in inds for i in range(original_dim)], dtype=bool)

        train_loader.dataset.set_inds(sparsity)
        val_loader.dataset.set_inds(sparsity)
        test_loader.dataset.set_inds(sparsity)

        if imputation:
            train_loader.dataset.set_output_inds(np.logical_not(sparsity))
            val_loader.dataset.set_output_inds(np.logical_not(sparsity))
            test_loader.dataset.set_output_inds(np.logical_not(sparsity))

        # Warm start, except for the first model training.
        if prev_model is not None:
            # Shrink inputs.
            indicator = torch.tensor(np.array(
                [i in inds for i in prev_inds], dtype=int)).byte()
            model.shrink_inputs(indicator)

            if imputation:
                # Grow outputs.
                num_outputs = original_dim - num
                prev_hidden = int(model.fc[-2].weight.shape[0])
                output_inds = [i for i in range(original_dim) if i not in inds]
                prev_output_inds = [i for i in range(original_dim)
                                    if i not in prev_inds]
                indicator = torch.tensor(
                    np.array([i in prev_output_inds for i in output_inds],
                             dtype=int)).byte()
                linear = nn.Linear(prev_hidden, num_outputs).cuda(device=device)
                linear.weight.data[indicator] = model.fc[-1].weight
                linear.bias.data[indicator] = model.fc[-1].bias
                model.fc[-1] = linear

        # Train model.
        trainer = MLPTrain(model)
        trainer.train(train_loader,
                      val_loader,
                      lr,
                      mbsize,
                      nepochs,
                      lam=0.0,
                      check_every=check_every,
                      check_convergence=check_convergence,
                      lookback=lookback,
                      loss_fn=loss_fn,
                      verbose=verbose)

        # Record performance.
        subset['nonlinear'] = {
            'train': validate(model, train_loader, loss_fn).item(),
            'val': validate(model, val_loader, loss_fn).item(),
            'test': validate(model, test_loader, loss_fn).item()
        }

        # For next model
        prev_inds = np.sort(inds)
        prev_model = model

        print('Done with {} variables'.format(num))

    # Reset data loaders.
    train_loader.dataset.set_inds(None)
    val_loader.dataset.set_inds(None)
    test_loader.dataset.set_inds(None)

    if imputation:
        train_loader.dataset.set_output_inds(None)
        val_loader.dataset.set_output_inds(None)
        test_loader.dataset.set_output_inds(None)
