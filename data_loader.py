import torch
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler


class RNASeq(Dataset):
    '''
    Dataset wrapper for AIBS single cell RNA sequencing data.

    Args:
      data: np.ndarray containing x.
      labels: np.ndarray containing y. If None, then x will be used as y
        (self reconstruction).
      mean: whether to adjust x for mean.
      std: whether to adjust x for standard deviation.
      device: device for torch.Tensors.
      inds: features to include in x. If None, all features are included.
    '''
    def __init__(self,
                 data,
                 labels=None,
                 mean=None,
                 std=None,
                 device=None,
                 inds=None):
        super(RNASeq, self).__init__()
        self.data = data
        self.original_dim = data.shape[1]
        self.device = device

        # For input normalization.
        if mean is not None:
            assert mean.shape == (self.original_dim,)
            self.mean_adjustment = True
            self.mean = mean[np.newaxis]
        else:
            self.mean_adjustment = False
        if std is not None:
            assert std.shape == (self.original_dim,)
            self.normalization = True
            self.std = std[np.newaxis]
        else:
            self.normalization = False

        # Set up Y.
        if labels is not None:
            self.supervised = True
            if len(labels.shape) == 1:
                assert len(labels) == len(data)
                y = torch.tensor(labels, dtype=torch.long, device=device)
            else:
                self.output_inds = None
                y = torch.tensor(labels, dtype=torch.float32, device=device)
        else:
            self.supervised = False
            y = torch.tensor(data, dtype=torch.float32, device=device)
        self.y = y

        # Set up X.
        self.set_inds(inds)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def set_inds(self, inds):
        x = self.data
        if self.mean_adjustment:
            x = x - self.mean
        if self.normalization:
            x = x / self.std

        # Set inputs.
        if inds is not None:
            if isinstance(inds, list):
                if len(inds) < self.original_dim:
                    inds = np.array(
                        [i in inds for i in range(self.original_dim)],
                        dtype=bool)
                else:
                    inds = np.array(inds, dtype=bool)
            elif isinstance(inds, np.ndarray):
                if len(inds) < self.original_dim:
                    inds = np.array(
                        [i in inds for i in range(self.original_dim)],
                        dtype=bool)
            else:
                raise ValueError('inds must be list or np.ndarray')
            x = x[:, inds]

        # Move to PyTorch.
        x = torch.tensor(x, dtype=torch.float32, device=self.device)

        self.x = x
        self.inds = inds

    def set_output_inds(self, inds):
        assert not self.supervised

        y = self.data

        # Setup inputs.
        if inds is not None:
            if isinstance(inds, list):
                if len(inds) < self.original_dim:
                    inds = np.array(
                        [i in inds for i in range(self.original_dim)],
                        dtype=bool)
                else:
                    inds = np.array(inds, dtype=bool)
            elif isinstance(inds, np.ndarray):
                if len(inds) < self.original_dim:
                    inds = np.array(
                        [i in inds for i in range(self.original_dim)],
                        dtype=bool)
            else:
                raise ValueError('inds must be list or np.ndarray')
            y = y[:, inds]

        # Move to PyTorch.
        y = torch.tensor(y, dtype=torch.float32, device=self.device)

        self.y = y
        self.output_inds = inds

    def get_shape(self):
        return tuple(self.x.shape)


def split_data(data, seed=123, val_portion=0.1, test_portion=0.1):
    N = data.shape[0]
    N_val = int(val_portion * N)
    N_test = int(test_portion * N)
    train, test = train_test_split(data, test_size=N_test, random_state=seed)
    train, val = train_test_split(train, test_size=N_val, random_state=seed+1)
    return train, val, test


def unsupervised_dataloaders(seed=123,
                             val_portion=0.1,
                             test_portion=0.1,
                             device=None,
                             mean_adjustment=False,
                             normalization=False):
    '''
    Create dataloaders with no label information (for self reconstruction).

    Args:
      seed: for shuffling dataset.
      val_portion: portion for validation.
      test_portion: portion for test.
      mean_adjustment: whether to adjust x for mean.
      normalization: whether to adjust x for standard deviation.
      device: device for torch.Tensors.
    '''
    # Load data.
    data = sio.loadmat('Mouse-V1-ALM-20180520_thr_7-5.mat')
    lopge = data['lOPGE']

    # Split data.
    train, val, test = split_data(lopge, seed=seed, val_portion=val_portion,
                                  test_portion=test_portion)

    # Calculate mean and std from training data.
    train_mean = np.mean(train, axis=0)
    train_std = np.std(train, axis=0)
    mean = train_mean if mean_adjustment else None
    std = train_std if normalization else None
    train_mean = torch.tensor(train_mean, device=device, dtype=torch.float32)
    train_std = torch.tensor(train_std, device=device, dtype=torch.float32)

    # Calculate total variance.
    full_mean = np.mean(lopge, axis=0, keepdims=True)
    total_variance = np.mean(np.sum((lopge - full_mean) ** 2, axis=1))

    # Create datasets.
    train_set = RNASeq(train, mean=mean, std=std, device=device)
    val_set = RNASeq(val, mean=mean, std=std, device=device)
    test_set = RNASeq(test, mean=mean, std=std, device=device)

    # Create data loaders.
    random_sampler = RandomSampler(train_set, replacement=True)
    batch_sampler = BatchSampler(random_sampler, batch_size=512, drop_last=True)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler)
    val_loader = DataLoader(val_set, batch_size=len(val))
    test_loader = DataLoader(test_set, batch_size=len(test))

    return (train_loader, val_loader, test_loader,
            train_mean, train_std, total_variance)


def supervised_dataloaders(seed=123,
                           val_portion=0.1,
                           test_portion=0.1,
                           mean_adjustment=False,
                           normalization=False,
                           device=None):
    '''
    Create dataloaders with cell type classification labels.

    Args:
      seed: for shuffling dataset.
      val_portion: portion for validation.
      test_portion: portion for test.
      mean_adjustment: whether to adjust x for mean.
      normalization: whether to adjust x for standard deviation.
      device: device for torch.Tensors.
    '''
    # Load data.
    data = sio.loadmat('Mouse-V1-ALM-20180520_thr_7-5.mat')
    lopge = data['lOPGE']

    # Add labels.
    labels = pd.Categorical(data['cluster']).codes[:, np.newaxis]
    lopge = np.concatenate((lopge, labels), axis=1)

    # Split data.
    train, val, test = split_data(lopge, seed=seed, val_portion=val_portion,
                                  test_portion=test_portion)

    # Separate X and Y.
    Y_train = train[:, -1]
    Y_val = val[:, -1]
    Y_test = test[:, -1]
    train = train[:, :-1]
    val = val[:, :-1]
    test = test[:, :-1]

    # Calculate mean and std from training data.
    train_mean = np.mean(train, axis=0)
    train_std = np.std(train, axis=0)
    mean = train_mean if mean_adjustment else None
    std = train_std if normalization else None
    train_mean = torch.tensor(train_mean, device=device, dtype=torch.float32)
    train_std = torch.tensor(train_std, device=device, dtype=torch.float32)

    # Create datasets.
    train_set = RNASeq(train, Y_train, mean, std, device)
    val_set = RNASeq(val, Y_val, mean, std, device)
    test_set = RNASeq(test, Y_test, mean, std, device)

    # Create data loaders.
    random_sampler = RandomSampler(train_set, replacement=True)
    batch_sampler = BatchSampler(random_sampler, batch_size=512, drop_last=True)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler)
    val_loader = DataLoader(val_set, batch_size=len(val))
    test_loader = DataLoader(test_set, batch_size=len(test))

    return train_loader, val_loader, test_loader, train_mean, train_std
