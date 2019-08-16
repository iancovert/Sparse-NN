# Sparse-NN

The `Sparse-NN` repository implements methods for sparsifying neural networks (i.e., making them use a small number of inputs). The methods are based on backward elimination and feature ranking, which provide a tractable approximate solution to a challenging combinatorial optimization problem.

## Installation

Please download the code by cloning the repository. To run it you'll need `Python >= 3`, `numpy`, `scipy`, `pandas`, `sklearn`, and `PyTorch >= 1.0.0`.

## How it works

The code is structured so that users can specify a base model that relies on all features, and then learn a sparse input neural network (SPINN) through either recursive feature elimination or a single ranking step. Base models are implemented in `models/mlp.py`, and SPINNs are implemented in `models/spinn.py`.

Five feature ranking methods are available here to use as part of the feature selection algorithm:

- **Learnable Bernoulli noise** (learn per-feature dropout rates)
- **Learnable Gaussian noise** (learn per-feature additive noise standard deviations)
- **Feature imputation** (measure increase in loss when features are imputed)
- **First layer sensitivity** (measure sensitivity of first hidden layer to each feature)
- **Jacobian sensitivity** (measure sensitivity of model output to each feature)


Each of the ranking methods leads to slightly different feature selection. The methods that tend to achieve the best performance with a small number of features are **Bernoulli noise**, **Gaussian noise**, and **feature imputation**. The methods that are most consistent with the features they select are **Gaussian noise** and **Bernoulli noise**.

To perform feature selection with these methods, a small amount of hyperparameter tuning is required. In our experiments we chose the model architecture and optimization hyperparameters that led to an accurate model when relying on all features.

## Unsupervised feature selection

In a paper that is under review, we proposed a method for selecting features *without* a specific supervised learning task in mind. The methods we developed are promising for selecting a small number of features in biological applications, e.g., high dimensional genetic data.

The mathematical theory leads to an intuitive conclusion: when forced to observe a subset of features, the most informative features are those that are highly predictive of the unobserved ones. In practice, we suggest learning an input-sparse autoencoder (ISAE). Feel free to reach out for a preprint.

## Demos

See the Jupyter notebooks `gaussian rfe.ipynb`, `bernoulli rfe.ipynb`, `imputation rfe.ipynb` for examples of feature selection using recursive feature elimination. See `bernoulli ranking.ipynb` as an example of feature selection based on a single ranking step.

## Authors

- Ian Covert (<icovert@cs.washington.edu>)
- Uygar Sumbul
- Su-In Lee