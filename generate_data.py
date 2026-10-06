import numpy as np
from scipy.stats import bernoulli, multivariate_normal, pearsonr, norm, multivariate_t, t
from sklearn import feature_selection, preprocessing,linear_model
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def simulate_linreg(n_train=100, n_test=100, sigma=0.5, rho=1.5, p=0.5, d=2):
    """

    :param n_train:  training sample size
    :param n_test:
    :param sigma: noise level
    :param rho: correlation level
    :param p:
    :return:
    """

    n = n_train + n_test

    # generate binary sensitive attribute
    S=bernoulli.rvs(p=p,size=n)

    # generate Gaussian covariates independent of S
    d1 = 2 * d
    X1 = multivariate_normal.rvs(mean=np.zeros(d1), cov=np.eye(d1), size=n)
    X1=preprocessing.scale(X1, axis=0)

    # generate Gaussian covariates correlated with S
    X2 = multivariate_normal.rvs(mean=np.zeros(d1), cov=np.eye(d1), size=n) + rho * S.reshape(n, 1)
    X2 = preprocessing.scale(X2, axis=0)

    # aggregate covariates into matrix
    X = np.concatenate([X1, X2], axis=1)
    print(f"Dimension of design matrix: {X.shape}")

    # generate outcome
    beta = np.concatenate([2.0 * np.ones(d),  0.2 * np.ones(d), -2.0 * np.ones(d), -0.2 * np.ones(d)])  # reg coeffs
    alpha = 0.  # coeff of S in y
    y = (np.dot(X, beta) + S * alpha + norm.rvs(scale=sigma, size=n)).reshape(-1, 1)

    X_train, y_train, S_train = X[:n_train, :], y[:n_train], S[:n_train]
    X_test, y_test, S_test = X[n_train:, :], y[n_train:], S[n_train:]

    data = { "beta": beta, "train": [X_train, y_train, S_train], "test": [X_test, y_test, S_test]}

    return data




def simulate_logreg(n_train=100, n_test=100, rho=1.5, p=0.5, d=2, intercept=1.0):
    """

    :param n_train:  training sample size
    :param n_test:
    :param sigma: noise level
    :param rho: correlation level
    :param p:
    :return:
    """

    n = n_train + n_test

    # generate binary sensitive attribute
    S=bernoulli.rvs(p=p,size=n)

    # generate Gaussian covariates independent of S
    d1 = 2 * d
    X1 = multivariate_normal.rvs(mean=np.zeros(d1), cov=np.eye(d1), size=n)
    X1=preprocessing.scale(X1, axis=0)

    # generate Gaussian covariates correlated with S
    X2 = multivariate_normal.rvs(mean=np.zeros(d1), cov=np.eye(d1), size=n) + rho * S.reshape(n, 1)
    X2 = preprocessing.scale(X2, axis=0)

    # aggregate covariates into matrix
    X = np.concatenate([X1, X2], axis=1)
    print(f"Dimension of design matrix: {X.shape}")

    # generate outcome
    beta = np.concatenate([2.0 * np.ones(d),  0.2 * np.ones(d), -2.0 * np.ones(d), 0.2 * np.ones(d)])  # reg coeffs
    alpha = 0.  # coeff of S in y

    logits = intercept + X @ beta + alpha * S
    y = np.random.binomial(1, 1 / (1 + np.exp(- logits)) )

    X_train, y_train, S_train, l_train = X[:n_train, :], y[:n_train], S[:n_train], logits[:n_train]
    X_test, y_test, S_test, l_test = X[n_train:, :], y[n_train:], S[n_train:], logits[n_train:]

    # true probas
    p_train, p_test = 1 / (1 + np.exp(- logits[:n_train])),  1 / (1 + np.exp(- logits[n_train:]))

    data = { "beta": beta, "train": [X_train, y_train, S_train, p_train], "test": [X_test, y_test, S_test, p_test]}

    return data


