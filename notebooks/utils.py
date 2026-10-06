import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


def safe_min(arr):
    return np.min(arr) if len(arr) > 0 else np.nan

def safe_argmin(arr):
    return np.argmin(arr) if len(arr) > 0 else np.nan


def get_frequencies(S):
    S = pd.Series(S)
    p = []
    for p_s in sorted(S.value_counts(1)):
        p.append(p_s)
    return p

def cramers_v(x, y, bias_correction=True):
    """Cramér's V between two categorical series (optionally bias-corrected, Bergsma 2013)."""
    table = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = chi2 / n
    if bias_correction:
        phi2 = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r = r - (r - 1) ** 2 / (n - 1)
        k = k - (k - 1) ** 2 / (n - 1)
    denom = min(k - 1, r - 1)
    return np.sqrt(phi2 / denom) if denom > 0 else np.nan


def plot_distributions_compare(fair_model, base_model, X, S):
    S_val = sorted(np.unique(S))

    y_pred_fair = fair_model.predict(X)
    y_pred_base = base_model.predict(X)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    ax1.set_title("Fair model")
    for s in S_val:
        ax1.hist(y_pred_fair[S == s], label='s=' + str(s), alpha=0.5, density=True, stacked=True, bins=20)
    ax1.legend()

    ax2.set_title("Base model")
    for s in S_val:
        ax2.hist(y_pred_base[S == s], label='s=' + str(s), alpha=0.5, density=True, stacked=True, bins=20)
    ax2.legend()

    plt.show()



#def fair_projection_l2(v: np.ndarray, w: np.ndarray, lam: float = 1.0, radius: float = 1.0) -> np.ndarray:
def projection_l1_ball(v: np.ndarray, w: np.ndarray, #lam: float = 1.0,
                       radius: float = 1.0) -> np.ndarray:
    """
    Project vector v onto the weighted L1 ball: { x : sum(w_i * |x_i|) <= radius }

    Args:
        v:      Input vector to project (shape: [n])
        w:      Positive weight vector  (shape: [n])
        radius: Radius of the L1 ball   (scalar > 0)

    Returns:
        x: Projected vector (shape: [n])
    """

    # if not lam > 0:
    #     return 0
    #
    # w = lam * w

    n = len(v)

    x = np.zeros(n)
    zero_mask = (w.copy() == 0)
    pos_mask = (w.copy() > 0)
    x[ zero_mask ] = v.copy()[ zero_mask ]

    vpos = v[w>0]
    wpos = w[w>0]

    if not radius > 0:
        return np.zeros( len(v) )

    # If already inside the ball, return as-is
    if np.sum(wpos * np.abs(vpos)) <= radius:
        return v.copy()

    # Work with absolute values; restore signs at the end
    u = np.abs(vpos)

    # Sort breakpoints: lambda candidates are u_i / w_i (descending)
    idx = np.argsort(u / wpos)[::-1]
    u_s = u[idx]
    w_s = wpos[idx]

    # Find lambda* such that sum_i w_i * max(u_i - lambda * w_i, 0) = radius
    # Cumulative sums to evaluate each linear piece efficiently
    w2_cumsum = np.cumsum(w_s ** 2)       # sum of w_i^2 for active set
    wu_cumsum = np.cumsum(w_s * u_s)      # sum of w_i * u_i for active set

    # Candidate lambda for each prefix (active set = first k+1 elements)
    lambdas = (wu_cumsum - radius) / w2_cumsum

    # Find the largest k where lambda < u_{k} / w_{k} (breakpoint condition)
    breakpoints = u_s / w_s
    # Valid if lambda_k < breakpoint_k (element k is still active)
    valid = lambdas < breakpoints
    k = np.where(valid)[0][-1]            # last valid index
    lam = lambdas[k]

    # Soft-threshold and restore original signs
    x[pos_mask] = np.sign(vpos) * np.maximum(u - lam * wpos, 0)

    return x


def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

# def fair_projection_l2(
#     beta_ref,
#     weights,
#     lam=1.0,
#     beta_init=None,
#     step_size=1.0,
#     max_iter=1000,
#     tol=1e-6,
#     eps=1e-8
# ):
#     """
#     Minimize:
#         ||beta - beta_ref||_2^2 + lam * sum_j w_j |beta_j|
#
#     Parameters
#     ----------
#     beta_ref : (p,) ndarray
#         Reference vector
#     weights : (p,) ndarray
#         User-specified nonnegative penalty weights
#     lam : float
#         Regularization parameter
#     beta_init : (p,) ndarray or None
#         Initial beta
#     step_size : float
#         Gradient step size
#     max_iter : int
#     tol : float
#         Convergence tolerance
#     eps : float
#         Numerical stability constant
#
#     Returns
#     -------
#     beta : (p,) ndarray
#     """
#
#     beta_ref = np.asarray(beta_ref)
#     weights = np.asarray(weights)
#
#     if beta_ref.shape != weights.shape:
#         raise ValueError("beta_ref and weights must have the same shape")
#
#     if np.any(weights < 0):
#         raise ValueError("weights must be nonnegative")
#
#     p = beta_ref.shape[0]
#
#     if beta_init is None:
#         beta = beta_ref.copy()
#     else:
#         beta = np.asarray(beta_init).copy()
#
#     for i in range(max_iter):
#
#         beta_old = beta.copy()
#
#         # Gradient of ||beta - beta_ref||_2^2
#         grad = 2 * (beta - beta_ref)
#
#         # Gradient step
#         z = beta - step_size * grad
#
#         # Proximal step: weighted soft-thresholding
#         beta = soft_threshold(
#             z,
#             step_size * lam * weights
#         )
#
#         # Convergence check
#         if np.linalg.norm(beta - beta_old) < tol:
#             break
#
#         if i == (max_iter - 1):
#             print("Algorithm has not converged")
#     return beta