import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import HMC, MCMC, MixedHMC, NUTS
import numpy as np


def weighted_l1_ball_projection(beta, weights, r):
    abs_beta = jnp.abs(beta)
    nonzero_w = weights > 0

    weighted_l1 = jnp.sum(jnp.where(nonzero_w, weights * abs_beta, 0.0))
    inside = weighted_l1 <= r

    # Breakpoints: |β_i| / w_i, with -inf for zero-weight (excluded)
    safe_w = jnp.where(nonzero_w, weights, 1.0)
    breakpoints = jnp.where(nonzero_w, abs_beta / safe_w, -jnp.inf)
    idx = jnp.argsort(breakpoints)[::-1]

    abs_beta_sort  = abs_beta[idx]
    weights_sort   = weights[idx]
    nonzero_w_sort = nonzero_w[idx]

    w_abs_beta = jnp.where(nonzero_w_sort, weights_sort * abs_beta_sort, 0.0)
    w2         = jnp.where(nonzero_w_sort, weights_sort ** 2,            0.0)

    cumsum_wab = jnp.cumsum(w_abs_beta)
    cumsum_w2  = jnp.cumsum(w2)

    bp_sort = jnp.where(nonzero_w_sort, abs_beta_sort / safe_w[idx], -jnp.inf)

    # λ candidate after including k elements in the active set
    lam_candidates = (cumsum_wab - r) / jnp.where(cumsum_w2 > 0, cumsum_w2, 1.0)

    # Valid prefix: this element is active (λ_k < θ_k) and nonzero weight
    active = nonzero_w_sort & (lam_candidates < bp_sort)

    # ---- FIX: use the LAST valid index, or fall back to full-sum lambda ----
    # Count valid indices; if none exist, all nonzero components are active
    n_active = jnp.sum(active)
    has_active = n_active > 0

    # Last active index one-hot
    active_shifted = jnp.concatenate([active[1:], jnp.array([False])])
    one_hot_c = active & ~active_shifted

    lam_from_active = jnp.dot(one_hot_c.astype(jnp.float32), lam_candidates)

    # Fallback: use the full prefix over all nonzero-weight elements
    total_wab = jnp.sum(w_abs_beta)
    total_w2  = jnp.sum(w2)
    lam_full  = (total_wab - r) / jnp.where(total_w2 > 0, total_w2, 1.0)

    lam = jnp.where(has_active, lam_from_active, lam_full)

    theta = jnp.where(
        nonzero_w,
        jnp.sign(beta) * jnp.maximum(abs_beta - lam * safe_w, 0.0),
        beta
    )

    return jnp.where(inside, beta, theta)


def linear_model(X, y, weights, a, b, beta_0, beta_var, r):
    """
    Bayesian linear regression with weighted L1 ball projection.

    Priors:
        sigma2 ~ InverseGamma(sig_p1, sig_p2)
        beta   ~ DoublExponential(0, sqrt(sigma2 * beta_s))   [Laplace]

    Likelihood:
        y ~ Normal(X @ theta, sqrt(sigma2))

    where theta = projection of beta onto { x : Σ wᵢ|xᵢ| ≤ r }.
    """
    p = X.shape[1]

    sigma2 = numpyro.sample("sigma2", dist.InverseGamma(a, b))

    z = numpyro.sample("z", dist.Normal(jnp.zeros(p), 1.0).to_event(1))
    beta = numpyro.deterministic("beta", beta_0 + jnp.sqrt(sigma2 * beta_var) * z)

    # beta = numpyro.sample(
    #     "beta",
    #     dist.Normal(beta_0 * jnp.ones(p), jnp.sqrt(sigma2 * beta_var)).to_event(1)
    # )

    # Project beta onto the weighted L1 ball
    theta = numpyro.deterministic(
        "theta", weighted_l1_ball_projection(beta, weights, r)
    )

    # Likelihood
    mu_y = X @ theta
    #mu_y = X @ beta
    numpyro.sample("y", dist.Normal(mu_y, jnp.sqrt(sigma2)), obs=y)



def fit_lm(X, y, weights, a, b, beta_0, beta_var, r,
        num_warmup=1000, num_samples=2000, num_chains = 1, seed=0):
    """
    Run NUTS sampling and return the MCMC object.
    """
    kernel = NUTS(linear_model, dense_mass=True)
    mcmc   = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(
        jax.random.PRNGKey(seed),
        X, y, weights, a, b, beta_0, beta_var, r
    )
    return mcmc



def logistic_model(X, y, weights, beta_0, beta_var, r = jnp.inf, projection = True):
    """
    Bayesian logistic regression with weighted L1 ball projection.

    Priors:
        sigma2 ~ InverseGamma(sig_p1, sig_p2)
        beta   ~ Normal(0, sqrt(beta_s))

    Likelihood:
        y ~ Bernoulli(sigmoid(X @ theta))

    where theta = projection of beta onto { x : Σ wᵢ|xᵢ| ≤ r }.
    """
    p = X.shape[1]

    if projection:
        beta = numpyro.sample(
        "beta",
        dist.Normal(jnp.zeros(p), jnp.sqrt(beta_var)).to_event(1)
        )
        theta = numpyro.deterministic(
            "theta", weighted_l1_ball_projection(beta, weights, r)
        )
    else:
        theta = numpyro.sample(
        "theta",
        dist.Normal(beta_0 * jnp.ones(p), jnp.sqrt(beta_var)).to_event(1)
        )

    # Logistic likelihood — BernoulliLogits is numerically more stable
    # than Bernoulli(sigmoid(...))
    logits = X @ theta
    numpyro.sample("y", dist.BernoulliLogits(logits).to_event(1), obs=y)


# ── 3. Fitting helper (unchanged) ────────────────────────────────────────────

def fit_lc(X, y, weights, beta_0=0.0, beta_var = 1.0, r = jnp.inf,
        num_warmup=1000, num_samples=2000, seed=0, projection = True):
    kernel = NUTS(logistic_model)
    mcmc   = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(
        jax.random.PRNGKey(seed),
        X, y, weights, beta_0, beta_var, r, projection
    )
    return mcmc