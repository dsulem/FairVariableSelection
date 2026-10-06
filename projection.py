import numpy as np
from metrics import absolute_covariance, mean_distance, DP_unfairness_WD, DP_unfairness_TV, individual_unfairness, penalisation_function, parity_gap
from scipy.stats import wasserstein_distance
from scipy.stats import bernoulli, multivariate_normal, pearsonr, norm, multivariate_t, t, gamma, invgamma
from sklearn import feature_selection, preprocessing,linear_model
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from utils import projection_l1_ball #fair_projection_l2
from mcmc import fit_lc, fit_lm
import jax.numpy as jnp
from scipy.special import expit



def sample_projection_posterior(X_train, y_train, S_train, a=1.0, b=1.0, beta_var=1.0, radius=[1.0], intercept=True,
                                num_samples = 1000, task = "reg", beta_0=0.0, metric ='correlation', weights = None,
                                num_warmup=1000):

    n = X_train.shape[0]

    if weights is None:
        if metric == 'correlation':
            weights = np.abs( feature_selection.r_regression(X_train,S_train) )
        elif metric == 'wasserstein':
            weights = [wasserstein_distance(X_train[S_train==1,i], X_train[S_train==0,i]) for i in range(X_train.shape[1])]
        elif metric == 'meandifference':
            weights = [mean_distance(X_train[:,i], S_train) for i in range(X_train.shape[1])]
        elif metric is None:
            weights = np.ones(X_train.shape[1])

    if intercept:
        dim = X_train.shape[1] + 1
        weights = np.insert( weights, 0 , 0.0)
        X_train = np.concatenate([np.ones((len(S_train),1)), X_train], axis=1)

    else:
        dim = X_train.shape[1]
        #weights = np.abs(feature_selection.r_regression(X_train,S_train))

    if task == "reg":

        if y_train.ndim == 1:
            y_train = y_train.reshape(-1,1)

        Sigma0 = beta_var * np.eye(dim) # prior covariance matrix
        beta0 = beta_0 * np.ones((dim,1))

        Lambda0 = np.linalg.inv(Sigma0)
        Lambda_n = X_train.T @ X_train + Lambda0
        Sigma_n = np.linalg.inv(Lambda_n)

        mu_n = Sigma_n @ (X_train.T @ y_train + Lambda0 @ beta0)

        a_n = a + n / 2
        b_n = b + 0.5 * (y_train.T @ y_train + beta0.T @ Lambda0 @ beta0 - mu_n.T @ Lambda_n @ mu_n).item()

        # marginal posteriors
        beta_samples = multivariate_t.rvs(loc=mu_n.ravel(), shape=(b_n / a_n) * Sigma_n,
                                          df=2 * a_n, size=num_samples)
        sigma2_samples = invgamma.rvs(a=a_n, scale=b_n, size=num_samples)

        # precompute products
        # prod_Xy = ( X_train.transpose() @ y_train ).reshape(-1,1)
        # Sigma_1 = np.linalg.inv( X_train.transpose() @ X_train + np.linalg.inv(Sigma0) )
        # # posterios covariance
        # pre_fact = ( (2 * b + y_train.transpose() @ y_train - prod_Xy.transpose() @ Sigma_1 @ prod_Xy) / (2 * a + n) ).item()
        # tmp =  np.linalg.solve(Sigma0, beta0).reshape(-1,1)
        # mu_post = Sigma_1 @ (prod_Xy + tmp)
        # #print(Sigma_1.shape, prod_Xy.shape, tmp.shape, mu_post.shape)
        # cov = pre_fact * Sigma_1
        # #print("Posterior mean: ", mu_post.flatten())
        # b_post = b + ( y_train.transpose() @ y_train - mu_post.transpose() @ np.linalg.inv(Sigma_1) @ mu_post ) / 2
        # conjugate posterior samples from multivariate student
        #post_samples = multivariate_t.rvs(loc = mu_post.flatten(), shape= cov, df = 2 * a + n, size = num_samples)
        #post_samples_var = invgamma.rvs( a=a + n/2, scale = 1./ b_post, size = num_samples)

        conjugate_posterior = {"mean": mu_n.ravel(), "samples": beta_samples, "samples var": sigma2_samples}

    else:
        # run mcmc
        mcmc = fit_lc(
            jnp.array(X_train), jnp.array(y_train.flatten()),
            jnp.array(weights), beta_var = beta_var, r = np.inf,
            num_warmup=num_warmup, num_samples=num_samples, projection=False
        )
        #mcmc.print_summary()
        beta_samples = mcmc.get_samples()["theta"]
        sigma2_samples = None

        mu_post = np.mean(beta_samples, axis = 0 )

        conjugate_posterior = {"mean": mu_post, "samples":  beta_samples, "samples var": sigma2_samples}


    # project samples
    proj_samples = np.zeros( (len(radius), beta_samples.shape[0], dim))

    for j, r in enumerate(radius):

        for i in range(beta_samples.shape[0]):
            samp = beta_samples[i,:]
            psamp = projection_l1_ball(samp, weights, r) #fair_projection_l2(samp,weights,alpha)
            proj_samples[j,i,:] = psamp

    projection_posterior = {"radius": radius, "mean": np.mean(proj_samples,axis=1), "samples": proj_samples, "weights" : weights,
                            "samples var": sigma2_samples}

    return projection_posterior, conjugate_posterior


def sample_posterior_with_projection_prior(X_train, y_train, S_train, a=1.0, b=1.0, beta_0 = 0.0, beta_var=1.0, radius=[1.0],
                                           intercept=True, num_samples = 1000, task = "reg", metric="correlation", weights = None,
                                           num_warmup=1000):

    n = X_train.shape[0]

    if weights is None:
        if metric == 'correlation':
            weights = np.abs( feature_selection.r_regression(X_train,S_train) )
        elif metric == 'wasserstein':
            weights = [wasserstein_distance(X_train[S_train==1,i], X_train[S_train==0,i]) for i in range(X_train.shape[1])]
        elif metric == 'meandifference':
            weights = [mean_distance(X_train[:,i], S_train) for i in range(X_train.shape[1])]
        elif metric is None:
            weights = np.ones(X_train.shape[1])

    if intercept:
        dim = X_train.shape[1] + 1
        weights = np.insert( weights, 0, 0.0)
        X_train = np.concatenate([np.ones((len(S_train), 1)), X_train], axis=1)

    else:
        dim = X_train.shape[1]


    class_post_samples = np.zeros((len(radius), num_samples, X_train.shape[1]))
    class_post_samples_var = np.zeros((len(radius), num_samples))

    for i, r in enumerate(radius):

        if task == "reg":

            mcmc = fit_lm(
                jnp.array(X_train), jnp.array(y_train.flatten()),
                jnp.array(weights), a, b, beta_0, beta_var, r,
                num_warmup=num_warmup, num_samples=num_samples
            )

            class_post_samples_var[i] = mcmc.get_samples()["sigma2"]

        elif task == "class":

            mcmc = fit_lc(
                jnp.array(X_train), jnp.array(y_train.flatten()),
                jnp.array(weights), beta_0, beta_var, r,
                num_warmup=num_warmup, num_samples=num_samples
            )

        else:
            print("task not implemented")

        class_post_samples[i, :] = mcmc.get_samples()["theta"]


    class_proj_prior = {"samples": class_post_samples, "mean": np.mean( class_post_samples, axis=1), "weights": weights, "radius": radius,
                        "samples var": class_post_samples_var}

    return class_proj_prior


def evaluate_posterior_per_budget(proj_posterior, X_train, S_train, y_train, X_val, S_val, y_val,intercept=True, coeff=5.0,
                                  task="reg", weights = None, p_train = None, p_val = None, coeff2=1.0, sample_from_pp = False):

    if not isinstance(y_train, np.ndarray):
        try:
            y_train = y_train.to_numpy()
        except:
            pass

    if intercept:
        X_train = np.concatenate([ np.ones((len(S_train),1)), X_train], axis=1)
        X_val = np.concatenate([np.ones((len(S_val), 1)), X_val], axis=1)

    samples = proj_posterior["samples"]
    samples_var = proj_posterior["samples var"]

    if weights is None:
        weights = proj_posterior["weights"]

    pmean = {"MSE": [], "BCE": [], "ACC": [], "AUC": [], "WDU":[], "TVU":[], "IU": [], "Corr": [], "norm_pen":[], "Parity Gap":[],
             "MSE-test": [], "BCE-test": [], "ACC-test": [], "AUC-test": [], "WDU-test":[], "TVU-test":[], "IU-test": [], "Corr-test": [],
             "Parity Gap-test": []
             }

    logloss, mse, acc, auc, meandiff, wassdist, totalvar, indivfair,  indivfair2, indivfair3, corr, norm_pen, pgap = np.zeros( (samples.shape[0], samples.shape[1]) ), \
        np.zeros( (samples.shape[0], samples.shape[1]) ), np.zeros( (samples.shape[0], samples.shape[1]) ), \
        np.zeros( (samples.shape[0], samples.shape[1]) ), np.zeros( (samples.shape[0], samples.shape[1]) ), \
        np.zeros( (samples.shape[0], samples.shape[1]) ), np.zeros( (samples.shape[0], samples.shape[1]) ), \
        np.zeros( (samples.shape[0], samples.shape[1]) ), np.zeros( (samples.shape[0], samples.shape[1]) ), \
        np.zeros((samples.shape[0], samples.shape[1])), np.zeros((samples.shape[0], samples.shape[1])), \
        np.zeros((samples.shape[0], samples.shape[1])), np.zeros((samples.shape[0], samples.shape[1]))
    loglossval, mseval, accval, aucval, meandiffval, wassdistval, totalvarval, indivfairval, indivfairval2, indivfairval3, corrval, pgapval =  np.zeros( (samples.shape[0], samples.shape[1]) ), \
        np.zeros( (samples.shape[0], samples.shape[1]) ), np.zeros( (samples.shape[0], samples.shape[1]) ), \
        np.zeros( (samples.shape[0], samples.shape[1]) ), np.zeros( (samples.shape[0], samples.shape[1]) ), \
        np.zeros( (samples.shape[0], samples.shape[1]) ), np.zeros( (samples.shape[0], samples.shape[1]) ), \
        np.zeros((samples.shape[0], samples.shape[1])), np.zeros( (samples.shape[0], samples.shape[1]) ), \
        np.zeros((samples.shape[0], samples.shape[1])),  np.zeros((samples.shape[0], samples.shape[1])), \
        np.zeros((samples.shape[0], samples.shape[1]))

    # performance of posterior predictive
    if task == "reg":
        posterior_pred, posterior_pred_val = {"MSE": [], "WDU":[], "IU":[], "IU2":[]}, {"MSE": [], "WDU":[], "IU":[], "IU2":[]}
    else:
        posterior_pred, posterior_pred_val = {"BCE": [], "WDU": [], "IU": [], "IU2": []}, {"BCE": [], "WDU": [], "IU": [],                                                                               "IU2": []}

    for i in range(samples.shape[0]):

        if task == "class":

            y_pred_train = np.mean(expit(X_train @ np.transpose(samples[i])), axis=1)
            y_pred_val = np.mean(expit(X_val @ np.transpose(samples[i])), axis=1)
            posterior_pred["BCE"].append(log_loss(y_train, y_pred_train))
            posterior_pred_val["BCE"].append(log_loss(y_val, y_pred_val))
            posterior_pred["WDU"].append(DP_unfairness_WD(y_pred_train, S_train))
            posterior_pred_val["WDU"].append(DP_unfairness_WD(y_pred_val, S_val))
            posterior_pred["IU"].append(individual_unfairness(y_train, y_pred_train, S_train, coeff=coeff, task=task))
            posterior_pred_val["IU"].append(individual_unfairness(y_val, y_pred_val, S_val, coeff=coeff, task=task))
            if p_train is not None and p_val is not None:
                posterior_pred["IU2"].append( individual_unfairness(p_train, y_pred_train, S_train, coeff=coeff, task="reg"))
                posterior_pred_val["IU2"].append( individual_unfairness(p_val, y_pred_val, S_val, coeff=coeff, task="reg") )

        for j in range(samples.shape[1]): # evaluation per sample

            w = samples[i,j,:]

            if task == "reg":

                if samples_var.ndim > 1:
                    sigma2 = samples_var[i,j]
                else:
                    sigma2 = samples_var[j]

                y_pred_train = X_train @ w
                y_pred_val = X_val @ w

                # sample from posterior predictive
                if sample_from_pp:
                    y_pred_train = y_pred_train + np.sqrt(sigma2) * np.random.standard_normal(size=len(y_pred_train))
                    y_pred_val = y_pred_val + np.sqrt(sigma2) * np.random.standard_normal(size=len(y_pred_val))

                mse[i,j] = mean_squared_error(y_pred_train, y_train)
                mseval[i,j] = mean_squared_error(y_pred_val, y_val)
                #print(sigma2, y_train.shape, y_pred_train.shape)
                logloss[i,j] = np.log(2*np.pi) / 2 + 0.5 * np.log(sigma2) + np.mean( (y_train.flatten() - y_pred_train.flatten()) ** 2 ) / (2 * sigma2)
                loglossval[i, j] = np.log(2 * np.pi) / 2 + 0.5 * np.log(sigma2) + np.mean( (y_val.flatten() - y_pred_val.flatten() ) ** 2 ) / (2 * sigma2)

            elif task == "class":

                y_pred_train = 1 / ( 1 + np.exp( - X_train @ w ) )
                y_pred_val = 1 / (1 + np.exp( -X_val @ w ))
                logloss[i, j] = log_loss(y_train, y_pred_train)
                loglossval[i, j] =  log_loss(y_val, y_pred_val)

                acc[i, j] = accuracy_score(y_train, y_pred_train > 0.5)
                accval[i, j] =   accuracy_score(y_val, y_pred_val > 0.5)

                auc[i, j] = roc_auc_score(y_train, y_pred_train)
                aucval[i, j] = roc_auc_score(y_val, y_pred_val)

                pgap[i,j] = parity_gap(y_pred_train > 0.5, S_train)
                pgapval[i, j] = parity_gap(y_pred_val > 0.5, S_val)

                if p_train is not None and p_val is not None:
                    indivfair2[i, j] = individual_unfairness(p_train, y_pred_train, S_train, coeff=coeff, task="reg")
                    indivfairval2[i, j] = individual_unfairness(p_val, y_pred_val, S_val, coeff=coeff, task="reg")


            else:
                print("task not implemented")

            #totalvar[i,j] = DP_unfairness_TV(y_pred_train, S_train)
            #totalvarval[i,j] = DP_unfairness_TV(y_pred_val, S_val)

            wassdist[i,j] = DP_unfairness_WD(y_pred_train, S_train)
            wassdistval[i,j] = DP_unfairness_WD(y_pred_val, S_val)

            indivfair[i,j] = individual_unfairness(y_train, y_pred_train, S_train, coeff=coeff, task = task)
            indivfairval[i,j] = individual_unfairness(y_val, y_pred_val, S_val, coeff=coeff, task = task)

            #indivfair3[i, j] = individual_unfairness(y_train, y_pred_train, S_train, X = X_train, coeff=coeff2, task="covariates")
            #indivfairval3[i, j] = individual_unfairness(y_val, y_pred_val, S_val, X=X_val, coeff=coeff2, task="covariates")

            #corr[i,j] = absolute_covariance( y_pred_train, S_train)
            #corrval[i,j] = absolute_covariance( y_pred_val, S_val)

            #norm_pen[i,j] = penalisation_function(w, weights)

        # evaluation of the posterior mean for a given budget
        m_est = np.mean(samples[i,:,:], axis=0)
        if task == "reg":
            y_pred_train = X_train @ m_est
            y_pred_val = X_val @ m_est
            pmean["MSE"].append( mean_squared_error(y_pred_train, y_train) )
            pmean["MSE-test"].append( mean_squared_error(y_pred_val, y_val) )
            pmean["WDU"].append(DP_unfairness_WD(y_pred_train, S_train))
            pmean["WDU-test"].append(DP_unfairness_WD(y_pred_val, S_val))

        elif task == "class":
            y_pred_train = 1 / (1 + np.exp(- X_train @ m_est))
            y_pred_val = 1 / (1 + np.exp(-X_val @ m_est))
            pmean["BCE"].append( log_loss(y_train, y_pred_train))
            pmean["BCE-test"].append(log_loss(y_val, y_pred_val))
            pmean["ACC"].append( accuracy_score(y_train, y_pred_train > 0.5) )
            pmean["ACC-test"].append( accuracy_score(y_val, y_pred_val > 0.5) )
            pmean["AUC"].append(  roc_auc_score(y_train, y_pred_train) )
            pmean["AUC-test"].append(  roc_auc_score(y_val, y_pred_val) )
            pmean["Parity Gap"].append( parity_gap(y_pred_train > 0.5, S_train))
            pmean["Parity Gap-test"].append(parity_gap(y_pred_val > 0.5, S_val))
        else:
            print("task not implemented")

        pmean["TVU"].append( DP_unfairness_TV(y_pred_train, S_train) )
        pmean["TVU-test"].append( DP_unfairness_TV(y_pred_val, S_val) )
        pmean["IU"].append(individual_unfairness(y_train, y_pred_train, S_train, coeff=coeff, task=task) )
        pmean["IU-test"].append( individual_unfairness(y_val, y_pred_val, S_val, coeff=coeff, task=task) )
        pmean["Corr"].append( absolute_covariance(y_pred_train, S_train))
        pmean["Corr-test"].append( absolute_covariance(y_pred_val, S_val))
        pmean["norm_pen"].append( penalisation_function(m_est, weights) )


    for key in pmean:
        pmean[key] = np.array(pmean[key])

    for key in posterior_pred:
        posterior_pred[key] = np.array(posterior_pred[key]).reshape(-1,1)
        posterior_pred_val[key] = np.array(posterior_pred_val[key]).reshape(-1,1)

    #norm_pen = norm_pen / np.max(norm_pen)

    results_train = {"budget": proj_posterior["radius"], "MSE": np.array(mse), "BCE": np.array(logloss), "TVU": np.array(totalvar), "WDU": np.array(wassdist),
                     "IU": np.array(indivfair), "IU2": np.array(indivfair2), "IU3": np.array(indivfair3), "Cov": np.array(corr), "pen-samples": np.array(norm_pen), "ACC": np.array(acc),
                     "AUC": np.array(auc), "mean": pmean, "Parity Gap": np.array(pgap), "PP": posterior_pred }
    results_val = {"budget": proj_posterior["radius"], "BCE": np.array(loglossval), "TVU": np.array(totalvar),
                   "WDU": np.array(wassdistval), "IU": np.array(indivfairval), "IU2": np.array(indivfairval2), "IU3": np.array(indivfairval3), "Cov": np.array(corrval),
                   "ACC": np.array(accval), "AUC": np.array(aucval), "Parity Gap": np.array(pgapval), #"Pen": np.array(pen),
                   "MSE": np.array(mseval), "PP": posterior_pred_val}

    return results_train, results_val




def evaluate_conjugate_posterior(posterior, X_train, S_train, y_train, X_val, S_val, y_val, weights=None, intercept=True,
                                 coeff=5.0, task = "reg", p_train = None, p_val = None, coeff2=1.0, sample_from_pp = False):

    samples = posterior["samples"]
    samples_var = posterior["samples var"]

    if intercept:
        if weights is None:
            weights = np.insert(np.abs(feature_selection.r_regression(X_train, S_train)), 0, 0.0)
        X_train = np.concatenate([ np.ones((len(S_train),1)), X_train], axis=1)
        X_val = np.concatenate([np.ones((len(S_val), 1)), X_val], axis=1)

    else:
        if weights is None:
            weights = np.abs(feature_selection.r_regression(X_train,S_train))

    mse, logloss, acc, auc, meandiff, wassdist, totalvar, indivfair, indivfair2, indivfair3, corr, pen, pgap = np.zeros( samples.shape[0] ), np.zeros( samples.shape[0] ), \
        np.zeros( samples.shape[0] ), np.zeros( samples.shape[0] ), np.zeros( samples.shape[0] ), np.zeros( samples.shape[0] ), \
        np.zeros( samples.shape[0] ), np.zeros( samples.shape[0] ), np.zeros( samples.shape[0] ), np.zeros( samples.shape[0] ), \
        np.zeros(samples.shape[0]), np.zeros(samples.shape[0]), np.zeros(samples.shape[0])

    mseval, loglossval, accval, aucval, meandiffval, wassdistval, totalvarval, indivfairval, indivfairval2,  indivfairval3, corrval, pgap_val =  np.zeros( samples.shape[0] ), \
        np.zeros( samples.shape[0] ), np.zeros( samples.shape[0] ), np.zeros( samples.shape[0] ), \
        np.zeros( samples.shape[0] ), np.zeros( samples.shape[0] ), np.zeros(samples.shape[0]), \
        np.zeros(samples.shape[0]), np.zeros(samples.shape[0]), np.zeros(samples.shape[0]), np.zeros(samples.shape[0]), \
        np.zeros(samples.shape[0])

    # performance of posterior predictive
    posterior_pred, posterior_pred_val = {}, {}
    if task == "class":
        y_pred_train = np.mean(expit(X_train @ np.transpose(samples) ), axis=1)
        y_pred_val = np.mean(expit(X_val @ np.transpose(samples) ), axis=1)
        posterior_pred["BCE"] = log_loss(y_train, y_pred_train)
        posterior_pred_val["BCE"] = log_loss(y_val, y_pred_val)
        posterior_pred["WDU"] = DP_unfairness_WD(y_pred_train, S_train)
        posterior_pred_val["WDU"] = DP_unfairness_WD(y_pred_val, S_val)
        posterior_pred["IU"] = individual_unfairness(y_train, y_pred_train, S_train, coeff=coeff, task = task)
        posterior_pred_val["IU"] = individual_unfairness(y_val, y_pred_val, S_val, coeff=coeff, task = task)


    for i in range(samples.shape[0]):

        w = samples[i,:]

        if task == "reg":
            y_pred_train = X_train @ w
            y_pred_val = X_val @ w

            # sample from posterior predictive
            if sample_from_pp:
                sigma2 = samples_var[i]
                y_pred_train = y_pred_train + np.sqrt(sigma2) * np.random.standard_normal(size=len(y_pred_train))
                y_pred_val = y_pred_val + np.sqrt(sigma2) * np.random.standard_normal(size=len(y_pred_val))

            mse[i] = mean_squared_error(y_pred_train, y_train)
            mseval[i] = mean_squared_error(y_pred_val, y_val)

        elif task == "class":

            y_pred_train = 1 / (1 + np.exp(- X_train @ w))
            y_pred_val = 1 / (1 + np.exp(-X_val @ w))
            logloss[i] = log_loss(y_train, y_pred_train)
            loglossval[i] = log_loss(y_val, y_pred_val)
            acc[i] = accuracy_score(y_train, y_pred_train > 0.5)
            accval[i] = accuracy_score(y_val, y_pred_val > 0.5)
            auc[i] = roc_auc_score(y_train, y_pred_train)
            aucval[i] = roc_auc_score(y_val, y_pred_val)
            pgap[i] = parity_gap(y_pred_train > 0.5, S_train)
            pgap_val[i] = parity_gap(y_pred_val > 0.5, S_val)

            if p_train is not None and p_val is not None:
                indivfair2[i] = individual_unfairness(p_train, y_pred_train, S_train, coeff=coeff, task="reg")
                indivfairval2[i] = individual_unfairness(p_val, y_pred_val, S_val, coeff=coeff, task="reg")

        else:
            print("task not implemented")

        #totalvar[i] = DP_unfairness_TV(y_pred_train, S_train)
        #totalvarval[i] = DP_unfairness_TV(y_pred_val, S_val)

        wassdist[i] = DP_unfairness_WD(y_pred_train, S_train)
        wassdistval[i] = DP_unfairness_WD(y_pred_val, S_val)

        indivfair[i] = individual_unfairness(y_train, y_pred_train, S_train, coeff=coeff, task = task)
        indivfairval[i] = individual_unfairness(y_val, y_pred_val, S_val, coeff=coeff, task = task)

        indivfair3[i] = individual_unfairness(y_train, y_pred_train, S_train, X=X_train, coeff=coeff2,
                                                 task="covariates")
        indivfairval3[i] = individual_unfairness(y_val, y_pred_val, S_val, X=X_val, coeff=coeff2,
                                                    task="covariates")

        #corr[i] = absolute_covariance( y_pred_train, S_train)
        #corrval[i] = absolute_covariance( y_pred_val, S_val)
        #pen[i] = penalisation_function(w, weights)

    pmean = posterior["mean"]

    if task == "reg":
        y_pred_train = X_train @ pmean
        y_pred_val = X_val @ pmean
        pmean_perf = [mean_squared_error(y_pred_train, y_train), DP_unfairness_WD(y_pred_train, S_train),
                      individual_unfairness(y_train, y_pred_train, S_train, coeff=coeff, task=task),
                      penalisation_function(pmean, weights)]
        pmean_perf_val = [mean_squared_error(y_pred_val, y_val), DP_unfairness_WD(y_pred_val, S_val),
                          individual_unfairness(y_val, y_pred_val, S_val, coeff=coeff, task=task),
                          penalisation_function(pmean, weights)]

    else:
        y_pred_train = 1 / (1 + np.exp(- X_train @ pmean))
        y_pred_val = 1 / (1 + np.exp(-X_val @ pmean))
        pmean_perf = [log_loss(y_train, y_pred_train), DP_unfairness_WD(y_pred_train, S_train),
                      individual_unfairness(y_train, y_pred_train, S_train, coeff=coeff, task=task),
                      accuracy_score(y_train, y_pred_train > 0.5), roc_auc_score(y_train, y_pred_train),
                      penalisation_function(pmean, weights), parity_gap(y_pred_train > 0.5, S_train)]
        pmean_perf_val = [log_loss(y_val, y_pred_val), DP_unfairness_WD(y_pred_val, S_val),
                          individual_unfairness(y_val, y_pred_val, S_val, coeff=coeff, task=task), roc_auc_score(y_val, y_pred_val),
                          accuracy_score(y_val, y_pred_val > 0.5), penalisation_function(pmean, weights),
                          parity_gap(y_pred_val > 0.5, S_val)]

    results_train = {"MSE": np.array(mse), "BCE": np.array(logloss), "TVU": np.array(totalvar), "WDU": np.array(wassdist),
                     "IU": np.array(indivfair), "IU2": np.array(indivfair2),  "IU3": np.array(indivfair3), "Cov": np.array(corr), "Pen": np.array(pen),
                     "pmean": np.array(pmean_perf), "ACC": np.array(acc), "AUC": np.array(auc), "Parity Gap": np.array(pgap), "PP": posterior_pred}
    results_val = {"MSE": np.array(mseval), "BCE": np.array(loglossval), "TVU": np.array(totalvar), "WDU": np.array(wassdistval), "IU": np.array(indivfairval),
                   "IU2": np.array(indivfairval2), "IU3": np.array(indivfairval3), "Cov": np.array(corrval),
                   "pmean": np.array(pmean_perf_val), "ACC": np.array(accval), "AUC": np.array(aucval), "Pen": np.array(pen), "Parity Gap": np.array(pgap_val),
                   "PP": posterior_pred_val}

    return results_train, results_val