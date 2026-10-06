import numpy as np
from metrics import absolute_covariance, mean_distance, DP_unfairness_WD, DP_unfairness_TV, individual_unfairness, penalisation_function, parity_gap
from scipy.stats import wasserstein_distance
from scipy.stats import bernoulli, multivariate_normal, pearsonr, norm, multivariate_t, t
from sklearn import feature_selection, preprocessing,linear_model
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter

from fairfs.fairfs_new import run_all, run_experiment
Fairml = importr("fairml")



class fairlasso():

    def __init__(self, alpha = None):
        self.alpha = alpha

    def fit(self,X_train, S_train, y_train, fit_intercept=True, task = "reg", weights=None):

        if weights is None:
            weights = np.abs(feature_selection.r_regression(X_train, S_train))

        # Adaptive lasso
        W = np.diag(1.0 / weights)
        X_star = X_train @ W

        if task == "reg":
            fair_lasso = linear_model.Lasso(alpha=self.alpha, fit_intercept=fit_intercept)
            fair_lasso.fit(X_star, y_train)
            self.coeffs = W @ fair_lasso.coef_

        elif task == "class":
            fair_lasso = LogisticRegression(penalty="l1", C=1/self.alpha, solver="liblinear", fit_intercept=fit_intercept, max_iter=50000)
            fair_lasso.fit(X_star, y_train)
            self.coeffs = W @ fair_lasso.coef_[0]


        self.weights = weights
        self.W = W
        self.model = fair_lasso
        self.intercept = fair_lasso.intercept_
        self.task = task

    def predict(self, X):

        X_star = X @ self.W
        return self.fit.predict(X_star)

    def evaluate(self, X_train, S_train, y_train, X_val, S_val, y_val, alphas=[1.0], metric='correlation', plot=True, coeff=5.0,
                            fit_intercept=True, task="reg", p_train = None, p_val = None, coeff2 = 0.5, weights=None, max_iter=10000):

        self.task = task

        if weights is None:
            if metric == 'correlation':
                lamdas = np.abs(feature_selection.r_regression(X_train,S_train))
            elif metric == 'wasserstein':
                lamdas = np.array([wasserstein_distance(X_train[S_train==1,i], X_train[S_train==0,i]) for i in range(X_train.shape[1])])
            elif metric == 'meandifference':
                lamdas = np.array([mean_distance(X_train[:,i], S_train) for i in range(X_train.shape[1])])
        else:
            lamdas = weights

        logloss, mse, acc, auc, meandiff, wassdist, \
            totalvar, indivfair, indivfair2, indivfair3, corr, pen, pgap = [], [], [], [], [], [], [], [], [], [], [], [], []
        loglossval, mseval, accval, aucval, meandiffval, \
            wassdistval, totalvarval, indivfairval,  indivfairval2, indivfairval3, corrval, pgapval = [], [], [], [], [], [], [], [], [], [], [], []
        coeffs_path = []

        for alpha in alphas:

            W = np.diag(1.0/lamdas)
            X_star= X_train @ W
            X_star_val = X_val @ W

            if task == "reg":
                fair_lasso = linear_model.Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
                fair_lasso.fit(X_star, y_train.reshape(-1, ))
                fair_lasso_coeff = W @ fair_lasso.coef_

            elif task == "class":
                fair_lasso = LogisticRegression(penalty="l1",
                                                C=1.0/alpha, solver="liblinear", fit_intercept=True,
                                                max_iter=max_iter, tol=1e-6)
                fair_lasso.fit(X_star, y_train.reshape(-1, ))
                fair_lasso_coeff = W @ fair_lasso.coef_[0]


            coeffs_path.append(np.insert(fair_lasso_coeff, 0, fair_lasso.intercept_))
            #coeffs_path.append(fair_lasso_coeff)

            y_pred_train = fair_lasso.predict(X_star)
            y_pred_val = fair_lasso.predict(X_star_val)

            if task == "reg":
                mse.append(mean_squared_error(y_pred_train, y_train))
                mseval.append(mean_squared_error(y_pred_val, y_val))

                wassdist.append(DP_unfairness_WD(y_pred_train, S_train))
                wassdistval.append(DP_unfairness_WD(y_pred_val, S_val))

                corr.append(absolute_covariance(y_pred_train, S_train))
                corrval.append(absolute_covariance(y_pred_val, S_val))

                indivfair.append(individual_unfairness(y_train, y_pred_train, S_train, coeff=coeff, task=task))
                indivfairval.append(individual_unfairness(y_val, y_pred_val, S_val, coeff=coeff, task=task))

                indivfair3.append(
                    individual_unfairness(y_train, y_pred_train, S_train, X=X_train, coeff=coeff2, task="covariates"))
                indivfairval3.append(
                    individual_unfairness(y_val, y_pred_val, S_val, X=X_val, coeff=coeff2, task="covariates"))


            elif task == "class":
                y_proba_train = np.exp(fair_lasso.predict_log_proba(X_star))[:, 1]
                y_proba_val = np.exp(fair_lasso.predict_log_proba(X_star_val))[:, 1]

                logloss.append(log_loss(y_train, y_proba_train))
                loglossval.append(log_loss(y_val, y_proba_val))

                acc.append(accuracy_score(y_train, y_pred_train))
                accval.append(accuracy_score(y_val, y_pred_val))

                auc.append(roc_auc_score(y_train, y_proba_train))
                aucval.append(roc_auc_score(y_val, y_proba_val))

                pgap.append(parity_gap(y_pred_train, S_train))
                pgapval.append(parity_gap(y_pred_val, S_val))

                wassdist.append(DP_unfairness_WD(y_proba_train, S_train))
                wassdistval.append(DP_unfairness_WD(y_proba_val, S_val))

                corr.append(absolute_covariance(y_proba_train, S_train))
                corrval.append(absolute_covariance(y_proba_val, S_val))

                indivfair.append(individual_unfairness(y_train, y_proba_train, S_train, coeff=coeff, task=task))
                indivfairval.append(individual_unfairness(y_val, y_proba_val, S_val, coeff=coeff, task=task))

                if p_train is not None and p_val is not None:
                    indivfair2.append(individual_unfairness(p_train, y_proba_train, S_train, coeff=coeff, task="reg"))
                    indivfairval2.append(individual_unfairness(p_val, y_proba_val, S_val, coeff=coeff, task="reg"))

                indivfair3.append(individual_unfairness(y_train, y_proba_train, S_train, X = X_train, coeff=coeff2, task="covariates"))
                indivfairval3.append(individual_unfairness(y_val, y_proba_val, S_val, X = X_val, coeff=coeff2, task="covariates"))

            totalvar.append(DP_unfairness_TV(y_pred_train, S_train))
            totalvarval.append(DP_unfairness_TV(y_pred_val, S_val))

            pen.append(penalisation_function(fair_lasso_coeff, lamdas))

        results_train = {"alphas": alphas, "MSE": np.array(mse), "BCE": np.array(logloss), "TVU": np.array(totalvar), "WDU": np.array(wassdist),
                         "IU": np.array(indivfair), "IU2": np.array(indivfair2), "IU3": np.array(indivfair3), "Cov": np.array(corr), "budget": np.array(pen), "CoeffPath": np.array(coeffs_path),
                         "ACC": np.array(acc), "AUC": np.array(auc), "Parity Gap": np.array(pgap), "weights": lamdas}
        results_val = {"alphas": alphas, "MSE": np.array(mseval), "BCE": np.array(loglossval), "TVU": np.array(totalvarval),"WDU": np.array(wassdistval), "IU": np.array(indivfairval),
                       "IU2": np.array(indivfairval2), "IU3": np.array(indivfairval3),
                       "Cov": np.array(corrval), "ACC": np.array(accval), "budget": np.array(pen), "AUC": np.array(aucval), "Parity Gap": np.array(pgapval),
                       "weights": lamdas
                       }


        if plot:
            dim = len(fair_lasso_coeff)
            colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
            coeffs_path = np.array(coeffs_path)
            plt.figure(1)
            #max_pen = max(pen)
            for i in range( coeffs_path.shape[1]):
                if fit_intercept:
                    if i > 0:
                        #l1 = plt.semilogx(np.array(pen), coeffs_path[:, i], label=fr"$\beta_{i}$", color=colors[(i - 1) // 2])
                        l1 = plt.semilogx( alphas, coeffs_path[:, i], label=fr"$\beta_{i }$", color=colors[(i-1)//(dim//4)])
                else:
                    l1 = plt.semilogx(np.array(pen), coeffs_path[:,i], label=fr"$\beta_{i + 1}$", color=colors[i//(dim//4)])
                    #l1 = plt.semilogx(alphas, coeffs_path[:, i], label=fr"$\beta_{i + 1}$", color=colors[i // 2])
            #plt.plot(results_train["alphas"], results_train["Pen"] / max_pen, label="Penalty")
            plt.xlabel(r"$\lambda$")
            # if task == "reg":
            #     plt.xlabel(r"$\lambda$")
            # elif task == "class":
            #     plt.xlabel(r"$1/\lambda$")
            # plt.ylabel("coefficients")
            plt.title("Fair Lasso Path")
            plt.legend()
            # plt.legend((l1[-1]), ("Lasso"), loc="lower right")
            plt.axis("tight")
            plt.savefig(f"/Users/deborah/PycharmProjects/FairModelSelection/figures/simulation_fair_lasso_path_{task}.pdf")

        return results_train, results_val


class standardlasso():

    def __init__(self, alpha = None):
        self.alpha = alpha

    def fit(self,X_train, S_train, y_train, fit_intercept=True, task = "reg"):

        # Adaptive lasso
        weights = np.abs(feature_selection.r_regression(X_train, S_train))
        W = np.diag(1.0 / weights)
        X_star = X_train @ W

        if task == "reg":
            lasso = linear_model.Lasso(alpha=self.alpha, fit_intercept=fit_intercept, max_iter=50000)
            lasso.fit(X_train, y_train.reshape(-1, ))
            self.coeffs = lasso.coef_

        elif task == "class":
            lasso = LogisticRegression(penalty="l1", C=1.0/self.alpha, solver="liblinear", fit_intercept=True, max_iter=50000)
            lasso.fit(X_train, y_train.reshape(-1, ))
            self.coeffs = lasso.coef_[0]

        self.weights = weights
        self.W = W
        self.model = lasso
        self.intercept = lasso.intercept_
        self.task = task

    def evaluate(self, X_train, S_train, y_train, X_val, S_val, y_val, p_train = None, p_val = None, alphas=None, metric="correlation", plot=True, coeff=1.0,
                                fit_intercept=True, task="reg", coeff2 = 0.5, weights = None):

        self.task = task

        #alpha_path, coeff_path,_= linear_model.lasso_path(X_train, y_train, alphas=None)
        if weights is None:
            if metric == 'correlation':
                lamdas = np.abs( feature_selection.r_regression(X_train,S_train) )
            elif metric == 'wasserstein':
                lamdas = [wasserstein_distance(X_train[S_train==1,i], X_train[S_train==0,i]) for i in range(X_train.shape[1])]
            elif metric == 'meandifference':
                lamdas = [mean_distance(X_train[:,i], S_train) for i in range(X_train.shape[1])]
        else:
            lamdas = weights


        logloss, mse, acc, auc, meandiff, wassdist, \
            totalvar, indivfair, indivfair2, indivfair3, corr, pen, pgap = [], [], [], [], [], [], [], [], [], [], [], [], []
        loglossval, mseval, accval, aucval, meandiffval, wassdistval, \
            totalvarval, indivfairval, indivfairval2,  indivfairval3, corrval,  pgapval = [], [], [], [], [], [], [], [], [], [], [], []
        coeffs_path = []

        for i,alpha in enumerate(alphas):
        #for i,alpha in enumerate(alpha_path):

            # model = make_pipeline(
            #     StandardScaler(),
            #     Lasso(alpha=alpha, max_iter=50000)
            # )
            #
            # model.fit(X_train, y_train)
            # lasso = model.named_steps["lasso"]
            # coeffs_path.append(lasso.coef_)

            if task == "reg":
                lasso = linear_model.Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=10000)

            elif task == "class":
                lasso = LogisticRegression(penalty="l1", C=1.0/alpha, solver="liblinear", fit_intercept=True, max_iter=10000)

            lasso.fit(X_train, y_train.reshape(-1,) )
            coeffs_path.append( np.insert(lasso.coef_ , 0, lasso.intercept_))
            #w = coeff_path[0,:,i]
            #y_pred_train = X_train @ w
            #y_pred_val = X_val @ w

            y_pred_train = lasso.predict(X_train)
            y_pred_val = lasso.predict(X_val)

            if self.task == "reg":
                mse.append(mean_squared_error(y_pred_train, y_train))
                mseval.append(mean_squared_error(y_pred_val, y_val))

                wassdist.append(DP_unfairness_WD(y_pred_train, S_train))
                wassdistval.append(DP_unfairness_WD(y_pred_val, S_val))

                corr.append(absolute_covariance(y_pred_train, S_train))
                corrval.append(absolute_covariance(y_pred_val, S_val))

                indivfair.append(individual_unfairness(y_train, y_pred_train, S_train, coeff=coeff, task=task))
                indivfairval.append(individual_unfairness(y_val, y_pred_val, S_val, coeff=coeff, task=task))

                indivfair3.append(
                    individual_unfairness(y_train, y_pred_train, S_train, X=X_train, coeff=coeff2, task="covariates"))
                indivfairval3.append(
                    individual_unfairness(y_val, y_pred_val, S_val, X=X_val, coeff=coeff2, task="covariates"))

            elif self.task == "class":
                y_proba_train = np.exp(lasso.predict_log_proba(X_train))[:,1]
                y_proba_val = np.exp(lasso.predict_log_proba(X_val))[:, 1]

                logloss.append(log_loss(y_train, y_proba_train))
                loglossval.append(log_loss(y_val, y_proba_val))

                acc.append( accuracy_score(y_train, y_pred_train) )
                accval.append( accuracy_score(y_val, y_pred_val) )

                auc.append(  roc_auc_score(y_train, y_proba_train) )
                aucval.append( roc_auc_score(y_val, y_proba_val) )

                pgap.append( parity_gap(y_pred_train, S_train) )
                pgapval.append( parity_gap(y_pred_val, S_val) )

                wassdist.append(DP_unfairness_WD(y_proba_train, S_train))
                wassdistval.append(DP_unfairness_WD(y_proba_val, S_val))

                corr.append(absolute_covariance( y_proba_train, S_train ))
                corrval.append(absolute_covariance( y_proba_val, S_val))

                indivfair.append(individual_unfairness(y_train, y_proba_train, S_train, coeff=coeff, task = task))
                indivfairval.append(individual_unfairness(y_val, y_proba_val, S_val, coeff=coeff, task = task))

                indivfair3.append(
                    individual_unfairness(y_train, y_proba_train, S_train, X=X_train, coeff=coeff2, task="covariates"))
                indivfairval3.append(
                    individual_unfairness(y_val, y_proba_val, S_val, X=X_val, coeff=coeff2, task="covariates"))

                if p_train is not None and p_val is not None:
                    indivfair2.append(individual_unfairness(p_train, y_proba_train, S_train, coeff=coeff, task = "reg"))
                    indivfairval2.append(individual_unfairness(p_val, y_proba_val, S_val, coeff=coeff, task="reg"))

                #pen.append(penalisation_function(lasso.coef_[0], lamdas))

            totalvar.append(DP_unfairness_TV(y_pred_train, S_train))
            totalvarval.append(DP_unfairness_TV(y_pred_val, S_val))



        results_train = {"alphas": alphas, "MSE": np.array(mse), "BCE": np.array(logloss), "TVU": np.array(totalvar), "WDU": np.array(wassdist),
                         "IU": np.array(indivfair), "IU2": np.array(indivfair2), "IU3": np.array(indivfair3), "Cov": np.array(corr), "budget":np.array(pen), "CoeffPath": np.array(coeffs_path),
                         "ACC": np.array(acc), "AUC": np.array(auc), "Parity Gap": np.array(pgap), "weights": lamdas}
        results_val = {"alphas": alphas, "MSE": np.array(mseval), "BCE": np.array(loglossval), "TVU": np.array(totalvarval),"WDU": np.array(wassdistval),
                       "IU": np.array(indivfairval),  "IU2": np.array(indivfairval2), "IU3": np.array(indivfairval3),
                       "Cov": np.array(corrval), "ACC": np.array(accval), "AUC": np.array(aucval), "budget":np.array(pen), "Parity Gap": np.array(pgapval),
                       "weights": lamdas}

        if plot:
            # plot lasso path
            colors = ['tab:blue', 'tab:green', 'tab:orange',  'tab:red']
            coeffs_path = np.array(coeffs_path)
            plt.figure(1)
            #max_pen = max(pen)
            for i in range(coeffs_path.shape[1]):
                if fit_intercept:
                    if i>0:
                        #l1 = plt.semilogx(np.array(pen), coeffs_path[:, i], label=fr"$\beta_{i}$", color=colors[(i - 1) // 2])
                        l1 = plt.semilogx(alphas, coeffs_path[:, i], label=fr"$\beta_{i }$", color=colors[(i-1)//2])
                else:
                    #l1 = plt.semilogx(np.array(pen), coeffs_path[:,i], label=fr"$\beta_{i + 1}$", color=colors[i//2])
                    l1 = plt.semilogx(alphas, coeffs_path[:,i], label=fr"$\beta_{i + 1}$", color=colors[i//2])
            #plt.plot(alphas, pen / max_pen, label="Penalty")
            #plt.xlabel(r"Unfairness budget $\epsilon$")
            plt.xlabel(r"$\lambda$")
            # if task == "reg":
            #     plt.xlabel(r"$\lambda$")
            # elif task == "class":
            #     plt.xlabel(r"$1/\lambda$")
            plt.title("(Standard) Lasso Path")
            plt.legend()
            # plt.legend((l1[-1]), ("Lasso"), loc="lower right")
            plt.axis("tight")
            plt.savefig(f"/Users/deborah/PycharmProjects/FairModelSelection/figures/simulation_lasso_path_{task}.pdf")

        return results_train, results_val


def evaluate_calders(X_train, S_train, y_train, X_val, S_val, y_val, alphas=[1.0], coeff=5.0):

    XS_train_1 = np.concatenate([S_train.reshape(-1,1), np.ones((len(S_train),1)), X_train], axis=1)
    d = np.mean(XS_train_1[S_train==0,:], axis=0) - np.mean(XS_train_1[S_train == 1,:], axis=0)
    d = d.reshape(-1,1)
    cov = XS_train_1.transpose() @ XS_train_1

    #wOLS = np.linalg.solve(cov, XS_train_1.transpose() @ y_train)
    aux = np.linalg.solve(cov, d)
    #wEM = wOLS - (d.transpose() @ wOLS) * aux / (d.transpose() @ aux)

    wPEN = []
    for alpha in alphas:

        wPEN.append(np.linalg.solve( cov + alpha * (d @ d.transpose()), XS_train_1.transpose() @ y_train))

    perf, meandiff, wassdist, totalvar, indivfair, corr = [], [], [], [], [], []
    perfval, meandiffval, wassdistval, totalvarval, indivfairval, corrval = [], [], [], [], [], []

    XS_val_1 = np.concatenate([S_val.reshape(-1,1), np.ones((len(S_val),1)), X_val], axis=1)

    #for w in ([wOLS, wEM] + wPEN):
    for w in (wPEN):

        y_pred_train = XS_train_1 @ w
        y_pred_val = XS_val_1 @ w

        perf.append(mean_squared_error(y_pred_train, y_train))
        perfval.append(mean_squared_error(y_pred_val, y_val))

        totalvar.append(DP_unfairness_TV(y_pred_train, S_train))
        totalvarval.append(DP_unfairness_TV(y_pred_val, S_val))

        wassdist.append(DP_unfairness_WD(y_pred_train, S_train))
        wassdistval.append(DP_unfairness_WD(y_pred_val, S_val))

        indivfair.append(individual_unfairness(y_train, y_pred_train, S_train, coeff=coeff))
        indivfairval.append(individual_unfairness(y_val, y_pred_val, S_val, coeff=coeff))

        corr.append(absolute_covariance( y_pred_train, S_train))
        corrval.append(absolute_covariance( y_pred_val, S_val))


    results_train = {"alphas": alphas, "MSE": np.array(perf), "TVU": np.array(totalvar), "WDU": np.array(wassdist), "IU": np.array(indivfair), "Cov": np.array(corr)}
    results_val = {"alphas": alphas, "MSE": np.array(perfval), "TVU": np.array(totalvarval),"WDU": np.array(wassdistval), "IU": np.array(indivfairval), "Cov": np.array(corrval) }


    return results_train, results_val



def evaluate_frrm(X_train, y_train, S_train, X_val, y_val, S_val, budget=None, metric="correlation", coeff=5.0, task = "reg",
                  p_train=None, p_val = None, coeff2 = 0.5):

    if task == "reg":
        ry_train = ro.FloatVector(y_train)
        ry_val = ro.FloatVector(y_val)
    elif task == "class":
        ry_train = ro.r.factor(ro.IntVector(y_train.astype(int)), levels=ro.IntVector([0, 1]))
        ry_val = ro.r.factor(ro.IntVector(y_val.astype(int)), levels=ro.IntVector([0, 1]))
    else:
        raise ValueError("task not implemented")

    rS_train = ro.FloatVector(S_train)
    rS_val = ro.FloatVector(S_val)

    with localconverter(default_converter + numpy2ri.converter):
        rX_train = ro.conversion.py2rpy(X_train)
        rX_val = ro.conversion.py2rpy(X_val)

    if budget is None:
        #budget = list(np.linspace(0.001, 1, num=30))
        budget = np.logspace(-5, 0, num=20, endpoint=True, base=10.0)

    list_frrm = []
    for i in range(len(budget)):
        r_p = ro.FloatVector([budget[i]])

        if task == "reg":
            frrm = Fairml.frrm(ry_train, rX_train, rS_train, r_p)
        else:
            frrm = Fairml.fgrrm(ry_train, rX_train, rS_train, r_p)
        list_frrm.append(frrm)

    list_pred_train = []
    list_pred_test = []
    with localconverter(default_converter + numpy2ri.converter):
        for i in range(len(budget)):
            pred_train = ro.conversion.rpy2py(ro.r["predict"](list_frrm[i], rX_train, rS_train))
            pred_test = ro.conversion.rpy2py(ro.r["predict"](list_frrm[i], rX_val, rS_val))
            list_pred_train.append(pred_train)
            list_pred_test.append(pred_test)

    pred_train = np.array(list_pred_train)
    pred_val = np.array(list_pred_test)

    #alpha_path, coeff_path,_= linear_model.lasso_path(X_train, y_train, alphas=None)
    # if metric == 'correlation':
    #     lamdas = feature_selection.r_regression(X_train,S_train)
    # elif metric == 'wasserstein':
    #     lamdas = [wasserstein_distance(X_train[S_train==1,i], X_train[S_train==0,i]) for i in range(X_train.shape[0])]
    # elif metric == 'meandifference':
    #     lamdas = [mean_distance(X_train[:,i], S_train) for i in range(X_train.shape[0])]


    mse, logloss, meandiff, wassdist, totalvar, indivfair, indivfair2, indivfair3, corr, acc, auc, pgap = [], [], [], [], [], [], [], [], [], [], [], []
    mseval, loglossval, meandiffval, wassdistval, totalvarval, indivfairval, \
        indivfairval2, indivfairval3, corrval, accval, aucval, pgapval = [], [], [], [], [], [], [], [], [], [], [], []

    for i in range(pred_train.shape[0]):
    #for i,alpha in enumerate(alpha_path):

        y_pred_train = pred_train[i,:]
        y_pred_val = pred_val[i,:]

        #print(y_pred_val)

        totalvar.append(DP_unfairness_TV(y_pred_train, S_train))
        totalvarval.append(DP_unfairness_TV(y_pred_val, S_val))

        wassdist.append(DP_unfairness_WD(y_pred_train, S_train))
        wassdistval.append(DP_unfairness_WD(y_pred_val, S_val))

        indivfair.append(individual_unfairness(y_train, y_pred_train, S_train, coeff=coeff, task = task))
        indivfairval.append(individual_unfairness(y_val, y_pred_val, S_val, coeff=coeff, task = task))

        indivfair3.append(individual_unfairness(y_train, y_pred_train, S_train, X=X_train, coeff=coeff2, task="covariates"))
        indivfairval3.append(individual_unfairness(y_val, y_pred_val, S_val, X=X_val, coeff=coeff2, task="covariates"))

        corr.append(absolute_covariance( y_pred_train, S_train))
        corrval.append(absolute_covariance( y_pred_val, S_val))

        if task == "class":

            logloss.append(log_loss(y_train, y_pred_train))
            loglossval.append( log_loss(y_val, y_pred_val))
            acc.append(accuracy_score(y_train, y_pred_train > 0.5) )
            accval.append(accuracy_score(y_val, y_pred_val > 0.5) )
            auc.append(roc_auc_score(y_train, y_pred_train))
            aucval.append(roc_auc_score(y_val, y_pred_val))
            pgap.append( parity_gap (y_pred_train > 0.5, S_train) )
            pgapval.append(parity_gap(y_pred_val > 0.5, S_val))

            if p_train is not None and p_val is not None:
                indivfair2.append(individual_unfairness(p_train, y_pred_train, S_train, coeff=coeff, task="reg"))
                indivfairval2.append(individual_unfairness(p_val, y_pred_val, S_val, coeff=coeff, task="reg"))

        else:
            mse.append(mean_squared_error(y_pred_train, y_train))
            mseval.append(mean_squared_error(y_pred_val, y_val))


    results_train = {"budget": budget, "MSE": np.array(mse), "BCE": np.array(logloss), "TVU": np.array(totalvar), "WDU": np.array(wassdist),
                     "IU": np.array(indivfair), "IU2": np.array(indivfair2), "IU3": np.array(indivfair3), "Cov": np.array(corr), "ACC": np.array(acc), "AUC": np.array(auc), "Parity Gap" : np.array(pgap)}
    results_val = {"budget": budget, "MSE": np.array(mseval), "BCE": np.array(loglossval), "TVunfair": np.array(totalvarval),"WDU": np.array(wassdistval),
                   "IU": np.array(indivfairval), "IU2": np.array(indivfairval2), "IU3": np.array(indivfairval3), "Cov": np.array(corrval), "ACC": np.array(accval), "AUC": np.array(aucval), "Parity Gap" : np.array(pgapval)}

    return results_train, results_val



def evaluate_fairfs(X_train, S_train, y_train, X_val, S_val, y_val, alphas=[1.0], coeff=5.0, unfairness_metrics=["statistical_parity"],
                    p_train=None, p_val = None, coeff2 = 0.5):


    results, predictions = run_all(X_train, S_train, y_train, X_val, S_val, y_val, unfairness_weights=alphas,
                                   unfairness_metric_names=unfairness_metrics)

    y_pred = (predictions[predictions.unfairness_metric == 'statistical_parity']
         .pivot(index='sample_index', columns='unfairness_weight', values='y_prob')
         .reindex(columns=alphas)
         .to_numpy()).T

    perfval, wassdistval,  indivfairval, indivfairval2, indivfairval3 = [], [], [], [], []


    #for w in ([wOLS, wEM] + wPEN):
    for i in range(len(alphas)):

        wassdistval.append(DP_unfairness_WD(y_pred[i], S_val))
        indivfairval.append(individual_unfairness(y_val, y_pred[i], S_val, coeff=coeff, task="class"))
        perfval.append(log_loss(y_val, y_pred[i]))

        indivfairval3.append(individual_unfairness(y_val, y_pred[i], S_val, X=X_val, coeff=coeff2, task="covariates"))

        if p_train is not None and p_val is not None:
            indivfairval2.append(individual_unfairness(p_val, y_pred[i], S_val, coeff=coeff, task="reg"))

    results_val = {"alphas": alphas, "logloss": results["log_loss"],  "BCE": np.array(perfval), "WDU": np.array(wassdistval), "IU": np.array(indivfairval),
                   "IU2": np.array(indivfairval2), "IU3": np.array(indivfairval3),
                   }


    return results_val