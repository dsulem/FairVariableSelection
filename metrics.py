import numpy as np
from scipy.stats import wasserstein_distance
from sklearn import feature_selection
from scipy.spatial import distance_matrix

def absolute_covariance(y,S):

    return np.abs( np.cov(y.flatten(), S)[0,1] )

    #return np.abs(feature_selection.r_regression(y.reshape(-1,1), S))

def mean_distance(y,S):

    S_val = sorted(np.unique(S))

    if len(S_val) == 2:
        return np.abs(np.mean(y[S == S_val[1]]) - np.mean(y[S == S_val[0]]))

    else:
        m = np.mean(y)
        return np.mean([ np.abs(np.mean(y[S == s]) - m) for s in S_val])

def parity_gap(y, S):

    S_val = sorted(np.unique(S))

    if len(S_val) == 2:
        return np.abs(np.mean(y[S == S_val[1]] > 0.5) - np.mean(y[S == S_val[0]] > 0.5))

    else:
        print("not implemented")

def DP_unfairness_TV(y, S, bins=20):

    S_val = sorted(np.unique(S))

    if len(S_val) == 2:
        hist1, bin_edges = np.histogram(y[S==S_val[1]],bins=bins)
        CDF_1 = np.cumsum(hist1/sum(hist1))

        hist0, bin_edges = np.histogram(y[S==S_val[0]],bins=bins)
        CDF_0 = np.cumsum(hist0/sum(hist0))

        return(max(abs(CDF_1-CDF_0)))

    else:
        hist, bin_edges = np.histogram(y, bins=bins)
        CDF = np.cumsum(hist / len(y))

        Unfairness = []
        S_val = sorted(np.unique(S))
        for s in S_val:
            hist, bin_edges = np.histogram(y[S == s], bins=bins)
            CDF_s = np.cumsum(hist / sum(hist))
            Unfairness.append(max(abs(CDF_s - CDF)))

        return np.mean(Unfairness)

def DP_unfairness_WD(y, S):

    S_val = sorted(np.unique(S))

    if len(S_val) == 2:
        return wasserstein_distance(y[S==S_val[0]].flatten(), y[S==S_val[1]].flatten())

    else:
        print("Not implemented")


def individual_unfairness(y_true, y_pred, S, X = None, coeff = 1.0, task = "reg"):

    S_val = sorted(np.unique(S))

    if len(S_val) == 2:
        y_true_1, y_pred_1 = y_true[S==S_val[0]], y_pred[S==S_val[0]]
        y_true_2, y_pred_2 = y_true[S == S_val[1]], y_pred[S == S_val[1]]

        if task == "reg":
            diff = y_true_1[:,None] - y_true_2[None,:]
            #dist_matrix = coeff * np.exp( - coeff * np.abs(diff.squeeze()) )
            dist_matrix = coeff * np.exp(- coeff * diff.squeeze() ** 2 )
            #diff_pred = (y_pred_1[:, None] - y_pred_2[None, :]) ** 2


        elif task == "class":
            dist_matrix =  (y_true_1[:,None] == y_true_2[None,:])
            #diff_pred = np.abs( np.int32(y_pred_1[:, None] > 0.5) - np.int32(y_pred_2[None, :] > 0.5) )
            #diff_pred =  np.abs(y_pred_1[:, None] - y_pred_2[None, :])
            #diff_pred = (y_pred_1[:, None] - y_pred_2[None, :]) ** 2

        elif task == "covariates" and X is not None:
            X_1, X_2 = X[S == S_val[0]], X[S == S_val[1]]
            diff = distance_matrix(X_1, X_2)
            dist_matrix = coeff * np.exp(- coeff * np.abs(diff.squeeze()) ) #** 2 )

        else:
            return "task not implemented"

        diff_pred = (y_pred_1[:, None] - y_pred_2[None, :]) ** 2

        return np.mean( dist_matrix * diff_pred.squeeze())


    else:
        print("Not implemented")


def penalisation_function(beta, pen_coeffs):

    return( np.linalg.norm(beta * pen_coeffs, ord=1))