import numpy as np
import statsmodels.api as sm
from collections import UserDict


class CausalData(UserDict):
    def __init__(self, outcome, treatment, covariates):
        """
        Initialize the causal data
        --------------------------
        :param outcome: N*1 nparray, the outcome to be analyzed
        :param treatment: N*1 nparray with 0 and 1, indicating the treatment following hypothetical assumptions
        :param covariates: N*K nparray, covariates/confounders that may affect the outcome
        """
        UserDict.__init__(self)
        Y, T, X = self.preprocess(outcome, treatment, covariates)
        self["Y"] = Y
        self["T"] = T
        self["X"] = X
        self["N"], self["K"] = covariates.shape

        self["N_t"] = T.sum()
        self["N_c"] = self["N"] - self["N_t"]
        if self["K"] + 1 > self["N_c"]:
            raise ValueError("Too few control units: N_c < K+1")
        if self["K"] + 1 > self["N_t"]:
            raise ValueError("Too few treatmend units: N_t < K+1")

    def preprocess(self, Y, T, X):
        """
        Preprocess the data
        -------------------
        :param Y: N*1 nparray, the outcome to be analyzed
        :param T: N*1 nparray with 0 and 1, indicating the treatment following hypothetical assumptions
        :param X: N*K nparray, covariates/confounders that may affect the outcome
        :return: a tuple of three np arrays: outcome, treatment and covariates
        """
        if Y.shape[0] == T.shape[0] == X.shape[0]:
            N = Y.shape[0]
        else:
            raise IndexError('Inconsistent number of rows among outcome, treatment and covariates')
        if Y.shape != (N,):
            Y.shape = (N,)
        if T.shape != (N,):
            T.shape = (N,)
        if T.dtype != 'int':
            T = T.astype(int)
        if X.shape == (N,):
            X.shape = (N,)
        return Y, T, X


class CausalModel(object):
    """
    class that conduct casual inference modeling given cleaned data
    """
    def __init__(self, Y, T, X, interact_index=None, obs_cluster=None, N_bt=30):
        """
        Initialize the causality model
        ------------------------------
        :param Y: N*1 numpy array, the outcome to be analyzed
        :param T: N*1 numpy array with 0 and 1, indicating the treatment following hypothetical assumptions
        :param X: N*K numpy array, covariates/confounders that may affect the outcome
        :param interact_index: a numpy array of column index used to build the interactive feature with the treatments
                               when applying standardization estimation
        :param obs_cluster: N*1 numpy array containing the cluster of each observation
        """
        self.data = CausalData(Y, T, X)
        self.Y = self.data["Y"]
        self.T = self.data["T"]
        self.X = self.data["X"]
        self.N = self.data["N"]
        self.K = self.data["K"]
        self.N_bt = N_bt

        self.interact_index = interact_index
        self.obs_cluster = obs_cluster

        self.ate_ip = None
        self.ate_ip_ci = None
        self.ate_std = None
        self.ate_std_ci = None

    def get_ip_weight(self, T, X):
        """
        Calculate the stablized inverse weighting P(T|X) from logistic regression and update ip_weights
        -------
        :param T: N*1 nparray, the treatment
        :param X: N*k nparray, the covariates that affect the outcome
        :return: N*1 nparray of stablized inverse weight
        """
        N = X.shape[0]
        Y = T
        try:
            model = sm.Logit(Y, X)
            res = model.fit()
            w_denoms = np.zeros(N)
            w_denoms[Y == 1] = res.predict(X[Y == 1])
            w_denoms[Y == 0] = (1 - res.predict(X[Y == 0]))
            raw_ip_weight = 1 / w_denoms
            s_weights = np.zeros(N)
            s_weights[Y == 1] = Y.mean() * raw_ip_weight[Y == 1]
            s_weights[Y == 0] = (1 - Y).mean() * raw_ip_weight[Y == 0]
            return s_weights
        except Exception as error:
            print("An exception occurred:", error)
            print("The ip weighting is not estimated properly and is set to uniform weights")
            return np.ones(N)

    def counterfactual_data_expansion(self, T, X, method="ip_weight"):
        """
        Build additional interactions with treatments
        ---------------------
        :param T: N*1 nparray
        :param X: N*K nparray, not containing the treatment T, but should include the constant column
        :param method: either ip_weight or standardization
        :return: a tuple of three np arrays: array of actual treatment interactions, array of counterfactual zero
                 treatment interactions and array of counterfactual ones treatment interactions
        """
        X_obs = X.copy()
        X_ones = X.copy()
        X_zeros = X.copy()
        N = X.shape[0]
        zeros = np.zeros(N)
        ones = np.ones(N)

        X_obs = np.concatenate((X_obs, T.reshape(-1, 1)), axis=1)
        X_zeros = np.concatenate((X_zeros, zeros.reshape(-1, 1)), axis=1)
        X_ones = np.concatenate((X_ones, ones.reshape(-1, 1)), axis=1)

        if method == "standardization" and self.interact_index is not None:
            for i in self.interact_index:
                X_obs = np.concatenate((X_obs, np.multiply(X[:, i], T).reshape(-1, 1)), axis=1)
                X_zeros = np.concatenate((X_zeros, np.multiply(X[:, i], zeros).reshape(-1, 1)), axis=1)
                X_ones = np.concatenate((X_ones, np.multiply(X[:, i], ones).reshape(-1, 1)), axis=1)
        return X_obs, X_zeros, X_ones

    def bootstrap_sample(self, Y, T, X, obs_cluster=None, ip_weight=None, method="ip_weight"):
        """
        get bootstrapping samples for ate
        ---------------------
        :param Y: N*1 nparray that contains the outcome variable
        :param T: N*1 nparray that conatins the treatment
        :param X: the covariates that do not contain treatment T
        :param interact_index: a numpy array of column index used to build the interactive feature with the treatments
                               when applying standardization estimation
        :param obs_cluster: N*1 numpy array containing the cluster of each observation
        :param method: either ip_weight or standardization
        :return: a list of bootstrapping samples
        """
        boot_samples = []
        N = X.shape[0]
        for i in range(self.N_bt):
            try:
                index = np.random.choice(N, N, replace=True)
                X_bt = X[index, :]
                T_bt = T[index]
                Y_bt = Y[index]
                X_obs_bt, X_zeros_bt, X_ones_bt = self.counterfactual_data_expansion(T_bt, X_bt, method)
                if method == "ip_weight":
                    ip_weight_bt = ip_weight
                    obs_cluster_bt = obs_cluster
                    if ip_weight is not None:
                        ip_weight_bt = ip_weight[index]
                    if obs_cluster is not None:
                        obs_cluster_bt = obs_cluster[index]
                    model_res_bt = self.fit_ip(Y_bt, X_obs_bt, obs_cluster_bt, ip_weight_bt)
                elif method == "standardization":
                    model_res_bt = self.fit_std(Y_bt, X_obs_bt)
                else:
                    raise Exception(
                        "Incorrect method, please choose the method between \'ip_weight\' and \'standardization\'")
                ones_pred_bt = model_res_bt.predict(X_ones_bt)
                zeros_pred_bt = model_res_bt.predict(X_zeros_bt)
                boot_samples.append(ones_pred_bt.mean() - zeros_pred_bt.mean())
            except:
                print(f"bootstrap failed for the {i}th trail")
        return boot_samples

    def fit_ip(self, Y, X, obs_cluster=None, ip_weight=None):
        """
        fit causal model using inverse weights
        -----------
        :param Y: N*1 nparray that contains the outcome variable
        :param X: N*k nparray that contains the constant covariates and the treatment
        :param obs_cluster: N*1 numpy array containing the cluster of each observation
        :param ip_weight: N*1 nparray that contains the inverse probability weights
        :return: fitted model
        """

        N = X.shape[0]
        if ip_weight is None:
            ip_weight = np.ones(N)
        if len(np.unique(Y)) != 2:
            # continuous outcome variable
            if not obs_cluster:
                obs_cluster = np.arange(1, N + 1)
            wls = sm.WLS(Y, X, weights=ip_weight)
            res = wls.fit(cov_type="cluster", cov_kwds={"groups": obs_cluster})
        else:
            # binary outcome variable
            wlg = sm.GLM(Y, X, family=sm.families.Binomial(), freq_weights=ip_weight)
            res = wlg.fit()
        return res

    def fit_std(self, Y, X):
        """
        fit causal model using standardization
        -----------
        :param Y: N*1 nparray that contains the outcome variable
        :param X: N*k nparray that conatins the covariates and the treatment
        :return: fitted model
        """
        if len(np.unique(Y)) != 2:
            ols = sm.OLS(Y, X)
            res = ols.fit()
        else:
            lg = sm.GLM(Y, X, family=sm.families.Binomial())
            res = lg.fit()
        return res

    def fit(self, method="ip_weight"):
        """
        Estimate the causal effect given the data
        ---------------------
        :param method: either ip_weight or standardization
        """
        X_const = np.ones(self.N).reshape(-1, 1)
        if method == "ip_weight":
            print('fitting causal model via inverse probability weighting')
            ip_weight = self.get_ip_weight(self.T, self.X)
            X_obs, X_zeros, X_ones = self.counterfactual_data_expansion(self.T, X_const, method)
            model_res = self.fit_ip(self.Y, X_obs, self.obs_cluster, ip_weight)
            if len(np.unique(self.Y)) != 2:
                results_df = model_res.summary2().tables[1]
                # the treatment is built after const, thus the coefficient of the treatment is the second one
                self.ate_ip = results_df.iloc[1, 0]
                self.ate_ip_ci = (results_df.iloc[1, 4], results_df.iloc[1, 5])
            else:
                ones_pred = model_res.predict(X_ones)
                zeros_pred = model_res.predict(X_zeros)

                self.ate_ip = ones_pred.mean() - zeros_pred.mean()
                bt_samples = self.bootstrap_sample(self.Y, self.T, X_const, self.obs_cluster, ip_weight, method)
                bt_std = np.std(bt_samples)
                self.ate_ip_ci = (self.ate_ip - 1.96 * bt_std, self.ate_ip + 1.96 * bt_std)

        elif method == "standardization":
            print('fitting causal model via standardization')
            X_std_base = np.concatenate((self.X, X_const), axis=1)
            X_obs, X_zeros, X_ones = self.counterfactual_data_expansion(self.T, X_std_base, method)
            model_res = self.fit_std(self.Y, X_obs)

            ones_pred = model_res.predict(X_ones)
            zeros_pred = model_res.predict(X_zeros)

            self.ate_std = ones_pred.mean() - zeros_pred.mean()
            bt_samples = self.bootstrap_sample(self.Y, self.T, X_std_base, self.obs_cluster, None, method)
            bt_std = np.std(bt_samples)
            self.ate_std_ci = (self.ate_std - 1.96 * bt_std, self.ate_std + 1.96 * bt_std)

        else:
            raise Exception("Incorrect method, please choose the method between \'ip_weight\' and \'standardization\'")
