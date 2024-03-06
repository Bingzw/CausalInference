import numpy as np
import pandas as pd
from core.causality import CausalModel
from utils.causal_utils import is_match_parenthesis


class RootCauseDetector(object):
    """
    class that conducts root cause detection given cleaned data
    """

    def __init__(self, data, outcome, treatment_variable, condition_expression=None, **kwargs):
        """
        Initialize the root cause detection model
        -----------------------------------------
        data: pandas dataframe, the data to be analyzed
        outcome: string, the column name of the outcome to be analyzed
        treatment_variable: column name of the treatment variable
        condition_expression: a string of the column name and their combination logic.
                              For example, "(A or B) and (C or D)"
        """
        self.data = data
        self.N, self.K = self.data.shape
        self.outcome = outcome
        self.treatment_variable = treatment_variable
        self.condition = condition_expression
        self.cond_variables = None
        self.causal_effects = None
        self.causal_thresholds = None
        self.condition_const = set(["(", ")", "and", "or"])
        self.interact_index = kwargs.get('interact_index', None)
        self.obs_cluster = kwargs.get('obs_cluster', None)
        self.N_bt = kwargs.get('N_bt', 30)
        self.initialization_check()

    def initialization_check(self):
        """
        check if the input data is valid
        """
        if self.data is None:
            raise Exception("The input data is None")
        elif self.data.shape[0] == 0:
            raise Exception("The input data is empty")
        elif self.outcome is None:
            raise Exception("The outcome variable is None")
        elif self.outcome not in set(self.data.columns):
            raise Exception("The outcome variable is not in the data")
        elif self.treatment_variable is None:
            raise Exception("The treatment variable is None")
        elif self.treatment_variable not in set(self.data.columns):
            raise Exception("The treatment variable is not in the data")
        elif self.condition is not None:
            if not self.validate_condition():
                print("The condition expression is not valid")
                raise Exception("The condition expression is not valid")
        print("Input data is valid to go")

    def split_condition(self):
        """
        split condition string to lists
        -------------------------------
        :return: list of decomposed string ["(", "impression", "and", "click", ")", "and", "cov"]
        """
        str_list = []
        i = 0
        n = len(self.condition)
        while i < n:
            if self.condition[i] == "(" or self.condition[i] == ")":
                str_list.append(self.condition[i])
                i += 1
            elif self.condition[i] == " ":
                i += 1
            else:
                j = i
                while j < n and self.condition[j] != " " and self.condition[j] != "(" and self.condition[j] != ")":
                    j += 1
                str_list.append(self.condition[i:j])
                i = j
        return str_list

    def validate_variables(self):
        """
        extract the variables from the condition expression, check if they exist in data
        --------------------------------------------------------------------------------
        :return: Boolean, True if the variables in the condition_expression exist in the data, False otherwise
        """
        str_list = self.split_condition()
        variables = []
        for str in str_list:
            if str not in self.condition_const:
                variables.append(str)
        for var in variables:
            if var not in set(self.data.columns):
                return False
        self.cond_variables = variables
        return True

    def validate_condition(self):
        """
        given a sql where condition expression, check if the expression is valid, no need to evaluate the final result
        --------------------------------------------------------------------------------------------------------------
        :return: Boolean, True if the condition_expression is valid, False otherwise
        """
        if self.condition is None:
            return True
        str_list = self.split_condition()
        if len(str_list) == 0:
            print("The condition expression is empty, set condition_expression as None instead")
            return False
        is_matched_parenthesis = is_match_parenthesis(str_list)
        is_validated_variables = self.validate_variables()
        if not is_matched_parenthesis:
            print("The condition expression contains unmatched parenthesis")
            return False
        if not is_validated_variables:
            print("The condition expression contains invalid variables")
            return False
        n = len(str_list)
        for i in range(n):
            cur_str = str_list[i]
            if cur_str == "(":
                if i > 0 and (str_list[i - 1] == ")" or str_list[i - 1] not in set(["and", "or", "(", ")"])):
                    return False
            elif cur_str == ")":
                if i > 0 and str_list[i - 1] in set(["and", "or", "("]):
                    return False
            elif cur_str == "and" or cur_str == "or":
                if i == 0 or i == n - 1 or str_list[i - 1] in set(["and", "or", "("]):
                    return False
            else:
                if i > 0 and str_list[i - 1] not in set(["and", "or", "("]):
                    return False
        return True

    def logical_eval(self, array_a, array_b, logical_op):
        """
        perform logical operation for two arrays of boolean values
        ----------------------------------------------------------
        :param array_a: a nparray of boolean values
        :param array_b: a nparray of boolean values
        :param logical_op: a string of "and" or "or"
        :return: a nparray of boolean values
        """
        if logical_op == "and":
            return np.logical_and(array_a, array_b)
        elif logical_op == "or":
            return np.logical_or(array_a, array_b)
        else:
            raise Exception("Incorrect logical operation, please choose between \'and\' and \'or\'")

    def evaluate_conditions(self, thresholds):
        """
        evaluate the condition expression given the threshold
        -----------------------------------------------------
        :param thresholds: a dictionary of variable name and their thresholds
        :return: a nparray of boolean values indicating whether the condition is satisfied
        """
        N = self.data.shape[0]
        res = np.ones(N, dtype=bool)
        if self.condition is None:
            return res
        logical_ops = 'and'
        str_list = self.split_condition()
        n = len(str_list)
        stack = []
        for i in range(n):
            cur_str = str_list[i]
            if cur_str not in self.condition_const:
                cur_cond_val = self.data[cur_str] <= thresholds[cur_str]
                res = self.logical_eval(res, cur_cond_val, logical_ops)
            elif cur_str == "and":
                logical_ops = "and"
            elif cur_str == "or":
                logical_ops = "or"
            elif cur_str == "(":
                stack.append((res, logical_ops))
                res = np.ones(N, dtype=bool)
                logical_ops = "and"
            elif cur_str == ")":
                prev_res, prev_logical_ops = stack.pop()
                res = self.logical_eval(prev_res, res, prev_logical_ops)
            else:
                continue
        return res

    def eval_treatment(self, thresholds):
        """
        evaluate the treatment assignments given the threshold
         -----------------------------------------------------
        :param thresholds: a dictionary of variable name and their thresholds
        :return: a nparray 1 and 0, indicating the treatment and control assignments
        """
        return (self.data[self.treatment_variable] <= thresholds[self.treatment_variable]).astype('int').values

    def grid_search_optimization(self, k_cond=5, k_treat=5):
        """
        apply grid search optimization to find the optimal threshold given the condition expression and the treatment
        variable
        ----------------------------------------------------------------------------------------------
        :k_cond: the number of intervals (including low/high ends) to split the condition search space, default to 5
        :k_treat: the number of intervals (including low/high ends) to split the treatment search space, default to 5
        :return: a dict of optimal parameters and a data frame of the optimal threshold and the corresponding causal
                 effects
        """
        if k_cond is not None and k_cond <= 2:
            raise Exception("k_cond should be greater than 2")
        if k_treat is not None and k_treat <= 2:
            raise Exception("k_treat should be greater than 2")

        search_space_dict = {}
        if self.condition is not None:
            str_list = self.split_condition()
            cond_percentiles = np.linspace(0, 100, k_cond + 1)[1:-1]
            for cur_str in str_list:
                if cur_str not in self.condition_const:
                    cur_perc_values = np.percentile(self.data[cur_str], cond_percentiles)
                    search_space_dict[cur_str] = np.unique(cur_perc_values)

        treatment_percentiles = np.linspace(0, 100, k_treat + 1)[1:-1]
        treatment_perc_values = np.percentile(self.data[self.treatment_variable], treatment_percentiles)
        search_space_dict[self.treatment_variable] = np.unique(treatment_perc_values)
        search_values = [v for k, v in search_space_dict.items()]
        search_space_arr = np.array(np.meshgrid(*search_values)).T.reshape(-1, len(search_values))
        search_space_df = pd.DataFrame(search_space_arr, columns=search_space_dict.keys())
        thresholds = search_space_df.to_dict(orient="records")
        global_Y = self.data[self.outcome].values
        for cur_thresholds in thresholds:
            print("---------------------")
            print("current thresholds are: " + str(cur_thresholds))
            cur_condition = self.evaluate_conditions(cur_thresholds)
            cur_T = self.eval_treatment(cur_thresholds)
            cur_T = cur_T[cur_condition]
            cur_data = self.data[cur_condition]
            cur_Y = cur_data[self.outcome].values
            if self.cond_variables is not None:
                to_drop_variables = [self.outcome, self.treatment_variable] + self.cond_variables
            else:
                to_drop_variables = [self.outcome, self.treatment_variable]
            cur_X = cur_data.drop(to_drop_variables, axis=1).values
            cur_global_recall = None
            cur_cond_recall = None
            cur_precision = None
            cur_ate_ip = None
            cur_ate_ip_ci = None
            cur_ate_std = None
            cur_ate_std_ci = None

            if len(np.unique(global_Y)) == 2:
                n_tp = np.sum(cur_T & cur_Y)
                n_t = np.sum(global_Y)
                n_p = np.sum(cur_T)
                n_cond_t = np.sum(cur_Y)

                cur_global_recall = n_tp * 1.0 / n_t
                cur_cond_recall = n_tp * 1.0 / n_cond_t
                cur_precision = n_tp * 1.0 / n_p
            try:
                cur_causal_model = CausalModel(cur_Y, cur_T, cur_X, self.interact_index, self.obs_cluster, self.N_bt)
                cur_causal_model.fit(method="ip_weight")
                cur_ate_ip = cur_causal_model.ate_ip
                cur_ate_ip_ci = cur_causal_model.ate_ip_ci
                cur_causal_model.fit(method="standardization")
                cur_ate_std = cur_causal_model.ate_std
                cur_ate_std_ci = cur_causal_model.ate_std_ci
            except ValueError as error:
                print(f"An error occurred: {error}")
            except:
                print(f"Could not fit causal inference model with this threshold: {cur_thresholds}")
            cur_thresholds["ip_weight_ate"] = cur_ate_ip
            cur_thresholds["ip_weight_ate_ci"] = cur_ate_ip_ci
            cur_thresholds["standardization_ate"] = cur_ate_std
            cur_thresholds["standardization_ate_ci"] = cur_ate_std_ci
            cur_thresholds["recall"] = cur_global_recall
            cur_thresholds["precision"] = cur_precision
            cur_thresholds["cond_coverage"] = cur_cond_recall

        res = pd.DataFrame.from_records(thresholds)
        df_res_all = res.copy()
        df_res_all.sort_values(by=["ip_weight_ate", "standardization_ate", "recall", "precision", "cond_coverage"],
                               ascending=False, inplace=True)
        # extract the optimal value
        df_res_optimal =res.copy()
        df_res_optimal = df_res_optimal.loc[(df_res_optimal['ip_weight_ate'] > 0) |
                                            (df_res_optimal['standardization_ate'] > 0)]
        process_stat_sig = np.vectorize(lambda ip_ci, std_ci: (ip_ci[0] > 0) | (std_ci[0] > 0))
        df_res_optimal['is_stat_sig'] = process_stat_sig(df_res_optimal['ip_weight_ate_ci'],
                                                         df_res_optimal['standardization_ate_ci'])
        df_res_optimal = df_res_optimal.loc[df_res_optimal['is_stat_sig']]
        if len(np.unique(global_Y)) == 2:
            df_res_optimal.sort_values(by=['recall', 'ip_weight_ate', 'standardization_ate'], ascending=False,
                                       inplace=True)
        else:
            df_res_optimal.sort_values(by=['ip_weight_ate', 'standardization_ate'], ascending=False,
                                       inplace=True)

        optimal_res_dict = df_res_optimal.to_dict('records')[0]
        return optimal_res_dict, df_res_all
