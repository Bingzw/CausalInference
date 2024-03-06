import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from core.root_cause_detector import RootCauseDetector
from utils.causal_utils import random_data


class TestRootCauseDetector(unittest.TestCase):
    def setUp(self):
        self.N = 10000
        self.K = 3
        self.Y, self.T, self.X, self.Y0, self.Y1, self.pscore = random_data(N=self.N, K=self.K, unobservables=True)
        self.df = pd.DataFrame({"AA": self.Y, "B":self.T, "C": self.Y+1, "CC": self.Y-1, "DD": self.T+1, "D": self.T-1})

    def test_condition_validation(self):
        # valid condition expression
        str_1 = "(AA and B) and C"
        str_2 = "AA and (B or C)"
        str_3 = "AA and B"
        str_4 = "(AA or B or C)"
        str_5 = "AA"
        str_6 = "(((AA and B) and CC) and DD)"
        str_7 = "(AA and (B and (CC and DD)))"
        str_8 = None
        # invalid condition expression
        str_9 = "()"
        str_10 = ""
        str_11 = "and AA"
        str_12 = "AA and B (CC and DD)"
        str_13 = ")("
        str_14 = "AA and B)"
        str_15 = "AA B"
        str_16 = "AA and or B"
        str_17 = "AA and (or B)"
        str_18 = "AA and"
        str_19 = "AA and ()"
        str_20 = "(AA and (B and CC (and DD)))"
        str_21 = "AA and (B and CC and DD))"

        valid_test_cases = [str_1, str_2, str_3, str_4, str_5, str_6, str_7, str_8]
        invalid_test_cases = [str_9, str_10, str_11, str_12, str_13, str_14, str_15, str_16, str_17, str_18, str_19, str_20, str_21]
        for case in valid_test_cases:
            test_model = RootCauseDetector(data=self.df, outcome="AA", treatment_variable="B",
                                           condition_expression=case)
            self.assertTrue(test_model.validate_condition())

        for case in invalid_test_cases:
            with self.assertRaises(Exception):
                test_model = RootCauseDetector(data=self.df, outcome="AA", treatment_variable="B",
                                               condition_expression=case)

    def test_condition_evaluation(self):
        test_df = self.df.copy()
        cond_expr1 = 'AA or B'
        cond_expr2 = '(AA and CC) or B'
        cond_expr3 = 'AA and (D and B) or C'
        cond_expr4 = 'AA and (B or CC) or (C and D)'

        test_thresholds = {'AA': 0, 'B': 0, 'C': 3, 'CC': 1, 'DD': 0, 'D': 0}
        # evaluate the true result given the test thresholds
        res_1_true = np.logical_or(test_df['AA'] <= test_thresholds['AA'], test_df['B'] <= test_thresholds['B'])

        res_2_true = np.logical_and(test_df['AA'] <= test_thresholds['AA'], test_df['CC'] <= test_thresholds['CC'])
        res_2_true = np.logical_or(res_2_true, test_df['B'] <= test_thresholds['B'])

        res_3_true = np.logical_and(test_df['D'] <= test_thresholds['D'], test_df['B'] <= test_thresholds['B'])
        res_3_true = np.logical_and(res_3_true, test_df['AA'] <= test_thresholds['AA'])
        res_3_true = np.logical_or(res_3_true, test_df['C'] <= test_thresholds['C'])

        res_4_1_true = np.logical_or(test_df['B'] <= test_thresholds['B'], test_df['CC'] <= test_thresholds['CC'])
        res_4_2_true = np.logical_and(test_df['C'] <= test_thresholds['C'], test_df['D'] <= test_thresholds['D'])
        res_4_true = np.logical_and(res_4_1_true, test_df['AA'] <= test_thresholds['AA'])
        res_4_true = np.logical_or(res_4_true, res_4_2_true)

        # evaluate the condition result through the root cause detector
        test_model_1 = RootCauseDetector(test_df, 'AA', 'B', cond_expr1)
        res_1 = test_model_1.evaluate_conditions(test_thresholds)
        assert_array_equal(res_1_true, res_1)

        test_model_2 = RootCauseDetector(test_df, 'AA', 'B', cond_expr2)
        res_2 = test_model_2.evaluate_conditions(test_thresholds)
        assert_array_equal(res_2_true, res_2)

        test_model_3 = RootCauseDetector(test_df, 'AA', 'B', cond_expr3)
        res_3 = test_model_3.evaluate_conditions(test_thresholds)
        assert_array_equal(res_3_true, res_3)

        test_model_4 = RootCauseDetector(test_df, 'AA', 'B', cond_expr4)
        res_4 = test_model_4.evaluate_conditions(test_thresholds)
        assert_array_equal(res_4_true, res_4)


if __name__ == '__main__':
    unittest.main()
