import unittest
import numpy as np
from core.causality import CausalModel
from utils.causal_utils import random_data

class TestCausalModel(unittest.TestCase):
    """
    Test the CausalModel class
    """
    def setUp(self):
        self.N = 10000
        self.K = 3
        self.Y, self.T, self.X, self.Y0, self.Y1, self.pscore = random_data(N=self.N, K=self.K, unobservables=True)
        self.true_ate = np.mean(self.Y1 - self.Y0)
        self.cm = CausalModel(self.Y, self.T, self.X)

    def test_fit_ip_weight(self):
        self.cm.fit(method="ip_weight")
        self.assertAlmostEqual(self.cm.ate_ip, self.true_ate, delta=0.1)
        self.assertGreater(self.cm.ate_ip_ci[0], 0)

    def test_fit_std(self):
        self.cm.fit(method="standardization")
        self.assertAlmostEqual(self.cm.ate_std, self.true_ate, delta=0.1)
        self.assertGreater(self.cm.ate_std_ci[0], 0)

    def test_fit_invalid_method(self):
        with self.assertRaises(Exception):
            self.cm.fit(method="invalid_method")

if __name__ == '__main__':
    unittest.main()







