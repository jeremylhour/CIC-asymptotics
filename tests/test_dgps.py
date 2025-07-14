import unittest
import numpy as np
from cic_asymptotics.dgps import LimitCaseDGP


# Unit tests
class TestLimitCaseDGP(unittest.TestCase):
    def test_valid_initialization(self):
        """Test that valid parameters do not raise exceptions."""
        dgp = LimitCaseDGP(d1=0.1, d2=0.1, b1=0.1, b2=0.1)
        self.assertIsInstance(dgp, LimitCaseDGP)
        self.assertTrue(hasattr(dgp, "name"))

    def test_invalid_sum_b1_b2(self):
        """Test that b1 + b2 >= 0.5 raises ValueError."""
        with self.assertRaises(ValueError):
            LimitCaseDGP(d1=0.1, d2=0.1, b1=0.3, b2=0.3)

    def test_invalid_sum_d1_d2(self):
        """Test that d1 + d2 >= 0.5 raises ValueError."""
        with self.assertRaises(ValueError):
            LimitCaseDGP(d1=0.3, d2=0.3, b1=0.1, b2=0.1)

    def test_invalid_max_sum(self):
        """Test that max(b1, b2) + max(d1, d2) >= 0.5 raises ValueError."""
        with self.assertRaises(ValueError):
            LimitCaseDGP(d1=0.25, d2=0.1, b1=0.3, b2=0.1)

    def test_generate_output_shapes(self):
        """Test that generate() returns arrays of correct shape."""
        dgp = LimitCaseDGP(n=500, d1=0.1, d2=0.1, b1=0.1, b2=0.1)
        y, z, x = dgp.generate()
        self.assertEqual(y.shape, (500,))
        self.assertEqual(z.shape, (500,))
        self.assertEqual(x.shape, (500,))

    def test_y_quantile_values(self):
        """Test y_quantile produces expected results on known input."""
        dgp = LimitCaseDGP(d1=0.1, d2=0.2, b1=0.1, b2=0.1)
        x = np.array([0.1, 0.5, 0.9])
        y = dgp.y_quantile(x)
        self.assertEqual(y.shape, (3,))
        self.assertTrue(np.all(np.isfinite(y)))


if __name__ == "__main__":
    unittest.main()
