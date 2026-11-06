"""CPU-only regression checks for evidence admitted to reports."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evidence_audit import finite_numbers, wilson, pareto_dominated, recovery


class EvidenceTests(unittest.TestCase):
    def test_nan_cannot_be_published(self):
        self.assertEqual(finite_numbers({"stages": [{"projection": math.nan}]}), ["$.stages[0].projection"])

    def test_infinity_cannot_be_published(self):
        self.assertEqual(finite_numbers({"a": [1.0, math.inf]}), ["$.a[1]"])

    def test_counts_not_rounded_percentages(self):
        lo, hi = wilson(20, 40)
        self.assertLess(lo, .5)
        self.assertGreater(hi, .5)
        self.assertRaises(ValueError, wilson, 1, 0)

    def test_global_not_dominated_by_less_refusal(self):
        self.assertFalse(pareto_dominated((.5, .742), [(.2, .508)]))
        self.assertTrue(pareto_dominated((.5, .742), [(.2, .75)]))

    def test_gap_recovery_is_baseline_adjusted(self):
        self.assertAlmostEqual(recovery(.7416666667, .7583333333, .2166666667), .9692307693)
        self.assertRaises(ValueError, recovery, .5, .2, .2)


if __name__ == "__main__":
    unittest.main()
