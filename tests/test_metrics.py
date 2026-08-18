from __future__ import annotations

import unittest

from evaluation.metrics import compute_pass_metrics


class ComputePassMetricsTests(unittest.TestCase):
    def test_all_fail(self) -> None:
        result = compute_pass_metrics([[False] * 4, [0] * 4], k=2)

        self.assertEqual(result.pass_power_k_per_task, (0.0, 0.0))
        self.assertEqual(result.pass_at_k_per_task, (0.0, 0.0))
        self.assertEqual(result.pass_power_k, 0.0)
        self.assertEqual(result.pass_at_k, 0.0)

    def test_all_pass(self) -> None:
        result = compute_pass_metrics([[True] * 4, [1] * 4], k=3)

        self.assertEqual(result.pass_power_k_per_task, (1.0, 1.0))
        self.assertEqual(result.pass_at_k_per_task, (1.0, 1.0))
        self.assertEqual(result.pass_power_k, 1.0)
        self.assertEqual(result.pass_at_k, 1.0)

    def test_mixed_known_combinatorial_fixture(self) -> None:
        result = compute_pass_metrics(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
            ],
            k=2,
        )

        expected_power = (0.0, 1.0 / 6.0, 1.0 / 2.0)
        expected_at = (1.0 / 2.0, 5.0 / 6.0, 1.0)
        for actual, expected in zip(
            result.pass_power_k_per_task, expected_power, strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            result.pass_at_k_per_task, expected_at, strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result.pass_power_k, 2.0 / 9.0)
        self.assertAlmostEqual(result.pass_at_k, 7.0 / 9.0)

    def test_pass_power_one_equals_pass_at_one(self) -> None:
        result = compute_pass_metrics(
            [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
            k=1,
        )

        self.assertEqual(result.pass_power_k_per_task, result.pass_at_k_per_task)
        self.assertEqual(result.pass_power_k, result.pass_at_k)

    def test_invalid_shapes_and_values(self) -> None:
        invalid_arrays = (
            [],
            [[]],
            [[1, 0], [1]],
            [[1, 2]],
            [[1, -1]],
            [[1, 0.0]],
            [[1, "0"]],
        )
        for invalid in invalid_arrays:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    compute_pass_metrics(invalid, k=1)

    def test_invalid_k(self) -> None:
        for invalid_k in (0, 3, True, 1.0):
            with self.subTest(k=invalid_k):
                with self.assertRaises(ValueError):
                    compute_pass_metrics([[1, 0]], k=invalid_k)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
