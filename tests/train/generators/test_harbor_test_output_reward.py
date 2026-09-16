import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.train_integrations.harbor.reward_shaping import (
    read_bounded_verifier_output,
    shape_reward_from_output,
)


class TestHarborTestOutputReward(unittest.TestCase):
    def test_pytest_pass_ratio_replaces_binary_failure(self):
        output = "================ 2 failed, 3 passed in 0.12s ================\n"

        result = shape_reward_from_output(output, 0.0, parser_name="pytest")

        self.assertTrue(result.parsed)
        self.assertEqual((result.passed, result.total), (3, 5))
        self.assertAlmostEqual(result.reward, 0.6)
        self.assertTrue(result.is_partial_credit)

    def test_single_test_and_collection_errors_keep_original_reward(self):
        one_test = shape_reward_from_output(
            "================ 1 passed in 0.01s ================\n",
            0.0,
            parser_name="pytest",
        )
        collection_error = shape_reward_from_output(
            "ERROR collecting tests/test_x.py\nInterrupted: 1 error during collection\n",
            0.25,
            parser_name="pytest",
        )

        self.assertEqual(one_test.reward, 0.0)
        self.assertTrue(one_test.parsed)
        self.assertEqual(collection_error.reward, 0.25)
        self.assertFalse(collection_error.parsed)

    def test_unparseable_output_respects_fallback(self):
        fallback = shape_reward_from_output("compiler crashed", 0.4)
        no_fallback = shape_reward_from_output(
            "compiler crashed", 0.4, fallback_to_original=False
        )

        self.assertEqual(fallback.reward, 0.4)
        self.assertEqual(no_fallback.reward, 0.0)

    def test_bounded_reader_keeps_pytest_summary_at_tail(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "test-stdout.txt"
            path.write_bytes(
                b"x" * 100_000
                + b"\n================ 1 failed, 3 passed in 1.0s ================\n"
            )

            output = read_bounded_verifier_output(path, max_bytes=4096)
            result = shape_reward_from_output(output, 0.0, parser_name="pytest")

        self.assertIsNotNone(output)
        self.assertAlmostEqual(result.reward, 0.75)


if __name__ == "__main__":
    unittest.main()
