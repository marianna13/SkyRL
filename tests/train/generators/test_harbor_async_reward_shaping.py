import asyncio
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.train_integrations.harbor import reward_shaping


class AsyncRewardScoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocking_scorer_leaves_event_loop_responsive(self):
        started = threading.Event()
        release = threading.Event()
        result = reward_shaping.ProcessRewardResult(raw_reward=1.0)

        def score(*args):
            started.set()
            if not release.wait(2):
                raise AssertionError("Event loop did not release the scoring worker")
            return result

        async def release_from_event_loop():
            while not started.is_set():
                await asyncio.sleep(0.001)
            release.set()

        with patch.object(reward_shaping, "score_harbor_trajectory", side_effect=score):
            heartbeat = asyncio.create_task(release_from_event_loop())
            try:
                actual = await reward_shaping.score_harbor_trajectory_async(Path("unused"), 1.0, {})
                self.assertIs(actual, result)
                await heartbeat
            finally:
                release.set()
                heartbeat.cancel()

    async def test_cancellation_does_not_allow_overlapping_scorers(self):
        started = threading.Event()
        release = threading.Event()
        second_started = threading.Event()

        def score(path, *args):
            if path.name == "first":
                started.set()
                if not release.wait(2):
                    raise AssertionError("Scoring worker was not released")
            else:
                second_started.set()
            return reward_shaping.ProcessRewardResult(raw_reward=1.0)

        with patch.object(reward_shaping, "score_harbor_trajectory", side_effect=score):
            first = asyncio.create_task(
                reward_shaping.score_harbor_trajectory_async(Path("first"), 1.0, {})
            )
            second = None
            try:
                async with asyncio.timeout(1):
                    while not started.is_set():
                        await asyncio.sleep(0.001)
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                second = asyncio.create_task(
                    reward_shaping.score_harbor_trajectory_async(Path("second"), 1.0, {})
                )
                await asyncio.sleep(0.05)
                self.assertFalse(second_started.is_set())
            finally:
                release.set()
                if second is not None:
                    await second
                if not first.done():
                    await first
            self.assertTrue(second_started.is_set())


if __name__ == "__main__":
    unittest.main()
