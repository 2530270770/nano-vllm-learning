import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "chunkprefill" / "summarize_results.py"
SPEC = importlib.util.spec_from_file_location("summarize_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AggregationTests(unittest.TestCase):
    def test_nearest_rank_percentile(self):
        self.assertEqual(MODULE.percentile([1, 2, 3, 4, 5], 0.50), 3)
        self.assertEqual(MODULE.percentile([1, 2, 3, 4, 5], 0.95), 5)
        self.assertIsNone(MODULE.percentile([], 0.95))

    def test_stats(self):
        result = MODULE.stats([1, 2, 3])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mean"], 2)
        self.assertEqual(result["median"], 2)
        self.assertEqual(result["sd"], 1)


class ValidationTests(unittest.TestCase):
    def make_result(self):
        request = {
            "request_id": 0, "prompt_tokens": 4, "output_tokens": 2,
            "prefill_completion_step": 0, "prefill_total_allocated_tokens": 4,
            "prefill_chunks": [{
                "cached_tokens_before": 0, "allocated_tokens": 4,
                "cached_tokens_after": 4,
            }],
        }
        return {
            "scenario": "staggered",
            "config": {"max_num_decode_seqs": 8, "seed": 42, "num_requests": 1},
            "summary": {"prefill_fairness": {"request_count": 1},
                        "total_steps": 1, "total_output_tokens": 2},
            "requests": [request], "steps": [{}], "scheduler_steps": [{}],
            "model_runner_stats": {"decode_calls": 0, "decode_graph_calls": 0},
        }

    def test_valid_result(self):
        errors = MODULE.validate_result(Path("cap8_run1_seed42.json"),
                                        self.make_result(), (8, 1, 42))
        self.assertEqual(errors, [])

    def test_detect_chunk_cursor_mismatch(self):
        data = self.make_result()
        data["requests"][0]["prefill_chunks"][0]["cached_tokens_after"] = 3
        errors = MODULE.validate_result(Path("cap8_run1_seed42.json"), data, (8, 1, 42))
        self.assertTrue(any("chunk cursor mismatch" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
