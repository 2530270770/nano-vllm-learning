import tempfile
import unittest
from pathlib import Path

from benchmarks.chunkprefill.run_capacity_sweep import (
    build_command,
    parse_args,
    parse_int_list,
)


class CapacitySweepTests(unittest.TestCase):
    def test_parse_int_list_requires_increasing_unique_values(self):
        self.assertEqual(parse_int_list("32, 40,48"), [32, 40, 48])
        with self.assertRaises(Exception):
            parse_int_list("32,32")
        with self.assertRaises(Exception):
            parse_int_list("48,32")

    def test_defaults_are_conservative_and_staggered(self):
        args = parse_args([])
        self.assertEqual(args.request_counts, [32, 40, 48])
        self.assertEqual(args.repeats, 1)
        self.assertEqual(args.decode_cap, 8)
        self.assertEqual(args.prefill_cap, 8)

    def test_command_preserves_gpu_benchmark_contract(self):
        args = parse_args(["--request-counts", "32,40"])
        command = build_command(
            args, 40, 1, 42, Path("out/requests40_run1_seed42.json")
        )
        self.assertIn("--scenario", command)
        self.assertEqual(command[command.index("--scenario") + 1], "staggered")
        self.assertEqual(command[command.index("--num-requests") + 1], "40")
        self.assertEqual(command[command.index("--max-num-decode-seqs") + 1], "8")
        self.assertEqual(command[command.index("--execution-mode") + 1], "cudagraph")

    def test_dry_run_writes_manifest_without_launching_gpu(self):
        from benchmarks.chunkprefill.run_capacity_sweep import main

        with tempfile.TemporaryDirectory() as tmp:
            result = main([
                "--request-counts", "32,40",
                "--output-dir", tmp,
                "--dry-run",
            ])
            self.assertEqual(result, 0)
            manifest = Path(tmp, "capacity_sweep_manifest.json")
            self.assertTrue(manifest.exists())
            import json
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(data["runs"]), 2)
            self.assertTrue(all(item["status"] == "dry_run" for item in data["runs"]))


if __name__ == "__main__":
    unittest.main()
