import tempfile
import unittest
from pathlib import Path

from benchmarks.compare_reports import compare
from benchmarks.roofline_report import counters
from benchmarks.sweep_memory import probe
from test_inference import tiny


class MeasurementTests(unittest.TestCase):
    def test_comparison_refuses_different_workload(self):
        a = {"identity": {"workload": "a"}, "gpu": "test"}
        b = {"identity": {"workload": "b"}, "gpu": "test"}
        with self.assertRaises(ValueError):
            compare(a, b)

    def test_ncu_counter_units(self):
        text = '"ID","Metric Name","Metric Unit","Metric Value"\n"0","dram__bytes.sum","byte","1,000"\n"0","gpu__time_duration.sum","nsecond","2,000"\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            path.write_text(text)
            traffic, duration = counters(path)
        self.assertEqual(traffic, 1000)
        self.assertAlmostEqual(duration, .000002)

    def test_executed_capacity_and_waste(self):
        model = tiny()
        dense = probe(model, "contiguous", 2, 5, 16, 100000, 4, 3, 2)
        paged = probe(model, "paged", 2, 5, 16, 100000, 4, 3, 2)
        self.assertTrue(dense["fits"] and paged["fits"])
        self.assertAlmostEqual(dense["prefill_snapshot"]["waste_fraction"], 1 - 5 / 16)
        self.assertAlmostEqual(paged["prefill_snapshot"]["waste_fraction"], 1 - 5 / 8)
        shared = probe(model, "paged", 2, 5, 16, 100000, 4, 3, 2, shared_prefix=5)
        self.assertLess(shared["prefill_snapshot"]["assigned_bytes"], paged["prefill_snapshot"]["assigned_bytes"])
