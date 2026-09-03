"""CPU-only reference-schema checks. These fixtures are not GPU evidence."""

import copy
import unittest

from benchmarks.fullscan_volume_iteration import require_reference_metadata


class ReferenceMetadataTests(unittest.TestCase):
    def setUp(self):
        self.input = {
            "metadata_sha256": "input-hash",
            "metadata": {"geometry": [{"source": [0, -600, 0]}]},
            "selected_view_indices": list(range(360)),
            "selected_geometry": [{"source": [0, -600, 0]}],
            "cotangent_subset_renormalised": False,
            "cotangent": {"sha256": "seed-file-hash", "raw_sha256": "seed-values-hash"},
        }
        self.result = {
            "status": "complete",
            "completed_at": "2026-10-19T00:00:00+00:00",
            "source_unchanged": True,
            "source": {"sha256": "source-hash", "files": {"python/dpt/example.py": "file-hash"}},
            "source_after": {
                "sha256": "source-hash",
                "files": {"python/dpt/example.py": "file-hash"},
            },
            "precision": "fp32",
            "input": copy.deepcopy(self.input),
            "measurements": [{"configuration": {"samples_per_ray": 256}}],
            "checks": {"before_timing": {"volume_adjoint": {"passed": True}}},
        }
        self.outputs = {
            "source_sha256": "source-hash",
            "input_metadata_sha256": "input-hash",
            "cotangent_sha256": "seed-file-hash",
            "precision": "fp32",
            "views": 360,
            "samples_per_ray": 256,
        }

    def validate(self, samples=256):
        require_reference_metadata(self.result, self.outputs, self.input, "fp32", samples)

    def test_same_sample_reference_is_accepted(self):
        self.validate()

    def test_1024_reference_is_not_a_256_accuracy_gate(self):
        self.result["measurements"][0]["configuration"]["samples_per_ray"] = 1024
        self.outputs["samples_per_ray"] = 1024
        with self.assertRaisesRegex(ValueError, "sample count"):
            self.validate()
        self.validate(samples=1024)

    def test_legacy_fullscan_sample_count_is_read_from_measurements(self):
        del self.outputs["samples_per_ray"]
        self.validate()

    def test_incomplete_and_changed_source_fail(self):
        for key, value in (("status", "failed"), ("source_unchanged", False)):
            with self.subTest(key=key):
                original = self.result[key]
                self.result[key] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.result[key] = original

    def test_changed_output_manifest_source_fails(self):
        self.outputs["source_sha256"] = "different"
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_source_identity_fails(self):
        del self.result["source"]
        del self.result["source_after"]
        del self.outputs["source_sha256"]
        with self.assertRaisesRegex(ValueError, "source hash and source file map"):
            self.validate()

    def test_empty_source_file_map_fails(self):
        self.result["source"]["files"] = {}
        self.result["source_after"]["files"] = {}
        with self.assertRaisesRegex(ValueError, "source hash and source file map"):
            self.validate()

    def test_changed_seed_hash_fails(self):
        self.outputs["cotangent_sha256"] = "different"
        with self.assertRaises(ValueError):
            self.validate()

    def test_changed_seed_values_fail(self):
        self.result["input"]["cotangent"]["raw_sha256"] = "different"
        with self.assertRaises(ValueError):
            self.validate()

    def test_geometry_and_view_selection_changes_fail(self):
        for key, value in (
            ("selected_geometry", [{"source": [0, -601, 0]}]),
            ("selected_view_indices", list(range(359))),
            ("cotangent_subset_renormalised", True),
        ):
            with self.subTest(key=key):
                original = self.result["input"][key]
                self.result["input"][key] = value
                with self.assertRaises(ValueError):
                    self.validate()
                self.result["input"][key] = original

    def test_wrong_precision_fails(self):
        self.result["precision"] = "fp64"
        with self.assertRaises(ValueError):
            self.validate()

    def test_failed_numerical_check_fails(self):
        self.result["checks"]["before_timing"]["images"] = {"passed": False}
        with self.assertRaisesRegex(ValueError, "failed numerical"):
            self.validate()

    def test_missing_adjoint_fails(self):
        self.result["checks"] = {"before_timing": {"passed": True}}
        with self.assertRaisesRegex(ValueError, "no passed volume-adjoint"):
            self.validate()

    def test_empty_measurement_set_fails(self):
        self.result["measurements"] = []
        with self.assertRaisesRegex(ValueError, "sample count"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
