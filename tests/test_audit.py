"""Tests write tiny safetensors files by hand (8-byte length, JSON header, raw bytes). No real model needed."""
import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model_diff_audit as m

NAMES = {np.dtype("float32"): "F32", np.dtype("float16"): "F16", np.dtype("uint16"): "BF16"}


def write(path, tensors):
    header, blob = {}, b""
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {"dtype": NAMES[arr.dtype], "shape": list(arr.shape), "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    head = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(head)) + head + blob)
    return path


def base_model(rng):
    t = {f"layers.{i}.weight": rng.standard_normal((64, 32)).astype(np.float32) for i in range(12)}
    t["embed.weight"] = rng.standard_normal((1000, 32)).astype(np.float32)
    return t


class Audit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rng = np.random.default_rng(1)
        self.base = base_model(self.rng)
        self.base_path = write(Path(self.tmp.name) / "base.safetensors", self.base)

    def tearDown(self):
        self.tmp.cleanup()

    def suspect(self, tensors):
        return m.audit(self.base_path, write(Path(self.tmp.name) / "suspect.safetensors", tensors))

    def test_identical(self):
        r = self.suspect(self.base)
        self.assertTrue(r["summary"].startswith("identical"))
        self.assertEqual(r["findings"], [])

    def test_ordinary_fine_tune_is_not_flagged(self):
        tuned = {k: (v + self.rng.standard_normal(v.shape).astype(np.float32) * 0.01) for k, v in self.base.items()}
        r = self.suspect(tuned)
        self.assertTrue(r["summary"].startswith("ordinary fine-tune"), r["summary"])
        self.assertEqual(r["findings"], [])

    def test_data_hidden_in_low_bits_is_flagged(self):
        t = dict(self.base)
        bits = t["layers.5.weight"].view(np.uint32).copy()
        secret = self.rng.integers(0, 2, bits.shape, dtype=np.uint32)
        t["layers.5.weight"] = ((bits & ~np.uint32(1)) | secret).view(np.float32)
        r = self.suspect(t)
        hit = [f for f in r["findings"] if f["tensor"] == "layers.5.weight"]
        self.assertTrue(hit and "low-bit edits" in hit[0]["finding"], r["findings"])

    def test_a_few_edited_embedding_rows_are_flagged_and_named(self):
        t = dict(self.base)
        e = t["embed.weight"].copy()
        e[[17, 903]] += 0.5
        t["embed.weight"] = e
        finding = [f for f in self.suspect(t)["findings"] if f["tensor"] == "embed.weight"][0]["finding"]
        self.assertIn("sparse row edits: 2 of 1,000 rows", finding)
        self.assertIn("17, 903", finding)

    def test_one_changed_layer_among_many_is_flagged(self):
        t = dict(self.base)
        t["layers.3.weight"] = t["layers.3.weight"] + self.rng.standard_normal((64, 32)).astype(np.float32)
        self.assertTrue(any("lone change" in f["finding"] for f in self.suspect(t)["findings"]))

    def test_missing_new_and_reshaped_tensors(self):
        t = dict(self.base)
        del t["layers.0.weight"]
        t["extra.weight"] = np.zeros((2, 2), dtype=np.float32)
        t["layers.1.weight"] = np.zeros((8, 8), dtype=np.float32)
        text = " | ".join(f["finding"] for f in self.suspect(t)["findings"])
        for expected in ("missing from the suspect", "new tensor", "dtype/shape differ"):
            self.assertIn(expected, text)

    def test_half_precision_and_sharded_folders(self):
        a, b = Path(self.tmp.name) / "A", Path(self.tmp.name) / "B"
        a.mkdir(), b.mkdir()
        half = {k: v.astype(np.float16) for k, v in self.base.items()}
        names = sorted(half)
        for folder, tensors in ((a, half), (b, dict(half))):
            write(folder / "model-00001-of-00002.safetensors", {k: tensors[k] for k in names[:6]})
            write(folder / "model-00002-of-00002.safetensors", {k: tensors[k] for k in names[6:]})
        r = m.audit(a, b)
        self.assertEqual(r["tensors_compared"], 13)
        self.assertTrue(r["summary"].startswith("identical"))

    def test_cli_exit_codes(self):
        t = dict(self.base)
        bits = t["layers.5.weight"].view(np.uint32).copy()
        t["layers.5.weight"] = (bits ^ np.uint32(1)).view(np.float32)
        bad = write(Path(self.tmp.name) / "bad.safetensors", t)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(m.main([str(self.base_path), str(self.base_path)]), 0)
            self.assertEqual(m.main([str(self.base_path), str(bad)]), 1)
            self.assertEqual(m.main([str(self.base_path), str(Path(self.tmp.name) / "nope.safetensors")]), 2)


if __name__ == "__main__":
    unittest.main()
