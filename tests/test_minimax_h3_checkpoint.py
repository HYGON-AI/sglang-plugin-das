import ast
import json
import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]

class H3CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "hcu_sglang/multimodal_gen/minimax_h3_quant.py"
        tree = ast.parse(path.read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("backbone_layers", "inspect_int8_checkpoint")]
        namespace = dict(json=json, Path=Path, FORMAT="sglang-minimax-h3-int8-v1", MANIFEST="minimax_h3_int8.json")
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
        cls.inspect = staticmethod(namespace["inspect_int8_checkpoint"])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "model.safetensors"
        self.layers = [f"blocks.{i}.{name}" for i in range(50) for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")]
        self.tensors = {}
        for name in self.layers:
            self.tensors[name + ".weight"] = torch.ones(16, 16, dtype=torch.int8)
            self.tensors[name + ".weight_scale_inv"] = torch.ones(16, 1, dtype=torch.float32)
        self.manifest = dict(format="sglang-minimax-h3-int8-v1", status="complete", profile="backbone",
            layout="native-qkv", variant="fl2va", quantized_layers=self.layers, shards=[self.path.name])

    def save(self):
        save_file(self.tensors, str(self.path), metadata={"format": self.manifest["format"]})
        (self.root / "minimax_h3_int8.json").write_text(json.dumps(self.manifest))

    def test_valid_full_adaln_metadata(self):
        self.save()
        self.assertEqual(self.inspect([str(self.path)]), self.manifest)

    def test_valid_compressed_adaln_metadata(self):
        self.manifest["profile"] = "backbone-adaln"
        self.tensors["adaln_t_table"] = torch.zeros(1025, 8, dtype=torch.float32)
        for prefix in [f"blocks.{i}" for i in range(50)] + ["final_layer"]:
            self.tensors[prefix + ".adaln_proj.linear.weight"] = torch.zeros(16, 8, dtype=torch.float32)
        self.save()
        self.assertEqual(self.inspect([str(self.path)])["profile"], "backbone-adaln")
        self.tensors["adaln_t_table"] = torch.zeros(1025, 8, dtype=torch.bfloat16)
        self.save()
        with self.assertRaisesRegex(ValueError, "FP32"):
            self.inspect([str(self.path)])

    def test_missing_backbone_layer_is_rejected(self):
        del self.tensors[self.layers[-1] + ".weight"]
        self.save()
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.inspect([str(self.path)])

    def test_wrong_scale_layout_is_rejected(self):
        self.tensors[self.layers[0] + ".weight_scale_inv"] = torch.ones(1, 16)
        self.save()
        with self.assertRaisesRegex(ValueError, "per-channel weight scale"):
            self.inspect([str(self.path)])

    def test_quantized_checkpoint_without_manifest_is_rejected(self):
        self.save()
        (self.root / "minimax_h3_int8.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing minimax_h3_int8.json"):
            self.inspect([str(self.path)])

if __name__ == "__main__":
    unittest.main()
