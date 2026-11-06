from __future__ import annotations

import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[2]


def load_definitions(relative_path, names, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError((relative_path, names))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class GlmVisualCpuTests(unittest.TestCase):
    def setUp(self):
        self.dp_encode = Mock()
        definitions = load_definitions(
            "hcu_sglang/models/glm5_next/visual.py",
            ["GlmVisualEncoderMixin"],
            {
                "torch": torch,
                "math": math,
                "List": List,
                "MultimodalDataItem": object,
                "get_int_env_var": lambda name, default: int(os.environ.get(name, default)),
                "run_dp_sharded_mrope_vision_model": self.dp_encode,
            },
        )
        self.model = definitions["GlmVisualEncoderMixin"]()
        self.model.use_data_parallel = False
        self.calls = []
        calls = self.calls

        class Visual:
            dtype = torch.float32

            def __call__(self, values, *, grid_thw):
                calls.append((values.clone(), grid_thw.clone()))
                return values * 2

        self.model.visual = Visual()
        self.environment = patch.dict(os.environ, {
            "SGLANG_VLM_MAX_PATCHES_PER_VIT": "0",
            "SGLANG_VLM_MAX_IMAGES_PER_VIT": "0",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.values = torch.arange(14, dtype=torch.float32).reshape(7, 2)
        self.grid = torch.tensor([[1, 1, 2], [1, 2, 2], [1, 1, 1]])

    def test_unbounded_visual_preserves_order_and_values(self):
        actual = self.model._run_visual_chunked(self.values, self.grid)
        torch.testing.assert_close(actual, self.values * 2)
        self.assertEqual(len(self.calls), 1)
        torch.testing.assert_close(self.calls[0][1], self.grid)

    def test_patch_limit_keeps_oversized_image_whole(self):
        os.environ["SGLANG_VLM_MAX_PATCHES_PER_VIT"] = "3"
        actual = self.model._run_visual_chunked(self.values, self.grid)
        self.assertEqual([len(values) for values, _ in self.calls], [2, 4, 1])
        torch.testing.assert_close(actual, self.values * 2)
        torch.testing.assert_close(torch.cat([grid for _, grid in self.calls]), self.grid)

    def test_image_limit_and_patch_count_validation(self):
        os.environ["SGLANG_VLM_MAX_IMAGES_PER_VIT"] = "2"
        actual = self.model._run_visual_chunked(self.values, self.grid)
        self.assertEqual([len(values) for values, _ in self.calls], [6, 1])
        torch.testing.assert_close(actual, self.values * 2)
        with self.assertRaises(AssertionError):
            self.model._run_visual_chunked(self.values[:-1], self.grid)

    def test_image_features_convert_dtype_and_concatenate_items(self):
        items = [
            SimpleNamespace(feature=self.values[:2].double(), image_grid_thw=self.grid[:1]),
            SimpleNamespace(feature=self.values[2:].double(), image_grid_thw=self.grid[1:]),
        ]
        actual = self.model.get_image_feature(items)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, self.values * 2)

    def test_video_grid_is_split_into_frames(self):
        values = self.values[:4]
        item = SimpleNamespace(feature=values, video_grid_thw=torch.tensor([[2, 2, 1]]))
        actual = self.model.get_video_feature([item])
        torch.testing.assert_close(actual, values * 2)
        torch.testing.assert_close(self.calls[0][1], torch.tensor([[1, 2, 1], [1, 2, 1]]))

    def test_missing_visual_encoder_reports_required_embeddings(self):
        self.model.visual = None
        for method in [self.model.get_image_feature, self.model.get_video_feature]:
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(RuntimeError, "disaggregated encoder"):
                    method([])

    def test_data_parallel_uses_rope_3d_encoder_boundary(self):
        self.model.use_data_parallel = True
        self.dp_encode.return_value = self.values
        result = self.model._run_visual_chunked(self.values, self.grid)
        self.assertIs(result, self.values)
        self.dp_encode.assert_called_once_with(
            self.model.visual, self.values, self.grid.tolist(), rope_type="rope_3d"
        )


class MuseSparkCpuTests(unittest.TestCase):
    def setUp(self):
        self.parallel = SimpleNamespace(attn_tp_size=1, attn_tp_rank=0)
        self.assignment = Mock(return_value=([2, 0, 1], [2, 1], None))
        self.definitions = load_definitions(
            "hcu_sglang/models/muse_spark/omni.py",
            ["_RMSNormWithoutScale", "_NormalizedEmbedding", "_run_dp_sharded_encode"],
            {
                "torch": torch,
                "nn": nn,
                "F": F,
                "itertools": __import__("itertools"),
                "get_parallel": lambda: self.parallel,
                "get_dp_encoder_lb_assignment": self.assignment,
                "_pad_mrope_vision_embeddings_for_tp_gather": lambda value, size: F.pad(
                    value, (0, 0, 0, size - value.shape[0])
                ),
            },
        )

    def test_rms_normalization_matches_explicit_cpu_reference(self):
        value = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
        layer = self.definitions["_RMSNormWithoutScale"](eps=0.25)
        actual = layer(value)
        expected = value / torch.sqrt(torch.mean(value.square(), dim=-1, keepdim=True) + 0.25)
        torch.testing.assert_close(actual, expected)

    def test_normalized_embedding_reuses_weight_without_duplicate_registration(self):
        embedding = nn.Embedding.from_pretrained(torch.tensor([[3.0, 4.0], [5.0, 12.0]]))
        norm = self.definitions["_RMSNormWithoutScale"]()
        layer = self.definitions["_NormalizedEmbedding"](embedding, norm)
        ids = torch.tensor([1, 0])
        actual = layer(ids)
        expected = embedding.weight[ids]
        expected = expected / torch.sqrt(expected.square().mean(-1, keepdim=True) + 1e-5)
        torch.testing.assert_close(actual, expected)
        self.assertIs(layer.weight, embedding.weight)
        self.assertEqual(layer.num_embeddings, 2)
        self.assertEqual(layer.state_dict(), {})

    def test_single_rank_encoding_keeps_feature_objects(self):
        features = [torch.tensor([[10.0]]), torch.tensor([[20.0], [21.0]])]
        encode = Mock(side_effect=lambda values: torch.cat(values))
        actual = self.definitions["_run_dp_sharded_encode"](
            features, [3, 2], [1, 2], encode, 1, torch.device("cpu"), torch.float32
        )
        torch.testing.assert_close(actual, torch.tensor([[10.0], [20.0], [21.0]]))
        self.assertIs(encode.call_args.args[0], features)
        self.assignment.assert_not_called()

    def test_simulated_gather_restores_original_item_order(self):
        self.parallel.attn_tp_size = 2
        features = [torch.tensor([[10.0]]), torch.tensor([[20.0], [21.0]]), torch.tensor([[30.0]])]
        gathered = torch.tensor([[30.0], [10.0], [20.0], [21.0]])
        gather = Mock(return_value=gathered)
        self.parallel.attn_tp_group = SimpleNamespace(all_gather=gather)
        encode = Mock(side_effect=lambda values: torch.cat(values))
        actual = self.definitions["_run_dp_sharded_encode"](
            features, [1, 2, 1], [1, 2, 1], encode, 1, torch.device("cpu"), torch.float32
        )
        torch.testing.assert_close(actual, torch.tensor([[10.0], [20.0], [21.0], [30.0]]))
        self.assertIs(encode.call_args.args[0][0], features[2])
        self.assertIs(encode.call_args.args[0][1], features[0])
        torch.testing.assert_close(gather.call_args.args[0], torch.tensor([[30.0], [10.0]]))
        self.assertEqual(gather.call_args.kwargs, {"dim": 0})


if __name__ == "__main__":
    unittest.main()
