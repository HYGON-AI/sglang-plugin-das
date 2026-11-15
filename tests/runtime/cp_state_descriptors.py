import types
import unittest
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase
from sglang.srt.disaggregation.base import conn
from hcu_sglang.disagg import utils
from hcu_sglang.disagg.protocol import StateType
from hcu_sglang.mem_cache.hcu_v4_pool import DeepSeekV4TokenToKVPool
from hcu_sglang.mem_cache.cp_cache_layer_split import handshake, transfer


class CPStateDescriptorTests(CustomTestCase):
    def args(self, layer_ids, sharded, hidden=True):
        mapping = object()
        def pool(draft):
            p = object.__new__(DeepSeekV4TokenToKVPool)
            p._unified_kv = False
            p.requires_descriptor_matched_transfer = sharded
            p.compression_ratios = [0, 0, 0] if draft else [1, 2]
            p.full_to_swa_index_mapping = mapping
            p.page_size = 256
            p.sliding_window = 512
            p.request_window = types.SimpleNamespace(capacity=512)
            p.swa_req_ring_size = 512
            ids = [0, 1, 2] if draft else layer_ids
            p.get_kv_transfer_layout = lambda: [] if draft else [('c1_kv', i) for i in layer_ids]
            p.get_state_transfer_layout = lambda: [('swa', i) for i in ids]
            p.get_request_state_transfer_layout = lambda: []
            p.get_state_buf_infos = lambda: ([100+i for i in ids], [1024]*len(ids), [64]*len(ids))
            p.get_request_state_buf_infos = lambda: ([], [], [])
            p.get_swa_valid_from_buf_infos = lambda: ([901], [1024], [4])
            return p
        p, d = pool(False), pool(True)
        a = types.SimpleNamespace(kv_data_ptrs=[20+i for i in layer_ids], kv_item_lens=[128]*len(layer_ids), kv_layer_ids=layer_ids)
        hp = types.SimpleNamespace(get_state_buf_infos=lambda: ([801], [1024*5120], [10240])) if hidden else None
        with patch.object(conn, 'StateType', StateType), patch.object(utils, 'is_npu', return_value=False):
            utils._SOURCE_SETUP_STATE_KV_ARGS(a, p, d, pd_hidden_pool=hp)
            transfer.configure_v4_transfer(a, p, d)
        return a

    def layout(self, args, peers=None):
        group = types.SimpleNamespace(all_gather_object=lambda local: [local] if peers is None else peers)
        spec = types.SimpleNamespace(speculative_algorithm='DSPARK', speculative_num_draft_tokens=6)
        with patch.object(handshake, 'get_spec', return_value=spec), patch.object(handshake, 'get_parallel', return_value=types.SimpleNamespace(attn_cp_group=group)), patch.object(conn, 'StateType', StateType):
            return handshake.get_dsv41_spec_layout(args)

    def local(self, args):
        m = args.v4_transfer_metadata
        return [list(zip(keys, sizes)) for keys, sizes in zip([m['kv'], *m['states']], [args.kv_item_lens, *args.state_item_lens])]

    def test_full_producer_and_global_handshake(self):
        p0, p1, d = self.args([0], True), self.args([1], True), self.args([0, 1], False)
        self.assertEqual(p0.state_types, [StateType.SWA, StateType.PD_HIDDEN, StateType.SWA, StateType.REQUEST_ROW])
        peers = [self.local(p0), self.local(p1)]
        self.assertEqual(self.layout(p0, peers), self.layout(d))
        self.assertEqual(p0.v4_transfer_metadata['states'][1], [('pd_hidden', 0)])
        self.assertEqual(p0.v4_transfer_metadata['states'][3], [('request_row', 0)])

    def test_original_incomplete_snapshot_rejected(self):
        p = self.args([0], True)
        p.v4_transfer_metadata['states'][1] = []
        p.v4_transfer_metadata['states'][3] = []
        with self.assertRaisesRegex(RuntimeError, 'Incomplete LayerSplit DSPARK'):
            self.layout(p)

    def test_replicated_state_abi_disagreement_rejected(self):
        p, other = self.args([0], True), self.args([1], True)
        other.state_item_lens[1][0] += 1
        with self.assertRaisesRegex(RuntimeError, 'page ABI differs across CP ranks'):
            self.layout(p, [self.local(p), self.local(other)])

    def test_without_hidden_still_has_request_row(self):
        p = self.args([0, 1], True, hidden=False)
        self.assertEqual(self.layout(p), self.layout(self.args([0, 1], False, hidden=False)))
        self.assertEqual(p.v4_transfer_metadata['states'][-1], [('request_row', 0)])

    def test_wire_roundtrip_and_pair_validation(self):
        p = self.args([0], True)
        wire = [b'']*23
        wire[22] = transfer.encode_v4_transfer_metadata(p)
        metadata = transfer.decode_v4_transfer_metadata(wire)
        for i in (1, 3):
            keys = metadata['states'][i]
            sizes = p.state_item_lens[i]
            self.assertEqual(transfer.match_transfer_entries(keys, keys, sizes, sizes), [(0, 0)])
            with self.assertRaisesRegex(RuntimeError, 'page ABI mismatch'):
                transfer.match_transfer_entries(keys, keys, sizes, [sizes[0]+1])
            with self.assertRaisesRegex(RuntimeError, 'missing LayerSplit'):
                transfer.match_transfer_entries(keys, [('wrong', 0)], sizes, sizes)


if __name__ == '__main__':
    unittest.main()
