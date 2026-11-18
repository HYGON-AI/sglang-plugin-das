"""
Grouped-head split-KV attention for speculative *verify* (topk==1).
Following the pattern of ``python/sglang/kernels/ops/attention/verify_splitkv.py``.

Grid is ``(bs, n_head_blocks, split)``; each program handles ``BLOCK_H`` query
heads x ALL ``L_EXT`` draft queries. It supports absorbed MLA and ordinary
MHA/GQA when exactly one TP-local KV head is shared by all local query heads.

Correctness matches ``extend_attention_fwd`` for the topk==1 causal verify case.

score(h,i,t) = q_nope[i,h] · c_KV[t]  +  q_pe[i,h] · k_pe[t]      # 512 dot + 64 dot
out(h,i)     = Σ_t softmax_t · c_KV[t]                            # V = c_KV
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.verify_splitkv import _AMD_LAUNCH_KWARGS

MAX_N_SPLITS = 32
DEFAULT_BLOCK_H = 16
DEFAULT_BLOCK_N = 64
DEFAULT_NUM_WARPS = 4


@triton.jit
def _active_splits(seqlen, num_splits, BLOCK_N: tl.constexpr):
    """
    The launched split count, capped by ``seqlen // BLOCK_N``.
    Floor partitioning keeps every active split non-empty.
    """
    return tl.maximum(1, tl.minimum(num_splits, seqlen // BLOCK_N))


@triton.jit
def _verify_mla_prefix_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    k_scale,
    v_scale,
    qo_indptr,
    kv_indptr,
    kv_indices,
    Att_Out,  # [BS, H_Q, MAX_N_SPLITS, L_EXT, Dv] bf16
    Att_Lse,  # [BS, H_Q, MAX_N_SPLITS, L_EXT]     fp32
    num_splits,  # launched split count
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
    stride_buf_kh,
    stride_buf_vh,
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    stride_buf_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    H_Q: tl.constexpr,
    L_EXT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    HAS_KV_HEADS: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    PE_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DNOPE: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGED_V2: tl.constexpr = False,
    PRESERVE_Q_PRECISION: tl.constexpr = False,
    SLIDING_WINDOW_SIZE: tl.constexpr = -1,
):
    cur_batch = tl.program_id(0)
    head_block = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    # FP8 cache storage must not quantize BF16 Q or softmax probabilities.
    COMPUTE_DTYPE: tl.constexpr = (
        Q.dtype.element_ty if PRESERVE_Q_PRECISION else K_Buffer.dtype.element_ty
    )

    # row tile size for each workgroup
    R: tl.constexpr = BLOCK_H * L_EXT

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    active = _active_splits(cur_batch_seq_len, num_splits, BLOCK_N)

    # skip idle workgroups
    if split_kv_id < active:
        head_start = head_block * BLOCK_H
        offs_h = head_start + tl.arange(0, BLOCK_H)
        # The caller guarantees KV_GROUP_NUM % BLOCK_H == 0, so every query head
        # in this block maps to the same KV head and the K/V tiles stay 2D loads.
        # MLA has a single latent head; keeping the offset out of that path
        # entirely preserves the original address arithmetic in the inner loop.
        if HAS_KV_HEADS:
            kv_head_off_k = (head_start // KV_GROUP_NUM) * stride_buf_kh
            kv_head_off_v = (head_start // KV_GROUP_NUM) * stride_buf_vh
        else:
            kv_head_off_k = 0
            kv_head_off_v = 0
        offs_l = tl.arange(0, L_EXT)
        offs_dn = tl.arange(0, BLOCK_DNOPE)
        offs_dp = tl.arange(0, BLOCK_DPE)
        offs_dv = tl.arange(0, BLOCK_DV)

        cur_q_start = tl.load(qo_indptr + cur_batch)
        l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
        row_mask = tl.reshape((offs_h[:, None] < H_Q) & (offs_l[None, :] < l_ext), (R,))

        # the last kv split completes the remaining
        if PAGED_V2:
            # Keep each partition on a BLOCK_N (and thus page64) boundary.
            # active <= floor(seq_len / BLOCK_N), except the single short/empty
            # partition, so every non-empty prefix still has non-empty splits.
            kv_len_per_split = (cur_batch_seq_len // BLOCK_N // active) * BLOCK_N
        else:
            kv_len_per_split = cur_batch_seq_len // active
        split_start = kv_len_per_split * split_kv_id
        split_end = tl.where(
            split_kv_id == active - 1, cur_batch_seq_len, split_start + kv_len_per_split
        )

        # For absorbed MLA, NOPE_DIM is the latent width and PE_DIM is the
        # appended RoPE width. For ordinary shared-KV attention (Qwen3.5),
        # NOPE_DIM is the complete already-rotated Q/K head and PE_DIM is zero.
        q_row = tl.reshape(
            (cur_q_start + offs_l)[None, :] * stride_qbs + offs_h[:, None] * stride_qh,
            (R,),
        )
        q_nope = tl.load(
            Q + q_row[:, None] + offs_dn[None, :],
            mask=row_mask[:, None] & (offs_dn[None, :] < NOPE_DIM),
            other=0.0,
        ).to(COMPUTE_DTYPE)
        if PE_DIM > 0:
            q_pe = tl.load(
                Q + q_row[:, None] + (NOPE_DIM + offs_dp)[None, :],
                mask=row_mask[:, None] & (offs_dp[None, :] < PE_DIM),
                other=0.0,
            ).to(COMPUTE_DTYPE)

        e_max = tl.zeros([R], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([R], dtype=tl.float32)
        acc = tl.zeros([R, BLOCK_DV], dtype=tl.float32)

        # calculate attention scores for each kv split
        for start_n in tl.range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < split_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n, mask=n_mask, other=0
            )
            # Token IDs address pages, not whole-page strides.
            if PAGE_SIZE == 1:
                base = kv_loc[None, :] * stride_buf_kbs + kv_head_off_k
                v_base = kv_loc[:, None] * stride_buf_vbs + kv_head_off_v
            else:
                page_id = kv_loc // PAGE_SIZE
                token_offset = kv_loc % PAGE_SIZE
                base = (
                    page_id[None, :] * stride_buf_kpage
                    + token_offset[None, :] * stride_buf_ktok
                    + kv_head_off_k
                )
                v_base = (
                    page_id[:, None] * stride_buf_vpage
                    + token_offset[:, None] * stride_buf_vtok
                    + kv_head_off_v
                )
            k_nope = tl.load(
                K_Buffer + base + offs_dn[:, None],
                mask=(offs_dn[:, None] < NOPE_DIM) & n_mask[None, :],
                other=0.0,
            )
            k_nope = k_nope.to(COMPUTE_DTYPE)
            qk = tl.dot(q_nope, k_nope)
            if PE_DIM > 0:
                k_pe = tl.load(
                    K_Buffer + base + (NOPE_DIM + offs_dp)[:, None],
                    mask=(offs_dp[:, None] < PE_DIM) & n_mask[None, :],
                    other=0.0,
                )
                qk += tl.dot(q_pe, k_pe.to(COMPUTE_DTYPE))
            qk *= sm_scale * k_scale
            score_mask = n_mask[None, :]
            if SLIDING_WINDOW_SIZE > 0:
                # kv_indptr describes the clipped prefix, exactly as in extend.
                # Query i sees prefix key n iff prefix_len + i <= n + window.
                row_l = tl.reshape(
                    tl.broadcast_to(offs_l[None, :], (BLOCK_H, L_EXT)), (R,)
                )
                score_mask = score_mask & (
                    cur_batch_seq_len + row_l[:, None]
                    <= offs_n[None, :] + SLIDING_WINDOW_SIZE
                )
            qk = tl.where(score_mask, qk, float("-inf"))

            # MLA exposes its latent V through V_Buffer; ordinary shared-KV
            # attention has an independent V cache. Both use this same load.
            if PAGED_V2:
                page_offsets = tl.arange(0, BLOCK_N // 64) * 64
                page_first = tl.load(
                    kv_indices + cur_batch_kv_start_idx + start_n + page_offsets,
                    mask=start_n + page_offsets < split_end,
                    other=0,
                )
                expected = tl.reshape(
                    page_first[:, None] + tl.arange(0, 64)[None, :], (BLOCK_N,)
                )
                regular = tl.sum(((kv_loc != expected) & n_mask).to(tl.int32), 0) == 0
                aligned = tl.sum((page_first % 64).to(tl.int32), 0) == 0
                if regular & aligned:
                    # Runtime-verified page-contiguous IDs, including partial last page.
                    vb = (page_first // 64) * tl.multiple_of(stride_buf_vpage, 64)
                    vb += tl.multiple_of(kv_head_off_v, 64)
                    vb = tl.reshape(vb[:, None] + tl.arange(0, 64)[None, :], (BLOCK_N,))
                    vb = tl.max_contiguous(vb, 64)
                    v = tl.load(
                        V_Buffer
                        + vb[:, None]
                        + offs_dv[None, :] * tl.multiple_of(stride_buf_vd, 64),
                        mask=n_mask[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
                        other=0.0,
                    )
                else:
                    v = tl.load(
                        V_Buffer + v_base + offs_dv[None, :] * stride_buf_vd,
                        mask=n_mask[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
                        other=0.0,
                    )
            else:
                v = tl.load(
                    V_Buffer
                    # Legacy HCU V is [page, head, dim, token].
                    + v_base
                    + offs_dv[None, :] * stride_buf_vd,
                    mask=n_mask[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
                    other=0.0,
                )
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            if SLIDING_WINDOW_SIZE > 0:
                # An entire prefix split can be invisible to a later query.
                safe_max = tl.where(n_e_max == float("-inf"), 0.0, n_e_max)
                re_scale = tl.exp(e_max - safe_max)
                p = tl.exp(qk - safe_max[:, None])
            else:
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            if PRESERVE_Q_PRECISION:
                v = v.to(COMPUTE_DTYPE)
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        # fp8 dequant of prefix V: scale the accumulated (pre-normalised) output.
        acc *= v_scale
        o_row = tl.reshape(
            cur_batch * stride_ob
            + offs_h[:, None] * stride_oh
            + split_kv_id * stride_os
            + offs_l[None, :] * stride_ol,
            (R,),
        )
        tl.store(
            Att_Out + o_row[:, None] + offs_dv[None, :],
            # Empty prefix is an identity contribution, not 0/0.
            (acc / tl.where(e_sum > 0, e_sum, 1.0)[:, None]).to(
                Att_Out.dtype.element_ty
            ),
            mask=row_mask[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        )
        lse_row = tl.reshape(
            cur_batch * stride_lb
            + offs_h[:, None] * stride_lh
            + split_kv_id * stride_ls
            + offs_l[None, :],
            (R,),
        )
        tl.store(Att_Lse + lse_row, e_max + tl.log(e_sum), mask=row_mask)


@triton.jit
def _verify_mla_combine_stage2(
    Att_Out,
    Att_Lse,
    Q,
    K_Extend,
    V_Extend,
    O_Out,
    sm_scale,
    qo_indptr,
    kv_indptr,
    num_splits,
    stride_ob,
    stride_oh,
    stride_os,
    stride_ol,
    stride_lb,
    stride_lh,
    stride_ls,
    stride_qbs,
    stride_qh,
    stride_kebs,
    stride_vebs,
    stride_keh,
    stride_veh,
    stride_oobs,
    stride_ooh,
    L_EXT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    HAS_KV_HEADS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDING_WINDOW_SIZE: tl.constexpr = -1,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    if HAS_KV_HEADS:
        kv_head_off_ke = (cur_head // KV_GROUP_NUM) * stride_keh
        kv_head_off_ve = (cur_head // KV_GROUP_NUM) * stride_veh
    else:
        kv_head_off_ke = 0
        kv_head_off_ve = 0

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_l = tl.arange(0, L_EXT)

    cur_q_start = tl.load(qo_indptr + cur_batch)
    l_ext = tl.load(qo_indptr + cur_batch + 1) - cur_q_start
    mask_l = offs_l < l_ext

    # ---- (a) combine prefix splits (online logsumexp over active splits) ---
    seqlen = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    active = _active_splits(seqlen, num_splits, BLOCK_N)

    m = tl.zeros([L_EXT], dtype=tl.float32) - float("inf")
    l_acc = tl.zeros([L_EXT], dtype=tl.float32)
    acc = tl.zeros([L_EXT, BLOCK_DV], dtype=tl.float32)
    # Empty/padded prefixes must not merge (-inf)-(-inf).
    for s in range(tl.where(seqlen > 0, active, 0)):
        lse_s = tl.load(
            Att_Lse
            + cur_batch * stride_lb
            + cur_head * stride_lh
            + s * stride_ls
            + offs_l,
            mask=mask_l,
            other=float("-inf"),
        )
        o_s = tl.load(
            Att_Out
            + cur_batch * stride_ob
            + cur_head * stride_oh
            + s * stride_os
            + offs_l[:, None] * stride_ol
            + offs_dv[None, :],
            mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        new_m = tl.maximum(m, lse_s)
        if SLIDING_WINDOW_SIZE > 0:
            safe_m = tl.where(new_m == float("-inf"), 0.0, new_m)
            alpha = tl.exp(m - safe_m)
            beta = tl.exp(lse_s - safe_m)
        else:
            alpha = tl.exp(m - new_m)
            beta = tl.exp(lse_s - new_m)
        acc = acc * alpha[:, None] + o_s * beta[:, None]
        l_acc = l_acc * alpha + beta
        m = new_m
    o_prefix = acc / tl.where(l_acc > 0, l_acc, 1.0)[:, None]
    lse_prefix = m + tl.log(l_acc)

    # ---- (b) draft-draft causal attention (L_EXT x L_EXT) -----------------
    # load draft queries [L_EXT, D], draft K/V [L_EXT, D]/[L_EXT, Dv]
    offs_q = (
        (cur_q_start + offs_l)[:, None] * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q + offs_q, mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM), other=0.0
    ).to(tl.float32)
    offs_ke = (
        (cur_q_start + offs_l)[:, None] * stride_kebs + kv_head_off_ke + offs_d[None, :]
    )
    ke = tl.load(
        K_Extend + offs_ke,
        mask=mask_l[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    offs_ve = (
        (cur_q_start + offs_l)[:, None] * stride_vebs
        + kv_head_off_ve
        + offs_dv[None, :]
    )
    ve = tl.load(
        V_Extend + offs_ve,
        mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
        other=0.0,
    ).to(tl.float32)

    # scores[i,j] = q_i . k_j  (i query, j key)  -> [L_EXT, L_EXT]
    qk = tl.sum(q[:, None, :] * ke[None, :, :], 2) * sm_scale
    # causal: query i sees key j iff j <= i
    # non-causal: query i sees all keys
    both_valid = mask_l[None, :] & mask_l[:, None]
    if IS_CAUSAL:
        vis = (offs_l[None, :] <= offs_l[:, None]) & both_valid
    else:
        vis = both_valid
    if SLIDING_WINDOW_SIZE > 0:
        vis = vis & (offs_l[:, None] <= offs_l[None, :] + SLIDING_WINDOW_SIZE)
    qk = tl.where(vis, qk, float("-inf"))
    m_d = tl.max(qk, 1)
    pd = tl.exp(qk - m_d[:, None])
    denom_d = tl.sum(pd, 1)
    o_draft = tl.sum(pd[:, :, None] * ve[None, :, :], 1) / denom_d[:, None]
    lse_draft = m_d + tl.log(denom_d)

    # ---- (c) final LSE merge (prefix vs draft) ----------------------------
    mm = tl.maximum(lse_prefix, lse_draft)
    wp = tl.exp(lse_prefix - mm)
    wd = tl.exp(lse_draft - mm)
    o = (o_prefix * wp[:, None] + o_draft * wd[:, None]) / (wp + wd)[:, None]

    offs_oo = (
        (cur_q_start + offs_l)[:, None] * stride_oobs
        + cur_head * stride_ooh
        + offs_dv[None, :]
    )
    tl.store(
        O_Out + offs_oo,
        o.to(O_Out.dtype.element_ty),
        mask=mask_l[:, None] & (offs_dv[None, :] < V_HEAD_DIM),
    )


class VerifyMLA:
    def __init__(
        self,
        max_bs,
        h_q,
        head_dim,
        v_head_dim,
        l_ext,
        device="cuda",
        block_h=DEFAULT_BLOCK_H,
        block_n=DEFAULT_BLOCK_N,
        num_warps=DEFAULT_NUM_WARPS,
        kv_group_num=None,
    ):
        self.h_q = h_q
        # MLA is the h_kv == 1 case (kv_group_num == h_q); a GQA draft passes a
        # smaller group. block_h must divide it so a head block maps to one KV
        # head -- can_handle enforces that before this is constructed.
        self.kv_group_num = h_q if kv_group_num is None else kv_group_num
        # h_kv == 1 (MLA / MQA) needs no KV-head offset at all; making that a
        # constexpr keeps the inner-loop addressing identical to the original.
        self.has_kv_heads = self.kv_group_num < h_q
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.nope_dim = v_head_dim
        self.pe_dim = head_dim - v_head_dim
        self.l_ext = l_ext
        # tl.dot requires BLOCK_H * L_EXT to cover at least 16 rows.
        min_l_pad = triton.next_power_of_2(triton.cdiv(16, block_h))
        self.l_pad = max(min_l_pad, triton.next_power_of_2(l_ext))
        self.device = device
        self.block_h = block_h
        self.block_n = block_n
        self.num_warps = num_warps
        self.n_head_blocks = triton.cdiv(h_q, block_h)
        self._alloc(max_bs)

    def _alloc(self, max_bs):
        self.max_bs = max_bs
        # bf16 partials (halves scratch traffic vs fp32); lse stays fp32.
        self.att_out = torch.empty(
            (max_bs, self.h_q, MAX_N_SPLITS, self.l_pad, self.v_head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.att_lse = torch.empty(
            (max_bs, self.h_q, MAX_N_SPLITS, self.l_pad),
            dtype=torch.float32,
            device=self.device,
        )

    def grow_buffers(self, max_bs):
        if max_bs > self.max_bs:
            self._alloc(max_bs)

    def _num_splits(self, bs):
        return max(1, min(MAX_N_SPLITS, 128 // (bs * self.n_head_blocks)))

    def _run_prefix_kernel(
        self,
        bs,
        num_splits,
        q_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        k_scale,
        v_scale,
        page_size=1,
        sliding_window_size=-1,
    ):
        grid = (bs, self.n_head_blocks, num_splits)
        ks, kh, _ = k_buffer.stride()
        vs, vh, vd = v_buffer.stride()
        kp, kt = ks * page_size, ks
        vp, vt = vs * page_size, vs
        _verify_mla_prefix_stage1[grid](
            q_extend,
            k_buffer,
            v_buffer,
            sm_scale,
            k_scale,
            v_scale,
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.att_out,
            self.att_lse,
            num_splits,
            q_extend.stride(0),
            q_extend.stride(1),
            ks,
            vs,
            kh,
            vh,
            kp,
            kt,
            vp,
            vt,
            vd,
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            H_Q=self.h_q,
            L_EXT=self.l_pad,
            BLOCK_H=self.block_h,
            KV_GROUP_NUM=self.kv_group_num,
            HAS_KV_HEADS=self.has_kv_heads,
            NOPE_DIM=self.nope_dim,
            PE_DIM=self.pe_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DNOPE=triton.next_power_of_2(self.nope_dim),
            BLOCK_DPE=max(1, triton.next_power_of_2(self.pe_dim)),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=self.block_n,
            PAGE_SIZE=page_size,
            SLIDING_WINDOW_SIZE=sliding_window_size,
            PRESERVE_Q_PRECISION=True,
            num_warps=self.num_warps,
            num_stages=1,
            **_AMD_LAUNCH_KWARGS,
        )

    def _run_combine_kernel(
        self,
        bs,
        num_splits,
        q_extend,
        k_extend,
        v_extend,
        o_out,
        qo_indptr,
        kv_indptr,
        sm_scale,
        is_causal=True,
        sliding_window_size=-1,
    ):
        grid = (bs, self.h_q)
        _verify_mla_combine_stage2[grid](
            self.att_out,
            self.att_lse,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            sm_scale,
            qo_indptr,
            kv_indptr,
            num_splits,
            self.att_out.stride(0),
            self.att_out.stride(1),
            self.att_out.stride(2),
            self.att_out.stride(3),
            self.att_lse.stride(0),
            self.att_lse.stride(1),
            self.att_lse.stride(2),
            q_extend.stride(0),
            q_extend.stride(1),
            k_extend.stride(0),
            v_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(1),
            o_out.stride(0),
            o_out.stride(1),
            L_EXT=self.l_pad,
            HEAD_DIM=self.head_dim,
            V_HEAD_DIM=self.v_head_dim,
            BLOCK_DMODEL=triton.next_power_of_2(self.head_dim),
            BLOCK_DV=triton.next_power_of_2(self.v_head_dim),
            BLOCK_N=self.block_n,
            KV_GROUP_NUM=self.kv_group_num,
            HAS_KV_HEADS=self.has_kv_heads,
            IS_CAUSAL=is_causal,
            SLIDING_WINDOW_SIZE=sliding_window_size,
            num_warps=4,
            num_stages=1,
        )

    def __call__(
        self,
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        o_out=None,
        k_scale=1.0,
        v_scale=1.0,
        is_causal=True,
        page_size=1,
        sliding_window_size=-1,
    ):
        if o_out is None:
            o_out = torch.empty(
                (q_extend.shape[0], self.h_q, self.v_head_dim),
                dtype=q_extend.dtype,
                device=q_extend.device,
            )
        bs = qo_indptr.shape[0] - 1
        # One split count for both stages (they must agree on the active count).
        num_splits = self._num_splits(bs)
        self._run_prefix_kernel(
            bs,
            num_splits,
            q_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            k_scale,
            v_scale,
            page_size=page_size,
            sliding_window_size=sliding_window_size,
        )
        self._run_combine_kernel(
            bs,
            num_splits,
            q_extend,
            k_extend,
            v_extend,
            o_out,
            qo_indptr,
            kv_indptr,
            sm_scale,
            is_causal=is_causal,
            sliding_window_size=sliding_window_size,
        )
        return o_out
