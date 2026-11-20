// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/DeviceGuard.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/ops/_assert_async.h>
#include <torch/extension.h>

#include <cstdint>
#include <initializer_list>

#include "glm53_fp8_k1_wasp.cuh"

namespace hcu_sglang_glm53_fp8_k1 {
namespace {

constexpr int kExperts = 32;
constexpr int kTopK = 8;
constexpr int kHiddenSize = 6144;
constexpr int kGateUpSize = 4096;
constexpr int kExpertAlignment = 256;
constexpr int64_t kOffsetLimit = int64_t{1} << 31;

__global__ void initialize_metadata(const int* padded_counts, int* counters,
                                    int* expert_ends, int* row_src,
                                    int* m_indices, int packed_rows) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < packed_rows) {
    row_src[index] = -1;
    m_indices[index] = -1;
  }
  if (index != 0) {
    return;
  }
  int64_t offset = 0;
  bool valid = true;
  for (int expert = 0; expert < kExperts; ++expert) {
    const int count = padded_counts[expert];
    valid = valid && count >= 0 && count % kExpertAlignment == 0;
    counters[expert] = offset >= 0 && offset <= packed_rows ? static_cast<int>(offset) : -1;
    offset += count;
    expert_ends[expert] = offset >= 0 && offset <= packed_rows ? static_cast<int>(offset) : -1;
  }
  if (!valid || offset != packed_rows) {
    for (int expert = 0; expert < kExperts; ++expert) {
      expert_ends[expert] = -1;
    }
    expert_ends[kExperts] = 1;
  } else {
    expert_ends[kExperts] = 0;
  }
}

__global__ void map_routes(const int64_t* topk, int* counters, int* expert_ends,
                           int* row_src, int* m_indices, int* output_index,
                           int route_count) {
  const int route = blockIdx.x * blockDim.x + threadIdx.x;
  if (route >= route_count) {
    return;
  }
  output_index[route] = -1;
  const int64_t expert = topk[route];
  if (expert < 0 || expert >= kExperts || expert_ends[expert] < 0) {
    return;
  }
  const int packed_row = atomicAdd(counters + expert, 1);
  if (packed_row >= expert_ends[expert]) {
    atomicExch(expert_ends + kExperts, 2);
    return;
  }
  row_src[packed_row] = route / kTopK;
  m_indices[packed_row] = static_cast<int>(expert);
  output_index[route] = packed_row;
}

void check_storage(const torch::Tensor& anchor,
                   std::initializer_list<torch::Tensor> tensors) {
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == anchor.device() &&
                    tensor.is_contiguous(),
                "GLM-5.3 indirect K1 requires contiguous tensors on one HIP device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                "GLM-5.3 indirect K1 requires 16-byte aligned tensors");
  }
}

void check_row_counts(int64_t received, int64_t packed) {
  TORCH_CHECK(received > 0 && received <= (kOffsetLimit - 1) / kHiddenSize &&
                  packed > 0 && packed % kExpertAlignment == 0 &&
                  packed <= (kOffsetLimit - 1) / (kGateUpSize * 2),
              "GLM-5.3 indirect K1 requires positive R/M, M aligned to 256, "
              "and 32-bit activation/output byte offsets");
}

void metadata(const torch::Tensor& topk, const torch::Tensor& padded_counts,
              torch::Tensor counters, torch::Tensor row_src,
              torch::Tensor m_indices, torch::Tensor output_index) {
  using hcu_sglang_glm53_fp8_k1_wasp::pair32_wasp_is_gfx938;
  check_storage(topk, {topk, padded_counts, counters, row_src, m_indices, output_index});
  const c10::OptionalDeviceGuard guard(at::device_of(topk));
  TORCH_CHECK(pair32_wasp_is_gfx938(), "GLM-5.3 indirect K1 requires gfx938");
  const auto stream = at::cuda::getCurrentHIPStream();
  hipStreamCaptureStatus capture_status;
  TORCH_CHECK(hipStreamIsCapturing(stream, &capture_status) == hipSuccess &&
                  capture_status == hipStreamCaptureStatusNone,
              "GLM-5.3 indirect K1 metadata validation requires an eager stream");
  TORCH_CHECK(topk.scalar_type() == at::kLong && topk.dim() == 2 && topk.size(1) == kTopK,
              "GLM-5.3 indirect K1 topk must be int64 [R,8]");
  for (const auto& tensor : {padded_counts, counters, row_src, m_indices, output_index}) {
    TORCH_CHECK(tensor.scalar_type() == at::kInt, "GLM-5.3 indirect K1 metadata must be int32");
  }
  TORCH_CHECK(padded_counts.sizes() == at::IntArrayRef({kExperts}) &&
                  counters.sizes() == padded_counts.sizes() && row_src.dim() == 1 &&
                  m_indices.sizes() == row_src.sizes() && output_index.sizes() == topk.sizes(),
              "Expected counts/counters [32], row maps [M], and output_index [R,8]");
  const int64_t packed_rows = row_src.numel();
  check_row_counts(topk.size(0), packed_rows);
  const torch::Tensor outputs[] = {counters, row_src, m_indices, output_index};
  for (int index = 0; index < 4; ++index) {
    at::assert_no_overlap(outputs[index], topk);
    at::assert_no_overlap(outputs[index], padded_counts);
    for (int other = 0; other < index; ++other) {
      at::assert_no_overlap(outputs[index], outputs[other]);
    }
  }

  auto expert_ends = at::empty({kExperts + 1}, padded_counts.options());
  hipLaunchKernelGGL(initialize_metadata, dim3((packed_rows + 255) / 256), dim3(256),
                     0, stream, padded_counts.data_ptr<int>(), counters.data_ptr<int>(),
                     expert_ends.data_ptr<int>(), row_src.data_ptr<int>(),
                     m_indices.data_ptr<int>(), static_cast<int>(packed_rows));
  TORCH_CHECK(hipGetLastError() == hipSuccess, "GLM-5.3 indirect K1 metadata initialization failed");
  const int route_count = static_cast<int>(topk.numel());
  hipLaunchKernelGGL(map_routes, dim3((route_count + 255) / 256), dim3(256),
                     0, stream, topk.data_ptr<int64_t>(), counters.data_ptr<int>(),
                     expert_ends.data_ptr<int>(), row_src.data_ptr<int>(),
                     m_indices.data_ptr<int>(), output_index.data_ptr<int>(), route_count);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "GLM-5.3 indirect K1 route mapping failed");
  at::_assert_async(expert_ends.select(0, kExperts).eq(0),
                   "GLM-5.3 indirect K1 requires valid padded counts and route capacity");
}

torch::Tensor indirect(const torch::Tensor& input, const torch::Tensor& packed_weight,
                       torch::Tensor output, const torch::Tensor& input_scale,
                       const torch::Tensor& weight_scale, const torch::Tensor& m_indices,
                       const torch::Tensor& row_src) {
  using namespace hcu_sglang_glm53_fp8_k1_wasp;
  check_storage(input, {input, packed_weight, output, input_scale, weight_scale, m_indices, row_src});
  const c10::OptionalDeviceGuard guard(at::device_of(input));
  TORCH_CHECK(pair32_wasp_is_gfx938(), "GLM-5.3 indirect K1 requires gfx938");
  TORCH_CHECK(input.scalar_type() == at::kFloat8_e4m3fn &&
                  packed_weight.scalar_type() == at::kFloat8_e4m3fn &&
                  output.scalar_type() == at::kBFloat16 && input_scale.scalar_type() == at::kFloat &&
                  weight_scale.scalar_type() == at::kFloat && m_indices.scalar_type() == at::kInt &&
                  row_src.scalar_type() == at::kInt,
              "GLM-5.3 indirect K1 requires E4M3FN inputs, FP32 scales, BF16 output and int32 maps");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == kHiddenSize &&
                  output.dim() == 2 && output.size(1) == kGateUpSize,
              "GLM-5.3 indirect K1 expects [R,6144] input and [M,4096] output");
  TORCH_CHECK(packed_weight.sizes() == at::IntArrayRef({kExperts, 96, 128, 2, 4, 16, 16}),
              "GLM-5.3 indirect K1 expects pair32 weights [32,96,128,2,4,16,16]");
  const int64_t received_rows = input.size(0);
  const int64_t packed_rows = output.size(0);
  check_row_counts(received_rows, packed_rows);
  TORCH_CHECK((input_scale.sizes() == at::IntArrayRef({received_rows}) ||
                   input_scale.sizes() == at::IntArrayRef({received_rows, 1})) &&
                  (weight_scale.sizes() == at::IntArrayRef({kExperts, kGateUpSize}) ||
                   weight_scale.sizes() == at::IntArrayRef({kExperts, kGateUpSize, 1})) &&
                  m_indices.sizes() == at::IntArrayRef({packed_rows}) &&
                  row_src.sizes() == m_indices.sizes(),
              "GLM-5.3 indirect K1 scale/map shapes do not match input/output");
  for (const auto& tensor : {input, packed_weight, input_scale, weight_scale, m_indices, row_src}) {
    at::assert_no_overlap(output, tensor);
  }

  DeepGemm_params params{};
  params.a_ptr = input.data_ptr();
  params.b_ptr = packed_weight.data_ptr();
  params.d_ptr = output.data_ptr();
  params.scale_a_ptr = input_scale.data_ptr();
  params.scale_b_ptr = weight_scale.data_ptr();
  params.m_indices = m_indices.data_ptr<int32_t>();
  params.row_src = row_src.data_ptr<int32_t>();
  params.recv_rows = static_cast<int>(received_rows);
  params.m = static_cast<int>(packed_rows);
  params.n = kGateUpSize;
  params.k = kHiddenSize;
  params.a_stride = kHiddenSize;
  params.d_stride = kGateUpSize;
  params.experts = kExperts;
  params.alpha = 1.0f;
  hipLaunchKernelGGL((fp8_pair32_wasp_kernel<bhalf_t, true>),
                     dim3(kGateUpSize / GemmTraitsPC::kBlockN, packed_rows / GemmTraitsPC::kBlockM),
                     dim3(GemmTraitsPC::kNThreads), GemmTraitsPC::kSmemSize,
                     at::cuda::getCurrentHIPStream(), params);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "GLM-5.3 indirect K1 launch failed");
  return output;
}

}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("metadata", &hcu_sglang_glm53_fp8_k1::metadata);
  module.def("indirect", &hcu_sglang_glm53_fp8_k1::indirect);
}
