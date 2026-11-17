#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace dsv41_quant {
static void check(hipblasStatus_t status, const char* where) {
    if (status != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(where) + ": " + std::to_string(int(status)));
}
struct Plan {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    hipblasLtMatmulAlgo_t algo;
    uint64_t workspace = 0;
    int64_t m, groups, n, k;
    ~Plan() {
        if (pref) hipblasLtMatmulPreferenceDestroy(pref);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
        if (op) hipblasLtMatmulDescDestroy(op);
        if (handle) hipblasLtDestroy(handle);
    }
};
static void batch(hipblasLtMatrixLayout_t layout, int32_t groups, int64_t stride) {
    check(hipblasLtMatrixLayoutSetAttribute(layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &groups, sizeof(groups)), "batch_count");
    check(hipblasLtMatrixLayoutSetAttribute(layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)), "batch_stride");
}
__device__ float max_nan(float x, float y) {
    return isnan(y) || y > x ? y : x;
}
__global__ __launch_bounds__(512) void quantize_fixed_kernel(const hip_bfloat16* x, int8_t* q, float* scales, int64_t k) {
    const int64_t row = blockIdx.x;
    float peak = 0.0f;
    for (int64_t col = threadIdx.x; col < k; col += blockDim.x)
        peak = max_nan(peak, fabsf(float(x[row * k + col])));
    for (int delta = warpSize / 2; delta; delta /= 2)
        peak = max_nan(peak, __shfl_down(peak, delta, warpSize));
    __shared__ float partial[16];
    __shared__ float inverse, scale;
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;
    if (!lane) partial[warp] = peak;
    __syncthreads();
    if (!warp) {
        peak = lane < blockDim.x / warpSize ? partial[lane] : 0.0f;
        for (int delta = warpSize / 2; delta; delta /= 2)
            peak = max_nan(peak, __shfl_down(peak, delta, warpSize));
        if (!lane) {
            if (peak < 1e-10f) peak = 1e-10f;
            inverse = 127.0f * (1.0f / peak);
            scale = peak * (1.0f / 127.0f);
            scales[row] = scale;
        }
    }
    __syncthreads();
    for (int64_t col = threadIdx.x; col < k; col += blockDim.x) {
        float value = float(x[row * k + col]) * inverse;
        value = copysignf(floorf(fabsf(value) + 0.5f), value);
        q[row * k + col] = isfinite(value) ? int8_t(fminf(127.0f, fmaxf(-127.0f, value))) : int8_t(0);
    }
}
__global__ void scale_kernel(const int32_t* raw, const float* xs, const float* ws,
                             hip_bfloat16* output, int64_t size, int64_t groups, int64_t n) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) {
        const int64_t row = i / n;
        float value = float(raw[i]) * xs[row];
        value = value * ws[i % (groups * n)];
        output[i] = hip_bfloat16(value);
    }
}
}

extern "C" void* woa_plan(int device, int64_t m, int64_t groups, int64_t n, int64_t k, char* error, size_t length) {
    using namespace dsv41_quant;
    Plan* p = new Plan();
    try {
        if (hipSetDevice(device) != hipSuccess) throw std::runtime_error("hipSetDevice failed");
        p->m = m; p->groups = groups; p->n = n; p->k = k;
        check(hipblasLtCreate(&p->handle), "create_handle");
        check(hipblasLtMatmulDescCreate(&p->op, HIPBLAS_COMPUTE_32I, HIP_R_32I), "create_op");
        hipblasOperation_t trans_a = HIPBLAS_OP_T;
        check(hipblasLtMatmulDescSetAttribute(p->op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)), "transpose_weight");
        check(hipblasLtMatrixLayoutCreate(&p->a, HIP_R_8I, k, n, k), "weight_layout");
        check(hipblasLtMatrixLayoutCreate(&p->b, HIP_R_8I, k, m, groups * k), "activation_layout");
        check(hipblasLtMatrixLayoutCreate(&p->c, HIP_R_32I, n, m, groups * n), "output_layout");
        batch(p->a, int32_t(groups), n * k);
        batch(p->b, int32_t(groups), k);
        batch(p->c, int32_t(groups), n);
        check(hipblasLtMatmulPreferenceCreate(&p->pref), "create_preference");
        uint64_t limit = 8 * 1024 * 1024;
        check(hipblasLtMatmulPreferenceSetAttribute(p->pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)), "workspace_limit");
        hipblasLtMatmulHeuristicResult_t result;
        int count = 0;
        check(hipblasLtMatmulAlgoGetHeuristic(p->handle, p->op, p->a, p->b, p->c, p->c, p->pref, 1, &result, &count), "heuristic");
        if (!count) throw std::runtime_error("No hipBLASLt INT8 algorithm for this grouped shape");
        check(result.state, "heuristic_state");
        p->algo = result.algo; p->workspace = result.workspaceSize;
        if (m == 4096 && groups == 8 && n == 1024 && k == 4096) {
            std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
            auto status = hipblaslt_ext::getAllAlgos(p->handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I,
                HIPBLAS_COMPUTE_32I, candidates);
            if (status == HIPBLAS_STATUS_SUCCESS) {
                for (auto& candidate : candidates) {
                    if (hipblaslt_ext::getKernelNameFromAlgo(p->handle, candidate.algo) != "Cijk_Alik_Bljk_I8II_BH_Bias_AI_MT256x256x32_MI16x16x1_SN_AFC1_DTL1_EPS1_GRVWA16_GRVWB16_GSU1_GSUAMB_K1_LRVW8_MIWT8_4_NLCA1_NLCB1_OBAFM1_OLDSR0_ONLL1_PGR4_PLR1_SU0_SUS0_TLDS1_ULDSWM1_USFGRO0_USSB1_WSS1_WG32_16_1_WGM5") continue;
                    int32_t alpha = 1, beta = 0;
                    size_t workspace = 0;
                    status = hipblaslt_ext::matmulIsAlgoSupported(p->handle, p->op, &alpha, p->a, p->b,
                        &beta, p->c, p->c, candidate.algo, workspace);
                    if (status == HIPBLAS_STATUS_SUCCESS && workspace <= limit) {
                        p->algo = candidate.algo;
                        p->workspace = workspace;
                    }
                    break;
                }
            }
        }
        return p;
    } catch (const std::exception& exc) {
        std::snprintf(error, length, "%s", exc.what()); delete p; return nullptr;
    }
}
extern "C" uint64_t woa_workspace(void* plan) { return static_cast<dsv41_quant::Plan*>(plan)->workspace; }
extern "C" void woa_destroy(void* plan) { delete static_cast<dsv41_quant::Plan*>(plan); }
extern "C" int woa_run(void* plan, const void* x, void* q, void* xs, const void* w, const void* ws,
                        void* raw, void* output, void* workspace, void* stream, char* error, size_t length) {
    using namespace dsv41_quant;
    auto* p = static_cast<Plan*>(plan);
    auto s = static_cast<hipStream_t>(stream);
    try {
        quantize_fixed_kernel<<<p->m * p->groups, 512, 0, s>>>(static_cast<const hip_bfloat16*>(x), static_cast<int8_t*>(q), static_cast<float*>(xs), p->k);
        if (hipGetLastError() != hipSuccess) throw std::runtime_error("INT8 quantize launch failed");
        int32_t alpha = 1, beta = 0;
        check(hipblasLtMatmul(p->handle, p->op, &alpha, w, p->a, q, p->b, &beta, raw, p->c, raw, p->c, &p->algo, workspace, p->workspace, s), "grouped_gemm");
        int64_t size = p->m * p->groups * p->n;
        scale_kernel<<<(size + 255) / 256, 256, 0, s>>>(static_cast<const int32_t*>(raw), static_cast<const float*>(xs), static_cast<const float*>(ws), static_cast<hip_bfloat16*>(output), size, p->groups, p->n);
        if (hipGetLastError() != hipSuccess) throw std::runtime_error("INT8 scale launch failed");
        return 0;
    } catch (const std::exception& exc) { std::snprintf(error, length, "%s", exc.what()); return 1; }
}
