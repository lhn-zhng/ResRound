#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/hadamard_transform.h>
#include <transformer_engine/transformer_engine.h>
#include <hadamard_transform/hadamard_transform_utils.cuh>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

enum ColumnwiseSource : int {
  kColumnwiseDirect = 1,
  kColumnwiseOutlierReuse = 2,
};

constexpr int kLogHistThreads = 256;
constexpr int kLogHistWarps = kLogHistThreads / 32;
constexpr int kLogHistMaxBins = 256;
constexpr int kHardCapHistogramBins = 2048;
constexpr int kActiveRowSortEndBit = 32;

struct EventState {
  cudaEvent_t event = nullptr;

  EventState() {
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }

  ~EventState() {
    if (event != nullptr) {
      cudaEventDestroy(event);
    }
  }

  EventState(const EventState&) = delete;
  EventState& operator=(const EventState&) = delete;
};

void record_stream_then_wait(cudaStream_t record_stream, cudaStream_t wait_stream) {
  EventState event;
  C10_CUDA_CHECK(cudaEventRecord(event.event, record_stream));
  C10_CUDA_CHECK(cudaStreamWaitEvent(wait_stream, event.event, 0));
}

void record_tensor_on_stream(const at::Tensor& tensor, const c10::cuda::CUDAStream& stream) {
  if (tensor.defined() && tensor.is_cuda() && tensor.numel() > 0) {
    c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  }
}

int32_t* pinned_int32_buffer() {
  thread_local int32_t* host_ptr = nullptr;
  if (host_ptr == nullptr) {
    C10_CUDA_CHECK(cudaHostAlloc(&host_ptr, sizeof(int32_t), cudaHostAllocPortable));
  }
  return host_ptr;
}

int32_t copy_int32_to_host(const int32_t* device_ptr, cudaStream_t stream) {
  int32_t* host_value = pinned_int32_buffer();
  *host_value = 0;
  C10_CUDA_CHECK(cudaMemcpyAsync(
      host_value, device_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  return *host_value;
}

template <typename T>
__device__ inline float to_float(T value) {
  return static_cast<float>(value);
}

__device__ inline void atomic_max_float_nonnegative(float* addr, float value) {
  if (value > 0.0f && isfinite(value)) {
    atomicMax(reinterpret_cast<int*>(addr), __float_as_int(value));
  }
}

__device__ inline float warp_reduce_max_f32(float value) {
  constexpr unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(mask, value, offset));
  }
  return value;
}

__device__ inline float normal_inv_cdf_approx(float p) {
  p = fminf(fmaxf(p, 1.0e-12f), 1.0f - 1.0e-12f);
  constexpr float plow = 0.02425f;
  constexpr float phigh = 1.0f - plow;
  if (p < plow) {
    const float q = sqrtf(-2.0f * logf(p));
    return (((((-7.784894002430293e-03f * q - 3.223964580411365e-01f) * q -
               2.400758277161838e+00f) *
                  q -
              2.549732539343734e+00f) *
                 q +
             4.374664141464968e+00f) *
                q +
            2.938163982698783e+00f) /
           ((((7.784695709041462e-03f * q + 3.224671290700398e-01f) * q +
              2.445134137142996e+00f) *
                 q +
             3.754408661907416e+00f) *
                q +
            1.0f);
  }
  if (p > phigh) {
    const float q = sqrtf(-2.0f * logf(1.0f - p));
    return -(((((-7.784894002430293e-03f * q - 3.223964580411365e-01f) * q -
                2.400758277161838e+00f) *
                   q -
               2.549732539343734e+00f) *
                  q +
              4.374664141464968e+00f) *
                 q +
             2.938163982698783e+00f) /
           ((((7.784695709041462e-03f * q + 3.224671290700398e-01f) * q +
              2.445134137142996e+00f) *
                 q +
             3.754408661907416e+00f) *
                q +
            1.0f);
  }
  const float q = p - 0.5f;
  const float r = q * q;
  return (((((-3.969683028665376e+01f * r + 2.209460984245205e+02f) * r -
             2.759285104469687e+02f) *
                r +
            1.383577518672690e+02f) *
               r -
           3.066479806614716e+01f) *
              r +
          2.506628277459239e+00f) *
         q /
         (((((-5.447609879822406e+01f * r + 1.615858368580409e+02f) * r -
             1.556989798866e+02f) *
                r +
            6.680131188771972e+01f) *
               r -
           1.328068155288572e+01f) *
              r +
          1.0f);
}

template <int BINS>
__device__ __forceinline__ int bf16_abs_log_bin(uint16_t bits, int min_exp) {
  static_assert(BINS == 64 || BINS == 128 || BINS == 256, "unsupported log hist bin count");
  constexpr int subbins_per_exp = BINS / 16;
  constexpr int subbin_shift = BINS == 64 ? 5 : (BINS == 128 ? 4 : 3);
  const uint16_t mag = static_cast<uint16_t>(bits & 0x7fffu);
  if (mag == 0) {
    return 0;
  }
  const int exp_biased = static_cast<int>((mag >> 7) & 0xffu);
  if (exp_biased == 0) {
    return 0;
  }
  const int exp_unbiased = exp_biased - 127;
  const int mant_hi = static_cast<int>((mag & 0x7fu) >> subbin_shift);
  int bin = (exp_unbiased - min_exp) * subbins_per_exp + mant_hi;
  if (bin < 0) {
    return 0;
  }
  if (bin >= BINS) {
    return BINS - 1;
  }
  return bin;
}

__device__ __forceinline__ uint32_t mix_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__device__ inline float compute_nvfp4_global_encode_scale(float global_amax) {
  constexpr float fp8_max = 448.0f;
  constexpr float fp4_max = 6.0f;
  if (!(global_amax > 0.0f) || !isfinite(global_amax)) {
    return 1.0f;
  }
  const float scale = fp8_max * fp4_max / global_amax;
  return (scale > 0.0f && isfinite(scale)) ? scale : 1.0f;
}

__device__ __forceinline__ uint16_t fp4x4_e2m1_rn_ordered(float v0,
                                                          float v1,
                                                          float v2,
                                                          float v3) {
#if ((defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && \
     ((__CUDA_ARCH_HAS_FEATURE__(SM100_ALL)) ||             \
      (__CUDA_ARCH_HAS_FEATURE__(SM101_ALL)) ||             \
      (__CUDA_ARCH_HAS_FEATURE__(SM120_ALL))))
  uint16_t result;
  asm volatile(
      "{\n\t"
      ".reg .b8 lo8, hi8;\n\t"
      ".reg .b32 lo32, hi32, packed32;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 lo8, %2, %1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 hi8, %4, %3;\n\t"
      "cvt.u32.u8 lo32, lo8;\n\t"
      "cvt.u32.u8 hi32, hi8;\n\t"
      "shl.b32 hi32, hi32, 8;\n\t"
      "or.b32 packed32, lo32, hi32;\n\t"
      "cvt.u16.u32 %0, packed32;\n\t"
      "}"
      : "=h"(result)
      : "f"(v0), "f"(v1), "f"(v2), "f"(v3));
  return result;
#else
  const float4 quad = make_float4(v0, v1, v2, v3);
  const __nv_fp4x4_e2m1 packed(quad);
  return packed.__x;
#endif
}

inline int64_t round_up_int64(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

inline int normalize_threads(int64_t requested, int default_threads) {
  if (requested == 64 || requested == 128 || requested == 256 || requested == 512) {
    return static_cast<int>(requested);
  }
  return default_threads;
}

template <typename scalar_t, int N>
struct alignas(16) Vec {
  scalar_t elt[N];

  __device__ inline void load_from(const scalar_t* ptr) {
    *this = *reinterpret_cast<const Vec<scalar_t, N>*>(ptr);
  }

  __device__ inline void store_to(scalar_t* ptr) const {
    *reinterpret_cast<Vec<scalar_t, N>*>(ptr) = *this;
  }
};

struct NvteTensorHandle {
  explicit NvteTensorHandle(NVTEScalingMode scaling_mode)
      : tensor(nvte_create_tensor(scaling_mode)) {}

  ~NvteTensorHandle() {
    if (tensor != nullptr) {
      nvte_destroy_tensor(tensor);
      tensor = nullptr;
    }
  }

  NvteTensorHandle(const NvteTensorHandle&) = delete;
  NvteTensorHandle& operator=(const NvteTensorHandle&) = delete;

  NVTETensor tensor = nullptr;
};

struct NvteQuantConfigHandle {
  NvteQuantConfigHandle() : config(nvte_create_quantization_config()) {}

  ~NvteQuantConfigHandle() {
    if (config != nullptr) {
      nvte_destroy_quantization_config(config);
      config = nullptr;
    }
  }

  NvteQuantConfigHandle(const NvteQuantConfigHandle&) = delete;
  NvteQuantConfigHandle& operator=(const NvteQuantConfigHandle&) = delete;

  NVTEQuantizationConfig config = nullptr;
};

inline NVTEShape make_nvte_shape(std::initializer_list<size_t> dims) {
  std::vector<size_t> shape(dims);
  return nvte_make_shape(shape.data(), shape.size());
}

inline void set_nvte_tensor_param(NVTETensor tensor,
                                  NVTETensorParam param,
                                  void* ptr,
                                  NVTEDType dtype,
                                  std::initializer_list<size_t> dims) {
  auto shape = make_nvte_shape(dims);
  NVTEBasicTensor basic{ptr, dtype, shape};
  nvte_set_tensor_param(&tensor, param, &basic);
}

inline NVTEDType nvte_dtype_from_scalar_type(at::ScalarType scalar_type) {
  if (scalar_type == at::kBFloat16) {
    return kNVTEBFloat16;
  }
  TORCH_CHECK(false, "unsupported TE input dtype");
}

void launch_te_rht_columnwise_quant(const at::Tensor& source,
                                    at::Tensor& columnwise_data,
                                    at::Tensor& columnwise_scale,
                                    at::Tensor& columnwise_amax,
                                    int64_t rows,
                                    int64_t cols,
                                    int64_t column_scale_inner,
                                    int rht_random_sign_mask_t,
                                    cudaStream_t stream) {
  const auto input_dtype = nvte_dtype_from_scalar_type(source.scalar_type());
  NvteTensorHandle input_tensor(NVTE_DELAYED_TENSOR_SCALING);
  set_nvte_tensor_param(input_tensor.tensor,
                        kNVTERowwiseData,
                        const_cast<void*>(source.data_ptr()),
                        input_dtype,
                        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

  NvteTensorHandle amax_out(NVTE_NVFP4_1D_SCALING);
  set_nvte_tensor_param(amax_out.tensor,
                        kNVTEColumnwiseAmax,
                        columnwise_amax.data_ptr<float>(),
                        kNVTEFloat32,
                        {1});
  nvte_hadamard_transform_amax(
      input_tensor.tensor, amax_out.tensor, 0, rht_random_sign_mask_t, stream);

  auto rht_output_t = at::empty({cols, rows}, source.options());
  NvteTensorHandle rht_output_tensor(NVTE_DELAYED_TENSOR_SCALING);
  set_nvte_tensor_param(rht_output_tensor.tensor,
                        kNVTERowwiseData,
                        rht_output_t.data_ptr(),
                        input_dtype,
                        {static_cast<size_t>(cols), static_cast<size_t>(rows)});
  nvte_hadamard_transform(
      input_tensor.tensor, rht_output_tensor.tensor, 0, rht_random_sign_mask_t, stream);

  NvteTensorHandle out_transpose(NVTE_NVFP4_1D_SCALING);
  set_nvte_tensor_param(out_transpose.tensor,
                        kNVTERowwiseData,
                        columnwise_data.data_ptr<uint8_t>(),
                        kNVTEFloat4E2M1,
                        {static_cast<size_t>(cols), static_cast<size_t>(rows)});
  set_nvte_tensor_param(out_transpose.tensor,
                        kNVTERowwiseScaleInv,
                        columnwise_scale.data_ptr<uint8_t>(),
                        kNVTEFloat8E4M3,
                        {static_cast<size_t>(round_up_int64(cols, 128)),
                         static_cast<size_t>(column_scale_inner)});
  set_nvte_tensor_param(out_transpose.tensor,
                        kNVTEAmax,
                        columnwise_amax.data_ptr<float>(),
                        kNVTEFloat32,
                        {1});

  NvteQuantConfigHandle quant_config;
  bool nvfp4_2d_quantization = false;
  bool stochastic_rounding = false;
  nvte_set_quantization_config_attribute(quant_config.config,
                                         kNVTEQuantizationConfigNVFP42DQuantization,
                                         &nvfp4_2d_quantization,
                                         sizeof(bool));
  nvte_set_quantization_config_attribute(quant_config.config,
                                         kNVTEQuantizationConfigStochasticRounding,
                                         &stochastic_rounding,
                                         sizeof(bool));
  nvte_quantize_v2(rht_output_tensor.tensor, out_transpose.tensor, quant_config.config, stream);
}

template <bool APPLY_MASK>
__global__ void custom_hadamard_transform_t_amax_kernel(
    const c10::BFloat16* __restrict__ input,
    const uint16_t* __restrict__ selection_masks,
    c10::BFloat16* __restrict__ output_t,
    float* __restrict__ amax_t,
    uint16_t random_sign_mask_t,
    uint64_t num_input_rows,
    uint64_t num_input_cols) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int kHadamardDimension = 16;
  extern __shared__ __align__(16) uint8_t dynamic_smem[];
  __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(dynamic_smem);
  float* block_warp_amax = reinterpret_cast<float*>(
      smem + kHadamardDimension * kHadamardDimension * blockDim.y * blockDim.z);

  const int32_t tid = threadIdx.x;
  const int32_t warp_id = threadIdx.y * blockDim.z + threadIdx.z;
  const int32_t local_bx = threadIdx.y;
  const int32_t local_by = threadIdx.z;
  const uint32_t row = tid / (kHadamardDimension * sizeof(__nv_bfloat16) / sizeof(uint4));
  const uint32_t col = tid % (kHadamardDimension * sizeof(__nv_bfloat16) / sizeof(uint4));
  const uint32_t input_start_col = (blockIdx.x * blockDim.y + local_bx) * kHadamardDimension;
  const uint32_t input_start_row = (blockIdx.y * blockDim.z + local_by) * kHadamardDimension;
  if (input_start_col >= num_input_cols || input_start_row >= num_input_rows) {
    return;
  }

  uint16_t lane_mask = 0;
  if constexpr (APPLY_MASK) {
    const uint64_t blocks_per_row = num_input_cols / 16;
    const uint16_t row_mask =
        selection_masks[static_cast<uint64_t>(input_start_row + row) * blocks_per_row +
                        input_start_col / 16];
    lane_mask = static_cast<uint16_t>((row_mask >> (col * 8)) & 0xffu);
  }

  __nv_bfloat16* base_smem = smem + kHadamardDimension * kHadamardDimension * warp_id;
  uint32_t* smem_b32 = reinterpret_cast<uint32_t*>(base_smem);
  uint4* smem_b128 = reinterpret_cast<uint4*>(base_smem);

  union Bf16x8 {
    uint4 v;
    __nv_bfloat16 h[8];
  };

  const uint64_t elem_offset = static_cast<uint64_t>(input_start_row + row) * num_input_cols +
                               input_start_col + col * 8;
  Bf16x8 pack;
  pack.v = *reinterpret_cast<const uint4*>(input + elem_offset);
  if constexpr (APPLY_MASK) {
    if (lane_mask != 0) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if ((lane_mask & (1u << i)) != 0) {
          pack.h[i] = __float2bfloat16(0.0f);
        }
      }
    }
  }
  smem_b128[tid] = pack.v;

  uint32_t a_frag[4];
  uint32_t b_frag_t[4];
  uint32_t c_frag[4];
  uint32_t b_frag_i[4];
  transformer_engine::get_hadamard_matrix_fragment<false, true, false, false>(
      b_frag_i, 0, b_frag_t, random_sign_mask_t);

  float local_amax_t = 0.0f;
  uint32_t local_amax_t_reg = *reinterpret_cast<uint32_t*>(&local_amax_t);
  __syncwarp();

  transformer_engine::load_matrix_16x16_from_shared<true>(
      a_frag[0],
      a_frag[2],
      a_frag[1],
      a_frag[3],
      smem_b32,
      kHadamardDimension);
  transformer_engine::mma_m16_n16_k16_b16_b16_b16_noacc<true>(
      a_frag[0],
      a_frag[2],
      a_frag[1],
      a_frag[3],
      b_frag_t[0],
      b_frag_t[1],
      b_frag_t[2],
      b_frag_t[3],
      c_frag[0],
      c_frag[1],
      c_frag[2],
      c_frag[3],
      local_amax_t_reg);

  const uint64_t global_offset_t =
      static_cast<uint64_t>(input_start_row) + static_cast<uint64_t>(input_start_col) *
                                                num_input_rows;
  uint4* output_t_b128 = reinterpret_cast<uint4*>(output_t + global_offset_t);
  transformer_engine::store_matrix_16x16_to_global<false>(
      c_frag[0],
      c_frag[1],
      c_frag[2],
      c_frag[3],
      output_t_b128,
      static_cast<uint32_t>(num_input_rows));

  transformer_engine::unpack_max_of_packed_bf16(local_amax_t_reg, local_amax_t);
  local_amax_t = warp_reduce_max_f32(local_amax_t);
  if (tid == 0) {
    block_warp_amax[warp_id] = local_amax_t;
  }
  __syncthreads();
  if (warp_id == 0) {
    const int warps_per_block = blockDim.y * blockDim.z;
    float block_amax = tid < warps_per_block ? block_warp_amax[tid] : 0.0f;
    block_amax = warp_reduce_max_f32(block_amax);
    if (tid == 0) {
      atomic_max_float_nonnegative(amax_t, block_amax);
    }
  }
#endif
}

at::Tensor launch_custom_te_rht_columnwise_quant(const at::Tensor& source,
                                                 const at::Tensor& selection_masks,
                                                 at::Tensor& columnwise_data,
                                                 at::Tensor& columnwise_scale,
                                                 at::Tensor& columnwise_amax,
                                                 int64_t rows,
                                                 int64_t cols,
                                                 int64_t column_scale_inner,
                                                 int rht_random_sign_mask_t,
                                                 bool apply_mask,
                                                 cudaStream_t stream,
                                                 const c10::cuda::CUDAStream& allocator_stream) {
  C10_CUDA_CHECK(cudaMemsetAsync(columnwise_amax.data_ptr<float>(), 0, sizeof(float), stream));
  auto rht_output_t = at::empty({cols, rows}, source.options());
  record_tensor_on_stream(rht_output_t, allocator_stream);

  constexpr int kHadamardDimension = 16;
  constexpr int kThreadBlockX = 4;
  constexpr int kThreadBlockY = 4;
  constexpr int kThreadsPerWarpLocal = 32;
  const size_t shmem_bytes =
      kHadamardDimension * kHadamardDimension * sizeof(__nv_bfloat16) *
          kThreadBlockX * kThreadBlockY +
      kThreadBlockX * kThreadBlockY * sizeof(float);
  dim3 block(kThreadsPerWarpLocal, kThreadBlockX, kThreadBlockY);
  dim3 grid(static_cast<unsigned>((cols / kHadamardDimension + kThreadBlockX - 1) /
                                  kThreadBlockX),
            static_cast<unsigned>((rows / kHadamardDimension + kThreadBlockY - 1) /
                                  kThreadBlockY));
  const uint16_t* mask_ptr =
      apply_mask ? reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>())
                 : nullptr;
  if (apply_mask) {
    custom_hadamard_transform_t_amax_kernel<true><<<grid, block, shmem_bytes, stream>>>(
        source.data_ptr<c10::BFloat16>(),
        mask_ptr,
        rht_output_t.data_ptr<c10::BFloat16>(),
        columnwise_amax.data_ptr<float>(),
        static_cast<uint16_t>(rht_random_sign_mask_t),
        static_cast<uint64_t>(rows),
        static_cast<uint64_t>(cols));
  } else {
    custom_hadamard_transform_t_amax_kernel<false><<<grid, block, shmem_bytes, stream>>>(
        source.data_ptr<c10::BFloat16>(),
        mask_ptr,
        rht_output_t.data_ptr<c10::BFloat16>(),
        columnwise_amax.data_ptr<float>(),
        static_cast<uint16_t>(rht_random_sign_mask_t),
        static_cast<uint64_t>(rows),
        static_cast<uint64_t>(cols));
  }

  const auto input_dtype = nvte_dtype_from_scalar_type(source.scalar_type());
  NvteTensorHandle rht_output_tensor(NVTE_DELAYED_TENSOR_SCALING);
  set_nvte_tensor_param(rht_output_tensor.tensor,
                        kNVTERowwiseData,
                        rht_output_t.data_ptr(),
                        input_dtype,
                        {static_cast<size_t>(cols), static_cast<size_t>(rows)});
  NvteTensorHandle out_transpose(NVTE_NVFP4_1D_SCALING);
  set_nvte_tensor_param(out_transpose.tensor,
                        kNVTERowwiseData,
                        columnwise_data.data_ptr<uint8_t>(),
                        kNVTEFloat4E2M1,
                        {static_cast<size_t>(cols), static_cast<size_t>(rows)});
  set_nvte_tensor_param(out_transpose.tensor,
                        kNVTERowwiseScaleInv,
                        columnwise_scale.data_ptr<uint8_t>(),
                        kNVTEFloat8E4M3,
                        {static_cast<size_t>(round_up_int64(cols, 128)),
                         static_cast<size_t>(column_scale_inner)});
  set_nvte_tensor_param(out_transpose.tensor,
                        kNVTEAmax,
                        columnwise_amax.data_ptr<float>(),
                        kNVTEFloat32,
                        {1});
  NvteQuantConfigHandle quant_config;
  bool nvfp4_2d_quantization = false;
  bool stochastic_rounding = false;
  nvte_set_quantization_config_attribute(quant_config.config,
                                         kNVTEQuantizationConfigNVFP42DQuantization,
                                         &nvfp4_2d_quantization,
                                         sizeof(bool));
  nvte_set_quantization_config_attribute(quant_config.config,
                                         kNVTEQuantizationConfigStochasticRounding,
                                         &stochastic_rounding,
                                         sizeof(bool));
  nvte_quantize_v2(rht_output_tensor.tensor, out_transpose.tensor, quant_config.config, stream);
  return rht_output_t;
}

void launch_custom_te_rht_columnwise_quant_out(const at::Tensor& source,
                                               const at::Tensor& selection_masks,
                                               at::Tensor& rht_output_t,
                                               at::Tensor& columnwise_data,
                                               at::Tensor& columnwise_scale,
                                               at::Tensor& columnwise_amax,
                                               int64_t rows,
                                               int64_t cols,
                                               int64_t column_scale_inner,
                                               int rht_random_sign_mask_t,
                                               bool apply_mask,
                                               cudaStream_t stream) {
  C10_CUDA_CHECK(cudaMemsetAsync(columnwise_amax.data_ptr<float>(), 0, sizeof(float), stream));

  constexpr int kHadamardDimension = 16;
  constexpr int kThreadBlockX = 4;
  constexpr int kThreadBlockY = 4;
  constexpr int kThreadsPerWarpLocal = 32;
  const size_t shmem_bytes =
      kHadamardDimension * kHadamardDimension * sizeof(__nv_bfloat16) *
          kThreadBlockX * kThreadBlockY +
      kThreadBlockX * kThreadBlockY * sizeof(float);
  dim3 block(kThreadsPerWarpLocal, kThreadBlockX, kThreadBlockY);
  dim3 grid(static_cast<unsigned>((cols / kHadamardDimension + kThreadBlockX - 1) /
                                  kThreadBlockX),
            static_cast<unsigned>((rows / kHadamardDimension + kThreadBlockY - 1) /
                                  kThreadBlockY));
  const uint16_t* mask_ptr =
      apply_mask ? reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>())
                 : nullptr;
  if (apply_mask) {
    custom_hadamard_transform_t_amax_kernel<true><<<grid, block, shmem_bytes, stream>>>(
        source.data_ptr<c10::BFloat16>(),
        mask_ptr,
        rht_output_t.data_ptr<c10::BFloat16>(),
        columnwise_amax.data_ptr<float>(),
        static_cast<uint16_t>(rht_random_sign_mask_t),
        static_cast<uint64_t>(rows),
        static_cast<uint64_t>(cols));
  } else {
    custom_hadamard_transform_t_amax_kernel<false><<<grid, block, shmem_bytes, stream>>>(
        source.data_ptr<c10::BFloat16>(),
        mask_ptr,
        rht_output_t.data_ptr<c10::BFloat16>(),
        columnwise_amax.data_ptr<float>(),
        static_cast<uint16_t>(rht_random_sign_mask_t),
        static_cast<uint64_t>(rows),
        static_cast<uint64_t>(cols));
  }

  const auto input_dtype = nvte_dtype_from_scalar_type(source.scalar_type());
  NvteTensorHandle rht_output_tensor(NVTE_DELAYED_TENSOR_SCALING);
  set_nvte_tensor_param(rht_output_tensor.tensor,
                        kNVTERowwiseData,
                        rht_output_t.data_ptr(),
                        input_dtype,
                        {static_cast<size_t>(cols), static_cast<size_t>(rows)});
  NvteTensorHandle out_transpose(NVTE_NVFP4_1D_SCALING);
  set_nvte_tensor_param(out_transpose.tensor,
                        kNVTERowwiseData,
                        columnwise_data.data_ptr<uint8_t>(),
                        kNVTEFloat4E2M1,
                        {static_cast<size_t>(cols), static_cast<size_t>(rows)});
  set_nvte_tensor_param(out_transpose.tensor,
                        kNVTERowwiseScaleInv,
                        columnwise_scale.data_ptr<uint8_t>(),
                        kNVTEFloat8E4M3,
                        {static_cast<size_t>(round_up_int64(cols, 128)),
                         static_cast<size_t>(column_scale_inner)});
  set_nvte_tensor_param(out_transpose.tensor,
                        kNVTEAmax,
                        columnwise_amax.data_ptr<float>(),
                        kNVTEFloat32,
                        {1});
  NvteQuantConfigHandle quant_config;
  bool nvfp4_2d_quantization = false;
  bool stochastic_rounding = false;
  nvte_set_quantization_config_attribute(quant_config.config,
                                         kNVTEQuantizationConfigNVFP42DQuantization,
                                         &nvfp4_2d_quantization,
                                         sizeof(bool));
  nvte_set_quantization_config_attribute(quant_config.config,
                                         kNVTEQuantizationConfigStochasticRounding,
                                         &stochastic_rounding,
                                         sizeof(bool));
  nvte_quantize_v2(rht_output_tensor.tensor, out_transpose.tensor, quant_config.config, stream);
}

template <typename scalar_t>
__global__ void sum_sumsq_absmax_kernel(const scalar_t* __restrict__ input,
                                        float* __restrict__ stats,
                                        int64_t total) {
  extern __shared__ float shmem[];
  float* sh_sum = shmem;
  float* sh_sumsq = shmem + blockDim.x;
  float* sh_sum_abs = shmem + 2 * blockDim.x;
  float* sh_max_abs = shmem + 3 * blockDim.x;

  float local_sum = 0.0f;
  float local_sumsq = 0.0f;
  float local_sum_abs = 0.0f;
  float local_max_abs = 0.0f;
  const int64_t vec_items = total / 8;
  const int64_t vec_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t vec_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       vec_idx < vec_items;
       vec_idx += vec_stride) {
    Vec<scalar_t, 8> in_vec;
    in_vec.load_from(input + vec_idx * 8);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float value = to_float(in_vec.elt[e]);
      const float abs_value = fabsf(value);
      local_sum += value;
      local_sumsq += value * value;
      local_sum_abs += abs_value;
      local_max_abs = fmaxf(local_max_abs, abs_value);
    }
  }
  for (int64_t idx = vec_items * 8 + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = to_float(input[idx]);
    const float abs_value = fabsf(value);
    local_sum += value;
    local_sumsq += value * value;
    local_sum_abs += abs_value;
    local_max_abs = fmaxf(local_max_abs, abs_value);
  }
  sh_sum[threadIdx.x] = local_sum;
  sh_sumsq[threadIdx.x] = local_sumsq;
  sh_sum_abs[threadIdx.x] = local_sum_abs;
  sh_max_abs[threadIdx.x] = local_max_abs;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sh_sum[threadIdx.x] += sh_sum[threadIdx.x + stride];
      sh_sumsq[threadIdx.x] += sh_sumsq[threadIdx.x + stride];
      sh_sum_abs[threadIdx.x] += sh_sum_abs[threadIdx.x + stride];
      sh_max_abs[threadIdx.x] = fmaxf(sh_max_abs[threadIdx.x], sh_max_abs[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(stats, sh_sum[0]);
    atomicAdd(stats + 1, sh_sumsq[0]);
    atomicAdd(stats + 2, sh_sum_abs[0]);
    atomic_max_float_nonnegative(stats + 3, sh_max_abs[0]);
  }
}

__device__ inline void compute_adaptive_stats_from_raw(const float* __restrict__ raw_stats,
                                                       int64_t total,
                                                       float base_ratio,
                                                       float min_ratio,
                                                       float max_ratio,
                                                       float reference_heaviness,
                                                       float& mean,
                                                       float& threshold,
                                                       float& heaviness,
                                                       float& effective_ratio) {
  const float inv_total = total > 0 ? 1.0f / static_cast<float>(total) : 0.0f;
  mean = raw_stats[0] * inv_total;
  const float mean_squares = raw_stats[1] * inv_total;
  const float mean_abs = raw_stats[2] * inv_total;
  const float max_abs = raw_stats[3];
  const float variance = fmaxf(mean_squares - mean * mean, 0.0f);
  const float std = sqrtf(variance);

  effective_ratio = 0.0f;
  heaviness = 0.0f;
  if (mean_abs > 1.0e-12f && isfinite(mean_abs) && isfinite(max_abs)) {
    heaviness = max_abs / mean_abs;
    const float log_ref = logf(fmaxf(reference_heaviness, 1.0f));
    effective_ratio = log_ref > 0.0f
                          ? base_ratio * (logf(fmaxf(heaviness, 1.0f)) / log_ref)
                          : base_ratio;
  }
  effective_ratio =
      effective_ratio < 1.0e-5f ? 0.0f : fmaxf(min_ratio, fminf(max_ratio, effective_ratio));

  threshold = -1.0f;
  if (effective_ratio >= 1.0f) {
    threshold = 0.0f;
  } else if (effective_ratio > 0.0f && isfinite(std) && std > 0.0f) {
    const float tail_probability =
        fminf(fmaxf(effective_ratio * 0.5f, 1.0e-12f), 0.5f - 1.0e-12f);
    threshold = normal_inv_cdf_approx(1.0f - tail_probability) * std;
  }
}

__global__ void finalize_adaptive_stats_kernel(const float* __restrict__ raw_stats,
                                               float* __restrict__ stats_out,
                                               int64_t total,
                                               float base_ratio,
                                               float min_ratio,
                                               float max_ratio,
                                               float reference_heaviness) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  float mean = 0.0f;
  float threshold = -1.0f;
  float heaviness = 0.0f;
  float effective_ratio = 0.0f;
  compute_adaptive_stats_from_raw(raw_stats,
                                  total,
                                  base_ratio,
                                  min_ratio,
                                  max_ratio,
                                  reference_heaviness,
                                  mean,
                                  threshold,
                                  heaviness,
                                  effective_ratio);
  stats_out[0] = mean;
  stats_out[1] = threshold;
  stats_out[2] = heaviness;
  stats_out[3] = effective_ratio;
}

__global__ void override_threshold_sigma_kernel(const float* __restrict__ raw_stats,
                                                float* __restrict__ stats,
                                                int64_t total,
                                                float threshold_sigma) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || !(threshold_sigma > 0.0f)) {
    return;
  }
  const float inv_total = total > 0 ? 1.0f / static_cast<float>(total) : 0.0f;
  const float mean = raw_stats[0] * inv_total;
  const float mean_squares = raw_stats[1] * inv_total;
  const float variance = fmaxf(mean_squares - mean * mean, 0.0f);
  stats[0] = mean;
  stats[1] = threshold_sigma * sqrtf(variance);
}

template <typename scalar_t>
__global__ void row_count_main_amax_total_kernel(const scalar_t* __restrict__ input,
                                                 int32_t* __restrict__ row_counts,
                                                 uint16_t* __restrict__ selection_masks,
                                                 float* __restrict__ main_amax,
                                                 int32_t* __restrict__ num_selected,
                                                 const float* __restrict__ stats,
                                                 int64_t rows,
                                                 int64_t cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float mean = stats[0];
  const float threshold = stats[1];
  const bool select_all = threshold == 0.0f;
  const bool select_none = threshold < 0.0f;

  __shared__ int32_t block_counts[256];
  __shared__ float block_amax[256];

  int32_t local_count = 0;
  float local_amax = 0.0f;
  const int64_t row_base = row * cols;
  const int64_t blocks_per_row = cols / 16;
  for (int64_t block_col = threadIdx.x; block_col < blocks_per_row; block_col += blockDim.x) {
    const int64_t col0 = block_col * 16;
    const int64_t flat0 = row_base + col0;
    uint16_t selected_mask = 0;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      Vec<scalar_t, 8> in_wave;
      in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = w * 8 + e;
        const float value = to_float(in_wave.elt[e]);
        const bool selected = !select_none && (select_all || fabsf(value - mean) >= threshold);
        const float abs_value = fabsf(value);
        selected_mask |= static_cast<uint16_t>(selected ? (1u << i) : 0u);
        if (!selected) {
          local_amax = fmaxf(local_amax, abs_value);
        }
      }
    }
    selection_masks[row * blocks_per_row + block_col] = selected_mask;
    local_count += __popc(static_cast<unsigned>(selected_mask));
  }
  block_counts[threadIdx.x] = local_count;
  block_amax[threadIdx.x] = local_amax;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      block_counts[threadIdx.x] += block_counts[threadIdx.x + stride];
      block_amax[threadIdx.x] = fmaxf(block_amax[threadIdx.x], block_amax[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_counts[row] = block_counts[0];
    atomicAdd(num_selected, block_counts[0]);
    atomic_max_float_nonnegative(main_amax, block_amax[0]);
  }
}

template <typename scalar_t>
__global__ void row_count_main_amax_total_nomask_kernel(const scalar_t* __restrict__ input,
                                                        int32_t* __restrict__ row_counts,
                                                        float* __restrict__ main_amax,
                                                        int32_t* __restrict__ num_selected,
                                                        const float* __restrict__ stats,
                                                        int64_t rows,
                                                        int64_t cols) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float mean = stats[0];
  const float threshold = stats[1];
  const bool select_all = threshold == 0.0f;
  const bool select_none = threshold < 0.0f;

  __shared__ int32_t block_counts[256];
  __shared__ float block_amax[256];

  int32_t local_count = 0;
  float local_amax = 0.0f;
  const int64_t row_base = row * cols;
  const int64_t blocks_per_row = cols / 16;
  for (int64_t block_col = threadIdx.x; block_col < blocks_per_row; block_col += blockDim.x) {
    const int64_t col0 = block_col * 16;
    const int64_t flat0 = row_base + col0;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      Vec<scalar_t, 8> in_wave;
      in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float value = to_float(in_wave.elt[e]);
        const bool selected = !select_none && (select_all || fabsf(value - mean) >= threshold);
        const float abs_value = fabsf(value);
        local_count += selected ? 1 : 0;
        if (!selected) {
          local_amax = fmaxf(local_amax, abs_value);
        }
      }
    }
  }
  block_counts[threadIdx.x] = local_count;
  block_amax[threadIdx.x] = local_amax;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      block_counts[threadIdx.x] += block_counts[threadIdx.x + stride];
      block_amax[threadIdx.x] = fmaxf(block_amax[threadIdx.x], block_amax[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_counts[row] = block_counts[0];
    atomicAdd(num_selected, block_counts[0]);
    atomic_max_float_nonnegative(main_amax, block_amax[0]);
  }
}

__global__ void reset_loghist_workspace_kernel(float* __restrict__ stats_amax,
                                               int32_t* __restrict__ num_selected,
                                               int32_t* __restrict__ overflow,
                                               int32_t* __restrict__ log_hist,
                                               int64_t* __restrict__ log_params) {
  const int idx = threadIdx.x;
  if (idx < 10) {
    stats_amax[idx] = 0.0f;
  }
  if (idx < kLogHistMaxBins) {
    log_hist[idx] = 0;
  }
  if (idx < 8) {
    log_params[idx] = 0;
  }
  if (idx == 0) {
    num_selected[0] = 0;
    overflow[0] = 0;
  }
}

__global__ void init_active_row_sort_inputs_kernel(const int32_t* __restrict__ row_counts,
                                                   int64_t* __restrict__ sort_keys,
                                                   int32_t* __restrict__ sort_rows,
                                                   int32_t* __restrict__ active_row_count,
                                                   int64_t rows) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int32_t count = row_counts[row];
  sort_rows[row] = static_cast<int32_t>(row);
  if (count > 0) {
    // Count is the primary key; lower row id is the stable tie-breaker.
    sort_keys[row] =
        static_cast<int64_t>(count) * (rows + 1) + (rows - row);
    atomicAdd(active_row_count, 1);
  } else {
    sort_keys[row] = 0;
  }
}

__global__ void build_heavy_light_active_rows_kernel(const int32_t* __restrict__ sorted_rows,
                                                     int32_t* __restrict__ active_rows,
                                                     const int32_t* __restrict__ active_row_count) {
  const int32_t idx = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int32_t count = active_row_count[0];
  if (idx >= count) {
    return;
  }
  int32_t src;
  if ((idx & 1) == 0) {
    src = idx;
  } else {
    const int32_t last_odd = ((count & 1) == 0) ? (count - 1) : (count - 2);
    src = last_odd - 2 * (idx >> 1);
  }
  active_rows[idx] = sorted_rows[src];
}

__global__ void build_unsorted_active_rows_from_counts_kernel(
    const int32_t* __restrict__ row_counts,
    int32_t* __restrict__ active_rows,
    int32_t* __restrict__ active_row_count,
    int64_t rows) {
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = 8;
  __shared__ int32_t warp_counts[kWarpsPerBlock];
  __shared__ int32_t warp_offsets[kWarpsPerBlock];
  __shared__ int32_t block_base;

  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) & (kWarpSize - 1);
  const int32_t warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
  const bool active = row < rows && row_counts[row] > 0;
  const uint32_t active_mask = __ballot_sync(0xffffffffu, active);
  if (lane == 0) {
    warp_counts[warp] = __popc(active_mask);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t block_count = 0;
    #pragma unroll
    for (int32_t index = 0; index < kWarpsPerBlock; ++index) {
      warp_offsets[index] = block_count;
      block_count += warp_counts[index];
    }
    block_base = atomicAdd(active_row_count, block_count);
  }
  __syncthreads();

  if (active) {
    const uint32_t preceding_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
    const int32_t pos =
        block_base + warp_offsets[warp] + __popc(active_mask & preceding_lanes);
    active_rows[pos] = static_cast<int32_t>(row);
  }
}

std::vector<at::Tensor> build_r25_active_rows_schedule(const at::Tensor& row_counts,
                                                       int64_t rows) {
  auto active_rows = at::empty({rows}, row_counts.options());
  auto sorted_rows_in = at::empty({rows}, row_counts.options());
  auto sorted_rows_out = at::empty({rows}, row_counts.options());
  auto sort_keys_in = at::empty({rows}, row_counts.options().dtype(at::kLong));
  auto sort_keys_out = at::empty({rows}, row_counts.options().dtype(at::kLong));
  auto active_row_count = at::zeros({1}, row_counts.options());
  if (rows == 0) {
    return {active_rows, active_row_count};
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>((rows + threads - 1) / threads);
  auto stream = at::cuda::getDefaultCUDAStream();
  init_active_row_sort_inputs_kernel<<<blocks, threads, 0, stream>>>(
      row_counts.data_ptr<int32_t>(),
      sort_keys_in.data_ptr<int64_t>(),
      sorted_rows_in.data_ptr<int32_t>(),
      active_row_count.data_ptr<int32_t>(),
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  void* temp_storage = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(temp_storage,
                                            temp_bytes,
                                            sort_keys_in.data_ptr<int64_t>(),
                                            sort_keys_out.data_ptr<int64_t>(),
                                            sorted_rows_in.data_ptr<int32_t>(),
                                            sorted_rows_out.data_ptr<int32_t>(),
                                            static_cast<int>(rows),
                                            0,
                                            kActiveRowSortEndBit,
                                            stream);
  auto temp = at::empty({static_cast<int64_t>(temp_bytes)}, row_counts.options().dtype(at::kByte));
  cub::DeviceRadixSort::SortPairsDescending(temp.data_ptr(),
                                            temp_bytes,
                                            sort_keys_in.data_ptr<int64_t>(),
                                            sort_keys_out.data_ptr<int64_t>(),
                                            sorted_rows_in.data_ptr<int32_t>(),
                                            sorted_rows_out.data_ptr<int32_t>(),
                                            static_cast<int>(rows),
                                            0,
                                            kActiveRowSortEndBit,
                                            stream);

  build_heavy_light_active_rows_kernel<<<blocks, threads, 0, stream>>>(
      sorted_rows_out.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_row_count.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {active_rows, active_row_count};
}

void build_unsorted_active_rows_from_counts_into(const at::Tensor& row_counts,
                                                 int64_t rows,
                                                 const at::Tensor& active_rows,
                                                 const at::Tensor& active_row_count,
                                                 bool pad_inactive_rows = false) {
  TORCH_CHECK(active_rows.numel() >= rows, "active_rows buffer is too small");
  TORCH_CHECK(active_row_count.numel() >= 1, "active_row_count buffer is too small");
  auto stream = at::cuda::getDefaultCUDAStream();
  if (pad_inactive_rows && rows > 0) {
    C10_CUDA_CHECK(cudaMemsetAsync(
        const_cast<int32_t*>(active_rows.data_ptr<int32_t>()),
        0xff,
        static_cast<size_t>(rows) * sizeof(int32_t),
        stream.stream()));
  }
  auto* active_count_ptr = const_cast<int32_t*>(active_row_count.data_ptr<int32_t>());
  C10_CUDA_CHECK(cudaMemsetAsync(active_count_ptr, 0, sizeof(int32_t), stream.stream()));
  if (rows == 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((rows + threads - 1) / threads);
  build_unsorted_active_rows_from_counts_kernel<<<blocks, threads, 0, stream>>>(
      row_counts.data_ptr<int32_t>(),
      const_cast<int32_t*>(active_rows.data_ptr<int32_t>()),
      active_count_ptr,
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t r25_active_sort_temp_bytes(int64_t rows) {
  if (rows <= 0) {
    return 0;
  }
  void* temp_storage = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(temp_storage,
                                            temp_bytes,
                                            static_cast<int64_t*>(nullptr),
                                            static_cast<int64_t*>(nullptr),
                                            static_cast<int32_t*>(nullptr),
                                            static_cast<int32_t*>(nullptr),
                                            static_cast<int>(rows),
                                            0,
                                            kActiveRowSortEndBit,
                                            at::cuda::getDefaultCUDAStream());
  return static_cast<int64_t>(temp_bytes);
}

void build_r25_active_rows_schedule_into(const at::Tensor& row_counts,
                                         int64_t rows,
                                         const at::Tensor& active_rows,
                                         const at::Tensor& active_row_count,
                                         const at::Tensor& sorted_rows_in,
                                         const at::Tensor& sorted_rows_out,
                                         const at::Tensor& sort_keys_in,
                                         const at::Tensor& sort_keys_out,
                                         const at::Tensor& sort_temp) {
  TORCH_CHECK(active_rows.numel() >= rows, "active_rows buffer is too small");
  TORCH_CHECK(active_row_count.numel() >= 1, "active_row_count buffer is too small");
  TORCH_CHECK(sorted_rows_in.numel() >= rows && sorted_rows_out.numel() >= rows,
              "active row sort row buffers are too small");
  TORCH_CHECK(sort_keys_in.numel() >= rows && sort_keys_out.numel() >= rows,
              "active row sort key buffers are too small");
  TORCH_CHECK(sort_temp.numel() >= r25_active_sort_temp_bytes(rows),
              "active row sort temp buffer is too small");
  if (rows == 0) {
    return;
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>((rows + threads - 1) / threads);
  auto stream = at::cuda::getDefaultCUDAStream();
  auto* active_count_ptr = const_cast<int32_t*>(active_row_count.data_ptr<int32_t>());
  C10_CUDA_CHECK(cudaMemsetAsync(active_count_ptr, 0, sizeof(int32_t), stream.stream()));
  init_active_row_sort_inputs_kernel<<<blocks, threads, 0, stream>>>(
      row_counts.data_ptr<int32_t>(),
      const_cast<int64_t*>(sort_keys_in.data_ptr<int64_t>()),
      const_cast<int32_t*>(sorted_rows_in.data_ptr<int32_t>()),
      active_count_ptr,
      rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t temp_bytes = static_cast<size_t>(sort_temp.numel());
  cub::DeviceRadixSort::SortPairsDescending(
      const_cast<uint8_t*>(sort_temp.data_ptr<uint8_t>()),
      temp_bytes,
      sort_keys_in.data_ptr<int64_t>(),
      const_cast<int64_t*>(sort_keys_out.data_ptr<int64_t>()),
      sorted_rows_in.data_ptr<int32_t>(),
      const_cast<int32_t*>(sorted_rows_out.data_ptr<int32_t>()),
      static_cast<int>(rows),
      0,
      kActiveRowSortEndBit,
      stream);

  build_heavy_light_active_rows_kernel<<<blocks, threads, 0, stream>>>(
      sorted_rows_out.data_ptr<int32_t>(),
      const_cast<int32_t*>(active_rows.data_ptr<int32_t>()),
      active_count_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int BINS>
__global__ void build_loghist_kernel(const uint16_t* __restrict__ input_bits,
                                     int32_t* __restrict__ log_hist,
                                     int64_t total,
                                     int min_exp) {
  __shared__ int32_t warp_hist[kLogHistWarps][BINS];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;

  for (int i = tid; i < kLogHistWarps * BINS; i += blockDim.x) {
    reinterpret_cast<int32_t*>(warp_hist)[i] = 0;
  }
  __syncthreads();

  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; idx < total; idx += stride) {
    const int bin = bf16_abs_log_bin<BINS>(input_bits[idx], min_exp);
    atomicAdd(&warp_hist[warp][bin], 1);
  }
  __syncthreads();

  for (int bin = tid; bin < BINS; bin += blockDim.x) {
    int32_t sum = 0;
#pragma unroll
    for (int w = 0; w < kLogHistWarps; ++w) {
      sum += warp_hist[w][bin];
    }
    if (sum != 0) {
      atomicAdd(&log_hist[bin], sum);
    }
  }
}

template <int BINS>
__global__ void choose_loghist_boundary_kernel(const int32_t* __restrict__ log_hist,
                                               int64_t* __restrict__ log_params,
                                               float* __restrict__ stats,
                                               double ratio,
                                               int64_t total) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int64_t target = static_cast<int64_t>(ratio * static_cast<double>(total) + 0.999999);
  if (target < 0) {
    target = 0;
  }
  if (target > total) {
    target = total;
  }

  int boundary = -1;
  int64_t n_above = 0;
  int64_t n_boundary = 0;
  int64_t need = 0;
  uint32_t hash_threshold = 0;

  if (target > 0) {
    int64_t running = 0;
    for (int bin = BINS - 1; bin >= 0; --bin) {
      const int64_t count = static_cast<int64_t>(log_hist[bin]);
      if (running + count >= target) {
        boundary = bin;
        n_above = running;
        n_boundary = count;
        need = target - running;
        if (count > 0 && need >= count) {
          hash_threshold = 0xffffffffu;
        } else if (count > 0 && need > 0) {
          const double p = static_cast<double>(need) / static_cast<double>(count);
          hash_threshold = static_cast<uint32_t>(p * 4294967295.0);
        }
        break;
      }
      running += count;
    }
  }

  log_params[0] = target;
  log_params[1] = boundary;
  log_params[2] = n_above;
  log_params[3] = n_boundary;
  log_params[4] = need;
  log_params[5] = static_cast<int64_t>(hash_threshold);
  log_params[6] = total;
  log_params[7] = BINS;

  stats[0] = static_cast<float>(ratio);
  stats[1] = static_cast<float>(boundary);
  stats[2] = target > 0 ? static_cast<float>(static_cast<double>(n_boundary) /
                                             static_cast<double>(target))
                        : 0.0f;
  stats[3] = n_boundary > 0 ? static_cast<float>(static_cast<double>(need) /
                                                 static_cast<double>(n_boundary))
                            : 0.0f;
}

template <typename scalar_t, int BINS>
__global__ void row_count_loghist_main_amax_total_kernel(
    const scalar_t* __restrict__ input,
    const uint16_t* __restrict__ input_bits,
    int32_t* __restrict__ row_counts,
    uint16_t* __restrict__ selection_masks,
    float* __restrict__ main_amax,
    int32_t* __restrict__ num_selected,
    const int64_t* __restrict__ log_params,
    int64_t rows,
    int64_t cols,
    int min_exp,
    uint32_t seed) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int boundary = static_cast<int>(log_params[1]);
  const int64_t need = log_params[4];
  const int64_t n_boundary = log_params[3];
  const uint32_t hash_threshold = static_cast<uint32_t>(log_params[5]);
  const bool select_none = boundary < 0;

  __shared__ int32_t block_counts[256];
  __shared__ float block_amax[256];

  int32_t local_count = 0;
  float local_amax = 0.0f;
  const int64_t row_base = row * cols;
  const int64_t blocks_per_row = cols / 16;
  for (int64_t block_col = threadIdx.x; block_col < blocks_per_row; block_col += blockDim.x) {
    const int64_t col0 = block_col * 16;
    const int64_t flat0 = row_base + col0;
    uint16_t selected_mask = 0;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      Vec<scalar_t, 8> in_wave;
      in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = w * 8 + e;
        const int64_t flat = flat0 + w * 8 + e;
        const int bin = bf16_abs_log_bin<BINS>(input_bits[flat], min_exp);
        bool selected = !select_none && bin > boundary;
        if (!selected && !select_none && bin == boundary) {
          if (need >= n_boundary) {
            selected = true;
          } else if (need > 0) {
            const uint32_t h = mix_u32(static_cast<uint32_t>(flat) ^ seed);
            selected = h < hash_threshold;
          }
        }
        const float abs_value = fabsf(to_float(in_wave.elt[e]));
        selected_mask |= static_cast<uint16_t>(selected ? (1u << i) : 0u);
        if (!selected) {
          local_amax = fmaxf(local_amax, abs_value);
        }
      }
    }
    selection_masks[row * blocks_per_row + block_col] = selected_mask;
    local_count += __popc(static_cast<unsigned>(selected_mask));
  }
  block_counts[threadIdx.x] = local_count;
  block_amax[threadIdx.x] = local_amax;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      block_counts[threadIdx.x] += block_counts[threadIdx.x + stride];
      block_amax[threadIdx.x] = fmaxf(block_amax[threadIdx.x], block_amax[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_counts[row] = block_counts[0];
    atomicAdd(num_selected, block_counts[0]);
    atomic_max_float_nonnegative(main_amax, block_amax[0]);
  }
}

template <typename scalar_t>
__global__ void row_count_main_amax_total_if_overflow_kernel(
    const scalar_t* __restrict__ input,
    int32_t* __restrict__ row_counts,
    uint16_t* __restrict__ selection_masks,
    float* __restrict__ main_amax,
    int32_t* __restrict__ num_selected,
    const float* __restrict__ stats,
    const int32_t* __restrict__ overflow,
    int64_t rows,
    int64_t cols) {
  if (overflow[0] == 0) {
    return;
  }
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float mean = stats[0];
  const float threshold = stats[1];
  const bool select_all = threshold == 0.0f;
  const bool select_none = threshold < 0.0f;

  __shared__ int32_t block_counts[256];
  __shared__ float block_amax[256];

  int32_t local_count = 0;
  float local_amax = 0.0f;
  const int64_t row_base = row * cols;
  const int64_t blocks_per_row = cols / 16;
  for (int64_t block_col = threadIdx.x; block_col < blocks_per_row; block_col += blockDim.x) {
    const int64_t col0 = block_col * 16;
    const int64_t flat0 = row_base + col0;
    uint16_t selected_mask = 0;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      Vec<scalar_t, 8> in_wave;
      in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = w * 8 + e;
        const float value = to_float(in_wave.elt[e]);
        const bool selected = !select_none && (select_all || fabsf(value - mean) >= threshold);
        const float abs_value = fabsf(value);
        selected_mask |= static_cast<uint16_t>(selected ? (1u << i) : 0u);
        if (!selected) {
          local_amax = fmaxf(local_amax, abs_value);
        }
      }
    }
    selection_masks[row * blocks_per_row + block_col] = selected_mask;
    local_count += __popc(static_cast<unsigned>(selected_mask));
  }
  block_counts[threadIdx.x] = local_count;
  block_amax[threadIdx.x] = local_amax;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      block_counts[threadIdx.x] += block_counts[threadIdx.x + stride];
      block_amax[threadIdx.x] = fmaxf(block_amax[threadIdx.x], block_amax[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_counts[row] = block_counts[0];
    atomicAdd(num_selected, block_counts[0]);
    atomic_max_float_nonnegative(main_amax, block_amax[0]);
  }
}

template <typename scalar_t>
__global__ void row_count_main_amax_total_nomask_if_overflow_kernel(
    const scalar_t* __restrict__ input,
    int32_t* __restrict__ row_counts,
    float* __restrict__ main_amax,
    int32_t* __restrict__ num_selected,
    const float* __restrict__ stats,
    const int32_t* __restrict__ overflow,
    int64_t rows,
    int64_t cols) {
  if (overflow[0] == 0) {
    return;
  }
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float mean = stats[0];
  const float threshold = stats[1];
  const bool select_all = threshold == 0.0f;
  const bool select_none = threshold < 0.0f;

  __shared__ int32_t block_counts[256];
  __shared__ float block_amax[256];

  int32_t local_count = 0;
  float local_amax = 0.0f;
  const int64_t row_base = row * cols;
  const int64_t blocks_per_row = cols / 16;
  for (int64_t block_col = threadIdx.x; block_col < blocks_per_row; block_col += blockDim.x) {
    const int64_t col0 = block_col * 16;
    const int64_t flat0 = row_base + col0;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      Vec<scalar_t, 8> in_wave;
      in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float value = to_float(in_wave.elt[e]);
        const bool selected = !select_none && (select_all || fabsf(value - mean) >= threshold);
        const float abs_value = fabsf(value);
        local_count += selected ? 1 : 0;
        if (!selected) {
          local_amax = fmaxf(local_amax, abs_value);
        }
      }
    }
  }
  block_counts[threadIdx.x] = local_count;
  block_amax[threadIdx.x] = local_amax;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      block_counts[threadIdx.x] += block_counts[threadIdx.x + stride];
      block_amax[threadIdx.x] = fmaxf(block_amax[threadIdx.x], block_amax[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_counts[row] = block_counts[0];
    atomicAdd(num_selected, block_counts[0]);
    atomic_max_float_nonnegative(main_amax, block_amax[0]);
  }
}

__global__ void prepare_hardcap_workspace_kernel(const int32_t* __restrict__ num_selected,
                                                 int32_t* __restrict__ overflow,
                                                 float* __restrict__ score_max,
                                                 int32_t* __restrict__ histogram,
                                                 int32_t max_selected) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t selected = num_selected[0];
  const int32_t do_overflow = selected > max_selected ? 1 : 0;
  if (idx == 0) {
    overflow[0] = do_overflow;
    if (do_overflow != 0) {
      score_max[0] = 0.0f;
    }
  }
  if (do_overflow != 0 && idx < kHardCapHistogramBins) {
    histogram[idx] = 0;
  }
}

template <typename scalar_t>
__global__ void hardcap_score_max_kernel(const scalar_t* __restrict__ input,
                                         const float* __restrict__ stats,
                                         const int32_t* __restrict__ overflow,
                                         float* __restrict__ score_max,
                                         int64_t total) {
  if (overflow[0] == 0) {
    return;
  }

  const float mean = stats[0];
  const float threshold = stats[1];
  const bool select_all = threshold == 0.0f;
  const bool select_none = threshold < 0.0f;
  float local_max = 0.0f;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = to_float(input[idx]);
    const float score = fabsf(value - mean);
    const bool selected = !select_none && (select_all || score >= threshold);
    if (selected) {
      local_max = fmaxf(local_max, score);
    }
  }

  extern __shared__ float shmem[];
  shmem[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shmem[threadIdx.x] = fmaxf(shmem[threadIdx.x], shmem[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomic_max_float_nonnegative(score_max, shmem[0]);
  }
}

template <typename scalar_t>
__global__ void hardcap_histogram_kernel(const scalar_t* __restrict__ input,
                                         const float* __restrict__ stats,
                                         const int32_t* __restrict__ overflow,
                                         const float* __restrict__ score_max,
                                         int32_t* __restrict__ histogram,
                                         int64_t total) {
  if (overflow[0] == 0) {
    return;
  }

  const float mean = stats[0];
  const float threshold = stats[1];
  const float max_score = score_max[0];
  if (!(threshold >= 0.0f) || !(max_score > threshold) || !isfinite(max_score)) {
    return;
  }

  const float inv_width = static_cast<float>(kHardCapHistogramBins) / (max_score - threshold);
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = to_float(input[idx]);
    const float score = fabsf(value - mean);
    if (score < threshold) {
      continue;
    }
    int bin = static_cast<int>((score - threshold) * inv_width);
    bin = max(0, min(kHardCapHistogramBins - 1, bin));
    atomicAdd(histogram + bin, 1);
  }
}

__global__ void hardcap_update_strict_threshold_kernel(
    const int32_t* __restrict__ histogram,
    float* __restrict__ stats,
    const float* __restrict__ score_max,
    const int32_t* __restrict__ overflow,
    float* __restrict__ main_amax,
    int32_t* __restrict__ num_selected,
    int32_t max_selected) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || overflow[0] == 0) {
    return;
  }

  const float threshold = stats[1];
  const float max_score = score_max[0];
  if (max_selected <= 0 || !(threshold >= 0.0f) || !(max_score > threshold) ||
      !isfinite(max_score)) {
    stats[1] = nextafterf(fmaxf(max_score, threshold), INFINITY);
    main_amax[0] = 0.0f;
    num_selected[0] = 0;
    return;
  }

  const float width = (max_score - threshold) / static_cast<float>(kHardCapHistogramBins);
  int32_t kept_above = 0;
  for (int bin = kHardCapHistogramBins - 1; bin >= 0; --bin) {
    const int32_t count = histogram[bin];
    if (kept_above + count > max_selected) {
      if (bin == kHardCapHistogramBins - 1 && kept_above == 0) {
        stats[1] = nextafterf(max_score, INFINITY);
      } else {
        stats[1] = threshold + static_cast<float>(bin + 1) * width;
      }
      main_amax[0] = 0.0f;
      num_selected[0] = 0;
      return;
    }
    kept_above += count;
  }

  main_amax[0] = 0.0f;
  num_selected[0] = 0;
}

__global__ void reset_adaptive_workspace_kernel(float* __restrict__ stats_amax,
                                                int32_t* __restrict__ num_selected,
                                                int32_t* __restrict__ overflow) {
  const int idx = threadIdx.x;
  if (idx < 10) {
    stats_amax[idx] = 0.0f;
  }
  if (idx == 0) {
    num_selected[0] = 0;
    overflow[0] = 0;
  }
}

void launch_build_loghist(int64_t hist_bins,
                          const uint16_t* input_bits,
                          int32_t* log_hist,
                          int64_t total,
                          int min_exp,
                          cudaStream_t stream) {
  const int blocks =
      static_cast<int>(std::min<int64_t>(1024, (total + kLogHistThreads - 1) / kLogHistThreads));
  if (hist_bins == 64) {
    build_loghist_kernel<64><<<blocks, kLogHistThreads, 0, stream>>>(
        input_bits, log_hist, total, min_exp);
  } else if (hist_bins == 128) {
    build_loghist_kernel<128><<<blocks, kLogHistThreads, 0, stream>>>(
        input_bits, log_hist, total, min_exp);
  } else if (hist_bins == 256) {
    build_loghist_kernel<256><<<blocks, kLogHistThreads, 0, stream>>>(
        input_bits, log_hist, total, min_exp);
  } else {
    TORCH_CHECK(false, "hist_bins must be one of 64, 128, 256");
  }
}

void launch_choose_loghist_boundary(int64_t hist_bins,
                                    const int32_t* log_hist,
                                    int64_t* log_params,
                                    float* stats,
                                    double ratio,
                                    int64_t total,
                                    cudaStream_t stream) {
  if (hist_bins == 64) {
    choose_loghist_boundary_kernel<64><<<1, 1, 0, stream>>>(
        log_hist, log_params, stats, ratio, total);
  } else if (hist_bins == 128) {
    choose_loghist_boundary_kernel<128><<<1, 1, 0, stream>>>(
        log_hist, log_params, stats, ratio, total);
  } else if (hist_bins == 256) {
    choose_loghist_boundary_kernel<256><<<1, 1, 0, stream>>>(
        log_hist, log_params, stats, ratio, total);
  } else {
    TORCH_CHECK(false, "hist_bins must be one of 64, 128, 256");
  }
}

void launch_row_count_loghist(int64_t hist_bins,
                              const c10::BFloat16* input,
                              const uint16_t* input_bits,
                              int32_t* row_counts,
                              uint16_t* selection_masks,
                              float* main_amax,
                              int32_t* num_selected,
                              const int64_t* log_params,
                              int64_t rows,
                              int64_t cols,
                              int min_exp,
                              uint32_t seed,
                              int stats_threads,
                              cudaStream_t stream) {
  if (hist_bins == 64) {
    row_count_loghist_main_amax_total_kernel<c10::BFloat16, 64>
        <<<static_cast<int>(rows), stats_threads, 0, stream>>>(
            input,
            input_bits,
            row_counts,
            selection_masks,
            main_amax,
            num_selected,
            log_params,
            rows,
            cols,
            min_exp,
            seed);
  } else if (hist_bins == 128) {
    row_count_loghist_main_amax_total_kernel<c10::BFloat16, 128>
        <<<static_cast<int>(rows), stats_threads, 0, stream>>>(
            input,
            input_bits,
            row_counts,
            selection_masks,
            main_amax,
            num_selected,
            log_params,
            rows,
            cols,
            min_exp,
            seed);
  } else if (hist_bins == 256) {
    row_count_loghist_main_amax_total_kernel<c10::BFloat16, 256>
        <<<static_cast<int>(rows), stats_threads, 0, stream>>>(
            input,
            input_bits,
            row_counts,
            selection_masks,
            main_amax,
            num_selected,
            log_params,
            rows,
            cols,
            min_exp,
            seed);
  } else {
    TORCH_CHECK(false, "hist_bins must be one of 64, 128, 256");
  }
}

template <typename scalar_t, bool EMIT_DENSE>
__global__ void fill_csr_rowwise_quant_fast_kernel(const scalar_t* __restrict__ input,
                                                   int32_t* __restrict__ row_offsets,
                                                   const uint16_t* __restrict__ selection_masks,
                                                   int32_t* __restrict__ flat_indices,
                                                   scalar_t* __restrict__ outlier_values,
                                                   int16_t* __restrict__ outlier_cols,
                                                   int32_t* __restrict__ num_selected,
                                                   int32_t* __restrict__ overflow,
                                                   scalar_t* __restrict__ dense_main,
                                                   const float* __restrict__ main_amax,
                                                   uint8_t* __restrict__ rowwise_data,
                                                   uint8_t* __restrict__ rowwise_scale,
                                                   int64_t rows,
                                                   int64_t cols,
                                                   int64_t scale_stride,
                                                   int32_t capacity) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int cols_i = static_cast<int>(cols);
  const int blocks_per_row = cols_i / 16;
  const int rowwise_stride = cols_i / 2;
  const int scale_stride_i = static_cast<int>(scale_stride);
  const float s_enc = compute_nvfp4_global_encode_scale(main_amax[0]);

  __shared__ int32_t row_write;
  if (threadIdx.x == 0) {
    row_write = 0;
  }
  __syncthreads();

  const int32_t row_start = row_offsets[row];
  const int32_t selected_total = num_selected[0];
  if (row == 0 && threadIdx.x == 0) {
    row_offsets[rows] = selected_total;
    overflow[0] = selected_total > capacity ? 1 : 0;
  }
  const int32_t row_end = row + 1 == rows ? selected_total : row_offsets[row + 1];
  const int32_t row_quota = row_end - row_start;
  for (int block_col = threadIdx.x; block_col < blocks_per_row; block_col += blockDim.x) {
    const int col0 = block_col * 16;
    const int flat0 = row * cols_i + col0;
    float values[16];
    float block_amax = 0.0f;
    const uint16_t selected_mask = selection_masks[row * blocks_per_row + block_col];
    if (selected_mask == 0) {
      Vec<scalar_t, 8> raw_waves[2];
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        Vec<scalar_t, 8> in_wave;
        in_wave.load_from(&input[flat0 + w * 8]);
        if constexpr (EMIT_DENSE) {
          raw_waves[w] = in_wave;
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = w * 8 + e;
          const float raw = to_float(in_wave.elt[e]);
          values[i] = raw;
          block_amax = fmaxf(block_amax, fabsf(raw));
        }
      }
      if constexpr (EMIT_DENSE) {
        raw_waves[0].store_to(&dense_main[flat0]);
        raw_waves[1].store_to(&dense_main[flat0 + 8]);
      }
    } else {
      Vec<scalar_t, 8> dense_waves[2];
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        Vec<scalar_t, 8> in_wave;
        in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = w * 8 + e;
          const scalar_t raw_scalar = in_wave.elt[e];
          const float raw = to_float(raw_scalar);
          const bool selected = (selected_mask & (1u << i)) != 0;
          if (selected) {
            const int32_t row_pos = atomicAdd(&row_write, 1);
            if (row_pos < row_quota) {
              const int32_t out_pos = row_start + row_pos;
              if (out_pos < capacity) {
                flat_indices[out_pos] = static_cast<int32_t>(flat0 + i);
                outlier_values[out_pos] = raw_scalar;
                outlier_cols[out_pos] = static_cast<int16_t>(col0 + i);
              }
            }
          }
          const float dense_value = selected ? 0.0f : raw;
          values[i] = dense_value;
          block_amax = fmaxf(block_amax, fabsf(dense_value));
          if constexpr (EMIT_DENSE) {
            dense_waves[w].elt[e] = selected ? static_cast<scalar_t>(0) : raw_scalar;
          }
        }
      }
      if constexpr (EMIT_DENSE) {
        dense_waves[0].store_to(&dense_main[flat0]);
        dense_waves[1].store_to(&dense_main[flat0 + 8]);
      }
    }

    // Match TransformerEngine's FP32 operation order exactly.  Algebraically
    // equivalent regroupings can land on opposite sides of E2M1 halfway
    // points after rounding.
    const float scale_value = (block_amax / 6.0f) * s_enc;
    const __nv_fp8_e4m3 scale_e4m3(scale_value);
    rowwise_scale[row * scale_stride_i + block_col] = scale_e4m3.__x;
    const float scale_e4m3_float = static_cast<float>(scale_e4m3);
    const float global_decode_scale = 1.0f / s_enc;
    const float block_scale_inverse = scale_e4m3_float > 0.0f
        ? fminf(1.0f / (scale_e4m3_float * global_decode_scale), FLT_MAX)
        : FLT_MAX;

    uint32_t packed_lo = 0;
    uint32_t packed_hi = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int base = 4 * p;
      const uint16_t packed = fp4x4_e2m1_rn_ordered(
          values[base] * block_scale_inverse,
          values[base + 1] * block_scale_inverse,
          values[base + 2] * block_scale_inverse,
          values[base + 3] * block_scale_inverse);
      if (p < 2) {
        packed_lo |= static_cast<uint32_t>(packed) << (16 * p);
      } else {
        packed_hi |= static_cast<uint32_t>(packed) << (16 * (p - 2));
      }
    }
    uint32_t* out = reinterpret_cast<uint32_t*>(rowwise_data + row * rowwise_stride + col0 / 2);
    out[0] = packed_lo;
    out[1] = packed_hi;
  }
#endif
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void fill_csr_rowwise_quant_prefix_kernel(const scalar_t* __restrict__ input,
                                                     int32_t* __restrict__ row_offsets,
                                                     const uint16_t* __restrict__ selection_masks,
                                                     int32_t* __restrict__ flat_indices,
                                                     scalar_t* __restrict__ outlier_values,
                                                     int16_t* __restrict__ outlier_cols,
                                                     int64_t* __restrict__ packed_records,
                                                     const int32_t* __restrict__ direct_light_offsets,
                                                     const int32_t* __restrict__ direct_heavy_offsets,
                                                     const int32_t* __restrict__ direct_light_counts,
                                                     const int32_t* __restrict__ direct_heavy_counts,
                                                     int32_t* __restrict__ direct_light_cols,
                                                     uint16_t* __restrict__ direct_light_values,
                                                     int32_t* __restrict__ direct_light_flat_indices,
                                                     int32_t* __restrict__ direct_light_entry_records,
                                                     int32_t* __restrict__ direct_heavy_cols,
                                                     uint16_t* __restrict__ direct_heavy_values,
                                                     int32_t* __restrict__ direct_heavy_flat_indices,
                                                     const int32_t* __restrict__ num_selected,
                                                     int32_t* __restrict__ overflow,
                                                     const float* __restrict__ main_amax,
                                                     uint8_t* __restrict__ rowwise_data,
                                                     uint8_t* __restrict__ rowwise_scale,
                                                     int64_t rows,
                                                     int64_t cols,
                                                     int64_t scale_stride,
                                                     int32_t capacity) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  if (row == 0 && threadIdx.x == 0) {
    const int32_t selected_total = num_selected[0];
    row_offsets[rows] = selected_total;
    overflow[0] = selected_total > capacity ? 1 : 0;
  }

  const int cols_i = static_cast<int>(cols);
  const int blocks_per_row = cols_i / 16;
  const int rowwise_stride = cols_i / 2;
  const int scale_stride_i = static_cast<int>(scale_stride);
  const float s_enc = compute_nvfp4_global_encode_scale(main_amax[0]);
  const int32_t row_start = row_offsets[row];

  int32_t local_selected_count = 0;
  for (int block_col = threadIdx.x; block_col < blocks_per_row; block_col += BLOCK_THREADS) {
    const uint16_t selected_mask = selection_masks[row * blocks_per_row + block_col];
    local_selected_count += __popc(static_cast<unsigned>(selected_mask));
  }

  using BlockScan = cub::BlockScan<int32_t, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  int32_t thread_write_base = 0;
  BlockScan(scan_storage).ExclusiveSum(local_selected_count, thread_write_base);

  int32_t local_write = 0;
  for (int block_col = threadIdx.x; block_col < blocks_per_row; block_col += BLOCK_THREADS) {
    const int col0 = block_col * 16;
    const int flat0 = row * cols_i + col0;
    float values[16];
    float block_amax = 0.0f;
    const uint16_t selected_mask = selection_masks[row * blocks_per_row + block_col];
    if (selected_mask == 0) {
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        Vec<scalar_t, 8> in_wave;
        in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = w * 8 + e;
          const float raw = to_float(in_wave.elt[e]);
          values[i] = raw;
          block_amax = fmaxf(block_amax, fabsf(raw));
        }
      }
    } else {
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        Vec<scalar_t, 8> in_wave;
        in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = w * 8 + e;
          const scalar_t raw_scalar = in_wave.elt[e];
          const float raw = to_float(raw_scalar);
          const bool selected = (selected_mask & (1u << i)) != 0;
          if (selected) {
            const int32_t row_pos = thread_write_base + local_write;
            const int32_t out_pos = row_start + row_pos;
            ++local_write;
            if (out_pos < capacity) {
              if (flat_indices != nullptr) {
                flat_indices[out_pos] = static_cast<int32_t>(flat0 + i);
              }
              if (outlier_values != nullptr) {
                outlier_values[out_pos] = raw_scalar;
              }
              if (outlier_cols != nullptr) {
                outlier_cols[out_pos] = static_cast<int16_t>(col0 + i);
              }
              if (packed_records != nullptr) {
                const uint16_t value_bits = *reinterpret_cast<const uint16_t*>(&raw_scalar);
                const uint64_t record =
                    static_cast<uint64_t>(static_cast<uint32_t>(flat0 + i)) |
                    (static_cast<uint64_t>(value_bits) << 32);
                packed_records[out_pos] = static_cast<int64_t>(record);
              }
              if (direct_light_counts != nullptr) {
                const bool direct_light = direct_light_counts[row] > 0;
                const bool direct_heavy = !direct_light && direct_heavy_counts[row] > 0;
                if (direct_light || direct_heavy) {
                  const uint16_t col = static_cast<uint16_t>(col0 + i);
                  const uint16_t value_bits = *reinterpret_cast<const uint16_t*>(&raw_scalar);
                  if (direct_light) {
                    const int32_t dst = direct_light_offsets[row] + row_pos;
                    direct_light_cols[dst] = static_cast<int32_t>(col);
                    direct_light_values[dst] = value_bits;
                    direct_light_flat_indices[dst] = flat0 + i;
                    direct_light_entry_records[dst] =
                        static_cast<int32_t>((static_cast<uint32_t>(col) << 16) |
                                             static_cast<uint32_t>(value_bits));
                  } else {
                    const int32_t dst = direct_heavy_offsets[row] + row_pos;
                    direct_heavy_cols[dst] = static_cast<int32_t>(col);
                    direct_heavy_values[dst] = value_bits;
                    direct_heavy_flat_indices[dst] = flat0 + i;
                  }
                }
              }
            }
          }
          const float dense_value = selected ? 0.0f : raw;
          values[i] = dense_value;
          block_amax = fmaxf(block_amax, fabsf(dense_value));
        }
      }
    }

    const float scale_value = (block_amax / 6.0f) * s_enc;
    const __nv_fp8_e4m3 scale_e4m3(scale_value);
    rowwise_scale[row * scale_stride_i + block_col] = scale_e4m3.__x;
    const float scale_e4m3_float = static_cast<float>(scale_e4m3);
    const float global_decode_scale = 1.0f / s_enc;
    const float block_scale_inverse = scale_e4m3_float > 0.0f
        ? fminf(1.0f / (scale_e4m3_float * global_decode_scale), FLT_MAX)
        : FLT_MAX;

    uint32_t packed_lo = 0;
    uint32_t packed_hi = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int base = 4 * p;
      const uint16_t packed = fp4x4_e2m1_rn_ordered(
          values[base] * block_scale_inverse,
          values[base + 1] * block_scale_inverse,
          values[base + 2] * block_scale_inverse,
          values[base + 3] * block_scale_inverse);
      if (p < 2) {
        packed_lo |= static_cast<uint32_t>(packed) << (16 * p);
      } else {
        packed_hi |= static_cast<uint32_t>(packed) << (16 * (p - 2));
      }
    }
    uint32_t* out = reinterpret_cast<uint32_t*>(rowwise_data + row * rowwise_stride + col0 / 2);
    out[0] = packed_lo;
    out[1] = packed_hi;
  }
#endif
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void fill_csr_rowwise_quant_prefix_nomask_kernel(
    const scalar_t* __restrict__ input,
    int32_t* __restrict__ row_offsets,
    int32_t* __restrict__ flat_indices,
    scalar_t* __restrict__ outlier_values,
    int16_t* __restrict__ outlier_cols,
    const int32_t* __restrict__ num_selected,
    int32_t* __restrict__ overflow,
    const float* __restrict__ main_amax,
    uint8_t* __restrict__ rowwise_data,
    uint8_t* __restrict__ rowwise_scale,
    const float* __restrict__ stats,
    int64_t rows,
    int64_t cols,
    int64_t scale_stride,
    int32_t capacity) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  if (row == 0 && threadIdx.x == 0) {
    const int32_t selected_total = num_selected[0];
    row_offsets[rows] = selected_total;
    overflow[0] = selected_total > capacity ? 1 : 0;
  }

  const int cols_i = static_cast<int>(cols);
  const int blocks_per_row = cols_i / 16;
  const int rowwise_stride = cols_i / 2;
  const int scale_stride_i = static_cast<int>(scale_stride);
  const float s_enc = compute_nvfp4_global_encode_scale(main_amax[0]);
  const int32_t row_start = row_offsets[row];
  const float mean = stats[0];
  const float threshold = stats[1];
  const bool select_all = threshold == 0.0f;
  const bool select_none = threshold < 0.0f;

  if (blocks_per_row <= BLOCK_THREADS) {
    const bool has_block = threadIdx.x < blocks_per_row;
    const int block_col = threadIdx.x;
    const int col0 = block_col * 16;
    const int flat0 = row * cols_i + col0;
    scalar_t raw_values[16];
    float values[16];
    uint16_t selected_mask = 0;
    float block_amax = 0.0f;

    if (has_block) {
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        Vec<scalar_t, 8> in_wave;
        in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = w * 8 + e;
          const scalar_t raw_scalar = in_wave.elt[e];
          const float raw = to_float(raw_scalar);
          const bool selected = !select_none && (select_all || fabsf(raw - mean) >= threshold);
          selected_mask |= static_cast<uint16_t>(selected ? (1u << i) : 0u);
          raw_values[i] = raw_scalar;
          const float dense_value = selected ? 0.0f : raw;
          values[i] = dense_value;
          block_amax = fmaxf(block_amax, fabsf(dense_value));
        }
      }
    }

    const int32_t local_selected_count =
        has_block ? __popc(static_cast<unsigned>(selected_mask)) : 0;
    using BlockScanFast = cub::BlockScan<int32_t, BLOCK_THREADS>;
    __shared__ typename BlockScanFast::TempStorage fast_scan_storage;
    int32_t thread_write_base = 0;
    BlockScanFast(fast_scan_storage).ExclusiveSum(local_selected_count, thread_write_base);

    if (!has_block) {
      return;
    }

    int32_t local_write = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if ((selected_mask & (1u << i)) == 0) {
        continue;
      }
      const int32_t row_pos = thread_write_base + local_write;
      const int32_t out_pos = row_start + row_pos;
      ++local_write;
      if (out_pos < capacity) {
        flat_indices[out_pos] = static_cast<int32_t>(flat0 + i);
        outlier_values[out_pos] = raw_values[i];
        outlier_cols[out_pos] = static_cast<int16_t>(col0 + i);
      }
    }

    const float scale_value = (block_amax / 6.0f) * s_enc;
    const __nv_fp8_e4m3 scale_e4m3(scale_value);
    rowwise_scale[row * scale_stride_i + block_col] = scale_e4m3.__x;
    const float scale_e4m3_float = static_cast<float>(scale_e4m3);
    const float global_decode_scale = 1.0f / s_enc;
    const float block_scale_inverse = scale_e4m3_float > 0.0f
        ? fminf(1.0f / (scale_e4m3_float * global_decode_scale), FLT_MAX)
        : FLT_MAX;

    uint32_t packed_lo = 0;
    uint32_t packed_hi = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int base = 4 * p;
      const uint16_t packed = fp4x4_e2m1_rn_ordered(
          values[base] * block_scale_inverse,
          values[base + 1] * block_scale_inverse,
          values[base + 2] * block_scale_inverse,
          values[base + 3] * block_scale_inverse);
      if (p < 2) {
        packed_lo |= static_cast<uint32_t>(packed) << (16 * p);
      } else {
        packed_hi |= static_cast<uint32_t>(packed) << (16 * (p - 2));
      }
    }
    uint32_t* out = reinterpret_cast<uint32_t*>(rowwise_data + row * rowwise_stride + col0 / 2);
    out[0] = packed_lo;
    out[1] = packed_hi;
    return;
  }

  int32_t local_selected_count = 0;
  for (int block_col = threadIdx.x; block_col < blocks_per_row; block_col += BLOCK_THREADS) {
    const int col0 = block_col * 16;
    const int flat0 = row * cols_i + col0;
    uint16_t selected_mask = 0;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      Vec<scalar_t, 8> in_wave;
      in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = w * 8 + e;
        const float raw = to_float(in_wave.elt[e]);
        const bool selected = !select_none && (select_all || fabsf(raw - mean) >= threshold);
        selected_mask |= static_cast<uint16_t>(selected ? (1u << i) : 0u);
      }
    }
    local_selected_count += __popc(static_cast<unsigned>(selected_mask));
  }

  using BlockScan = cub::BlockScan<int32_t, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  int32_t thread_write_base = 0;
  BlockScan(scan_storage).ExclusiveSum(local_selected_count, thread_write_base);

  int32_t local_write = 0;
  for (int block_col = threadIdx.x; block_col < blocks_per_row; block_col += BLOCK_THREADS) {
    const int col0 = block_col * 16;
    const int flat0 = row * cols_i + col0;
    float values[16];
    float block_amax = 0.0f;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      Vec<scalar_t, 8> in_wave;
      in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = w * 8 + e;
        const scalar_t raw_scalar = in_wave.elt[e];
        const float raw = to_float(raw_scalar);
        const bool selected = !select_none && (select_all || fabsf(raw - mean) >= threshold);
        if (selected) {
          const int32_t row_pos = thread_write_base + local_write;
          const int32_t out_pos = row_start + row_pos;
          ++local_write;
          if (out_pos < capacity) {
            flat_indices[out_pos] = static_cast<int32_t>(flat0 + i);
            outlier_values[out_pos] = raw_scalar;
            outlier_cols[out_pos] = static_cast<int16_t>(col0 + i);
          }
        }
        const float dense_value = selected ? 0.0f : raw;
        values[i] = dense_value;
        block_amax = fmaxf(block_amax, fabsf(dense_value));
      }
    }

    const float scale_value = (block_amax / 6.0f) * s_enc;
    const __nv_fp8_e4m3 scale_e4m3(scale_value);
    rowwise_scale[row * scale_stride_i + block_col] = scale_e4m3.__x;
    const float scale_e4m3_float = static_cast<float>(scale_e4m3);
    const float global_decode_scale = 1.0f / s_enc;
    const float block_scale_inverse = scale_e4m3_float > 0.0f
        ? fminf(1.0f / (scale_e4m3_float * global_decode_scale), FLT_MAX)
        : FLT_MAX;

    uint32_t packed_lo = 0;
    uint32_t packed_hi = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int base = 4 * p;
      const uint16_t packed = fp4x4_e2m1_rn_ordered(
          values[base] * block_scale_inverse,
          values[base + 1] * block_scale_inverse,
          values[base + 2] * block_scale_inverse,
          values[base + 3] * block_scale_inverse);
      if (p < 2) {
        packed_lo |= static_cast<uint32_t>(packed) << (16 * p);
      } else {
        packed_hi |= static_cast<uint32_t>(packed) << (16 * (p - 2));
      }
    }
    uint32_t* out = reinterpret_cast<uint32_t*>(rowwise_data + row * rowwise_stride + col0 / 2);
    out[0] = packed_lo;
    out[1] = packed_hi;
  }
#endif
}

__global__ void reset_padded_spill_kernel(int32_t* __restrict__ overflow,
                                          int32_t* __restrict__ overflow_count) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    overflow[0] = 0;
    overflow_count[0] = 0;
  }
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void fill_padded_rowwise_quant_kernel(const scalar_t* __restrict__ input,
                                                 const int32_t* __restrict__ row_counts,
                                                 const uint16_t* __restrict__ selection_masks,
                                                 scalar_t* __restrict__ padded_values,
                                                 int16_t* __restrict__ padded_cols,
                                                 int32_t* __restrict__ overflow_rows,
                                                 int16_t* __restrict__ overflow_cols,
                                                 scalar_t* __restrict__ overflow_values,
                                                 int32_t* __restrict__ overflow_count,
                                                 int32_t* __restrict__ overflow,
                                                 const float* __restrict__ main_amax,
                                                 uint8_t* __restrict__ rowwise_data,
                                                 uint8_t* __restrict__ rowwise_scale,
                                                 int64_t rows,
                                                 int64_t cols,
                                                 int64_t scale_stride,
                                                 int32_t max_per_row,
                                                 int32_t overflow_capacity) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const int cols_i = static_cast<int>(cols);
  const int blocks_per_row = cols_i / 16;
  const int rowwise_stride = cols_i / 2;
  const int scale_stride_i = static_cast<int>(scale_stride);
  const float s_enc = compute_nvfp4_global_encode_scale(main_amax[0]);

  int32_t local_selected_count = 0;
  for (int block_col = threadIdx.x; block_col < blocks_per_row; block_col += BLOCK_THREADS) {
    const uint16_t selected_mask = selection_masks[row * blocks_per_row + block_col];
    local_selected_count += __popc(static_cast<unsigned>(selected_mask));
  }

  using BlockScan = cub::BlockScan<int32_t, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  int32_t thread_write_base = 0;
  BlockScan(scan_storage).ExclusiveSum(local_selected_count, thread_write_base);

  int32_t local_write = 0;
  for (int block_col = threadIdx.x; block_col < blocks_per_row; block_col += BLOCK_THREADS) {
    const int col0 = block_col * 16;
    const int flat0 = row * cols_i + col0;
    float values[16];
    float block_amax = 0.0f;
    const uint16_t selected_mask = selection_masks[row * blocks_per_row + block_col];
    if (selected_mask == 0) {
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        Vec<scalar_t, 8> in_wave;
        in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = w * 8 + e;
          const float raw = to_float(in_wave.elt[e]);
          values[i] = raw;
          block_amax = fmaxf(block_amax, fabsf(raw));
        }
      }
    } else {
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        Vec<scalar_t, 8> in_wave;
        in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = w * 8 + e;
          const scalar_t raw_scalar = in_wave.elt[e];
          const float raw = to_float(raw_scalar);
          const bool selected = (selected_mask & (1u << i)) != 0;
          if (selected) {
            const int32_t row_pos = thread_write_base + local_write;
            ++local_write;
            if (row_pos < max_per_row) {
              const int64_t out_pos = row * static_cast<int64_t>(max_per_row) + row_pos;
              padded_values[out_pos] = raw_scalar;
              padded_cols[out_pos] = static_cast<int16_t>(col0 + i);
            } else {
              const int32_t spill_pos = atomicAdd(overflow_count, 1);
              if (spill_pos < overflow_capacity) {
                overflow_rows[spill_pos] = static_cast<int32_t>(row);
                overflow_cols[spill_pos] = static_cast<int16_t>(col0 + i);
                overflow_values[spill_pos] = raw_scalar;
              } else {
                overflow[0] = 1;
              }
            }
          }
          const float dense_value = selected ? 0.0f : raw;
          values[i] = dense_value;
          block_amax = fmaxf(block_amax, fabsf(dense_value));
        }
      }
    }

    const float scale_value = (block_amax / 6.0f) * s_enc;
    const __nv_fp8_e4m3 scale_e4m3(scale_value);
    rowwise_scale[row * scale_stride_i + block_col] = scale_e4m3.__x;
    const float scale_e4m3_float = static_cast<float>(scale_e4m3);
    const float global_decode_scale = 1.0f / s_enc;
    const float block_scale_inverse = scale_e4m3_float > 0.0f
        ? fminf(1.0f / (scale_e4m3_float * global_decode_scale), FLT_MAX)
        : FLT_MAX;

    uint32_t packed_lo = 0;
    uint32_t packed_hi = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int base = 4 * p;
      const uint16_t packed = fp4x4_e2m1_rn_ordered(
          values[base] * block_scale_inverse,
          values[base + 1] * block_scale_inverse,
          values[base + 2] * block_scale_inverse,
          values[base + 3] * block_scale_inverse);
      if (p < 2) {
        packed_lo |= static_cast<uint32_t>(packed) << (16 * p);
      } else {
        packed_hi |= static_cast<uint32_t>(packed) << (16 * (p - 2));
      }
    }
    uint32_t* out = reinterpret_cast<uint32_t*>(rowwise_data + row * rowwise_stride + col0 / 2);
    out[0] = packed_lo;
    out[1] = packed_hi;
  }

  if (threadIdx.x == 0 && row_counts[row] > max_per_row && overflow_capacity <= 0) {
    overflow[0] = 1;
  }
#endif
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void fill_padded_rowwise_quant_nomask_kernel(
    const scalar_t* __restrict__ input,
    const int32_t* __restrict__ row_counts,
    scalar_t* __restrict__ padded_values,
    int16_t* __restrict__ padded_cols,
    int32_t* __restrict__ overflow_rows,
    int16_t* __restrict__ overflow_cols,
    scalar_t* __restrict__ overflow_values,
    int32_t* __restrict__ overflow_count,
    int32_t* __restrict__ overflow,
    const float* __restrict__ main_amax,
    const float* __restrict__ stats,
    uint8_t* __restrict__ rowwise_data,
    uint8_t* __restrict__ rowwise_scale,
    int64_t rows,
    int64_t cols,
    int64_t scale_stride,
    int32_t max_per_row,
    int32_t overflow_capacity) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const int cols_i = static_cast<int>(cols);
  const int blocks_per_row = cols_i / 16;
  const int rowwise_stride = cols_i / 2;
  const int scale_stride_i = static_cast<int>(scale_stride);
  const float s_enc = compute_nvfp4_global_encode_scale(main_amax[0]);
  const float mean = stats[0];
  const float threshold = stats[1];
  const bool select_all = threshold == 0.0f;
  const bool select_none = threshold < 0.0f;

  __shared__ int32_t row_write;
  if (threadIdx.x == 0) {
    row_write = 0;
  }
  __syncthreads();

  for (int block_col = threadIdx.x; block_col < blocks_per_row; block_col += BLOCK_THREADS) {
    const int col0 = block_col * 16;
    const int flat0 = row * cols_i + col0;
    float values[16];
    float block_amax = 0.0f;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      Vec<scalar_t, 8> in_wave;
      in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = w * 8 + e;
        const scalar_t raw_scalar = in_wave.elt[e];
        const float raw = to_float(raw_scalar);
        const bool selected = !select_none && (select_all || fabsf(raw - mean) >= threshold);
        if (selected) {
          const int32_t row_pos = atomicAdd(&row_write, 1);
          if (row_pos < max_per_row) {
            const int64_t out_pos = row * static_cast<int64_t>(max_per_row) + row_pos;
            padded_values[out_pos] = raw_scalar;
            padded_cols[out_pos] = static_cast<int16_t>(col0 + i);
          } else {
            const int32_t spill_pos = atomicAdd(overflow_count, 1);
            if (spill_pos < overflow_capacity) {
              overflow_rows[spill_pos] = static_cast<int32_t>(row);
              overflow_cols[spill_pos] = static_cast<int16_t>(col0 + i);
              overflow_values[spill_pos] = raw_scalar;
            } else {
              overflow[0] = 1;
            }
          }
        }
        const float dense_value = selected ? 0.0f : raw;
        values[i] = dense_value;
        block_amax = fmaxf(block_amax, fabsf(dense_value));
      }
    }

    const float scale_value = (block_amax / 6.0f) * s_enc;
    const __nv_fp8_e4m3 scale_e4m3(scale_value);
    rowwise_scale[row * scale_stride_i + block_col] = scale_e4m3.__x;
    const float scale_e4m3_float = static_cast<float>(scale_e4m3);
    const float global_decode_scale = 1.0f / s_enc;
    const float block_scale_inverse = scale_e4m3_float > 0.0f
        ? fminf(1.0f / (scale_e4m3_float * global_decode_scale), FLT_MAX)
        : FLT_MAX;

    uint32_t packed_lo = 0;
    uint32_t packed_hi = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int base = 4 * p;
      const uint16_t packed = fp4x4_e2m1_rn_ordered(
          values[base] * block_scale_inverse,
          values[base + 1] * block_scale_inverse,
          values[base + 2] * block_scale_inverse,
          values[base + 3] * block_scale_inverse);
      if (p < 2) {
        packed_lo |= static_cast<uint32_t>(packed) << (16 * p);
      } else {
        packed_hi |= static_cast<uint32_t>(packed) << (16 * (p - 2));
      }
    }
    uint32_t* out = reinterpret_cast<uint32_t*>(rowwise_data + row * rowwise_stride + col0 / 2);
    out[0] = packed_lo;
    out[1] = packed_hi;
  }

  if (threadIdx.x == 0 && row_counts[row] > max_per_row && overflow_capacity <= 0) {
    overflow[0] = 1;
  }
#endif
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void refill_csr_payload_prefix_kernel(const scalar_t* __restrict__ input,
                                                 const int32_t* __restrict__ row_offsets,
                                                 const uint16_t* __restrict__ selection_masks,
                                                 int32_t* __restrict__ flat_indices,
                                                 scalar_t* __restrict__ outlier_values,
                                                 int16_t* __restrict__ outlier_cols,
                                                 int64_t rows,
                                                 int64_t cols,
                                                 int32_t capacity,
                                                 int32_t* __restrict__ overflow) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  if (row == 0 && threadIdx.x == 0) {
    overflow[0] = row_offsets[rows] > capacity ? 1 : 0;
  }

  const int cols_i = static_cast<int>(cols);
  const int blocks_per_row = cols_i / 16;
  const int32_t row_start = row_offsets[row];

  int32_t local_selected_count = 0;
  for (int block_col = threadIdx.x; block_col < blocks_per_row; block_col += BLOCK_THREADS) {
    const uint16_t selected_mask = selection_masks[row * blocks_per_row + block_col];
    local_selected_count += __popc(static_cast<unsigned>(selected_mask));
  }

  using BlockScan = cub::BlockScan<int32_t, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  int32_t thread_write_base = 0;
  BlockScan(scan_storage).ExclusiveSum(local_selected_count, thread_write_base);

  int32_t local_write = 0;
  for (int block_col = threadIdx.x; block_col < blocks_per_row; block_col += BLOCK_THREADS) {
    const uint16_t selected_mask = selection_masks[row * blocks_per_row + block_col];
    if (selected_mask == 0) {
      continue;
    }
    const int col0 = block_col * 16;
    const int flat0 = row * cols_i + col0;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      Vec<scalar_t, 8> in_wave;
      in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = w * 8 + e;
        if ((selected_mask & (1u << i)) == 0) {
          continue;
        }
        const int32_t out_pos = row_start + thread_write_base + local_write;
        ++local_write;
        if (out_pos < capacity) {
          flat_indices[out_pos] = static_cast<int32_t>(flat0 + i);
          outlier_values[out_pos] = in_wave.elt[e];
          outlier_cols[out_pos] = static_cast<int16_t>(col0 + i);
        }
      }
    }
  }
#endif
}

std::vector<at::Tensor> adaptive_rowcol_quant_fast_impl(
    const at::Tensor& input,
    double base_ratio,
    double min_ratio,
    double max_ratio,
    double reference_heaviness,
    int64_t capacity,
    bool emit_dense_main,
    int64_t stats_threads_arg,
    int64_t fill_threads_arg,
    int64_t columnwise_source_arg,
    int64_t rht_random_sign_mask_t,
    bool overlap_columnwise,
    bool direct_nomask_arg,
    bool build_active_schedule,
    bool build_unsorted_active_rows,
    double threshold_sigma_override) {
  const auto rows = input.size(0);
  const auto cols = input.size(1);
  const int64_t total = rows * cols;
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "adaptive rowcol fast supports BF16 only");
  TORCH_CHECK(capacity >= 0 && capacity <= std::numeric_limits<int32_t>::max(),
              "capacity must fit int32");
  TORCH_CHECK(rows % 128 == 0, "rows must be divisible by 128");
  TORCH_CHECK(cols % 128 == 0, "cols must be divisible by 128");
  TORCH_CHECK(cols % 16 == 0, "cols must be divisible by 16");
  TORCH_CHECK(total <= std::numeric_limits<int32_t>::max(), "numel must fit int32");
  TORCH_CHECK(columnwise_source_arg == kColumnwiseDirect ||
                  columnwise_source_arg == kColumnwiseOutlierReuse,
              "columnwise_source must be direct or outlier_reuse");
  TORCH_CHECK(!direct_nomask_arg || columnwise_source_arg == kColumnwiseDirect,
              "direct_nomask requires columnwise_source=direct");
  TORCH_CHECK(!(build_active_schedule && build_unsorted_active_rows),
              "build_active_schedule and build_unsorted_active_rows are mutually exclusive");

  const bool masked_rht_outlier_reuse = columnwise_source_arg == kColumnwiseOutlierReuse;
  const bool need_dense_main = emit_dense_main;
  const bool direct_nomask = direct_nomask_arg && !need_dense_main;
  auto flat_indices = at::empty({capacity}, input.options().dtype(at::kInt));
  auto outlier_values = at::empty({capacity}, input.options());
  auto outlier_cols = at::empty({capacity}, input.options().dtype(at::kShort));
  auto row_counts = at::empty({rows}, input.options().dtype(at::kInt));
  auto row_offsets = at::empty({rows + 1}, input.options().dtype(at::kInt));
  auto selection_masks = direct_nomask
      ? at::empty({0}, input.options().dtype(at::kShort))
      : at::empty({rows, cols / 16}, input.options().dtype(at::kShort));
  auto num_selected = at::zeros({1}, input.options().dtype(at::kInt));
  auto overflow = at::empty({1}, input.options().dtype(at::kInt));
  auto stats_amax = at::zeros({10}, input.options().dtype(at::kFloat));
  auto stats = stats_amax.narrow(0, 0, 4);
  auto main_amax = stats_amax.narrow(0, 4, 1);
  auto raw_stats = stats_amax.narrow(0, 6, 4);

  const int64_t row_scale_outer = round_up_int64(rows, 128);
  const int64_t row_scale_inner = round_up_int64((cols + 15) / 16, 4);
  const int64_t column_scale_outer = round_up_int64(cols, 128);
  const int64_t column_scale_inner = round_up_int64((rows + 15) / 16, 4);
  auto rowwise_data = at::empty({rows, cols / 2}, input.options().dtype(at::kByte));
  auto rowwise_scale =
      at::empty({row_scale_outer, row_scale_inner}, input.options().dtype(at::kByte));
  auto dense_main = need_dense_main ? at::empty_like(input) : at::empty({0}, input.options());
  auto columnwise_data = at::empty({cols, rows / 2}, input.options().dtype(at::kByte));
  auto columnwise_scale =
      at::empty({column_scale_outer, column_scale_inner}, input.options().dtype(at::kByte));
  auto columnwise_amax = at::empty({1}, input.options().dtype(at::kFloat));

  if (total == 0) {
    row_offsets.zero_();
    num_selected.zero_();
    overflow.zero_();
    auto active_rows = at::empty({rows}, input.options().dtype(at::kInt));
    auto active_row_count = at::zeros({1}, input.options().dtype(at::kInt));
    return {flat_indices, outlier_values, outlier_cols, row_offsets, num_selected, overflow,
            rowwise_data, rowwise_scale, main_amax, stats, dense_main, columnwise_data,
            columnwise_scale, columnwise_amax, active_rows, active_row_count};
  }

  const auto main_stream = at::cuda::getDefaultCUDAStream(input.device().index());
  auto column_stream = at::cuda::getStreamFromPool(false, input.device().index());
  const cudaStream_t main_raw_stream = main_stream.stream();
  const cudaStream_t column_raw_stream = column_stream.stream();
  at::Tensor column_rht_keepalive;
  bool columnwise_launched_async = false;

  auto record_column_inputs = [&]() {
    record_tensor_on_stream(input, column_stream);
    record_tensor_on_stream(selection_masks, column_stream);
    record_tensor_on_stream(columnwise_data, column_stream);
    record_tensor_on_stream(columnwise_scale, column_stream);
    record_tensor_on_stream(columnwise_amax, column_stream);
  };

  if (overlap_columnwise && columnwise_source_arg == kColumnwiseDirect) {
    record_column_inputs();
    record_stream_then_wait(main_raw_stream, column_raw_stream);
    column_rht_keepalive = launch_custom_te_rht_columnwise_quant(input,
                                                                 selection_masks,
                                                                 columnwise_data,
                                                                 columnwise_scale,
                                                                 columnwise_amax,
                                                                 rows,
                                                                 cols,
                                                                 column_scale_inner,
                                                                 static_cast<int>(rht_random_sign_mask_t),
                                                                 false,
                                                                 column_raw_stream,
                                                                 column_stream);
    columnwise_launched_async = true;
  }

  const int reduce_threads = 256;
  const int reduce_blocks =
      static_cast<int>(std::min<int64_t>(1024, (total + reduce_threads - 1) / reduce_threads));
  const size_t adaptive_smem = 4 * reduce_threads * sizeof(float);
  sum_sumsq_absmax_kernel<c10::BFloat16>
      <<<reduce_blocks, reduce_threads, adaptive_smem, at::cuda::getDefaultCUDAStream()>>>(
          input.data_ptr<c10::BFloat16>(),
          raw_stats.data_ptr<float>(),
          total);
  finalize_adaptive_stats_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
      raw_stats.data_ptr<float>(),
      stats.data_ptr<float>(),
      total,
      static_cast<float>(base_ratio),
      static_cast<float>(min_ratio),
      static_cast<float>(max_ratio),
      static_cast<float>(reference_heaviness));
  if (threshold_sigma_override > 0.0) {
    override_threshold_sigma_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
        raw_stats.data_ptr<float>(),
        stats.data_ptr<float>(),
        total,
        static_cast<float>(threshold_sigma_override));
  }

  const int stats_threads = normalize_threads(stats_threads_arg, cols <= 4096 ? 128 : 256);
  if (direct_nomask) {
    row_count_main_amax_total_nomask_kernel<c10::BFloat16><<<
        static_cast<int>(rows),
        stats_threads,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            row_counts.data_ptr<int32_t>(),
            main_amax.data_ptr<float>(),
            num_selected.data_ptr<int32_t>(),
            stats.data_ptr<float>(),
            rows,
            cols);
  } else {
    row_count_main_amax_total_kernel<c10::BFloat16><<<
        static_cast<int>(rows),
        stats_threads,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            row_counts.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(selection_masks.data_ptr<int16_t>()),
            main_amax.data_ptr<float>(),
            num_selected.data_ptr<int32_t>(),
            stats.data_ptr<float>(),
            rows,
            cols);
  }

  if (overlap_columnwise && masked_rht_outlier_reuse) {
    record_column_inputs();
    record_stream_then_wait(main_raw_stream, column_raw_stream);
    column_rht_keepalive = launch_custom_te_rht_columnwise_quant(input,
                                                                 selection_masks,
                                                                 columnwise_data,
                                                                 columnwise_scale,
                                                                 columnwise_amax,
                                                                 rows,
                                                                 cols,
                                                                 column_scale_inner,
                                                                 static_cast<int>(rht_random_sign_mask_t),
                                                                 true,
                                                                 column_raw_stream,
                                                                 column_stream);
    columnwise_launched_async = true;
  }

  void* temp_storage = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      temp_storage,
      temp_bytes,
      row_counts.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      static_cast<int>(rows),
      at::cuda::getDefaultCUDAStream());
  auto temp = at::empty({static_cast<int64_t>(temp_bytes)}, input.options().dtype(at::kByte));
  cub::DeviceScan::ExclusiveSum(
      temp.data_ptr(),
      temp_bytes,
      row_counts.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      static_cast<int>(rows),
      at::cuda::getDefaultCUDAStream());

  const int fill_threads = normalize_threads(fill_threads_arg, cols <= 2048 ? 128 : (cols <= 4096 ? 256 : 512));
  if (need_dense_main) {
    fill_csr_rowwise_quant_fast_kernel<c10::BFloat16, true><<<
        static_cast<int>(rows),
        fill_threads,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            row_offsets.data_ptr<int32_t>(),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            flat_indices.data_ptr<int32_t>(),
            outlier_values.data_ptr<c10::BFloat16>(),
            outlier_cols.data_ptr<int16_t>(),
            num_selected.data_ptr<int32_t>(),
            overflow.data_ptr<int32_t>(),
            dense_main.data_ptr<c10::BFloat16>(),
            main_amax.data_ptr<float>(),
            rowwise_data.data_ptr<uint8_t>(),
            rowwise_scale.data_ptr<uint8_t>(),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(capacity));
  } else {
    if (direct_nomask && fill_threads == 64) {
      fill_csr_rowwise_quant_prefix_nomask_kernel<c10::BFloat16, 64><<<
          static_cast<int>(rows),
          64,
          0,
          at::cuda::getDefaultCUDAStream()>>>(
              input.data_ptr<c10::BFloat16>(),
              row_offsets.data_ptr<int32_t>(),
              flat_indices.data_ptr<int32_t>(),
              outlier_values.data_ptr<c10::BFloat16>(),
              outlier_cols.data_ptr<int16_t>(),
              num_selected.data_ptr<int32_t>(),
              overflow.data_ptr<int32_t>(),
              main_amax.data_ptr<float>(),
              rowwise_data.data_ptr<uint8_t>(),
              rowwise_scale.data_ptr<uint8_t>(),
              stats.data_ptr<float>(),
              rows,
              cols,
              row_scale_inner,
              static_cast<int32_t>(capacity));
    } else if (direct_nomask && fill_threads == 256) {
      fill_csr_rowwise_quant_prefix_nomask_kernel<c10::BFloat16, 256><<<
          static_cast<int>(rows),
          256,
          0,
          at::cuda::getDefaultCUDAStream()>>>(
              input.data_ptr<c10::BFloat16>(),
              row_offsets.data_ptr<int32_t>(),
              flat_indices.data_ptr<int32_t>(),
              outlier_values.data_ptr<c10::BFloat16>(),
              outlier_cols.data_ptr<int16_t>(),
              num_selected.data_ptr<int32_t>(),
              overflow.data_ptr<int32_t>(),
              main_amax.data_ptr<float>(),
              rowwise_data.data_ptr<uint8_t>(),
              rowwise_scale.data_ptr<uint8_t>(),
              stats.data_ptr<float>(),
              rows,
              cols,
              row_scale_inner,
              static_cast<int32_t>(capacity));
    } else if (direct_nomask && fill_threads == 512) {
      fill_csr_rowwise_quant_prefix_nomask_kernel<c10::BFloat16, 512><<<
          static_cast<int>(rows),
          512,
          0,
          at::cuda::getDefaultCUDAStream()>>>(
              input.data_ptr<c10::BFloat16>(),
              row_offsets.data_ptr<int32_t>(),
              flat_indices.data_ptr<int32_t>(),
              outlier_values.data_ptr<c10::BFloat16>(),
              outlier_cols.data_ptr<int16_t>(),
              num_selected.data_ptr<int32_t>(),
              overflow.data_ptr<int32_t>(),
              main_amax.data_ptr<float>(),
              rowwise_data.data_ptr<uint8_t>(),
              rowwise_scale.data_ptr<uint8_t>(),
              stats.data_ptr<float>(),
              rows,
              cols,
              row_scale_inner,
              static_cast<int32_t>(capacity));
    } else if (direct_nomask) {
      fill_csr_rowwise_quant_prefix_nomask_kernel<c10::BFloat16, 128><<<
          static_cast<int>(rows),
          128,
          0,
          at::cuda::getDefaultCUDAStream()>>>(
              input.data_ptr<c10::BFloat16>(),
              row_offsets.data_ptr<int32_t>(),
              flat_indices.data_ptr<int32_t>(),
              outlier_values.data_ptr<c10::BFloat16>(),
              outlier_cols.data_ptr<int16_t>(),
              num_selected.data_ptr<int32_t>(),
              overflow.data_ptr<int32_t>(),
              main_amax.data_ptr<float>(),
              rowwise_data.data_ptr<uint8_t>(),
              rowwise_scale.data_ptr<uint8_t>(),
              stats.data_ptr<float>(),
              rows,
              cols,
              row_scale_inner,
              static_cast<int32_t>(capacity));
    } else if (fill_threads == 64) {
      fill_csr_rowwise_quant_prefix_kernel<c10::BFloat16, 64><<<
          static_cast<int>(rows),
          64,
          0,
          at::cuda::getDefaultCUDAStream()>>>(
              input.data_ptr<c10::BFloat16>(),
              row_offsets.data_ptr<int32_t>(),
              reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            flat_indices.data_ptr<int32_t>(),
            outlier_values.data_ptr<c10::BFloat16>(),
            outlier_cols.data_ptr<int16_t>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            num_selected.data_ptr<int32_t>(),
              overflow.data_ptr<int32_t>(),
              main_amax.data_ptr<float>(),
              rowwise_data.data_ptr<uint8_t>(),
              rowwise_scale.data_ptr<uint8_t>(),
              rows,
              cols,
              row_scale_inner,
              static_cast<int32_t>(capacity));
    } else if (fill_threads == 256) {
      fill_csr_rowwise_quant_prefix_kernel<c10::BFloat16, 256><<<
          static_cast<int>(rows),
          256,
          0,
          at::cuda::getDefaultCUDAStream()>>>(
              input.data_ptr<c10::BFloat16>(),
              row_offsets.data_ptr<int32_t>(),
              reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            flat_indices.data_ptr<int32_t>(),
            outlier_values.data_ptr<c10::BFloat16>(),
            outlier_cols.data_ptr<int16_t>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            num_selected.data_ptr<int32_t>(),
              overflow.data_ptr<int32_t>(),
              main_amax.data_ptr<float>(),
              rowwise_data.data_ptr<uint8_t>(),
              rowwise_scale.data_ptr<uint8_t>(),
              rows,
              cols,
              row_scale_inner,
              static_cast<int32_t>(capacity));
    } else {
      fill_csr_rowwise_quant_prefix_kernel<c10::BFloat16, 128><<<
          static_cast<int>(rows),
          128,
          0,
          at::cuda::getDefaultCUDAStream()>>>(
              input.data_ptr<c10::BFloat16>(),
              row_offsets.data_ptr<int32_t>(),
              reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
              flat_indices.data_ptr<int32_t>(),
              outlier_values.data_ptr<c10::BFloat16>(),
              outlier_cols.data_ptr<int16_t>(),
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              num_selected.data_ptr<int32_t>(),
              overflow.data_ptr<int32_t>(),
              main_amax.data_ptr<float>(),
              rowwise_data.data_ptr<uint8_t>(),
              rowwise_scale.data_ptr<uint8_t>(),
              rows,
              cols,
              row_scale_inner,
              static_cast<int32_t>(capacity));
    }
  }

  at::Tensor active_rows;
  at::Tensor active_row_count;
  if (build_active_schedule) {
    auto r25_active_schedule = build_r25_active_rows_schedule(row_counts, rows);
    active_rows = r25_active_schedule[0];
    active_row_count = r25_active_schedule[1];
  } else if (build_unsorted_active_rows) {
    active_rows = at::empty({rows}, input.options().dtype(at::kInt));
    active_row_count = at::empty({1}, input.options().dtype(at::kInt));
    build_unsorted_active_rows_from_counts_into(
        row_counts, rows, active_rows, active_row_count, true);
  } else {
    active_rows = at::empty({0}, input.options().dtype(at::kInt));
    active_row_count = at::zeros({1}, input.options().dtype(at::kInt));
  }
  if (columnwise_launched_async) {
    record_stream_then_wait(column_raw_stream, main_raw_stream);
  } else if (columnwise_source_arg == kColumnwiseDirect || masked_rht_outlier_reuse) {
    column_rht_keepalive = launch_custom_te_rht_columnwise_quant(input,
                                                                 selection_masks,
                                                                 columnwise_data,
                                                                 columnwise_scale,
                                                                 columnwise_amax,
                                                                 rows,
                                                                 cols,
                                                                 column_scale_inner,
                                                                 static_cast<int>(rht_random_sign_mask_t),
                                                                 masked_rht_outlier_reuse,
                                                                 at::cuda::getDefaultCUDAStream(),
                                                                 main_stream);
  }

  return {flat_indices, outlier_values, outlier_cols, row_offsets, num_selected, overflow,
          rowwise_data, rowwise_scale, main_amax, stats, dense_main, columnwise_data,
          columnwise_scale, columnwise_amax, active_rows, active_row_count};
}

__global__ void cap_split_counts_kernel(const int32_t* __restrict__ row_offsets,
                                        int32_t* __restrict__ light_counts,
                                        int32_t* __restrict__ heavy_counts,
                                        int32_t rows,
                                        int32_t cap);
__global__ void cap_split_set_tail_kernel(const int32_t* __restrict__ counts,
                                          int32_t* __restrict__ offsets,
                                          int32_t rows);
__global__ void cap_split_active_rows_kernel(const int32_t* __restrict__ counts,
                                             int32_t* __restrict__ active_rows,
                                             int32_t* __restrict__ active_count,
                                             int32_t rows);
__global__ void cap_split_active_rows_records_kernel(const int32_t* __restrict__ counts,
                                                     const int32_t* __restrict__ row_offsets,
                                                     int32_t* __restrict__ active_rows,
                                                     int64_t* __restrict__ row_records,
                                                     int32_t* __restrict__ active_count,
                                                     int32_t rows);
__global__ void cap_split_row_records_kernel(const int32_t* __restrict__ row_offsets,
                                             const int32_t* __restrict__ counts,
                                             const int32_t* __restrict__ active_rows,
                                             int64_t* __restrict__ row_records,
                                             const int32_t* __restrict__ active_count);
__global__ void cap_split_rowblocks_kernel(const int32_t* __restrict__ counts,
                                           int32_t* __restrict__ rowblocks,
                                           int32_t* __restrict__ rowblock_count,
                                           int32_t rows,
                                           int32_t rows_per_block);
__global__ void cap_split_tile_offsets_kernel(int32_t* __restrict__ tile_offsets,
                                              const int32_t* __restrict__ active_count);
__global__ void policy_split_init_counts_sort_kernel(const int32_t* __restrict__ row_offsets,
                                                     int32_t* __restrict__ light_counts,
                                                     int32_t* __restrict__ heavy_counts,
                                                     int64_t* __restrict__ sort_keys,
                                                     int32_t* __restrict__ sort_rows,
                                                     int32_t rows,
                                                     int32_t policy_mode,
                                                     int32_t param0,
                                                     int32_t param1);
__global__ void policy_split_select_densepack_kernel(const int64_t* __restrict__ sorted_keys,
                                                     const int32_t* __restrict__ sorted_rows,
                                                     const int32_t* __restrict__ row_offsets,
                                                     int32_t* __restrict__ light_counts,
                                                     int32_t* __restrict__ heavy_counts,
                                                     int32_t rows,
                                                     int32_t row_budget);
__global__ void policy_split_select_entrybudget_kernel(const int64_t* __restrict__ sorted_keys,
                                                       const int32_t* __restrict__ sorted_rows,
                                                       const int32_t* __restrict__ row_offsets,
                                                       int32_t* __restrict__ light_counts,
                                                       int32_t* __restrict__ heavy_counts,
                                                       int32_t rows,
                                                       int32_t entry_budget);
__global__ void policy_split_init_counts_only_kernel(const int32_t* __restrict__ row_offsets,
                                                     int32_t* __restrict__ light_counts,
                                                     int32_t* __restrict__ heavy_counts,
                                                     int32_t rows);
__global__ void policy_split_hist_rows_kernel(const int32_t* __restrict__ row_offsets,
                                              int32_t* __restrict__ hist_rows,
                                              int32_t rows,
                                              int32_t max_bucket);
__global__ void policy_split_compute_densepack_limits_kernel(const int32_t* __restrict__ hist_rows,
                                                             int32_t* __restrict__ bucket_limits,
                                                             int32_t max_bucket,
                                                             int32_t min_count,
                                                             int32_t row_budget);
__global__ void policy_split_compute_entrybudget_limits_kernel(const int32_t* __restrict__ hist_rows,
                                                               int32_t* __restrict__ bucket_limits,
                                                               int32_t max_bucket,
                                                               int32_t entry_budget);
__global__ void policy_split_apply_bucket_limits_kernel(const int32_t* __restrict__ row_offsets,
                                                        const int32_t* __restrict__ bucket_limits,
                                                        int32_t* __restrict__ light_counts,
                                                        int32_t* __restrict__ heavy_counts,
                                                        int32_t rows);
__global__ void policy_split_apply_bucket_limits_atomic_kernel(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ bucket_limits,
    int32_t* __restrict__ bucket_taken,
    int32_t* __restrict__ light_counts,
    int32_t* __restrict__ heavy_counts,
    int32_t rows,
    int32_t max_bucket);
__global__ void direct_split_stats_kernel(const int32_t* __restrict__ light_offsets,
                                          const int32_t* __restrict__ heavy_offsets,
                                          const int32_t* __restrict__ light_active_count,
                                          const int32_t* __restrict__ heavy_active_count,
                                          const int32_t* __restrict__ light_rowblock_count,
                                          int32_t* __restrict__ light_rowblocks,
                                          int32_t* __restrict__ stats,
                                          int32_t rows);

std::vector<at::Tensor> adaptive_rowcol_quant_fast_out_impl(
    const at::Tensor& input,
    const at::Tensor& flat_indices,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& row_counts,
    const at::Tensor& row_offsets,
    const at::Tensor& selection_masks,
    const at::Tensor& num_selected,
    const at::Tensor& overflow,
    const at::Tensor& stats_amax,
    const at::Tensor& rowwise_data,
    const at::Tensor& rowwise_scale,
    const at::Tensor& columnwise_data,
    const at::Tensor& columnwise_scale,
    const at::Tensor& columnwise_amax,
    const at::Tensor& rht_output_t,
    const at::Tensor& scan_temp,
    const at::Tensor& packed_records,
    const at::Tensor& active_rows_heavy_light,
    const at::Tensor& active_row_count,
    const at::Tensor& active_sort_rows_in,
    const at::Tensor& active_sort_rows_out,
    const at::Tensor& active_sort_keys_in,
    const at::Tensor& active_sort_keys_out,
    const at::Tensor& active_sort_temp,
    const at::Tensor& hardcap_score_max,
    const at::Tensor& hardcap_histogram,
    double base_ratio,
    double min_ratio,
    double max_ratio,
    double reference_heaviness,
    int64_t capacity,
    int64_t stats_threads_arg,
    int64_t fill_threads_arg,
    int64_t columnwise_source_arg,
    int64_t rht_random_sign_mask_t,
    bool overlap_columnwise,
    bool auto_expand_capacity,
    bool emit_packed_records,
    bool build_active_schedule,
    bool emit_direct_split,
    int64_t direct_policy_mode,
    int64_t direct_param0,
    int64_t direct_param1,
    bool direct_use_bucket,
    bool direct_no_host_slice) {
  const auto rows = input.size(0);
  const auto cols = input.size(1);
  const int64_t total = rows * cols;
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "adaptive rowcol fast supports BF16 only");
  TORCH_CHECK(columnwise_source_arg == kColumnwiseDirect ||
                  columnwise_source_arg == kColumnwiseOutlierReuse,
              "columnwise_source must be direct or outlier_reuse");
  TORCH_CHECK(capacity == flat_indices.numel(), "capacity must match flat_indices size");
  if (emit_packed_records) {
    TORCH_CHECK(packed_records.scalar_type() == at::kLong, "packed_records must be int64");
    TORCH_CHECK(packed_records.numel() >= capacity, "packed_records capacity is too small");
  }
  TORCH_CHECK(row_counts.numel() == rows, "row_counts shape mismatch");
  TORCH_CHECK(row_offsets.numel() == rows + 1, "row_offsets shape mismatch");
  TORCH_CHECK(selection_masks.numel() == rows * (cols / 16), "selection_masks shape mismatch");
  TORCH_CHECK(rht_output_t.numel() == 0 ||
                  (rht_output_t.size(0) == cols && rht_output_t.size(1) == rows),
              "rht_output_t shape mismatch");
  TORCH_CHECK(active_rows_heavy_light.numel() >= rows, "active_rows_heavy_light shape mismatch");
  TORCH_CHECK(active_row_count.numel() >= 1, "active_row_count shape mismatch");
  TORCH_CHECK(active_sort_rows_in.numel() >= rows && active_sort_rows_out.numel() >= rows,
              "active sort row buffer shape mismatch");
  TORCH_CHECK(active_sort_keys_in.numel() >= rows && active_sort_keys_out.numel() >= rows,
              "active sort key buffer shape mismatch");
  TORCH_CHECK(hardcap_score_max.numel() >= 1, "hardcap_score_max shape mismatch");
  TORCH_CHECK(hardcap_histogram.numel() >= kHardCapHistogramBins,
              "hardcap_histogram shape mismatch");

  float* stats_amax_ptr = const_cast<float*>(stats_amax.data_ptr<float>());
  float* stats_ptr = stats_amax_ptr;
  float* main_amax_ptr = stats_amax_ptr + 4;
  float* raw_stats_ptr = stats_amax_ptr + 6;
  const bool masked_rht_outlier_reuse = columnwise_source_arg == kColumnwiseOutlierReuse;
  const int64_t row_scale_inner = round_up_int64((cols + 15) / 16, 4);
  const int64_t column_scale_inner = round_up_int64((rows + 15) / 16, 4);

  const auto main_stream = at::cuda::getDefaultCUDAStream(input.device().index());
  auto column_stream = at::cuda::getStreamFromPool(false, input.device().index());
  const cudaStream_t main_raw_stream = main_stream.stream();
  const cudaStream_t column_raw_stream = column_stream.stream();
  at::Tensor rht_output_keepalive;
  bool columnwise_launched_async = false;

  auto ensure_rht_output = [&](const c10::cuda::CUDAStream& stream) -> at::Tensor& {
    if (!rht_output_keepalive.defined()) {
      rht_output_keepalive = at::empty({cols, rows}, input.options());
    }
    record_tensor_on_stream(rht_output_keepalive, stream);
    return rht_output_keepalive;
  };

  reset_adaptive_workspace_kernel<<<1, 32, 0, at::cuda::getDefaultCUDAStream()>>>(
      const_cast<float*>(stats_amax.data_ptr<float>()),
      const_cast<int32_t*>(num_selected.data_ptr<int32_t>()),
      const_cast<int32_t*>(overflow.data_ptr<int32_t>()));

  if (overlap_columnwise && columnwise_source_arg == kColumnwiseDirect) {
    record_stream_then_wait(main_raw_stream, column_raw_stream);
    launch_custom_te_rht_columnwise_quant_out(input,
                                              selection_masks,
                                              ensure_rht_output(column_stream),
                                              const_cast<at::Tensor&>(columnwise_data),
                                              const_cast<at::Tensor&>(columnwise_scale),
                                              const_cast<at::Tensor&>(columnwise_amax),
                                              rows,
                                              cols,
                                              column_scale_inner,
                                              static_cast<int>(rht_random_sign_mask_t),
                                              false,
                                              column_raw_stream);
    columnwise_launched_async = true;
  }

  const int reduce_threads = 256;
  const int reduce_blocks =
      static_cast<int>(std::min<int64_t>(1024, (total + reduce_threads - 1) / reduce_threads));
  const size_t adaptive_smem = 4 * reduce_threads * sizeof(float);
  sum_sumsq_absmax_kernel<c10::BFloat16>
      <<<reduce_blocks, reduce_threads, adaptive_smem, at::cuda::getDefaultCUDAStream()>>>(
          input.data_ptr<c10::BFloat16>(),
          raw_stats_ptr,
          total);
  finalize_adaptive_stats_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
      raw_stats_ptr,
      stats_ptr,
      total,
      static_cast<float>(base_ratio),
      static_cast<float>(min_ratio),
      static_cast<float>(max_ratio),
      static_cast<float>(reference_heaviness));

  const int stats_threads = normalize_threads(stats_threads_arg, cols <= 4096 ? 128 : 256);
  row_count_main_amax_total_kernel<c10::BFloat16><<<
      static_cast<int>(rows),
      stats_threads,
      0,
      at::cuda::getDefaultCUDAStream()>>>(
          input.data_ptr<c10::BFloat16>(),
          const_cast<int32_t*>(row_counts.data_ptr<int32_t>()),
          reinterpret_cast<uint16_t*>(const_cast<int16_t*>(selection_masks.data_ptr<int16_t>())),
          main_amax_ptr,
          const_cast<int32_t*>(num_selected.data_ptr<int32_t>()),
          stats_ptr,
          rows,
          cols);

  int64_t hard_capacity_64 =
      static_cast<int64_t>(std::ceil(std::max(0.0, max_ratio) * static_cast<double>(total)));
  hard_capacity_64 = std::min<int64_t>(hard_capacity_64, total);
  hard_capacity_64 = std::min<int64_t>(hard_capacity_64, capacity);
  hard_capacity_64 = std::max<int64_t>(hard_capacity_64, 0);
  const int32_t hard_capacity = static_cast<int32_t>(hard_capacity_64);
  prepare_hardcap_workspace_kernel<<<
      (kHardCapHistogramBins + 255) / 256,
      256,
      0,
      at::cuda::getDefaultCUDAStream()>>>(
          num_selected.data_ptr<int32_t>(),
          const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
          const_cast<float*>(hardcap_score_max.data_ptr<float>()),
          const_cast<int32_t*>(hardcap_histogram.data_ptr<int32_t>()),
          hard_capacity);
  hardcap_score_max_kernel<c10::BFloat16><<<
      reduce_blocks,
      reduce_threads,
      reduce_threads * sizeof(float),
      at::cuda::getDefaultCUDAStream()>>>(
          input.data_ptr<c10::BFloat16>(),
          stats_ptr,
          overflow.data_ptr<int32_t>(),
          const_cast<float*>(hardcap_score_max.data_ptr<float>()),
          total);
  hardcap_histogram_kernel<c10::BFloat16><<<
      reduce_blocks,
      reduce_threads,
      0,
      at::cuda::getDefaultCUDAStream()>>>(
          input.data_ptr<c10::BFloat16>(),
          stats_ptr,
          overflow.data_ptr<int32_t>(),
          hardcap_score_max.data_ptr<float>(),
          const_cast<int32_t*>(hardcap_histogram.data_ptr<int32_t>()),
          total);
  hardcap_update_strict_threshold_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
      hardcap_histogram.data_ptr<int32_t>(),
      stats_ptr,
      hardcap_score_max.data_ptr<float>(),
      overflow.data_ptr<int32_t>(),
      main_amax_ptr,
      num_selected.data_ptr<int32_t>(),
      hard_capacity);
  row_count_main_amax_total_if_overflow_kernel<c10::BFloat16><<<
      static_cast<int>(rows),
      stats_threads,
      0,
      at::cuda::getDefaultCUDAStream()>>>(
          input.data_ptr<c10::BFloat16>(),
          const_cast<int32_t*>(row_counts.data_ptr<int32_t>()),
          reinterpret_cast<uint16_t*>(const_cast<int16_t*>(selection_masks.data_ptr<int16_t>())),
          main_amax_ptr,
          const_cast<int32_t*>(num_selected.data_ptr<int32_t>()),
          stats_ptr,
          overflow.data_ptr<int32_t>(),
          rows,
          cols);

  if (overlap_columnwise && masked_rht_outlier_reuse) {
    record_stream_then_wait(main_raw_stream, column_raw_stream);
    launch_custom_te_rht_columnwise_quant_out(input,
                                              selection_masks,
                                              ensure_rht_output(column_stream),
                                              const_cast<at::Tensor&>(columnwise_data),
                                              const_cast<at::Tensor&>(columnwise_scale),
                                              const_cast<at::Tensor&>(columnwise_amax),
                                              rows,
                                              cols,
                                              column_scale_inner,
                                              static_cast<int>(rht_random_sign_mask_t),
                                              true,
                                              column_raw_stream);
    columnwise_launched_async = true;
  }

  at::Tensor target_flat_indices = const_cast<at::Tensor&>(flat_indices);
  at::Tensor target_outlier_values = const_cast<at::Tensor&>(outlier_values);
  at::Tensor target_outlier_cols = const_cast<at::Tensor&>(outlier_cols);
  int64_t output_capacity = capacity;
  if (auto_expand_capacity) {
    const int32_t selected_host =
        copy_int32_to_host(num_selected.data_ptr<int32_t>(), main_raw_stream);
    if (selected_host > capacity) {
      output_capacity = static_cast<int64_t>(selected_host);
      target_flat_indices = at::empty({output_capacity}, input.options().dtype(at::kInt));
      target_outlier_values = at::empty({output_capacity}, input.options());
      target_outlier_cols = at::empty({output_capacity}, input.options().dtype(at::kShort));
    }
  }

  size_t temp_bytes = static_cast<size_t>(scan_temp.numel());
  cub::DeviceScan::ExclusiveSum(
      const_cast<uint8_t*>(scan_temp.data_ptr<uint8_t>()),
      temp_bytes,
      row_counts.data_ptr<int32_t>(),
      const_cast<int32_t*>(row_offsets.data_ptr<int32_t>()),
      static_cast<int>(rows),
      at::cuda::getDefaultCUDAStream());

  const int fill_threads = normalize_threads(fill_threads_arg, cols <= 4096 ? 128 : 256);
  const bool do_direct_split = emit_direct_split && direct_policy_mode >= 0;
  at::Tensor direct_light_counts;
  at::Tensor direct_heavy_counts;
  at::Tensor direct_light_offsets;
  at::Tensor direct_heavy_offsets;
  at::Tensor direct_light_cols;
  at::Tensor direct_heavy_cols;
  at::Tensor direct_light_flat_indices;
  at::Tensor direct_heavy_flat_indices;
  at::Tensor direct_light_values;
  at::Tensor direct_heavy_values;
  at::Tensor direct_light_entry_records;
  at::Tensor direct_light_active_rows;
  at::Tensor direct_heavy_active_rows;
  at::Tensor direct_light_active_count;
  at::Tensor direct_heavy_active_count;
  at::Tensor direct_light_row_records;
  at::Tensor direct_light_rowblocks;
  at::Tensor direct_light_rowblock_count;
  at::Tensor direct_tile_offsets;
  at::Tensor direct_stats_cuda;
  int32_t direct_light_entries = 0;
  int32_t direct_heavy_entries = 0;
  int32_t direct_light_rows = 0;
  int32_t direct_heavy_rows = 0;
  int32_t direct_light_rowblocks_count = 0;

  if (do_direct_split) {
    TORCH_CHECK(direct_policy_mode >= 0 && direct_policy_mode <= 2,
                "direct_policy_mode must be 0=cap, 1=densepack, or 2=entrybudget");
    const auto opts_i32 = row_offsets.options().dtype(at::kInt);
    const auto opts_i64 = row_offsets.options().dtype(at::kLong);
    const auto opts_val = input.options();
    const int32_t m = static_cast<int32_t>(rows);
    const int32_t rowblock_capacity = (m + 7) / 8;
    direct_light_counts = at::empty({rows}, opts_i32);
    direct_heavy_counts = at::empty({rows}, opts_i32);
    direct_light_offsets = at::empty({rows + 1}, opts_i32);
    direct_heavy_offsets = at::empty({rows + 1}, opts_i32);
    direct_light_cols = at::empty({output_capacity}, opts_i32);
    direct_heavy_cols = at::empty({output_capacity}, opts_i32);
    direct_light_flat_indices = at::empty({output_capacity}, opts_i32);
    direct_heavy_flat_indices = at::empty({output_capacity}, opts_i32);
    direct_light_values = at::empty({output_capacity}, opts_val);
    direct_heavy_values = at::empty({output_capacity}, opts_val);
    direct_light_entry_records = at::empty({output_capacity}, opts_i32);
    direct_light_active_rows = at::empty({rows}, opts_i32);
    direct_heavy_active_rows = at::empty({rows}, opts_i32);
    direct_light_active_count = at::zeros({1}, opts_i32);
    direct_heavy_active_count = at::zeros({1}, opts_i32);
    direct_light_row_records = at::empty({rows}, opts_i64);
    direct_light_rowblocks = at::empty({std::max<int32_t>(rowblock_capacity, 1)}, opts_i32);
    direct_light_rowblock_count = at::zeros({1}, opts_i32);
    direct_tile_offsets = at::empty({2}, opts_i32);
    direct_stats_cuda = at::empty({5}, opts_i32);

    if (direct_no_host_slice) {
      C10_CUDA_CHECK(cudaMemsetAsync(
          direct_light_active_rows.data_ptr<int32_t>(),
          0,
          static_cast<size_t>(rows) * sizeof(int32_t),
          main_raw_stream));
      C10_CUDA_CHECK(cudaMemsetAsync(
          direct_heavy_active_rows.data_ptr<int32_t>(),
          0,
          static_cast<size_t>(rows) * sizeof(int32_t),
          main_raw_stream));
      C10_CUDA_CHECK(cudaMemsetAsync(
          direct_light_row_records.data_ptr<int64_t>(),
          0,
          static_cast<size_t>(rows) * sizeof(int64_t),
          main_raw_stream));
      C10_CUDA_CHECK(cudaMemsetAsync(
          direct_light_rowblocks.data_ptr<int32_t>(),
          0,
          static_cast<size_t>(std::max<int32_t>(rowblock_capacity, 1)) * sizeof(int32_t),
          main_raw_stream));
      C10_CUDA_CHECK(cudaMemsetAsync(
          direct_tile_offsets.data_ptr<int32_t>(),
          0,
          2 * sizeof(int32_t),
          main_raw_stream));
    }

    const int count_threads = 256;
    const int count_blocks = (m + count_threads - 1) / count_threads;
    if (direct_policy_mode == 0) {
      cap_split_counts_kernel<<<count_blocks, count_threads, 0, at::cuda::getDefaultCUDAStream()>>>(
          row_offsets.data_ptr<int32_t>(),
          direct_light_counts.data_ptr<int32_t>(),
          direct_heavy_counts.data_ptr<int32_t>(),
          m,
          static_cast<int32_t>(direct_param0));
    } else if (direct_use_bucket) {
      policy_split_init_counts_only_kernel<<<
          count_blocks,
          count_threads,
          0,
          at::cuda::getDefaultCUDAStream()>>>(
              row_offsets.data_ptr<int32_t>(),
              direct_light_counts.data_ptr<int32_t>(),
              direct_heavy_counts.data_ptr<int32_t>(),
              m);
      const int32_t max_bucket = static_cast<int32_t>(
          direct_policy_mode == 1 ? std::max<int64_t>(1, cols)
                                  : std::max<int64_t>(1, direct_param1));
      auto bucket_hist = at::zeros({static_cast<int64_t>(max_bucket) + 1}, opts_i32);
      auto bucket_limits = at::zeros({static_cast<int64_t>(max_bucket) + 1}, opts_i32);
      auto bucket_taken = at::zeros({static_cast<int64_t>(max_bucket) + 1}, opts_i32);
      policy_split_hist_rows_kernel<<<
          count_blocks,
          count_threads,
          0,
          at::cuda::getDefaultCUDAStream()>>>(
              row_offsets.data_ptr<int32_t>(),
              bucket_hist.data_ptr<int32_t>(),
              m,
              max_bucket);
      if (direct_policy_mode == 1) {
        policy_split_compute_densepack_limits_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
            bucket_hist.data_ptr<int32_t>(),
            bucket_limits.data_ptr<int32_t>(),
            max_bucket,
            static_cast<int32_t>(direct_param0),
            static_cast<int32_t>(direct_param1));
      } else {
        policy_split_compute_entrybudget_limits_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
            bucket_hist.data_ptr<int32_t>(),
            bucket_limits.data_ptr<int32_t>(),
            max_bucket,
            static_cast<int32_t>(direct_param0));
      }
      policy_split_apply_bucket_limits_atomic_kernel<<<
          count_blocks,
          count_threads,
          0,
          at::cuda::getDefaultCUDAStream()>>>(
              row_offsets.data_ptr<int32_t>(),
              bucket_limits.data_ptr<int32_t>(),
              bucket_taken.data_ptr<int32_t>(),
              direct_light_counts.data_ptr<int32_t>(),
              direct_heavy_counts.data_ptr<int32_t>(),
              m,
              max_bucket);
    } else {
      auto sort_keys_in = at::empty({rows}, opts_i64);
      auto sort_keys_out = at::empty({rows}, opts_i64);
      auto sort_rows_in = at::empty({rows}, opts_i32);
      auto sort_rows_out = at::empty({rows}, opts_i32);
      policy_split_init_counts_sort_kernel<<<
          count_blocks,
          count_threads,
          0,
          at::cuda::getDefaultCUDAStream()>>>(
              row_offsets.data_ptr<int32_t>(),
              direct_light_counts.data_ptr<int32_t>(),
              direct_heavy_counts.data_ptr<int32_t>(),
              sort_keys_in.data_ptr<int64_t>(),
              sort_rows_in.data_ptr<int32_t>(),
              m,
              static_cast<int32_t>(direct_policy_mode),
              static_cast<int32_t>(direct_param0),
              static_cast<int32_t>(direct_param1));
      void* sort_temp_storage = nullptr;
      size_t sort_temp_bytes = 0;
      cub::DeviceRadixSort::SortPairsDescending(sort_temp_storage,
                                                sort_temp_bytes,
                                                sort_keys_in.data_ptr<int64_t>(),
                                                sort_keys_out.data_ptr<int64_t>(),
                                                sort_rows_in.data_ptr<int32_t>(),
                                                sort_rows_out.data_ptr<int32_t>(),
                                                m,
                                                0,
                                                64,
                                                at::cuda::getDefaultCUDAStream());
      auto sort_temp = at::empty({static_cast<int64_t>(sort_temp_bytes)},
                                 row_offsets.options().dtype(at::kByte));
      cub::DeviceRadixSort::SortPairsDescending(sort_temp.data_ptr(),
                                                sort_temp_bytes,
                                                sort_keys_in.data_ptr<int64_t>(),
                                                sort_keys_out.data_ptr<int64_t>(),
                                                sort_rows_in.data_ptr<int32_t>(),
                                                sort_rows_out.data_ptr<int32_t>(),
                                                m,
                                                0,
                                                64,
                                                at::cuda::getDefaultCUDAStream());
      if (direct_policy_mode == 1) {
        policy_split_select_densepack_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
            sort_keys_out.data_ptr<int64_t>(),
            sort_rows_out.data_ptr<int32_t>(),
            row_offsets.data_ptr<int32_t>(),
            direct_light_counts.data_ptr<int32_t>(),
            direct_heavy_counts.data_ptr<int32_t>(),
            m,
            static_cast<int32_t>(direct_param1));
      } else {
        policy_split_select_entrybudget_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
            sort_keys_out.data_ptr<int64_t>(),
            sort_rows_out.data_ptr<int32_t>(),
            row_offsets.data_ptr<int32_t>(),
            direct_light_counts.data_ptr<int32_t>(),
            direct_heavy_counts.data_ptr<int32_t>(),
            m,
            static_cast<int32_t>(direct_param0));
      }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cub::DeviceScan::ExclusiveSum(
        const_cast<uint8_t*>(scan_temp.data_ptr<uint8_t>()),
        temp_bytes,
        direct_light_counts.data_ptr<int32_t>(),
        direct_light_offsets.data_ptr<int32_t>(),
        static_cast<int>(rows),
        at::cuda::getDefaultCUDAStream());
    cub::DeviceScan::ExclusiveSum(
        const_cast<uint8_t*>(scan_temp.data_ptr<uint8_t>()),
        temp_bytes,
        direct_heavy_counts.data_ptr<int32_t>(),
        direct_heavy_offsets.data_ptr<int32_t>(),
        static_cast<int>(rows),
        at::cuda::getDefaultCUDAStream());
    cap_split_set_tail_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
        direct_light_counts.data_ptr<int32_t>(), direct_light_offsets.data_ptr<int32_t>(), m);
    cap_split_set_tail_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
        direct_heavy_counts.data_ptr<int32_t>(), direct_heavy_offsets.data_ptr<int32_t>(), m);
  }

  int64_t* packed_records_ptr =
      (emit_packed_records && !do_direct_split)
          ? const_cast<int64_t*>(packed_records.data_ptr<int64_t>())
          : nullptr;
  int32_t* full_flat_ptr =
      do_direct_split ? nullptr : target_flat_indices.data_ptr<int32_t>();
  c10::BFloat16* full_values_ptr =
      do_direct_split ? nullptr : target_outlier_values.data_ptr<c10::BFloat16>();
  int16_t* full_cols_ptr =
      do_direct_split ? nullptr : target_outlier_cols.data_ptr<int16_t>();
  if (fill_threads == 64) {
    fill_csr_rowwise_quant_prefix_kernel<c10::BFloat16, 64><<<
        static_cast<int>(rows),
        64,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            const_cast<int32_t*>(row_offsets.data_ptr<int32_t>()),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            full_flat_ptr,
            full_values_ptr,
            full_cols_ptr,
            packed_records_ptr,
            do_direct_split ? direct_light_offsets.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_heavy_offsets.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_light_counts.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_heavy_counts.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_light_cols.data_ptr<int32_t>() : nullptr,
            do_direct_split ? reinterpret_cast<uint16_t*>(direct_light_values.data_ptr<c10::BFloat16>()) : nullptr,
            do_direct_split ? direct_light_flat_indices.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_light_entry_records.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_heavy_cols.data_ptr<int32_t>() : nullptr,
            do_direct_split ? reinterpret_cast<uint16_t*>(direct_heavy_values.data_ptr<c10::BFloat16>()) : nullptr,
            do_direct_split ? direct_heavy_flat_indices.data_ptr<int32_t>() : nullptr,
            num_selected.data_ptr<int32_t>(),
            const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
            main_amax_ptr,
            const_cast<uint8_t*>(rowwise_data.data_ptr<uint8_t>()),
            const_cast<uint8_t*>(rowwise_scale.data_ptr<uint8_t>()),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(output_capacity));
  } else if (fill_threads == 256) {
    fill_csr_rowwise_quant_prefix_kernel<c10::BFloat16, 256><<<
        static_cast<int>(rows),
        256,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            const_cast<int32_t*>(row_offsets.data_ptr<int32_t>()),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            full_flat_ptr,
            full_values_ptr,
            full_cols_ptr,
            packed_records_ptr,
            do_direct_split ? direct_light_offsets.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_heavy_offsets.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_light_counts.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_heavy_counts.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_light_cols.data_ptr<int32_t>() : nullptr,
            do_direct_split ? reinterpret_cast<uint16_t*>(direct_light_values.data_ptr<c10::BFloat16>()) : nullptr,
            do_direct_split ? direct_light_flat_indices.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_light_entry_records.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_heavy_cols.data_ptr<int32_t>() : nullptr,
            do_direct_split ? reinterpret_cast<uint16_t*>(direct_heavy_values.data_ptr<c10::BFloat16>()) : nullptr,
            do_direct_split ? direct_heavy_flat_indices.data_ptr<int32_t>() : nullptr,
            num_selected.data_ptr<int32_t>(),
            const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
            main_amax_ptr,
            const_cast<uint8_t*>(rowwise_data.data_ptr<uint8_t>()),
            const_cast<uint8_t*>(rowwise_scale.data_ptr<uint8_t>()),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(output_capacity));
  } else if (fill_threads == 512) {
    fill_csr_rowwise_quant_prefix_kernel<c10::BFloat16, 512><<<
        static_cast<int>(rows),
        512,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            const_cast<int32_t*>(row_offsets.data_ptr<int32_t>()),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            full_flat_ptr,
            full_values_ptr,
            full_cols_ptr,
            packed_records_ptr,
            do_direct_split ? direct_light_offsets.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_heavy_offsets.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_light_counts.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_heavy_counts.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_light_cols.data_ptr<int32_t>() : nullptr,
            do_direct_split ? reinterpret_cast<uint16_t*>(direct_light_values.data_ptr<c10::BFloat16>()) : nullptr,
            do_direct_split ? direct_light_flat_indices.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_light_entry_records.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_heavy_cols.data_ptr<int32_t>() : nullptr,
            do_direct_split ? reinterpret_cast<uint16_t*>(direct_heavy_values.data_ptr<c10::BFloat16>()) : nullptr,
            do_direct_split ? direct_heavy_flat_indices.data_ptr<int32_t>() : nullptr,
            num_selected.data_ptr<int32_t>(),
            const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
            main_amax_ptr,
            const_cast<uint8_t*>(rowwise_data.data_ptr<uint8_t>()),
            const_cast<uint8_t*>(rowwise_scale.data_ptr<uint8_t>()),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(output_capacity));
  } else {
    fill_csr_rowwise_quant_prefix_kernel<c10::BFloat16, 128><<<
        static_cast<int>(rows),
        128,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            const_cast<int32_t*>(row_offsets.data_ptr<int32_t>()),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            full_flat_ptr,
            full_values_ptr,
            full_cols_ptr,
            packed_records_ptr,
            do_direct_split ? direct_light_offsets.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_heavy_offsets.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_light_counts.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_heavy_counts.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_light_cols.data_ptr<int32_t>() : nullptr,
            do_direct_split ? reinterpret_cast<uint16_t*>(direct_light_values.data_ptr<c10::BFloat16>()) : nullptr,
            do_direct_split ? direct_light_flat_indices.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_light_entry_records.data_ptr<int32_t>() : nullptr,
            do_direct_split ? direct_heavy_cols.data_ptr<int32_t>() : nullptr,
            do_direct_split ? reinterpret_cast<uint16_t*>(direct_heavy_values.data_ptr<c10::BFloat16>()) : nullptr,
            do_direct_split ? direct_heavy_flat_indices.data_ptr<int32_t>() : nullptr,
            num_selected.data_ptr<int32_t>(),
            const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
            main_amax_ptr,
            const_cast<uint8_t*>(rowwise_data.data_ptr<uint8_t>()),
            const_cast<uint8_t*>(rowwise_scale.data_ptr<uint8_t>()),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(output_capacity));
  }

  if (do_direct_split) {
    const int32_t m = static_cast<int32_t>(rows);
    const int count_threads = 256;
    const int count_blocks = (m + count_threads - 1) / count_threads;
    cap_split_active_rows_records_kernel<<<
        count_blocks,
        count_threads,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
        direct_light_counts.data_ptr<int32_t>(),
        direct_light_offsets.data_ptr<int32_t>(),
        direct_light_active_rows.data_ptr<int32_t>(),
        direct_light_row_records.data_ptr<int64_t>(),
        direct_light_active_count.data_ptr<int32_t>(),
        m);
    cap_split_active_rows_kernel<<<count_blocks, count_threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        direct_heavy_counts.data_ptr<int32_t>(),
        direct_heavy_active_rows.data_ptr<int32_t>(),
        direct_heavy_active_count.data_ptr<int32_t>(),
        m);

    const int32_t rowblock_capacity = (m + 7) / 8;
    const int rowblock_threads = 128;
    const int rowblock_blocks = (rowblock_capacity + rowblock_threads - 1) / rowblock_threads;
    cap_split_rowblocks_kernel<<<
        rowblock_blocks,
        rowblock_threads,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            direct_light_counts.data_ptr<int32_t>(),
            direct_light_rowblocks.data_ptr<int32_t>(),
            direct_light_rowblock_count.data_ptr<int32_t>(),
            m,
            8);
    cap_split_tile_offsets_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
        direct_tile_offsets.data_ptr<int32_t>(), direct_light_active_count.data_ptr<int32_t>());

    if (direct_no_host_slice) {
      direct_split_stats_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
          direct_light_offsets.data_ptr<int32_t>(),
          direct_heavy_offsets.data_ptr<int32_t>(),
          direct_light_active_count.data_ptr<int32_t>(),
          direct_heavy_active_count.data_ptr<int32_t>(),
          direct_light_rowblock_count.data_ptr<int32_t>(),
          direct_light_rowblocks.data_ptr<int32_t>(),
          direct_stats_cuda.data_ptr<int32_t>(),
          m);
    } else {
      direct_light_entries =
          copy_int32_to_host(direct_light_offsets.data_ptr<int32_t>() + m, main_raw_stream);
      direct_heavy_entries =
          copy_int32_to_host(direct_heavy_offsets.data_ptr<int32_t>() + m, main_raw_stream);
      direct_light_rows =
          copy_int32_to_host(direct_light_active_count.data_ptr<int32_t>(), main_raw_stream);
      direct_heavy_rows =
          copy_int32_to_host(direct_heavy_active_count.data_ptr<int32_t>(), main_raw_stream);
      direct_light_rowblocks_count =
          copy_int32_to_host(direct_light_rowblock_count.data_ptr<int32_t>(), main_raw_stream);
      if (direct_light_rowblocks_count == 0) {
        direct_light_rowblocks_count = 1;
        C10_CUDA_CHECK(cudaMemsetAsync(
            direct_light_rowblocks.data_ptr<int32_t>(), 0, sizeof(int32_t), main_raw_stream));
      }
    }
  }

  if (build_active_schedule) {
    build_r25_active_rows_schedule_into(row_counts,
                                        rows,
                                        active_rows_heavy_light,
                                        active_row_count,
                                        active_sort_rows_in,
                                        active_sort_rows_out,
                                        active_sort_keys_in,
                                        active_sort_keys_out,
                                        active_sort_temp);
  } else {
    build_unsorted_active_rows_from_counts_into(row_counts,
                                                rows,
                                                active_rows_heavy_light,
                                                active_row_count);
  }
  if (columnwise_launched_async) {
    record_stream_then_wait(column_raw_stream, main_raw_stream);
  } else {
    launch_custom_te_rht_columnwise_quant_out(input,
                                              selection_masks,
                                              ensure_rht_output(main_stream),
                                              const_cast<at::Tensor&>(columnwise_data),
                                              const_cast<at::Tensor&>(columnwise_scale),
                                              const_cast<at::Tensor&>(columnwise_amax),
                                              rows,
                                              cols,
                                              column_scale_inner,
                                              static_cast<int>(rht_random_sign_mask_t),
                                              masked_rht_outlier_reuse,
                                              at::cuda::getDefaultCUDAStream());
  }

  auto main_amax = stats_amax.narrow(0, 4, 1);
  auto stats = stats_amax.narrow(0, 0, 4);
  auto dense_main = at::empty({0}, input.options());
  std::vector<at::Tensor> result = {
      do_direct_split ? at::empty({0}, input.options().dtype(at::kInt)) : target_flat_indices,
      do_direct_split ? at::empty({0}, input.options()) : target_outlier_values,
      do_direct_split ? at::empty({0}, input.options().dtype(at::kShort)) : target_outlier_cols,
      const_cast<at::Tensor&>(row_offsets),
      const_cast<at::Tensor&>(num_selected),
      const_cast<at::Tensor&>(overflow),
      const_cast<at::Tensor&>(rowwise_data),
      const_cast<at::Tensor&>(rowwise_scale),
      main_amax,
      stats,
      dense_main,
      const_cast<at::Tensor&>(columnwise_data),
      const_cast<at::Tensor&>(columnwise_scale),
      const_cast<at::Tensor&>(columnwise_amax),
      const_cast<at::Tensor&>(active_rows_heavy_light),
      const_cast<at::Tensor&>(active_row_count)};
  if (do_direct_split) {
    if (direct_no_host_slice) {
      result.insert(result.end(),
                    {direct_light_offsets,
                     direct_light_cols,
                     direct_light_values,
                     direct_heavy_offsets,
                     direct_heavy_cols,
                     direct_heavy_values,
                     direct_light_active_rows,
                     direct_heavy_active_rows,
                     direct_tile_offsets,
                     direct_light_row_records,
                     direct_light_entry_records,
                     direct_light_rowblocks,
                     direct_light_flat_indices,
                     direct_heavy_flat_indices,
                     direct_stats_cuda});
    } else {
      result.insert(result.end(),
                    {direct_light_offsets,
                     direct_light_cols.slice(0, 0, direct_light_entries),
                     direct_light_values.slice(0, 0, direct_light_entries),
                     direct_heavy_offsets,
                     direct_heavy_cols.slice(0, 0, direct_heavy_entries),
                     direct_heavy_values.slice(0, 0, direct_heavy_entries),
                     direct_light_active_rows.slice(0, 0, direct_light_rows),
                     direct_heavy_active_rows.slice(0, 0, direct_heavy_rows),
                     direct_tile_offsets,
                     direct_light_row_records.slice(0, 0, direct_light_rows),
                     direct_light_entry_records.slice(0, 0, direct_light_entries),
                     direct_light_rowblocks.slice(0, 0, direct_light_rowblocks_count),
                     direct_light_flat_indices.slice(0, 0, direct_light_entries),
                     direct_heavy_flat_indices.slice(0, 0, direct_heavy_entries),
                     torch::tensor({direct_light_entries,
                                    direct_heavy_entries,
                                    direct_light_rows,
                                    direct_heavy_rows,
                                    direct_light_rowblocks_count},
                                   row_offsets.options().dtype(at::kInt))});
    }
  }
  return result;
}

std::vector<at::Tensor> adaptive_rowcol_quant_fast_padded_out_impl(
    const at::Tensor& input,
    const at::Tensor& padded_values,
    const at::Tensor& padded_cols,
    const at::Tensor& row_counts,
    const at::Tensor& selection_masks,
    const at::Tensor& num_selected,
    const at::Tensor& overflow,
    const at::Tensor& overflow_rows,
    const at::Tensor& overflow_cols,
    const at::Tensor& overflow_values,
    const at::Tensor& overflow_count,
    const at::Tensor& stats_amax,
    const at::Tensor& rowwise_data,
    const at::Tensor& rowwise_scale,
    const at::Tensor& columnwise_data,
    const at::Tensor& columnwise_scale,
    const at::Tensor& columnwise_amax,
    const at::Tensor& rht_output_t,
    const at::Tensor& active_rows_heavy_light,
    const at::Tensor& active_row_count,
    const at::Tensor& active_sort_rows_in,
    const at::Tensor& active_sort_rows_out,
    const at::Tensor& active_sort_keys_in,
    const at::Tensor& active_sort_keys_out,
    const at::Tensor& active_sort_temp,
    const at::Tensor& hardcap_score_max,
    const at::Tensor& hardcap_histogram,
    double base_ratio,
    double min_ratio,
    double max_ratio,
    double reference_heaviness,
    int64_t max_per_row_arg,
    int64_t stats_threads_arg,
    int64_t fill_threads_arg,
    int64_t columnwise_source_arg,
    int64_t rht_random_sign_mask_t,
    bool overlap_columnwise,
    bool direct_nomask) {
  const auto rows = input.size(0);
  const auto cols = input.size(1);
  const int64_t total = rows * cols;
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "adaptive rowcol fast supports BF16 only");
  TORCH_CHECK(columnwise_source_arg == kColumnwiseDirect ||
                  columnwise_source_arg == kColumnwiseOutlierReuse,
              "columnwise_source must be direct or outlier_reuse");
  TORCH_CHECK(!direct_nomask || columnwise_source_arg == kColumnwiseDirect,
              "direct_nomask requires columnwise_source=direct");
  TORCH_CHECK(max_per_row_arg >= 0 && max_per_row_arg <= cols,
              "max_per_row must be in [0, cols]");
  TORCH_CHECK(max_per_row_arg <= std::numeric_limits<int32_t>::max(),
              "max_per_row must fit int32");
  const int64_t max_per_row = max_per_row_arg;
  const int64_t padded_capacity = rows * max_per_row;
  TORCH_CHECK(padded_values.numel() == padded_capacity, "padded_values shape mismatch");
  TORCH_CHECK(padded_cols.numel() == padded_capacity, "padded_cols shape mismatch");
  TORCH_CHECK(row_counts.numel() == rows, "row_counts shape mismatch");
  TORCH_CHECK(selection_masks.numel() == rows * (cols / 16), "selection_masks shape mismatch");
  TORCH_CHECK(num_selected.numel() >= 1, "num_selected shape mismatch");
  TORCH_CHECK(overflow.numel() >= 1, "overflow shape mismatch");
  TORCH_CHECK(overflow_count.numel() >= 1, "overflow_count shape mismatch");
  TORCH_CHECK(overflow_rows.numel() == overflow_cols.numel() &&
                  overflow_rows.numel() == overflow_values.numel(),
              "overflow buffer size mismatch");
  TORCH_CHECK(overflow_rows.numel() <= std::numeric_limits<int32_t>::max(),
              "overflow capacity must fit int32");
  TORCH_CHECK(rht_output_t.numel() == 0 ||
                  (rht_output_t.size(0) == cols && rht_output_t.size(1) == rows),
              "rht_output_t shape mismatch");
  TORCH_CHECK(active_rows_heavy_light.numel() >= rows, "active_rows_heavy_light shape mismatch");
  TORCH_CHECK(active_row_count.numel() >= 1, "active_row_count shape mismatch");
  TORCH_CHECK(active_sort_rows_in.numel() >= rows && active_sort_rows_out.numel() >= rows,
              "active sort row buffer shape mismatch");
  TORCH_CHECK(active_sort_keys_in.numel() >= rows && active_sort_keys_out.numel() >= rows,
              "active sort key buffer shape mismatch");
  TORCH_CHECK(hardcap_score_max.numel() >= 1, "hardcap_score_max shape mismatch");
  TORCH_CHECK(hardcap_histogram.numel() >= kHardCapHistogramBins,
              "hardcap_histogram shape mismatch");

  float* stats_amax_ptr = const_cast<float*>(stats_amax.data_ptr<float>());
  float* stats_ptr = stats_amax_ptr;
  float* main_amax_ptr = stats_amax_ptr + 4;
  float* raw_stats_ptr = stats_amax_ptr + 6;
  const bool masked_rht_outlier_reuse = columnwise_source_arg == kColumnwiseOutlierReuse;
  const int64_t row_scale_inner = round_up_int64((cols + 15) / 16, 4);
  const int64_t column_scale_inner = round_up_int64((rows + 15) / 16, 4);
  const int32_t overflow_capacity = static_cast<int32_t>(overflow_rows.numel());

  const auto main_stream = at::cuda::getDefaultCUDAStream(input.device().index());
  auto column_stream = at::cuda::getStreamFromPool(false, input.device().index());
  const cudaStream_t main_raw_stream = main_stream.stream();
  const cudaStream_t column_raw_stream = column_stream.stream();
  at::Tensor rht_output_keepalive;
  bool columnwise_launched_async = false;

  auto ensure_rht_output = [&](const c10::cuda::CUDAStream& stream) -> at::Tensor& {
    if (!rht_output_keepalive.defined()) {
      rht_output_keepalive = at::empty({cols, rows}, input.options());
    }
    record_tensor_on_stream(rht_output_keepalive, stream);
    return rht_output_keepalive;
  };

  reset_adaptive_workspace_kernel<<<1, 32, 0, at::cuda::getDefaultCUDAStream()>>>(
      const_cast<float*>(stats_amax.data_ptr<float>()),
      const_cast<int32_t*>(num_selected.data_ptr<int32_t>()),
      const_cast<int32_t*>(overflow.data_ptr<int32_t>()));

  if (overlap_columnwise && columnwise_source_arg == kColumnwiseDirect) {
    record_stream_then_wait(main_raw_stream, column_raw_stream);
    launch_custom_te_rht_columnwise_quant_out(input,
                                              selection_masks,
                                              ensure_rht_output(column_stream),
                                              const_cast<at::Tensor&>(columnwise_data),
                                              const_cast<at::Tensor&>(columnwise_scale),
                                              const_cast<at::Tensor&>(columnwise_amax),
                                              rows,
                                              cols,
                                              column_scale_inner,
                                              static_cast<int>(rht_random_sign_mask_t),
                                              false,
                                              column_raw_stream);
    columnwise_launched_async = true;
  }

  const int reduce_threads = 256;
  const int reduce_blocks =
      static_cast<int>(std::min<int64_t>(1024, (total + reduce_threads - 1) / reduce_threads));
  const size_t adaptive_smem = 4 * reduce_threads * sizeof(float);
  sum_sumsq_absmax_kernel<c10::BFloat16>
      <<<reduce_blocks, reduce_threads, adaptive_smem, at::cuda::getDefaultCUDAStream()>>>(
          input.data_ptr<c10::BFloat16>(),
          raw_stats_ptr,
          total);
  finalize_adaptive_stats_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
      raw_stats_ptr,
      stats_ptr,
      total,
      static_cast<float>(base_ratio),
      static_cast<float>(min_ratio),
      static_cast<float>(max_ratio),
      static_cast<float>(reference_heaviness));

  const int stats_threads = normalize_threads(stats_threads_arg, cols <= 4096 ? 128 : 256);
  if (direct_nomask) {
    row_count_main_amax_total_nomask_kernel<c10::BFloat16><<<
        static_cast<int>(rows),
        stats_threads,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            const_cast<int32_t*>(row_counts.data_ptr<int32_t>()),
            main_amax_ptr,
            const_cast<int32_t*>(num_selected.data_ptr<int32_t>()),
            stats_ptr,
            rows,
            cols);
  } else {
    row_count_main_amax_total_kernel<c10::BFloat16><<<
        static_cast<int>(rows),
        stats_threads,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            const_cast<int32_t*>(row_counts.data_ptr<int32_t>()),
            reinterpret_cast<uint16_t*>(const_cast<int16_t*>(selection_masks.data_ptr<int16_t>())),
            main_amax_ptr,
            const_cast<int32_t*>(num_selected.data_ptr<int32_t>()),
            stats_ptr,
            rows,
            cols);
  }

  int64_t hard_capacity_64 =
      static_cast<int64_t>(std::ceil(std::max(0.0, max_ratio) * static_cast<double>(total)));
  hard_capacity_64 = std::min<int64_t>(hard_capacity_64, total);
  hard_capacity_64 = std::max<int64_t>(hard_capacity_64, 0);
  hard_capacity_64 = std::min<int64_t>(hard_capacity_64, std::numeric_limits<int32_t>::max());
  const int32_t hard_capacity = static_cast<int32_t>(hard_capacity_64);
  prepare_hardcap_workspace_kernel<<<
      (kHardCapHistogramBins + 255) / 256,
      256,
      0,
      at::cuda::getDefaultCUDAStream()>>>(
          num_selected.data_ptr<int32_t>(),
          const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
          const_cast<float*>(hardcap_score_max.data_ptr<float>()),
          const_cast<int32_t*>(hardcap_histogram.data_ptr<int32_t>()),
          hard_capacity);
  hardcap_score_max_kernel<c10::BFloat16><<<
      reduce_blocks,
      reduce_threads,
      reduce_threads * sizeof(float),
      at::cuda::getDefaultCUDAStream()>>>(
          input.data_ptr<c10::BFloat16>(),
          stats_ptr,
          overflow.data_ptr<int32_t>(),
          const_cast<float*>(hardcap_score_max.data_ptr<float>()),
          total);
  hardcap_histogram_kernel<c10::BFloat16><<<
      reduce_blocks,
      reduce_threads,
      0,
      at::cuda::getDefaultCUDAStream()>>>(
          input.data_ptr<c10::BFloat16>(),
          stats_ptr,
          overflow.data_ptr<int32_t>(),
          hardcap_score_max.data_ptr<float>(),
          const_cast<int32_t*>(hardcap_histogram.data_ptr<int32_t>()),
          total);
  hardcap_update_strict_threshold_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
      hardcap_histogram.data_ptr<int32_t>(),
      stats_ptr,
      hardcap_score_max.data_ptr<float>(),
      overflow.data_ptr<int32_t>(),
      main_amax_ptr,
      num_selected.data_ptr<int32_t>(),
      hard_capacity);
  if (direct_nomask) {
    row_count_main_amax_total_nomask_if_overflow_kernel<c10::BFloat16><<<
        static_cast<int>(rows),
        stats_threads,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            const_cast<int32_t*>(row_counts.data_ptr<int32_t>()),
            main_amax_ptr,
            const_cast<int32_t*>(num_selected.data_ptr<int32_t>()),
            stats_ptr,
            overflow.data_ptr<int32_t>(),
            rows,
            cols);
  } else {
    row_count_main_amax_total_if_overflow_kernel<c10::BFloat16><<<
        static_cast<int>(rows),
        stats_threads,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            const_cast<int32_t*>(row_counts.data_ptr<int32_t>()),
            reinterpret_cast<uint16_t*>(const_cast<int16_t*>(selection_masks.data_ptr<int16_t>())),
            main_amax_ptr,
            const_cast<int32_t*>(num_selected.data_ptr<int32_t>()),
            stats_ptr,
            overflow.data_ptr<int32_t>(),
            rows,
            cols);
  }

  if (overlap_columnwise && masked_rht_outlier_reuse) {
    record_stream_then_wait(main_raw_stream, column_raw_stream);
    launch_custom_te_rht_columnwise_quant_out(input,
                                              selection_masks,
                                              ensure_rht_output(column_stream),
                                              const_cast<at::Tensor&>(columnwise_data),
                                              const_cast<at::Tensor&>(columnwise_scale),
                                              const_cast<at::Tensor&>(columnwise_amax),
                                              rows,
                                              cols,
                                              column_scale_inner,
                                              static_cast<int>(rht_random_sign_mask_t),
                                              true,
                                              column_raw_stream);
    columnwise_launched_async = true;
  }

  reset_padded_spill_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
      const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
      const_cast<int32_t*>(overflow_count.data_ptr<int32_t>()));

  const int fill_threads = normalize_threads(fill_threads_arg, cols <= 4096 ? 128 : 256);
  if (direct_nomask && fill_threads == 64) {
    fill_padded_rowwise_quant_nomask_kernel<c10::BFloat16, 64><<<
        static_cast<int>(rows),
        64,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            row_counts.data_ptr<int32_t>(),
            const_cast<c10::BFloat16*>(padded_values.data_ptr<c10::BFloat16>()),
            const_cast<int16_t*>(padded_cols.data_ptr<int16_t>()),
            const_cast<int32_t*>(overflow_rows.data_ptr<int32_t>()),
            const_cast<int16_t*>(overflow_cols.data_ptr<int16_t>()),
            const_cast<c10::BFloat16*>(overflow_values.data_ptr<c10::BFloat16>()),
            const_cast<int32_t*>(overflow_count.data_ptr<int32_t>()),
            const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
            main_amax_ptr,
            stats_ptr,
            const_cast<uint8_t*>(rowwise_data.data_ptr<uint8_t>()),
            const_cast<uint8_t*>(rowwise_scale.data_ptr<uint8_t>()),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(max_per_row),
            overflow_capacity);
  } else if (direct_nomask && fill_threads == 256) {
    fill_padded_rowwise_quant_nomask_kernel<c10::BFloat16, 256><<<
        static_cast<int>(rows),
        256,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            row_counts.data_ptr<int32_t>(),
            const_cast<c10::BFloat16*>(padded_values.data_ptr<c10::BFloat16>()),
            const_cast<int16_t*>(padded_cols.data_ptr<int16_t>()),
            const_cast<int32_t*>(overflow_rows.data_ptr<int32_t>()),
            const_cast<int16_t*>(overflow_cols.data_ptr<int16_t>()),
            const_cast<c10::BFloat16*>(overflow_values.data_ptr<c10::BFloat16>()),
            const_cast<int32_t*>(overflow_count.data_ptr<int32_t>()),
            const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
            main_amax_ptr,
            stats_ptr,
            const_cast<uint8_t*>(rowwise_data.data_ptr<uint8_t>()),
            const_cast<uint8_t*>(rowwise_scale.data_ptr<uint8_t>()),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(max_per_row),
            overflow_capacity);
  } else if (direct_nomask) {
    fill_padded_rowwise_quant_nomask_kernel<c10::BFloat16, 128><<<
        static_cast<int>(rows),
        128,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            row_counts.data_ptr<int32_t>(),
            const_cast<c10::BFloat16*>(padded_values.data_ptr<c10::BFloat16>()),
            const_cast<int16_t*>(padded_cols.data_ptr<int16_t>()),
            const_cast<int32_t*>(overflow_rows.data_ptr<int32_t>()),
            const_cast<int16_t*>(overflow_cols.data_ptr<int16_t>()),
            const_cast<c10::BFloat16*>(overflow_values.data_ptr<c10::BFloat16>()),
            const_cast<int32_t*>(overflow_count.data_ptr<int32_t>()),
            const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
            main_amax_ptr,
            stats_ptr,
            const_cast<uint8_t*>(rowwise_data.data_ptr<uint8_t>()),
            const_cast<uint8_t*>(rowwise_scale.data_ptr<uint8_t>()),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(max_per_row),
            overflow_capacity);
  } else if (fill_threads == 64) {
    fill_padded_rowwise_quant_kernel<c10::BFloat16, 64><<<
        static_cast<int>(rows),
        64,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            row_counts.data_ptr<int32_t>(),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            const_cast<c10::BFloat16*>(padded_values.data_ptr<c10::BFloat16>()),
            const_cast<int16_t*>(padded_cols.data_ptr<int16_t>()),
            const_cast<int32_t*>(overflow_rows.data_ptr<int32_t>()),
            const_cast<int16_t*>(overflow_cols.data_ptr<int16_t>()),
            const_cast<c10::BFloat16*>(overflow_values.data_ptr<c10::BFloat16>()),
            const_cast<int32_t*>(overflow_count.data_ptr<int32_t>()),
            const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
            main_amax_ptr,
            const_cast<uint8_t*>(rowwise_data.data_ptr<uint8_t>()),
            const_cast<uint8_t*>(rowwise_scale.data_ptr<uint8_t>()),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(max_per_row),
            overflow_capacity);
  } else if (fill_threads == 256) {
    fill_padded_rowwise_quant_kernel<c10::BFloat16, 256><<<
        static_cast<int>(rows),
        256,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            row_counts.data_ptr<int32_t>(),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            const_cast<c10::BFloat16*>(padded_values.data_ptr<c10::BFloat16>()),
            const_cast<int16_t*>(padded_cols.data_ptr<int16_t>()),
            const_cast<int32_t*>(overflow_rows.data_ptr<int32_t>()),
            const_cast<int16_t*>(overflow_cols.data_ptr<int16_t>()),
            const_cast<c10::BFloat16*>(overflow_values.data_ptr<c10::BFloat16>()),
            const_cast<int32_t*>(overflow_count.data_ptr<int32_t>()),
            const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
            main_amax_ptr,
            const_cast<uint8_t*>(rowwise_data.data_ptr<uint8_t>()),
            const_cast<uint8_t*>(rowwise_scale.data_ptr<uint8_t>()),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(max_per_row),
            overflow_capacity);
  } else {
    fill_padded_rowwise_quant_kernel<c10::BFloat16, 128><<<
        static_cast<int>(rows),
        128,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            row_counts.data_ptr<int32_t>(),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            const_cast<c10::BFloat16*>(padded_values.data_ptr<c10::BFloat16>()),
            const_cast<int16_t*>(padded_cols.data_ptr<int16_t>()),
            const_cast<int32_t*>(overflow_rows.data_ptr<int32_t>()),
            const_cast<int16_t*>(overflow_cols.data_ptr<int16_t>()),
            const_cast<c10::BFloat16*>(overflow_values.data_ptr<c10::BFloat16>()),
            const_cast<int32_t*>(overflow_count.data_ptr<int32_t>()),
            const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
            main_amax_ptr,
            const_cast<uint8_t*>(rowwise_data.data_ptr<uint8_t>()),
            const_cast<uint8_t*>(rowwise_scale.data_ptr<uint8_t>()),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(max_per_row),
            overflow_capacity);
  }

  build_r25_active_rows_schedule_into(row_counts,
                                      rows,
                                      active_rows_heavy_light,
                                      active_row_count,
                                      active_sort_rows_in,
                                      active_sort_rows_out,
                                      active_sort_keys_in,
                                      active_sort_keys_out,
                                      active_sort_temp);
  if (columnwise_launched_async) {
    record_stream_then_wait(column_raw_stream, main_raw_stream);
  } else {
    launch_custom_te_rht_columnwise_quant_out(input,
                                              selection_masks,
                                              ensure_rht_output(main_stream),
                                              const_cast<at::Tensor&>(columnwise_data),
                                              const_cast<at::Tensor&>(columnwise_scale),
                                              const_cast<at::Tensor&>(columnwise_amax),
                                              rows,
                                              cols,
                                              column_scale_inner,
                                              static_cast<int>(rht_random_sign_mask_t),
                                              masked_rht_outlier_reuse,
                                              at::cuda::getDefaultCUDAStream());
  }

  auto main_amax = stats_amax.narrow(0, 4, 1);
  auto stats = stats_amax.narrow(0, 0, 4);

  return {const_cast<at::Tensor&>(padded_values),
          const_cast<at::Tensor&>(padded_cols),
          const_cast<at::Tensor&>(row_counts),
          const_cast<at::Tensor&>(num_selected),
          const_cast<at::Tensor&>(overflow),
          const_cast<at::Tensor&>(overflow_rows),
          const_cast<at::Tensor&>(overflow_cols),
          const_cast<at::Tensor&>(overflow_values),
          const_cast<at::Tensor&>(overflow_count),
          const_cast<at::Tensor&>(rowwise_data),
          const_cast<at::Tensor&>(rowwise_scale),
          main_amax,
          stats,
          const_cast<at::Tensor&>(columnwise_data),
          const_cast<at::Tensor&>(columnwise_scale),
          const_cast<at::Tensor&>(columnwise_amax),
          const_cast<at::Tensor&>(active_rows_heavy_light),
          const_cast<at::Tensor&>(active_row_count),
          const_cast<at::Tensor&>(selection_masks)};
}

std::vector<at::Tensor> adaptive_rowcol_loghist_quant_fast_out_impl(
    const at::Tensor& input,
    const at::Tensor& flat_indices,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& row_counts,
    const at::Tensor& row_offsets,
    const at::Tensor& selection_masks,
    const at::Tensor& num_selected,
    const at::Tensor& overflow,
    const at::Tensor& stats_amax,
    const at::Tensor& rowwise_data,
    const at::Tensor& rowwise_scale,
    const at::Tensor& columnwise_data,
    const at::Tensor& columnwise_scale,
    const at::Tensor& columnwise_amax,
    const at::Tensor& rht_output_t,
    const at::Tensor& scan_temp,
    const at::Tensor& log_hist,
    const at::Tensor& log_params,
    double ratio,
    int64_t hist_bins,
    int64_t min_exp,
    int64_t seed,
    int64_t capacity,
    int64_t stats_threads_arg,
    int64_t fill_threads_arg,
    int64_t columnwise_source_arg,
    int64_t rht_random_sign_mask_t,
    bool overlap_columnwise,
    bool auto_expand_capacity) {
  const auto rows = input.size(0);
  const auto cols = input.size(1);
  const int64_t total = rows * cols;
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "adaptive rowcol loghist fast supports BF16 only");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(hist_bins == 64 || hist_bins == 128 || hist_bins == 256,
              "hist_bins must be one of 64, 128, 256");
  TORCH_CHECK(min_exp >= -126 && min_exp <= 111, "min_exp out of BF16 normal exponent range");
  TORCH_CHECK(columnwise_source_arg == kColumnwiseDirect ||
                  columnwise_source_arg == kColumnwiseOutlierReuse,
              "columnwise_source must be direct or outlier_reuse");
  TORCH_CHECK(capacity == flat_indices.numel(), "capacity must match flat_indices size");
  TORCH_CHECK(row_counts.numel() == rows, "row_counts shape mismatch");
  TORCH_CHECK(row_offsets.numel() == rows + 1, "row_offsets shape mismatch");
  TORCH_CHECK(selection_masks.numel() == rows * (cols / 16), "selection_masks shape mismatch");
  TORCH_CHECK(rht_output_t.numel() == 0 ||
                  (rht_output_t.size(0) == cols && rht_output_t.size(1) == rows),
              "rht_output_t shape mismatch");
  TORCH_CHECK(log_hist.numel() >= kLogHistMaxBins, "log_hist must have at least 256 elements");
  TORCH_CHECK(log_params.numel() >= 8, "log_params must have at least 8 elements");

  float* stats_amax_ptr = const_cast<float*>(stats_amax.data_ptr<float>());
  float* stats_ptr = stats_amax_ptr;
  float* main_amax_ptr = stats_amax_ptr + 4;
  const bool masked_rht_outlier_reuse = columnwise_source_arg == kColumnwiseOutlierReuse;
  const int64_t row_scale_inner = round_up_int64((cols + 15) / 16, 4);
  const int64_t column_scale_inner = round_up_int64((rows + 15) / 16, 4);

  const auto main_stream = at::cuda::getDefaultCUDAStream(input.device().index());
  auto column_stream = at::cuda::getStreamFromPool(false, input.device().index());
  const cudaStream_t main_raw_stream = main_stream.stream();
  const cudaStream_t column_raw_stream = column_stream.stream();
  at::Tensor rht_output_keepalive;
  bool columnwise_launched_async = false;

  auto ensure_rht_output = [&](const c10::cuda::CUDAStream& stream) -> at::Tensor& {
    if (!rht_output_keepalive.defined()) {
      rht_output_keepalive = at::empty({cols, rows}, input.options());
    }
    record_tensor_on_stream(rht_output_keepalive, stream);
    return rht_output_keepalive;
  };

  reset_loghist_workspace_kernel<<<1, kLogHistThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
      stats_amax_ptr,
      const_cast<int32_t*>(num_selected.data_ptr<int32_t>()),
      const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
      const_cast<int32_t*>(log_hist.data_ptr<int32_t>()),
      const_cast<int64_t*>(log_params.data_ptr<int64_t>()));

  if (overlap_columnwise && columnwise_source_arg == kColumnwiseDirect) {
    record_stream_then_wait(main_raw_stream, column_raw_stream);
    launch_custom_te_rht_columnwise_quant_out(input,
                                              selection_masks,
                                              ensure_rht_output(column_stream),
                                              const_cast<at::Tensor&>(columnwise_data),
                                              const_cast<at::Tensor&>(columnwise_scale),
                                              const_cast<at::Tensor&>(columnwise_amax),
                                              rows,
                                              cols,
                                              column_scale_inner,
                                              static_cast<int>(rht_random_sign_mask_t),
                                              false,
                                              column_raw_stream);
    columnwise_launched_async = true;
  }

  const uint16_t* input_bits = reinterpret_cast<const uint16_t*>(input.data_ptr<c10::BFloat16>());
  launch_build_loghist(hist_bins,
                       input_bits,
                       const_cast<int32_t*>(log_hist.data_ptr<int32_t>()),
                       total,
                       static_cast<int>(min_exp),
                       at::cuda::getDefaultCUDAStream());
  launch_choose_loghist_boundary(hist_bins,
                                 log_hist.data_ptr<int32_t>(),
                                 const_cast<int64_t*>(log_params.data_ptr<int64_t>()),
                                 stats_ptr,
                                 ratio,
                                 total,
                                 at::cuda::getDefaultCUDAStream());

  const int stats_threads = normalize_threads(stats_threads_arg, cols <= 4096 ? 128 : 256);
  launch_row_count_loghist(hist_bins,
                           input.data_ptr<c10::BFloat16>(),
                           input_bits,
                           const_cast<int32_t*>(row_counts.data_ptr<int32_t>()),
                           reinterpret_cast<uint16_t*>(
                               const_cast<int16_t*>(selection_masks.data_ptr<int16_t>())),
                           main_amax_ptr,
                           const_cast<int32_t*>(num_selected.data_ptr<int32_t>()),
                           log_params.data_ptr<int64_t>(),
                           rows,
                           cols,
                           static_cast<int>(min_exp),
                           static_cast<uint32_t>(seed),
                           stats_threads,
                           at::cuda::getDefaultCUDAStream());

  if (overlap_columnwise && masked_rht_outlier_reuse) {
    record_stream_then_wait(main_raw_stream, column_raw_stream);
    launch_custom_te_rht_columnwise_quant_out(input,
                                              selection_masks,
                                              ensure_rht_output(column_stream),
                                              const_cast<at::Tensor&>(columnwise_data),
                                              const_cast<at::Tensor&>(columnwise_scale),
                                              const_cast<at::Tensor&>(columnwise_amax),
                                              rows,
                                              cols,
                                              column_scale_inner,
                                              static_cast<int>(rht_random_sign_mask_t),
                                              true,
                                              column_raw_stream);
    columnwise_launched_async = true;
  }

  at::Tensor target_flat_indices = const_cast<at::Tensor&>(flat_indices);
  at::Tensor target_outlier_values = const_cast<at::Tensor&>(outlier_values);
  at::Tensor target_outlier_cols = const_cast<at::Tensor&>(outlier_cols);
  int64_t output_capacity = capacity;
  if (auto_expand_capacity) {
    const int32_t selected_host =
        copy_int32_to_host(num_selected.data_ptr<int32_t>(), main_raw_stream);
    if (selected_host > capacity) {
      output_capacity = static_cast<int64_t>(selected_host);
      target_flat_indices = at::empty({output_capacity}, input.options().dtype(at::kInt));
      target_outlier_values = at::empty({output_capacity}, input.options());
      target_outlier_cols = at::empty({output_capacity}, input.options().dtype(at::kShort));
    }
  }

  size_t temp_bytes = static_cast<size_t>(scan_temp.numel());
  cub::DeviceScan::ExclusiveSum(
      const_cast<uint8_t*>(scan_temp.data_ptr<uint8_t>()),
      temp_bytes,
      row_counts.data_ptr<int32_t>(),
      const_cast<int32_t*>(row_offsets.data_ptr<int32_t>()),
      static_cast<int>(rows),
      at::cuda::getDefaultCUDAStream());

  const int fill_threads = normalize_threads(fill_threads_arg, cols <= 4096 ? 128 : 256);
  if (fill_threads == 64) {
    fill_csr_rowwise_quant_prefix_kernel<c10::BFloat16, 64><<<
        static_cast<int>(rows),
        64,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            const_cast<int32_t*>(row_offsets.data_ptr<int32_t>()),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            target_flat_indices.data_ptr<int32_t>(),
            target_outlier_values.data_ptr<c10::BFloat16>(),
            target_outlier_cols.data_ptr<int16_t>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            num_selected.data_ptr<int32_t>(),
            const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
            main_amax_ptr,
            const_cast<uint8_t*>(rowwise_data.data_ptr<uint8_t>()),
            const_cast<uint8_t*>(rowwise_scale.data_ptr<uint8_t>()),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(output_capacity));
  } else if (fill_threads == 256) {
    fill_csr_rowwise_quant_prefix_kernel<c10::BFloat16, 256><<<
        static_cast<int>(rows),
        256,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            const_cast<int32_t*>(row_offsets.data_ptr<int32_t>()),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            target_flat_indices.data_ptr<int32_t>(),
            target_outlier_values.data_ptr<c10::BFloat16>(),
            target_outlier_cols.data_ptr<int16_t>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            num_selected.data_ptr<int32_t>(),
            const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
            main_amax_ptr,
            const_cast<uint8_t*>(rowwise_data.data_ptr<uint8_t>()),
            const_cast<uint8_t*>(rowwise_scale.data_ptr<uint8_t>()),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(output_capacity));
  } else {
    fill_csr_rowwise_quant_prefix_kernel<c10::BFloat16, 128><<<
        static_cast<int>(rows),
        128,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            const_cast<int32_t*>(row_offsets.data_ptr<int32_t>()),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            target_flat_indices.data_ptr<int32_t>(),
            target_outlier_values.data_ptr<c10::BFloat16>(),
            target_outlier_cols.data_ptr<int16_t>(),
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            num_selected.data_ptr<int32_t>(),
            const_cast<int32_t*>(overflow.data_ptr<int32_t>()),
            main_amax_ptr,
            const_cast<uint8_t*>(rowwise_data.data_ptr<uint8_t>()),
            const_cast<uint8_t*>(rowwise_scale.data_ptr<uint8_t>()),
            rows,
            cols,
            row_scale_inner,
            static_cast<int32_t>(output_capacity));
  }

  if (columnwise_launched_async) {
    record_stream_then_wait(column_raw_stream, main_raw_stream);
  } else {
    launch_custom_te_rht_columnwise_quant_out(input,
                                              selection_masks,
                                              ensure_rht_output(main_stream),
                                              const_cast<at::Tensor&>(columnwise_data),
                                              const_cast<at::Tensor&>(columnwise_scale),
                                              const_cast<at::Tensor&>(columnwise_amax),
                                              rows,
                                              cols,
                                              column_scale_inner,
                                              static_cast<int>(rht_random_sign_mask_t),
                                              masked_rht_outlier_reuse,
                                              at::cuda::getDefaultCUDAStream());
  }

  auto main_amax = stats_amax.narrow(0, 4, 1);
  auto stats = stats_amax.narrow(0, 0, 4);
  auto dense_main = at::empty({0}, input.options());

  return {target_flat_indices,
          target_outlier_values,
          target_outlier_cols,
          const_cast<at::Tensor&>(row_offsets),
          const_cast<at::Tensor&>(num_selected),
          const_cast<at::Tensor&>(overflow),
          const_cast<at::Tensor&>(rowwise_data),
          const_cast<at::Tensor&>(rowwise_scale),
          main_amax,
          stats,
          dense_main,
          const_cast<at::Tensor&>(columnwise_data),
          const_cast<at::Tensor&>(columnwise_scale),
          const_cast<at::Tensor&>(columnwise_amax)};
}

std::vector<at::Tensor> adaptive_rowcol_refill_csr_impl(
    const at::Tensor& input,
    const at::Tensor& row_offsets,
    const at::Tensor& selection_masks,
    int64_t selected_nnz,
    int64_t fill_threads_arg) {
  const auto rows = input.size(0);
  const auto cols = input.size(1);
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "adaptive rowcol fast supports BF16 only");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(row_offsets.is_cuda() && selection_masks.is_cuda(),
              "row_offsets and selection_masks must be CUDA tensors");
  TORCH_CHECK(row_offsets.scalar_type() == at::kInt, "row_offsets must be int32");
  TORCH_CHECK(selection_masks.scalar_type() == at::kShort, "selection_masks must be int16");
  TORCH_CHECK(row_offsets.numel() == rows + 1, "row_offsets shape mismatch");
  TORCH_CHECK(cols % 16 == 0, "cols must be divisible by 16");
  TORCH_CHECK(selection_masks.numel() == rows * (cols / 16), "selection_masks shape mismatch");
  TORCH_CHECK(selected_nnz >= 0 && selected_nnz <= std::numeric_limits<int32_t>::max(),
              "selected_nnz must fit int32");

  auto flat_indices = at::empty({selected_nnz}, input.options().dtype(at::kInt));
  auto outlier_values = at::empty({selected_nnz}, input.options());
  auto outlier_cols = at::empty({selected_nnz}, input.options().dtype(at::kShort));
  auto overflow = at::zeros({1}, input.options().dtype(at::kInt));

  if (rows == 0 || cols == 0) {
    return {flat_indices, outlier_values, outlier_cols, overflow};
  }

  const int fill_threads = normalize_threads(fill_threads_arg, cols <= 4096 ? 128 : 256);
  if (fill_threads == 64) {
    refill_csr_payload_prefix_kernel<c10::BFloat16, 64><<<
        static_cast<int>(rows),
        64,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            row_offsets.data_ptr<int32_t>(),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            flat_indices.data_ptr<int32_t>(),
            outlier_values.data_ptr<c10::BFloat16>(),
            outlier_cols.data_ptr<int16_t>(),
            rows,
            cols,
            static_cast<int32_t>(selected_nnz),
            overflow.data_ptr<int32_t>());
  } else if (fill_threads == 256) {
    refill_csr_payload_prefix_kernel<c10::BFloat16, 256><<<
        static_cast<int>(rows),
        256,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            row_offsets.data_ptr<int32_t>(),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            flat_indices.data_ptr<int32_t>(),
            outlier_values.data_ptr<c10::BFloat16>(),
            outlier_cols.data_ptr<int16_t>(),
            rows,
            cols,
            static_cast<int32_t>(selected_nnz),
            overflow.data_ptr<int32_t>());
  } else {
    refill_csr_payload_prefix_kernel<c10::BFloat16, 128><<<
        static_cast<int>(rows),
        128,
        0,
        at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<c10::BFloat16>(),
            row_offsets.data_ptr<int32_t>(),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            flat_indices.data_ptr<int32_t>(),
            outlier_values.data_ptr<c10::BFloat16>(),
            outlier_cols.data_ptr<int16_t>(),
            rows,
            cols,
            static_cast<int32_t>(selected_nnz),
            overflow.data_ptr<int32_t>());
  }

  return {flat_indices, outlier_values, outlier_cols, overflow};
}

__global__ void cap_split_counts_kernel(const int32_t* __restrict__ row_offsets,
                                        int32_t* __restrict__ light_counts,
                                        int32_t* __restrict__ heavy_counts,
                                        int32_t rows,
                                        int32_t cap);
__global__ void cap_split_set_tail_kernel(const int32_t* __restrict__ counts,
                                          int32_t* __restrict__ offsets,
                                          int32_t rows);
__global__ void cap_split_active_rows_kernel(const int32_t* __restrict__ counts,
                                             int32_t* __restrict__ active_rows,
                                             int32_t* __restrict__ active_count,
                                             int32_t rows);
__global__ void cap_split_row_records_kernel(const int32_t* __restrict__ row_offsets,
                                             const int32_t* __restrict__ counts,
                                             const int32_t* __restrict__ active_rows,
                                             int64_t* __restrict__ row_records,
                                             const int32_t* __restrict__ active_count);
__global__ void cap_split_rowblocks_kernel(const int32_t* __restrict__ counts,
                                           int32_t* __restrict__ rowblocks,
                                           int32_t* __restrict__ rowblock_count,
                                           int32_t rows,
                                           int32_t rows_per_block);
__global__ void cap_split_tile_offsets_kernel(int32_t* __restrict__ tile_offsets,
                                              const int32_t* __restrict__ active_count);
__global__ void policy_split_init_counts_sort_kernel(const int32_t* __restrict__ row_offsets,
                                                     int32_t* __restrict__ light_counts,
                                                     int32_t* __restrict__ heavy_counts,
                                                     int64_t* __restrict__ sort_keys,
                                                     int32_t* __restrict__ sort_rows,
                                                     int32_t rows,
                                                     int32_t policy_mode,
                                                     int32_t param0,
                                                     int32_t param1);
__global__ void policy_split_select_densepack_kernel(const int64_t* __restrict__ sorted_keys,
                                                     const int32_t* __restrict__ sorted_rows,
                                                     const int32_t* __restrict__ row_offsets,
                                                     int32_t* __restrict__ light_counts,
                                                     int32_t* __restrict__ heavy_counts,
                                                     int32_t rows,
                                                     int32_t row_budget);
__global__ void policy_split_select_entrybudget_kernel(const int64_t* __restrict__ sorted_keys,
                                                       const int32_t* __restrict__ sorted_rows,
                                                       const int32_t* __restrict__ row_offsets,
                                                       int32_t* __restrict__ light_counts,
                                                       int32_t* __restrict__ heavy_counts,
                                                       int32_t rows,
                                                       int32_t entry_budget);
__global__ void policy_split_init_counts_only_kernel(const int32_t* __restrict__ row_offsets,
                                                     int32_t* __restrict__ light_counts,
                                                     int32_t* __restrict__ heavy_counts,
                                                     int32_t rows);
__global__ void policy_split_hist_rows_kernel(const int32_t* __restrict__ row_offsets,
                                              int32_t* __restrict__ hist_rows,
                                              int32_t rows,
                                              int32_t max_bucket);
__global__ void policy_split_compute_densepack_limits_kernel(const int32_t* __restrict__ hist_rows,
                                                             int32_t* __restrict__ bucket_limits,
                                                             int32_t max_bucket,
                                                             int32_t min_count,
                                                             int32_t row_budget);
__global__ void policy_split_compute_entrybudget_limits_kernel(const int32_t* __restrict__ hist_rows,
                                                               int32_t* __restrict__ bucket_limits,
                                                               int32_t max_bucket,
                                                               int32_t entry_budget);
__global__ void policy_split_apply_bucket_limits_kernel(const int32_t* __restrict__ row_offsets,
                                                        const int32_t* __restrict__ bucket_limits,
                                                        int32_t* __restrict__ light_counts,
                                                        int32_t* __restrict__ heavy_counts,
                                                        int32_t rows);

__global__ void cap_split_counts_kernel(const int32_t* __restrict__ row_offsets,
                                        int32_t* __restrict__ light_counts,
                                        int32_t* __restrict__ heavy_counts,
                                        int32_t rows,
                                        int32_t cap) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int32_t count = row_offsets[row + 1] - row_offsets[row];
  const bool is_light = count > 0 && count <= cap;
  light_counts[row] = is_light ? count : 0;
  heavy_counts[row] = (count > 0 && !is_light) ? count : 0;
}

__global__ void cap_split_set_tail_kernel(const int32_t* __restrict__ counts,
                                          int32_t* __restrict__ offsets,
                                          int32_t rows) {
  if (threadIdx.x == 0) {
    offsets[rows] = offsets[rows - 1] + counts[rows - 1];
  }
}

__global__ void cap_split_scatter_kernel(const int32_t* __restrict__ row_offsets,
                                         const int32_t* __restrict__ light_offsets,
                                         const int32_t* __restrict__ heavy_offsets,
                                         const int32_t* __restrict__ light_counts,
                                         const int32_t* __restrict__ heavy_counts,
                                         const int16_t* __restrict__ in_cols,
                                         const uint16_t* __restrict__ in_values,
                                         int32_t* __restrict__ light_cols,
                                         uint16_t* __restrict__ light_values,
                                         int32_t* __restrict__ light_flat_indices,
                                         int32_t* __restrict__ light_entry_records,
                                         int32_t* __restrict__ heavy_cols,
                                         uint16_t* __restrict__ heavy_values,
                                         int32_t* __restrict__ heavy_flat_indices,
                                         int32_t cols,
                                         int32_t rows) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  const int32_t count = end - start;
  if (count <= 0) {
    return;
  }
  const bool is_light = light_counts[row] > 0;
  const int32_t dst_base = is_light ? light_offsets[row] : heavy_offsets[row];
  for (int32_t j = threadIdx.x; j < count; j += blockDim.x) {
    const int32_t src = start + j;
    const uint16_t col = static_cast<uint16_t>(in_cols[src]);
    const uint16_t value_bits = in_values[src];
    const int32_t dst = dst_base + j;
    if (is_light) {
      light_cols[dst] = static_cast<int32_t>(col);
      light_values[dst] = value_bits;
      light_flat_indices[dst] = row * cols + static_cast<int32_t>(col);
      light_entry_records[dst] =
          static_cast<int32_t>((static_cast<uint32_t>(col) << 16) |
                               static_cast<uint32_t>(value_bits));
    } else {
      heavy_cols[dst] = static_cast<int32_t>(col);
      heavy_values[dst] = value_bits;
      heavy_flat_indices[dst] = row * cols + static_cast<int32_t>(col);
    }
  }
}

__global__ void cap_split_scatter_i32_kernel(const int32_t* __restrict__ row_offsets,
                                             const int32_t* __restrict__ light_offsets,
                                             const int32_t* __restrict__ heavy_offsets,
                                             const int32_t* __restrict__ light_counts,
                                             const int32_t* __restrict__ heavy_counts,
                                             const int32_t* __restrict__ in_cols,
                                             const uint16_t* __restrict__ in_values,
                                             int32_t* __restrict__ light_cols,
                                             uint16_t* __restrict__ light_values,
                                             int32_t* __restrict__ light_flat_indices,
                                             int32_t* __restrict__ light_entry_records,
                                             int32_t* __restrict__ heavy_cols,
                                             uint16_t* __restrict__ heavy_values,
                                             int32_t* __restrict__ heavy_flat_indices,
                                             int32_t cols,
                                             int32_t rows) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  const int32_t count = end - start;
  if (count <= 0) {
    return;
  }
  const bool is_light = light_counts[row] > 0;
  const int32_t dst_base = is_light ? light_offsets[row] : heavy_offsets[row];
  for (int32_t j = threadIdx.x; j < count; j += blockDim.x) {
    const int32_t src = start + j;
    const uint16_t col = static_cast<uint16_t>(in_cols[src]);
    const uint16_t value_bits = in_values[src];
    const int32_t dst = dst_base + j;
    if (is_light) {
      light_cols[dst] = static_cast<int32_t>(col);
      light_values[dst] = value_bits;
      light_flat_indices[dst] = row * cols + static_cast<int32_t>(col);
      light_entry_records[dst] =
          static_cast<int32_t>((static_cast<uint32_t>(col) << 16) |
                               static_cast<uint32_t>(value_bits));
    } else {
      heavy_cols[dst] = static_cast<int32_t>(col);
      heavy_values[dst] = value_bits;
      heavy_flat_indices[dst] = row * cols + static_cast<int32_t>(col);
    }
  }
}

__global__ void cap_split_active_rows_kernel(const int32_t* __restrict__ counts,
                                             int32_t* __restrict__ active_rows,
                                             int32_t* __restrict__ active_count,
                                             int32_t rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || counts[row] <= 0) {
    return;
  }
  const int32_t dst = atomicAdd(active_count, 1);
  active_rows[dst] = row;
}

__global__ void cap_split_active_rows_records_kernel(const int32_t* __restrict__ counts,
                                                     const int32_t* __restrict__ row_offsets,
                                                     int32_t* __restrict__ active_rows,
                                                     int64_t* __restrict__ row_records,
                                                     int32_t* __restrict__ active_count,
                                                     int32_t rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || counts[row] <= 0) {
    return;
  }
  const int32_t dst = atomicAdd(active_count, 1);
  active_rows[dst] = row;
  const uint64_t entry_start = static_cast<uint32_t>(row_offsets[row]);
  const uint64_t count = static_cast<uint32_t>(counts[row]) & 0xFFFFu;
  const uint64_t row_bits = static_cast<uint32_t>(row) & 0xFFFFu;
  row_records[dst] = static_cast<int64_t>((entry_start << 32) | (count << 16) | row_bits);
}

__global__ void cap_split_row_records_kernel(const int32_t* __restrict__ row_offsets,
                                             const int32_t* __restrict__ counts,
                                             const int32_t* __restrict__ active_rows,
                                             int64_t* __restrict__ row_records,
                                             const int32_t* __restrict__ active_count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = active_count[0];
  if (idx >= total) {
    return;
  }
  const int32_t row = active_rows[idx];
  const uint64_t entry_start = static_cast<uint32_t>(row_offsets[row]);
  const uint64_t count = static_cast<uint32_t>(counts[row]) & 0xFFFFu;
  const uint64_t row_bits = static_cast<uint32_t>(row) & 0xFFFFu;
  row_records[idx] = static_cast<int64_t>((entry_start << 32) | (count << 16) | row_bits);
}

__global__ void cap_split_rowblocks_kernel(const int32_t* __restrict__ counts,
                                           int32_t* __restrict__ rowblocks,
                                           int32_t* __restrict__ rowblock_count,
                                           int32_t rows,
                                           int32_t rows_per_block) {
  const int block = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_blocks = (rows + rows_per_block - 1) / rows_per_block;
  if (block >= total_blocks) {
    return;
  }
  const int row0 = block * rows_per_block;
  int32_t sum = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int row = row0 + i;
    if (i < rows_per_block && row < rows) {
      sum += counts[row];
    }
  }
  if (sum > 0) {
    const int32_t dst = atomicAdd(rowblock_count, 1);
    rowblocks[dst] = block;
  }
}

__global__ void cap_split_tile_offsets_kernel(int32_t* __restrict__ tile_offsets,
                                              const int32_t* __restrict__ active_count) {
  if (threadIdx.x == 0) {
    tile_offsets[0] = 0;
    tile_offsets[1] = active_count[0];
  }
}

__global__ void policy_split_init_counts_sort_kernel(const int32_t* __restrict__ row_offsets,
                                                     int32_t* __restrict__ light_counts,
                                                     int32_t* __restrict__ heavy_counts,
                                                     int64_t* __restrict__ sort_keys,
                                                     int32_t* __restrict__ sort_rows,
                                                     int32_t rows,
                                                     int32_t policy_mode,
                                                     int32_t param0,
                                                     int32_t param1) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int32_t count = row_offsets[row + 1] - row_offsets[row];
  light_counts[row] = 0;
  heavy_counts[row] = count > 0 ? count : 0;
  sort_rows[row] = row;

  bool candidate = false;
  if (policy_mode == 1) {
    // densepack:<min_count>:<row_budget>
    candidate = count >= param0;
  } else if (policy_mode == 2) {
    // entrybudget:<entry_budget>:<max_count>
    candidate = count > 0 && count <= param1;
  }
  if (candidate) {
    const uint64_t count_bits = static_cast<uint64_t>(static_cast<uint32_t>(count));
    const uint64_t row_tiebreak = 0xffffffffull - static_cast<uint32_t>(row);
    sort_keys[row] = static_cast<int64_t>((count_bits << 32) | row_tiebreak);
  } else {
    sort_keys[row] = 0;
  }
}

__global__ void policy_split_select_densepack_kernel(const int64_t* __restrict__ sorted_keys,
                                                     const int32_t* __restrict__ sorted_rows,
                                                     const int32_t* __restrict__ row_offsets,
                                                     int32_t* __restrict__ light_counts,
                                                     int32_t* __restrict__ heavy_counts,
                                                     int32_t rows,
                                                     int32_t row_budget) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int32_t selected_rows = 0;
  for (int32_t idx = 0; idx < rows && selected_rows < row_budget; ++idx) {
    if (sorted_keys[idx] == 0) {
      break;
    }
    const int32_t row = sorted_rows[idx];
    const int32_t count = row_offsets[row + 1] - row_offsets[row];
    light_counts[row] = count;
    heavy_counts[row] = 0;
    ++selected_rows;
  }
}

__global__ void policy_split_select_entrybudget_kernel(const int64_t* __restrict__ sorted_keys,
                                                       const int32_t* __restrict__ sorted_rows,
                                                       const int32_t* __restrict__ row_offsets,
                                                       int32_t* __restrict__ light_counts,
                                                       int32_t* __restrict__ heavy_counts,
                                                       int32_t rows,
                                                       int32_t entry_budget) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int32_t used = 0;
  for (int32_t idx = 0; idx < rows; ++idx) {
    if (sorted_keys[idx] == 0) {
      break;
    }
    const int32_t row = sorted_rows[idx];
    const int32_t count = row_offsets[row + 1] - row_offsets[row];
    if (used + count > entry_budget) {
      continue;
    }
    light_counts[row] = count;
    heavy_counts[row] = 0;
    used += count;
  }
}

__global__ void policy_split_init_counts_only_kernel(const int32_t* __restrict__ row_offsets,
                                                     int32_t* __restrict__ light_counts,
                                                     int32_t* __restrict__ heavy_counts,
                                                     int32_t rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int32_t count = row_offsets[row + 1] - row_offsets[row];
  light_counts[row] = 0;
  heavy_counts[row] = count > 0 ? count : 0;
}

__global__ void policy_split_select_densepack_bucket_kernel(const int32_t* __restrict__ row_offsets,
                                                            int32_t* __restrict__ light_counts,
                                                            int32_t* __restrict__ heavy_counts,
                                                            int32_t rows,
                                                            int32_t min_count,
                                                            int32_t row_budget) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int32_t max_count = 0;
  for (int32_t row = 0; row < rows; ++row) {
    const int32_t count = row_offsets[row + 1] - row_offsets[row];
    max_count = max(max_count, count);
  }
  int32_t selected_rows = 0;
  for (int32_t count_value = max_count; count_value >= min_count && selected_rows < row_budget;
       --count_value) {
    for (int32_t row = 0; row < rows && selected_rows < row_budget; ++row) {
      const int32_t count = row_offsets[row + 1] - row_offsets[row];
      if (count == count_value) {
        light_counts[row] = count;
        heavy_counts[row] = 0;
        ++selected_rows;
      }
    }
  }
}

__global__ void policy_split_select_entrybudget_bucket_kernel(const int32_t* __restrict__ row_offsets,
                                                              int32_t* __restrict__ light_counts,
                                                              int32_t* __restrict__ heavy_counts,
                                                              int32_t rows,
                                                              int32_t entry_budget,
                                                              int32_t max_count) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int32_t used = 0;
  for (int32_t count_value = max_count; count_value >= 1; --count_value) {
    for (int32_t row = 0; row < rows; ++row) {
      const int32_t count = row_offsets[row + 1] - row_offsets[row];
      if (count != count_value) {
        continue;
      }
      if (used + count > entry_budget) {
        continue;
      }
      light_counts[row] = count;
      heavy_counts[row] = 0;
      used += count;
    }
  }
}

__global__ void policy_split_hist_rows_kernel(const int32_t* __restrict__ row_offsets,
                                              int32_t* __restrict__ hist_rows,
                                              int32_t rows,
                                              int32_t max_bucket) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int32_t count = row_offsets[row + 1] - row_offsets[row];
  if (count > 0 && count <= max_bucket) {
    atomicAdd(hist_rows + count, 1);
  }
}

__global__ void policy_split_compute_densepack_limits_kernel(const int32_t* __restrict__ hist_rows,
                                                             int32_t* __restrict__ bucket_limits,
                                                             int32_t max_bucket,
                                                             int32_t min_count,
                                                             int32_t row_budget) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int32_t rows_left = row_budget;
  for (int32_t count_value = max_bucket; count_value >= min_count && rows_left > 0; --count_value) {
    const int32_t available = hist_rows[count_value];
    const int32_t take = min(available, rows_left);
    bucket_limits[count_value] = take;
    rows_left -= take;
  }
}

__global__ void policy_split_compute_entrybudget_limits_kernel(const int32_t* __restrict__ hist_rows,
                                                               int32_t* __restrict__ bucket_limits,
                                                               int32_t max_bucket,
                                                               int32_t entry_budget) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  int32_t entries_left = entry_budget;
  for (int32_t count_value = max_bucket; count_value >= 1; --count_value) {
    const int32_t available = hist_rows[count_value];
    if (available <= 0 || entries_left < count_value) {
      continue;
    }
    const int32_t take = min(available, entries_left / count_value);
    bucket_limits[count_value] = take;
    entries_left -= take * count_value;
  }
}

__global__ void policy_split_apply_bucket_limits_kernel(const int32_t* __restrict__ row_offsets,
                                                        const int32_t* __restrict__ bucket_limits,
                                                        int32_t* __restrict__ light_counts,
                                                        int32_t* __restrict__ heavy_counts,
                                                        int32_t rows) {
  const int32_t count_value = static_cast<int32_t>(blockIdx.x);
  if (count_value <= 0) {
    return;
  }
  const int32_t limit = bucket_limits[count_value];
  if (limit <= 0) {
    return;
  }
  int32_t selected = 0;
  for (int32_t row = 0; row < rows && selected < limit; ++row) {
    const int32_t count = row_offsets[row + 1] - row_offsets[row];
    if (count == count_value) {
      light_counts[row] = count;
      heavy_counts[row] = 0;
      ++selected;
    }
  }
}

__global__ void policy_split_apply_bucket_limits_atomic_kernel(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ bucket_limits,
    int32_t* __restrict__ bucket_taken,
    int32_t* __restrict__ light_counts,
    int32_t* __restrict__ heavy_counts,
    int32_t rows,
    int32_t max_bucket) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int32_t count = row_offsets[row + 1] - row_offsets[row];
  if (count <= 0 || count > max_bucket) {
    return;
  }
  const int32_t limit = bucket_limits[count];
  if (limit <= 0) {
    return;
  }
  const int32_t slot = atomicAdd(bucket_taken + count, 1);
  if (slot < limit) {
    light_counts[row] = count;
    heavy_counts[row] = 0;
  }
}

__global__ void direct_split_stats_kernel(const int32_t* __restrict__ light_offsets,
                                          const int32_t* __restrict__ heavy_offsets,
                                          const int32_t* __restrict__ light_active_count,
                                          const int32_t* __restrict__ heavy_active_count,
                                          const int32_t* __restrict__ light_rowblock_count,
                                          int32_t* __restrict__ light_rowblocks,
                                          int32_t* __restrict__ stats,
                                          int32_t rows) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  const int32_t rowblock_count = light_rowblock_count[0];
  if (rowblock_count == 0) {
    light_rowblocks[0] = 0;
  }
  stats[0] = light_offsets[rows];
  stats[1] = heavy_offsets[rows];
  stats[2] = light_active_count[0];
  stats[3] = heavy_active_count[0];
  stats[4] = rowblock_count > 0 ? rowblock_count : 1;
}

std::vector<at::Tensor> adaptive_rowcol_cap_split_packed_impl(
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    int64_t rows,
    int64_t cols,
    int64_t cap) {
  TORCH_CHECK(row_offsets.is_cuda() && row_ks.is_cuda() && row_values.is_cuda(),
              "cap split tensors must be CUDA tensors");
  TORCH_CHECK(row_offsets.scalar_type() == at::kInt, "row_offsets must be int32");
  TORCH_CHECK(row_ks.scalar_type() == at::kInt || row_ks.scalar_type() == at::kShort,
              "row_ks must be int32 or int16");
  TORCH_CHECK(row_values.scalar_type() == at::kBFloat16, "row_values must be bf16");
  TORCH_CHECK(row_offsets.numel() == rows + 1, "row_offsets must have rows + 1 elements");
  TORCH_CHECK(cap > 0 && cap <= 65535, "cap must be in (0, 65535]");

  const auto opts_i32 = row_offsets.options().dtype(at::kInt);
  const auto opts_i64 = row_offsets.options().dtype(at::kLong);
  const auto opts_val = row_values.options();
  const int32_t m = static_cast<int32_t>(rows);
  const int64_t selected = row_ks.numel();
  const int32_t rowblock_capacity = (m + 7) / 8;

  auto light_counts = at::empty({rows}, opts_i32);
  auto heavy_counts = at::empty({rows}, opts_i32);
  auto light_offsets = at::empty({rows + 1}, opts_i32);
  auto heavy_offsets = at::empty({rows + 1}, opts_i32);
  auto light_cols = at::empty({selected}, opts_i32);
  auto heavy_cols = at::empty({selected}, opts_i32);
  auto light_flat_indices = at::empty({selected}, opts_i32);
  auto heavy_flat_indices = at::empty({selected}, opts_i32);
  auto light_values = at::empty({selected}, opts_val);
  auto heavy_values = at::empty({selected}, opts_val);
  auto light_entry_records = at::empty({selected}, opts_i32);
  auto light_active_rows = at::empty({rows}, opts_i32);
  auto heavy_active_rows = at::empty({rows}, opts_i32);
  auto light_active_count = at::zeros({1}, opts_i32);
  auto heavy_active_count = at::zeros({1}, opts_i32);
  auto light_row_records = at::empty({rows}, opts_i64);
  auto light_rowblocks = at::empty({std::max<int32_t>(rowblock_capacity, 1)}, opts_i32);
  auto light_rowblock_count = at::zeros({1}, opts_i32);
  auto tile_offsets = at::empty({2}, opts_i32);

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int count_threads = 256;
  const int count_blocks = (m + count_threads - 1) / count_threads;
  cap_split_counts_kernel<<<count_blocks, count_threads, 0, stream>>>(
      row_offsets.data_ptr<int32_t>(),
      light_counts.data_ptr<int32_t>(),
      heavy_counts.data_ptr<int32_t>(),
      m,
      static_cast<int32_t>(cap));

  void* temp_storage = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      temp_storage,
      temp_bytes,
      light_counts.data_ptr<int32_t>(),
      light_offsets.data_ptr<int32_t>(),
      m,
      stream);
  auto scan_temp = at::empty({static_cast<int64_t>(temp_bytes)}, row_offsets.options().dtype(at::kByte));
  temp_storage = scan_temp.data_ptr();
  cub::DeviceScan::ExclusiveSum(
      temp_storage,
      temp_bytes,
      light_counts.data_ptr<int32_t>(),
      light_offsets.data_ptr<int32_t>(),
      m,
      stream);
  cub::DeviceScan::ExclusiveSum(
      temp_storage,
      temp_bytes,
      heavy_counts.data_ptr<int32_t>(),
      heavy_offsets.data_ptr<int32_t>(),
      m,
      stream);
  cap_split_set_tail_kernel<<<1, 1, 0, stream>>>(
      light_counts.data_ptr<int32_t>(), light_offsets.data_ptr<int32_t>(), m);
  cap_split_set_tail_kernel<<<1, 1, 0, stream>>>(
      heavy_counts.data_ptr<int32_t>(), heavy_offsets.data_ptr<int32_t>(), m);

  const int scatter_threads = 256;
  if (row_ks.scalar_type() == at::kInt) {
    cap_split_scatter_i32_kernel<<<m, scatter_threads, 0, stream>>>(
        row_offsets.data_ptr<int32_t>(),
        light_offsets.data_ptr<int32_t>(),
        heavy_offsets.data_ptr<int32_t>(),
        light_counts.data_ptr<int32_t>(),
        heavy_counts.data_ptr<int32_t>(),
        row_ks.data_ptr<int32_t>(),
        reinterpret_cast<const uint16_t*>(row_values.data_ptr<c10::BFloat16>()),
        light_cols.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(light_values.data_ptr<c10::BFloat16>()),
        light_flat_indices.data_ptr<int32_t>(),
        light_entry_records.data_ptr<int32_t>(),
        heavy_cols.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(heavy_values.data_ptr<c10::BFloat16>()),
        heavy_flat_indices.data_ptr<int32_t>(),
        static_cast<int32_t>(cols),
        m);
  } else {
    cap_split_scatter_kernel<<<m, scatter_threads, 0, stream>>>(
        row_offsets.data_ptr<int32_t>(),
        light_offsets.data_ptr<int32_t>(),
        heavy_offsets.data_ptr<int32_t>(),
        light_counts.data_ptr<int32_t>(),
        heavy_counts.data_ptr<int32_t>(),
        row_ks.data_ptr<int16_t>(),
        reinterpret_cast<const uint16_t*>(row_values.data_ptr<c10::BFloat16>()),
        light_cols.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(light_values.data_ptr<c10::BFloat16>()),
        light_flat_indices.data_ptr<int32_t>(),
        light_entry_records.data_ptr<int32_t>(),
        heavy_cols.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(heavy_values.data_ptr<c10::BFloat16>()),
        heavy_flat_indices.data_ptr<int32_t>(),
        static_cast<int32_t>(cols),
        m);
  }

  cap_split_active_rows_kernel<<<count_blocks, count_threads, 0, stream>>>(
      light_counts.data_ptr<int32_t>(),
      light_active_rows.data_ptr<int32_t>(),
      light_active_count.data_ptr<int32_t>(),
      m);
  cap_split_active_rows_kernel<<<count_blocks, count_threads, 0, stream>>>(
      heavy_counts.data_ptr<int32_t>(),
      heavy_active_rows.data_ptr<int32_t>(),
      heavy_active_count.data_ptr<int32_t>(),
      m);

  const int rowblock_threads = 128;
  const int rowblock_blocks = (rowblock_capacity + rowblock_threads - 1) / rowblock_threads;
  cap_split_rowblocks_kernel<<<rowblock_blocks, rowblock_threads, 0, stream>>>(
      light_counts.data_ptr<int32_t>(),
      light_rowblocks.data_ptr<int32_t>(),
      light_rowblock_count.data_ptr<int32_t>(),
      m,
      8);
  cap_split_tile_offsets_kernel<<<1, 1, 0, stream>>>(
      tile_offsets.data_ptr<int32_t>(), light_active_count.data_ptr<int32_t>());

  const int32_t light_entries = copy_int32_to_host(light_offsets.data_ptr<int32_t>() + m, stream);
  const int32_t heavy_entries = copy_int32_to_host(heavy_offsets.data_ptr<int32_t>() + m, stream);
  const int32_t light_rows = copy_int32_to_host(light_active_count.data_ptr<int32_t>(), stream);
  const int32_t heavy_rows = copy_int32_to_host(heavy_active_count.data_ptr<int32_t>(), stream);
  int32_t light_rowblocks_count = copy_int32_to_host(light_rowblock_count.data_ptr<int32_t>(), stream);
  if (light_rowblocks_count == 0) {
    light_rowblocks_count = 1;
    C10_CUDA_CHECK(cudaMemsetAsync(light_rowblocks.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  }

  const int record_threads = 256;
  const int record_blocks = std::max(1, (light_rows + record_threads - 1) / record_threads);
  cap_split_row_records_kernel<<<record_blocks, record_threads, 0, stream>>>(
      light_offsets.data_ptr<int32_t>(),
      light_counts.data_ptr<int32_t>(),
      light_active_rows.data_ptr<int32_t>(),
      light_row_records.data_ptr<int64_t>(),
      light_active_count.data_ptr<int32_t>());

  return {
      light_offsets,
      light_cols.slice(0, 0, light_entries),
      light_values.slice(0, 0, light_entries),
      heavy_offsets,
      heavy_cols.slice(0, 0, heavy_entries),
      heavy_values.slice(0, 0, heavy_entries),
      light_active_rows.slice(0, 0, light_rows),
      heavy_active_rows.slice(0, 0, heavy_rows),
      tile_offsets,
      light_row_records.slice(0, 0, light_rows),
      light_entry_records.slice(0, 0, light_entries),
      light_rowblocks.slice(0, 0, light_rowblocks_count),
      light_flat_indices.slice(0, 0, light_entries),
      heavy_flat_indices.slice(0, 0, heavy_entries),
      torch::tensor({light_entries, heavy_entries, light_rows, heavy_rows, light_rowblocks_count},
                    opts_i32)};
}

std::vector<at::Tensor> adaptive_rowcol_policy_split_packed_impl(
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    int64_t rows,
    int64_t cols,
    int64_t policy_mode,
    int64_t param0,
    int64_t param1) {
  TORCH_CHECK(row_offsets.is_cuda() && row_ks.is_cuda() && row_values.is_cuda(),
              "policy split tensors must be CUDA tensors");
  TORCH_CHECK(row_offsets.scalar_type() == at::kInt, "row_offsets must be int32");
  TORCH_CHECK(row_ks.scalar_type() == at::kInt || row_ks.scalar_type() == at::kShort,
              "row_ks must be int32 or int16");
  TORCH_CHECK(row_values.scalar_type() == at::kBFloat16, "row_values must be bf16");
  TORCH_CHECK(row_offsets.numel() == rows + 1, "row_offsets must have rows + 1 elements");
  TORCH_CHECK(policy_mode == 1 || policy_mode == 2,
              "policy_mode must be 1=densepack or 2=entrybudget");
  TORCH_CHECK(rows > 0, "rows must be positive");

  const auto opts_i32 = row_offsets.options().dtype(at::kInt);
  const auto opts_i64 = row_offsets.options().dtype(at::kLong);
  const auto opts_val = row_values.options();
  const int32_t m = static_cast<int32_t>(rows);
  const int64_t selected = row_ks.numel();
  const int32_t rowblock_capacity = (m + 7) / 8;

  auto light_counts = at::empty({rows}, opts_i32);
  auto heavy_counts = at::empty({rows}, opts_i32);
  auto light_offsets = at::empty({rows + 1}, opts_i32);
  auto heavy_offsets = at::empty({rows + 1}, opts_i32);
  auto light_cols = at::empty({selected}, opts_i32);
  auto heavy_cols = at::empty({selected}, opts_i32);
  auto light_flat_indices = at::empty({selected}, opts_i32);
  auto heavy_flat_indices = at::empty({selected}, opts_i32);
  auto light_values = at::empty({selected}, opts_val);
  auto heavy_values = at::empty({selected}, opts_val);
  auto light_entry_records = at::empty({selected}, opts_i32);
  auto light_active_rows = at::empty({rows}, opts_i32);
  auto heavy_active_rows = at::empty({rows}, opts_i32);
  auto light_active_count = at::zeros({1}, opts_i32);
  auto heavy_active_count = at::zeros({1}, opts_i32);
  auto light_row_records = at::empty({rows}, opts_i64);
  auto light_rowblocks = at::empty({std::max<int32_t>(rowblock_capacity, 1)}, opts_i32);
  auto light_rowblock_count = at::zeros({1}, opts_i32);
  auto tile_offsets = at::empty({2}, opts_i32);
  auto sort_keys_in = at::empty({rows}, opts_i64);
  auto sort_keys_out = at::empty({rows}, opts_i64);
  auto sort_rows_in = at::empty({rows}, opts_i32);
  auto sort_rows_out = at::empty({rows}, opts_i32);

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int count_threads = 256;
  const int count_blocks = (m + count_threads - 1) / count_threads;
  policy_split_init_counts_sort_kernel<<<count_blocks, count_threads, 0, stream>>>(
      row_offsets.data_ptr<int32_t>(),
      light_counts.data_ptr<int32_t>(),
      heavy_counts.data_ptr<int32_t>(),
      sort_keys_in.data_ptr<int64_t>(),
      sort_rows_in.data_ptr<int32_t>(),
      m,
      static_cast<int32_t>(policy_mode),
      static_cast<int32_t>(param0),
      static_cast<int32_t>(param1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  void* sort_temp_storage = nullptr;
  size_t sort_temp_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(sort_temp_storage,
                                            sort_temp_bytes,
                                            sort_keys_in.data_ptr<int64_t>(),
                                            sort_keys_out.data_ptr<int64_t>(),
                                            sort_rows_in.data_ptr<int32_t>(),
                                            sort_rows_out.data_ptr<int32_t>(),
                                            m,
                                            0,
                                            64,
                                            stream);
  auto sort_temp = at::empty({static_cast<int64_t>(sort_temp_bytes)},
                             row_offsets.options().dtype(at::kByte));
  cub::DeviceRadixSort::SortPairsDescending(sort_temp.data_ptr(),
                                            sort_temp_bytes,
                                            sort_keys_in.data_ptr<int64_t>(),
                                            sort_keys_out.data_ptr<int64_t>(),
                                            sort_rows_in.data_ptr<int32_t>(),
                                            sort_rows_out.data_ptr<int32_t>(),
                                            m,
                                            0,
                                            64,
                                            stream);

  if (policy_mode == 1) {
    policy_split_select_densepack_kernel<<<1, 1, 0, stream>>>(
        sort_keys_out.data_ptr<int64_t>(),
        sort_rows_out.data_ptr<int32_t>(),
        row_offsets.data_ptr<int32_t>(),
        light_counts.data_ptr<int32_t>(),
        heavy_counts.data_ptr<int32_t>(),
        m,
        static_cast<int32_t>(param1));
  } else {
    policy_split_select_entrybudget_kernel<<<1, 1, 0, stream>>>(
        sort_keys_out.data_ptr<int64_t>(),
        sort_rows_out.data_ptr<int32_t>(),
        row_offsets.data_ptr<int32_t>(),
        light_counts.data_ptr<int32_t>(),
        heavy_counts.data_ptr<int32_t>(),
        m,
        static_cast<int32_t>(param0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  void* scan_temp_storage = nullptr;
  size_t scan_temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      scan_temp_storage,
      scan_temp_bytes,
      light_counts.data_ptr<int32_t>(),
      light_offsets.data_ptr<int32_t>(),
      m,
      stream);
  auto scan_temp = at::empty({static_cast<int64_t>(scan_temp_bytes)},
                             row_offsets.options().dtype(at::kByte));
  scan_temp_storage = scan_temp.data_ptr();
  cub::DeviceScan::ExclusiveSum(
      scan_temp_storage,
      scan_temp_bytes,
      light_counts.data_ptr<int32_t>(),
      light_offsets.data_ptr<int32_t>(),
      m,
      stream);
  cub::DeviceScan::ExclusiveSum(
      scan_temp_storage,
      scan_temp_bytes,
      heavy_counts.data_ptr<int32_t>(),
      heavy_offsets.data_ptr<int32_t>(),
      m,
      stream);
  cap_split_set_tail_kernel<<<1, 1, 0, stream>>>(
      light_counts.data_ptr<int32_t>(), light_offsets.data_ptr<int32_t>(), m);
  cap_split_set_tail_kernel<<<1, 1, 0, stream>>>(
      heavy_counts.data_ptr<int32_t>(), heavy_offsets.data_ptr<int32_t>(), m);

  const int scatter_threads = 256;
  if (row_ks.scalar_type() == at::kInt) {
    cap_split_scatter_i32_kernel<<<m, scatter_threads, 0, stream>>>(
        row_offsets.data_ptr<int32_t>(),
        light_offsets.data_ptr<int32_t>(),
        heavy_offsets.data_ptr<int32_t>(),
        light_counts.data_ptr<int32_t>(),
        heavy_counts.data_ptr<int32_t>(),
        row_ks.data_ptr<int32_t>(),
        reinterpret_cast<const uint16_t*>(row_values.data_ptr<c10::BFloat16>()),
        light_cols.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(light_values.data_ptr<c10::BFloat16>()),
        light_flat_indices.data_ptr<int32_t>(),
        light_entry_records.data_ptr<int32_t>(),
        heavy_cols.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(heavy_values.data_ptr<c10::BFloat16>()),
        heavy_flat_indices.data_ptr<int32_t>(),
        static_cast<int32_t>(cols),
        m);
  } else {
    cap_split_scatter_kernel<<<m, scatter_threads, 0, stream>>>(
        row_offsets.data_ptr<int32_t>(),
        light_offsets.data_ptr<int32_t>(),
        heavy_offsets.data_ptr<int32_t>(),
        light_counts.data_ptr<int32_t>(),
        heavy_counts.data_ptr<int32_t>(),
        row_ks.data_ptr<int16_t>(),
        reinterpret_cast<const uint16_t*>(row_values.data_ptr<c10::BFloat16>()),
        light_cols.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(light_values.data_ptr<c10::BFloat16>()),
        light_flat_indices.data_ptr<int32_t>(),
        light_entry_records.data_ptr<int32_t>(),
        heavy_cols.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(heavy_values.data_ptr<c10::BFloat16>()),
        heavy_flat_indices.data_ptr<int32_t>(),
        static_cast<int32_t>(cols),
        m);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cap_split_active_rows_kernel<<<count_blocks, count_threads, 0, stream>>>(
      light_counts.data_ptr<int32_t>(),
      light_active_rows.data_ptr<int32_t>(),
      light_active_count.data_ptr<int32_t>(),
      m);
  cap_split_active_rows_kernel<<<count_blocks, count_threads, 0, stream>>>(
      heavy_counts.data_ptr<int32_t>(),
      heavy_active_rows.data_ptr<int32_t>(),
      heavy_active_count.data_ptr<int32_t>(),
      m);

  const int rowblock_threads = 128;
  const int rowblock_blocks = (rowblock_capacity + rowblock_threads - 1) / rowblock_threads;
  cap_split_rowblocks_kernel<<<rowblock_blocks, rowblock_threads, 0, stream>>>(
      light_counts.data_ptr<int32_t>(),
      light_rowblocks.data_ptr<int32_t>(),
      light_rowblock_count.data_ptr<int32_t>(),
      m,
      8);
  cap_split_tile_offsets_kernel<<<1, 1, 0, stream>>>(
      tile_offsets.data_ptr<int32_t>(), light_active_count.data_ptr<int32_t>());

  const int32_t light_entries = copy_int32_to_host(light_offsets.data_ptr<int32_t>() + m, stream);
  const int32_t heavy_entries = copy_int32_to_host(heavy_offsets.data_ptr<int32_t>() + m, stream);
  const int32_t light_rows = copy_int32_to_host(light_active_count.data_ptr<int32_t>(), stream);
  const int32_t heavy_rows = copy_int32_to_host(heavy_active_count.data_ptr<int32_t>(), stream);
  int32_t light_rowblocks_count = copy_int32_to_host(light_rowblock_count.data_ptr<int32_t>(), stream);
  if (light_rowblocks_count == 0) {
    light_rowblocks_count = 1;
    C10_CUDA_CHECK(cudaMemsetAsync(light_rowblocks.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  }

  const int record_threads = 256;
  const int record_blocks = std::max(1, (light_rows + record_threads - 1) / record_threads);
  cap_split_row_records_kernel<<<record_blocks, record_threads, 0, stream>>>(
      light_offsets.data_ptr<int32_t>(),
      light_counts.data_ptr<int32_t>(),
      light_active_rows.data_ptr<int32_t>(),
      light_row_records.data_ptr<int64_t>(),
      light_active_count.data_ptr<int32_t>());

  return {
      light_offsets,
      light_cols.slice(0, 0, light_entries),
      light_values.slice(0, 0, light_entries),
      heavy_offsets,
      heavy_cols.slice(0, 0, heavy_entries),
      heavy_values.slice(0, 0, heavy_entries),
      light_active_rows.slice(0, 0, light_rows),
      heavy_active_rows.slice(0, 0, heavy_rows),
      tile_offsets,
      light_row_records.slice(0, 0, light_rows),
      light_entry_records.slice(0, 0, light_entries),
      light_rowblocks.slice(0, 0, light_rowblocks_count),
      light_flat_indices.slice(0, 0, light_entries),
      heavy_flat_indices.slice(0, 0, heavy_entries),
      torch::tensor({light_entries, heavy_entries, light_rows, heavy_rows, light_rowblocks_count},
                    opts_i32)};
}

std::vector<at::Tensor> adaptive_rowcol_policy_bucket_split_packed_impl(
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    int64_t rows,
    int64_t cols,
    int64_t policy_mode,
    int64_t param0,
    int64_t param1) {
  TORCH_CHECK(row_offsets.is_cuda() && row_ks.is_cuda() && row_values.is_cuda(),
              "policy bucket split tensors must be CUDA tensors");
  TORCH_CHECK(row_offsets.scalar_type() == at::kInt, "row_offsets must be int32");
  TORCH_CHECK(row_ks.scalar_type() == at::kInt || row_ks.scalar_type() == at::kShort,
              "row_ks must be int32 or int16");
  TORCH_CHECK(row_values.scalar_type() == at::kBFloat16, "row_values must be bf16");
  TORCH_CHECK(row_offsets.numel() == rows + 1, "row_offsets must have rows + 1 elements");
  TORCH_CHECK(policy_mode == 1 || policy_mode == 2,
              "policy_mode must be 1=densepack or 2=entrybudget");
  TORCH_CHECK(rows > 0, "rows must be positive");

  const auto opts_i32 = row_offsets.options().dtype(at::kInt);
  const auto opts_i64 = row_offsets.options().dtype(at::kLong);
  const auto opts_val = row_values.options();
  const int32_t m = static_cast<int32_t>(rows);
  const int64_t selected = row_ks.numel();
  const int32_t rowblock_capacity = (m + 7) / 8;

  auto light_counts = at::empty({rows}, opts_i32);
  auto heavy_counts = at::empty({rows}, opts_i32);
  auto light_offsets = at::empty({rows + 1}, opts_i32);
  auto heavy_offsets = at::empty({rows + 1}, opts_i32);
  auto light_cols = at::empty({selected}, opts_i32);
  auto heavy_cols = at::empty({selected}, opts_i32);
  auto light_flat_indices = at::empty({selected}, opts_i32);
  auto heavy_flat_indices = at::empty({selected}, opts_i32);
  auto light_values = at::empty({selected}, opts_val);
  auto heavy_values = at::empty({selected}, opts_val);
  auto light_entry_records = at::empty({selected}, opts_i32);
  auto light_active_rows = at::empty({rows}, opts_i32);
  auto heavy_active_rows = at::empty({rows}, opts_i32);
  auto light_active_count = at::zeros({1}, opts_i32);
  auto heavy_active_count = at::zeros({1}, opts_i32);
  auto light_row_records = at::empty({rows}, opts_i64);
  auto light_rowblocks = at::empty({std::max<int32_t>(rowblock_capacity, 1)}, opts_i32);
  auto light_rowblock_count = at::zeros({1}, opts_i32);
  auto tile_offsets = at::empty({2}, opts_i32);
  const int32_t max_bucket = static_cast<int32_t>(
      policy_mode == 1 ? std::max<int64_t>(1, cols) : std::max<int64_t>(1, param1));
  auto bucket_hist = at::zeros({static_cast<int64_t>(max_bucket) + 1}, opts_i32);
  auto bucket_limits = at::zeros({static_cast<int64_t>(max_bucket) + 1}, opts_i32);
  auto bucket_taken = at::zeros({static_cast<int64_t>(max_bucket) + 1}, opts_i32);

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int count_threads = 256;
  const int count_blocks = (m + count_threads - 1) / count_threads;
  policy_split_init_counts_only_kernel<<<count_blocks, count_threads, 0, stream>>>(
      row_offsets.data_ptr<int32_t>(),
      light_counts.data_ptr<int32_t>(),
      heavy_counts.data_ptr<int32_t>(),
      m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  policy_split_hist_rows_kernel<<<count_blocks, count_threads, 0, stream>>>(
      row_offsets.data_ptr<int32_t>(),
      bucket_hist.data_ptr<int32_t>(),
      m,
      max_bucket);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (policy_mode == 1) {
    policy_split_compute_densepack_limits_kernel<<<1, 1, 0, stream>>>(
        bucket_hist.data_ptr<int32_t>(),
        bucket_limits.data_ptr<int32_t>(),
        max_bucket,
        static_cast<int32_t>(param0),
        static_cast<int32_t>(param1));
  } else {
    policy_split_compute_entrybudget_limits_kernel<<<1, 1, 0, stream>>>(
        bucket_hist.data_ptr<int32_t>(),
        bucket_limits.data_ptr<int32_t>(),
        max_bucket,
        static_cast<int32_t>(param0));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  policy_split_apply_bucket_limits_atomic_kernel<<<count_blocks, count_threads, 0, stream>>>(
      row_offsets.data_ptr<int32_t>(),
      bucket_limits.data_ptr<int32_t>(),
      bucket_taken.data_ptr<int32_t>(),
      light_counts.data_ptr<int32_t>(),
      heavy_counts.data_ptr<int32_t>(),
      m,
      max_bucket);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  void* scan_temp_storage = nullptr;
  size_t scan_temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      scan_temp_storage,
      scan_temp_bytes,
      light_counts.data_ptr<int32_t>(),
      light_offsets.data_ptr<int32_t>(),
      m,
      stream);
  auto scan_temp = at::empty({static_cast<int64_t>(scan_temp_bytes)},
                             row_offsets.options().dtype(at::kByte));
  scan_temp_storage = scan_temp.data_ptr();
  cub::DeviceScan::ExclusiveSum(
      scan_temp_storage,
      scan_temp_bytes,
      light_counts.data_ptr<int32_t>(),
      light_offsets.data_ptr<int32_t>(),
      m,
      stream);
  cub::DeviceScan::ExclusiveSum(
      scan_temp_storage,
      scan_temp_bytes,
      heavy_counts.data_ptr<int32_t>(),
      heavy_offsets.data_ptr<int32_t>(),
      m,
      stream);
  cap_split_set_tail_kernel<<<1, 1, 0, stream>>>(
      light_counts.data_ptr<int32_t>(), light_offsets.data_ptr<int32_t>(), m);
  cap_split_set_tail_kernel<<<1, 1, 0, stream>>>(
      heavy_counts.data_ptr<int32_t>(), heavy_offsets.data_ptr<int32_t>(), m);

  const int scatter_threads = 256;
  if (row_ks.scalar_type() == at::kInt) {
    cap_split_scatter_i32_kernel<<<m, scatter_threads, 0, stream>>>(
        row_offsets.data_ptr<int32_t>(),
        light_offsets.data_ptr<int32_t>(),
        heavy_offsets.data_ptr<int32_t>(),
        light_counts.data_ptr<int32_t>(),
        heavy_counts.data_ptr<int32_t>(),
        row_ks.data_ptr<int32_t>(),
        reinterpret_cast<const uint16_t*>(row_values.data_ptr<c10::BFloat16>()),
        light_cols.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(light_values.data_ptr<c10::BFloat16>()),
        light_flat_indices.data_ptr<int32_t>(),
        light_entry_records.data_ptr<int32_t>(),
        heavy_cols.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(heavy_values.data_ptr<c10::BFloat16>()),
        heavy_flat_indices.data_ptr<int32_t>(),
        static_cast<int32_t>(cols),
        m);
  } else {
    cap_split_scatter_kernel<<<m, scatter_threads, 0, stream>>>(
        row_offsets.data_ptr<int32_t>(),
        light_offsets.data_ptr<int32_t>(),
        heavy_offsets.data_ptr<int32_t>(),
        light_counts.data_ptr<int32_t>(),
        heavy_counts.data_ptr<int32_t>(),
        row_ks.data_ptr<int16_t>(),
        reinterpret_cast<const uint16_t*>(row_values.data_ptr<c10::BFloat16>()),
        light_cols.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(light_values.data_ptr<c10::BFloat16>()),
        light_flat_indices.data_ptr<int32_t>(),
        light_entry_records.data_ptr<int32_t>(),
        heavy_cols.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(heavy_values.data_ptr<c10::BFloat16>()),
        heavy_flat_indices.data_ptr<int32_t>(),
        static_cast<int32_t>(cols),
        m);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cap_split_active_rows_kernel<<<count_blocks, count_threads, 0, stream>>>(
      light_counts.data_ptr<int32_t>(),
      light_active_rows.data_ptr<int32_t>(),
      light_active_count.data_ptr<int32_t>(),
      m);
  cap_split_active_rows_kernel<<<count_blocks, count_threads, 0, stream>>>(
      heavy_counts.data_ptr<int32_t>(),
      heavy_active_rows.data_ptr<int32_t>(),
      heavy_active_count.data_ptr<int32_t>(),
      m);

  const int rowblock_threads = 128;
  const int rowblock_blocks = (rowblock_capacity + rowblock_threads - 1) / rowblock_threads;
  cap_split_rowblocks_kernel<<<rowblock_blocks, rowblock_threads, 0, stream>>>(
      light_counts.data_ptr<int32_t>(),
      light_rowblocks.data_ptr<int32_t>(),
      light_rowblock_count.data_ptr<int32_t>(),
      m,
      8);
  cap_split_tile_offsets_kernel<<<1, 1, 0, stream>>>(
      tile_offsets.data_ptr<int32_t>(), light_active_count.data_ptr<int32_t>());

  const int32_t light_entries = copy_int32_to_host(light_offsets.data_ptr<int32_t>() + m, stream);
  const int32_t heavy_entries = copy_int32_to_host(heavy_offsets.data_ptr<int32_t>() + m, stream);
  const int32_t light_rows = copy_int32_to_host(light_active_count.data_ptr<int32_t>(), stream);
  const int32_t heavy_rows = copy_int32_to_host(heavy_active_count.data_ptr<int32_t>(), stream);
  int32_t light_rowblocks_count = copy_int32_to_host(light_rowblock_count.data_ptr<int32_t>(), stream);
  if (light_rowblocks_count == 0) {
    light_rowblocks_count = 1;
    C10_CUDA_CHECK(cudaMemsetAsync(light_rowblocks.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  }

  const int record_threads = 256;
  const int record_blocks = std::max(1, (light_rows + record_threads - 1) / record_threads);
  cap_split_row_records_kernel<<<record_blocks, record_threads, 0, stream>>>(
      light_offsets.data_ptr<int32_t>(),
      light_counts.data_ptr<int32_t>(),
      light_active_rows.data_ptr<int32_t>(),
      light_row_records.data_ptr<int64_t>(),
      light_active_count.data_ptr<int32_t>());

  return {
      light_offsets,
      light_cols.slice(0, 0, light_entries),
      light_values.slice(0, 0, light_entries),
      heavy_offsets,
      heavy_cols.slice(0, 0, heavy_entries),
      heavy_values.slice(0, 0, heavy_entries),
      light_active_rows.slice(0, 0, light_rows),
      heavy_active_rows.slice(0, 0, heavy_rows),
      tile_offsets,
      light_row_records.slice(0, 0, light_rows),
      light_entry_records.slice(0, 0, light_entries),
      light_rowblocks.slice(0, 0, light_rowblocks_count),
      light_flat_indices.slice(0, 0, light_entries),
      heavy_flat_indices.slice(0, 0, heavy_entries),
      torch::tensor({light_entries, heavy_entries, light_rows, heavy_rows, light_rowblocks_count},
                    opts_i32)};
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void direct_split_fill_packed_kernel(const scalar_t* __restrict__ input,
                                                const int32_t* __restrict__ row_offsets,
                                                const uint16_t* __restrict__ selection_masks,
                                                const int32_t* __restrict__ light_offsets,
                                                const int32_t* __restrict__ heavy_offsets,
                                                const int32_t* __restrict__ light_counts,
                                                const int32_t* __restrict__ heavy_counts,
                                                int32_t* __restrict__ light_cols,
                                                uint16_t* __restrict__ light_values,
                                                int32_t* __restrict__ light_flat_indices,
                                                int32_t* __restrict__ light_entry_records,
                                                int32_t* __restrict__ heavy_cols,
                                                uint16_t* __restrict__ heavy_values,
                                                int32_t* __restrict__ heavy_flat_indices,
                                                int32_t rows,
                                                int32_t cols,
                                                int32_t capacity) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int32_t count = row_offsets[row + 1] - row_offsets[row];
  if (count <= 0) {
    return;
  }
  const bool is_light = light_counts[row] > 0;
  if (!is_light && heavy_counts[row] <= 0) {
    return;
  }

  const int blocks_per_row = cols / 16;
  int32_t local_selected_count = 0;
  for (int block_col = threadIdx.x; block_col < blocks_per_row; block_col += BLOCK_THREADS) {
    const uint16_t selected_mask = selection_masks[row * blocks_per_row + block_col];
    local_selected_count += __popc(static_cast<unsigned>(selected_mask));
  }

  using BlockScan = cub::BlockScan<int32_t, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  int32_t thread_write_base = 0;
  BlockScan(scan_storage).ExclusiveSum(local_selected_count, thread_write_base);

  const int32_t dst_base = is_light ? light_offsets[row] : heavy_offsets[row];
  int32_t local_write = 0;
  for (int block_col = threadIdx.x; block_col < blocks_per_row; block_col += BLOCK_THREADS) {
    const uint16_t selected_mask = selection_masks[row * blocks_per_row + block_col];
    if (selected_mask == 0) {
      continue;
    }
    const int col0 = block_col * 16;
    const int flat0 = row * cols + col0;
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      Vec<scalar_t, 8> in_wave;
      in_wave.load_from(&input[flat0 + w * 8]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int i = w * 8 + e;
        if ((selected_mask & (1u << i)) == 0) {
          continue;
        }
        const int32_t dst = dst_base + thread_write_base + local_write;
        ++local_write;
        if (dst >= capacity) {
          continue;
        }
        const uint16_t col = static_cast<uint16_t>(col0 + i);
        const uint16_t value_bits = *reinterpret_cast<const uint16_t*>(&in_wave.elt[e]);
        if (is_light) {
          light_cols[dst] = static_cast<int32_t>(col);
          light_values[dst] = value_bits;
          light_flat_indices[dst] = flat0 + i;
          light_entry_records[dst] =
              static_cast<int32_t>((static_cast<uint32_t>(col) << 16) |
                                   static_cast<uint32_t>(value_bits));
        } else {
          heavy_cols[dst] = static_cast<int32_t>(col);
          heavy_values[dst] = value_bits;
          heavy_flat_indices[dst] = flat0 + i;
        }
      }
    }
  }
#endif
}

std::vector<at::Tensor> adaptive_rowcol_direct_split_packed_impl(
    const at::Tensor& input,
    const at::Tensor& row_offsets,
    const at::Tensor& selection_masks,
    int64_t selected_nnz,
    int64_t policy_mode,
    int64_t param0,
    int64_t param1,
    bool use_bucket,
    int64_t fill_threads_arg) {
  const auto rows = input.size(0);
  const auto cols = input.size(1);
  TORCH_CHECK(input.is_cuda() && row_offsets.is_cuda() && selection_masks.is_cuda(),
              "direct split tensors must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "direct split supports BF16 input only");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(row_offsets.scalar_type() == at::kInt, "row_offsets must be int32");
  TORCH_CHECK(selection_masks.scalar_type() == at::kShort, "selection_masks must be int16");
  TORCH_CHECK(row_offsets.numel() == rows + 1, "row_offsets must have rows + 1 elements");
  TORCH_CHECK(cols % 16 == 0, "cols must be divisible by 16");
  TORCH_CHECK(selection_masks.numel() == rows * (cols / 16), "selection_masks shape mismatch");
  TORCH_CHECK(policy_mode >= 0 && policy_mode <= 2,
              "policy_mode must be 0=cap, 1=densepack, or 2=entrybudget");
  TORCH_CHECK(selected_nnz >= 0 && selected_nnz <= std::numeric_limits<int32_t>::max(),
              "selected_nnz must fit int32");
  TORCH_CHECK(rows > 0, "rows must be positive");

  const auto opts_i32 = row_offsets.options().dtype(at::kInt);
  const auto opts_i64 = row_offsets.options().dtype(at::kLong);
  const auto opts_val = input.options();
  const int32_t m = static_cast<int32_t>(rows);
  const int32_t k = static_cast<int32_t>(cols);
  const int32_t selected = static_cast<int32_t>(selected_nnz);
  const int32_t rowblock_capacity = (m + 7) / 8;

  auto light_counts = at::empty({rows}, opts_i32);
  auto heavy_counts = at::empty({rows}, opts_i32);
  auto light_offsets = at::empty({rows + 1}, opts_i32);
  auto heavy_offsets = at::empty({rows + 1}, opts_i32);
  auto light_cols = at::empty({selected_nnz}, opts_i32);
  auto heavy_cols = at::empty({selected_nnz}, opts_i32);
  auto light_flat_indices = at::empty({selected_nnz}, opts_i32);
  auto heavy_flat_indices = at::empty({selected_nnz}, opts_i32);
  auto light_values = at::empty({selected_nnz}, opts_val);
  auto heavy_values = at::empty({selected_nnz}, opts_val);
  auto light_entry_records = at::empty({selected_nnz}, opts_i32);
  auto light_active_rows = at::empty({rows}, opts_i32);
  auto heavy_active_rows = at::empty({rows}, opts_i32);
  auto light_active_count = at::zeros({1}, opts_i32);
  auto heavy_active_count = at::zeros({1}, opts_i32);
  auto light_row_records = at::empty({rows}, opts_i64);
  auto light_rowblocks = at::empty({std::max<int32_t>(rowblock_capacity, 1)}, opts_i32);
  auto light_rowblock_count = at::zeros({1}, opts_i32);
  auto tile_offsets = at::empty({2}, opts_i32);

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  const int count_threads = 256;
  const int count_blocks = (m + count_threads - 1) / count_threads;
  if (policy_mode == 0) {
    cap_split_counts_kernel<<<count_blocks, count_threads, 0, stream>>>(
        row_offsets.data_ptr<int32_t>(),
        light_counts.data_ptr<int32_t>(),
        heavy_counts.data_ptr<int32_t>(),
        m,
        static_cast<int32_t>(param0));
  } else if (use_bucket) {
    policy_split_init_counts_only_kernel<<<count_blocks, count_threads, 0, stream>>>(
        row_offsets.data_ptr<int32_t>(),
        light_counts.data_ptr<int32_t>(),
        heavy_counts.data_ptr<int32_t>(),
        m);
    const int32_t max_bucket = static_cast<int32_t>(
        policy_mode == 1 ? std::max<int64_t>(1, cols) : std::max<int64_t>(1, param1));
    auto bucket_hist = at::zeros({static_cast<int64_t>(max_bucket) + 1}, opts_i32);
    auto bucket_limits = at::zeros({static_cast<int64_t>(max_bucket) + 1}, opts_i32);
    auto bucket_taken = at::zeros({static_cast<int64_t>(max_bucket) + 1}, opts_i32);
    policy_split_hist_rows_kernel<<<count_blocks, count_threads, 0, stream>>>(
        row_offsets.data_ptr<int32_t>(),
        bucket_hist.data_ptr<int32_t>(),
        m,
        max_bucket);
    if (policy_mode == 1) {
      policy_split_compute_densepack_limits_kernel<<<1, 1, 0, stream>>>(
          bucket_hist.data_ptr<int32_t>(),
          bucket_limits.data_ptr<int32_t>(),
          max_bucket,
          static_cast<int32_t>(param0),
          static_cast<int32_t>(param1));
    } else {
      policy_split_compute_entrybudget_limits_kernel<<<1, 1, 0, stream>>>(
          bucket_hist.data_ptr<int32_t>(),
          bucket_limits.data_ptr<int32_t>(),
          max_bucket,
          static_cast<int32_t>(param0));
    }
    policy_split_apply_bucket_limits_atomic_kernel<<<count_blocks, count_threads, 0, stream>>>(
        row_offsets.data_ptr<int32_t>(),
        bucket_limits.data_ptr<int32_t>(),
        bucket_taken.data_ptr<int32_t>(),
        light_counts.data_ptr<int32_t>(),
        heavy_counts.data_ptr<int32_t>(),
        m,
        max_bucket);
  } else {
    auto sort_keys_in = at::empty({rows}, opts_i64);
    auto sort_keys_out = at::empty({rows}, opts_i64);
    auto sort_rows_in = at::empty({rows}, opts_i32);
    auto sort_rows_out = at::empty({rows}, opts_i32);
    policy_split_init_counts_sort_kernel<<<count_blocks, count_threads, 0, stream>>>(
        row_offsets.data_ptr<int32_t>(),
        light_counts.data_ptr<int32_t>(),
        heavy_counts.data_ptr<int32_t>(),
        sort_keys_in.data_ptr<int64_t>(),
        sort_rows_in.data_ptr<int32_t>(),
        m,
        static_cast<int32_t>(policy_mode),
        static_cast<int32_t>(param0),
        static_cast<int32_t>(param1));

    void* sort_temp_storage = nullptr;
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(sort_temp_storage,
                                              sort_temp_bytes,
                                              sort_keys_in.data_ptr<int64_t>(),
                                              sort_keys_out.data_ptr<int64_t>(),
                                              sort_rows_in.data_ptr<int32_t>(),
                                              sort_rows_out.data_ptr<int32_t>(),
                                              m,
                                              0,
                                              64,
                                              stream);
    auto sort_temp = at::empty({static_cast<int64_t>(sort_temp_bytes)},
                               row_offsets.options().dtype(at::kByte));
    cub::DeviceRadixSort::SortPairsDescending(sort_temp.data_ptr(),
                                              sort_temp_bytes,
                                              sort_keys_in.data_ptr<int64_t>(),
                                              sort_keys_out.data_ptr<int64_t>(),
                                              sort_rows_in.data_ptr<int32_t>(),
                                              sort_rows_out.data_ptr<int32_t>(),
                                              m,
                                              0,
                                              64,
                                              stream);
    if (policy_mode == 1) {
      policy_split_select_densepack_kernel<<<1, 1, 0, stream>>>(
          sort_keys_out.data_ptr<int64_t>(),
          sort_rows_out.data_ptr<int32_t>(),
          row_offsets.data_ptr<int32_t>(),
          light_counts.data_ptr<int32_t>(),
          heavy_counts.data_ptr<int32_t>(),
          m,
          static_cast<int32_t>(param1));
    } else {
      policy_split_select_entrybudget_kernel<<<1, 1, 0, stream>>>(
          sort_keys_out.data_ptr<int64_t>(),
          sort_rows_out.data_ptr<int32_t>(),
          row_offsets.data_ptr<int32_t>(),
          light_counts.data_ptr<int32_t>(),
          heavy_counts.data_ptr<int32_t>(),
          m,
          static_cast<int32_t>(param0));
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  void* scan_temp_storage = nullptr;
  size_t scan_temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      scan_temp_storage,
      scan_temp_bytes,
      light_counts.data_ptr<int32_t>(),
      light_offsets.data_ptr<int32_t>(),
      m,
      stream);
  auto scan_temp = at::empty({static_cast<int64_t>(scan_temp_bytes)},
                             row_offsets.options().dtype(at::kByte));
  scan_temp_storage = scan_temp.data_ptr();
  cub::DeviceScan::ExclusiveSum(
      scan_temp_storage,
      scan_temp_bytes,
      light_counts.data_ptr<int32_t>(),
      light_offsets.data_ptr<int32_t>(),
      m,
      stream);
  cub::DeviceScan::ExclusiveSum(
      scan_temp_storage,
      scan_temp_bytes,
      heavy_counts.data_ptr<int32_t>(),
      heavy_offsets.data_ptr<int32_t>(),
      m,
      stream);
  cap_split_set_tail_kernel<<<1, 1, 0, stream>>>(
      light_counts.data_ptr<int32_t>(), light_offsets.data_ptr<int32_t>(), m);
  cap_split_set_tail_kernel<<<1, 1, 0, stream>>>(
      heavy_counts.data_ptr<int32_t>(), heavy_offsets.data_ptr<int32_t>(), m);

  const int fill_threads = normalize_threads(fill_threads_arg, cols <= 4096 ? 128 : 256);
  if (fill_threads == 64) {
    direct_split_fill_packed_kernel<c10::BFloat16, 64>
        <<<m, 64, 0, stream>>>(
            input.data_ptr<c10::BFloat16>(),
            row_offsets.data_ptr<int32_t>(),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            light_offsets.data_ptr<int32_t>(),
            heavy_offsets.data_ptr<int32_t>(),
            light_counts.data_ptr<int32_t>(),
            heavy_counts.data_ptr<int32_t>(),
            light_cols.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(light_values.data_ptr<c10::BFloat16>()),
            light_flat_indices.data_ptr<int32_t>(),
            light_entry_records.data_ptr<int32_t>(),
            heavy_cols.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(heavy_values.data_ptr<c10::BFloat16>()),
            heavy_flat_indices.data_ptr<int32_t>(),
            m,
            k,
            selected);
  } else if (fill_threads == 256) {
    direct_split_fill_packed_kernel<c10::BFloat16, 256>
        <<<m, 256, 0, stream>>>(
            input.data_ptr<c10::BFloat16>(),
            row_offsets.data_ptr<int32_t>(),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            light_offsets.data_ptr<int32_t>(),
            heavy_offsets.data_ptr<int32_t>(),
            light_counts.data_ptr<int32_t>(),
            heavy_counts.data_ptr<int32_t>(),
            light_cols.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(light_values.data_ptr<c10::BFloat16>()),
            light_flat_indices.data_ptr<int32_t>(),
            light_entry_records.data_ptr<int32_t>(),
            heavy_cols.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(heavy_values.data_ptr<c10::BFloat16>()),
            heavy_flat_indices.data_ptr<int32_t>(),
            m,
            k,
            selected);
  } else {
    direct_split_fill_packed_kernel<c10::BFloat16, 128>
        <<<m, 128, 0, stream>>>(
            input.data_ptr<c10::BFloat16>(),
            row_offsets.data_ptr<int32_t>(),
            reinterpret_cast<const uint16_t*>(selection_masks.data_ptr<int16_t>()),
            light_offsets.data_ptr<int32_t>(),
            heavy_offsets.data_ptr<int32_t>(),
            light_counts.data_ptr<int32_t>(),
            heavy_counts.data_ptr<int32_t>(),
            light_cols.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(light_values.data_ptr<c10::BFloat16>()),
            light_flat_indices.data_ptr<int32_t>(),
            light_entry_records.data_ptr<int32_t>(),
            heavy_cols.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(heavy_values.data_ptr<c10::BFloat16>()),
            heavy_flat_indices.data_ptr<int32_t>(),
            m,
            k,
            selected);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cap_split_active_rows_kernel<<<count_blocks, count_threads, 0, stream>>>(
      light_counts.data_ptr<int32_t>(),
      light_active_rows.data_ptr<int32_t>(),
      light_active_count.data_ptr<int32_t>(),
      m);
  cap_split_active_rows_kernel<<<count_blocks, count_threads, 0, stream>>>(
      heavy_counts.data_ptr<int32_t>(),
      heavy_active_rows.data_ptr<int32_t>(),
      heavy_active_count.data_ptr<int32_t>(),
      m);

  const int rowblock_threads = 128;
  const int rowblock_blocks = (rowblock_capacity + rowblock_threads - 1) / rowblock_threads;
  cap_split_rowblocks_kernel<<<rowblock_blocks, rowblock_threads, 0, stream>>>(
      light_counts.data_ptr<int32_t>(),
      light_rowblocks.data_ptr<int32_t>(),
      light_rowblock_count.data_ptr<int32_t>(),
      m,
      8);
  cap_split_tile_offsets_kernel<<<1, 1, 0, stream>>>(
      tile_offsets.data_ptr<int32_t>(), light_active_count.data_ptr<int32_t>());

  const int32_t light_entries = copy_int32_to_host(light_offsets.data_ptr<int32_t>() + m, stream);
  const int32_t heavy_entries = copy_int32_to_host(heavy_offsets.data_ptr<int32_t>() + m, stream);
  const int32_t light_rows = copy_int32_to_host(light_active_count.data_ptr<int32_t>(), stream);
  const int32_t heavy_rows = copy_int32_to_host(heavy_active_count.data_ptr<int32_t>(), stream);
  int32_t light_rowblocks_count = copy_int32_to_host(light_rowblock_count.data_ptr<int32_t>(), stream);
  if (light_rowblocks_count == 0) {
    light_rowblocks_count = 1;
    C10_CUDA_CHECK(cudaMemsetAsync(light_rowblocks.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  }

  const int record_threads = 256;
  const int record_blocks = std::max(1, (light_rows + record_threads - 1) / record_threads);
  cap_split_row_records_kernel<<<record_blocks, record_threads, 0, stream>>>(
      light_offsets.data_ptr<int32_t>(),
      light_counts.data_ptr<int32_t>(),
      light_active_rows.data_ptr<int32_t>(),
      light_row_records.data_ptr<int64_t>(),
      light_active_count.data_ptr<int32_t>());

  return {
      light_offsets,
      light_cols.slice(0, 0, light_entries),
      light_values.slice(0, 0, light_entries),
      heavy_offsets,
      heavy_cols.slice(0, 0, heavy_entries),
      heavy_values.slice(0, 0, heavy_entries),
      light_active_rows.slice(0, 0, light_rows),
      heavy_active_rows.slice(0, 0, heavy_rows),
      tile_offsets,
      light_row_records.slice(0, 0, light_rows),
      light_entry_records.slice(0, 0, light_entries),
      light_rowblocks.slice(0, 0, light_rowblocks_count),
      light_flat_indices.slice(0, 0, light_entries),
      heavy_flat_indices.slice(0, 0, heavy_entries),
      torch::tensor({light_entries, heavy_entries, light_rows, heavy_rows, light_rowblocks_count},
                    opts_i32)};
}

}  // namespace

int64_t adaptive_rowcol_scan_temp_bytes_cuda(int64_t rows) {
  void* temp_storage = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      temp_storage,
      temp_bytes,
      static_cast<int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      static_cast<int>(rows),
      at::cuda::getDefaultCUDAStream());
  return static_cast<int64_t>(temp_bytes);
}

int64_t adaptive_rowcol_active_sort_temp_bytes_cuda(int64_t rows) {
  return r25_active_sort_temp_bytes(rows);
}

std::vector<at::Tensor> adaptive_rowcol_quant_fast_cuda(
    const at::Tensor& input,
    double base_ratio,
    double min_ratio,
    double max_ratio,
    double reference_heaviness,
    int64_t capacity,
    bool emit_dense_main,
    int64_t stats_threads,
    int64_t fill_threads,
    int64_t columnwise_source,
    int64_t rht_random_sign_mask_t,
    bool overlap_columnwise,
    bool direct_nomask,
    bool build_active_schedule,
    bool build_unsorted_active_rows,
    double threshold_sigma_override) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return adaptive_rowcol_quant_fast_impl(input,
                                         base_ratio,
                                         min_ratio,
                                         max_ratio,
                                         reference_heaviness,
                                         capacity,
                                         emit_dense_main,
                                         stats_threads,
                                         fill_threads,
                                         columnwise_source,
                                         rht_random_sign_mask_t,
                                         overlap_columnwise,
                                         direct_nomask,
                                         build_active_schedule,
                                         build_unsorted_active_rows,
                                         threshold_sigma_override);
}

std::vector<at::Tensor> adaptive_rowcol_quant_fast_out_cuda(
    const at::Tensor& input,
    const at::Tensor& flat_indices,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& row_counts,
    const at::Tensor& row_offsets,
    const at::Tensor& selection_masks,
    const at::Tensor& num_selected,
    const at::Tensor& overflow,
    const at::Tensor& stats_amax,
    const at::Tensor& rowwise_data,
    const at::Tensor& rowwise_scale,
    const at::Tensor& columnwise_data,
    const at::Tensor& columnwise_scale,
    const at::Tensor& columnwise_amax,
    const at::Tensor& rht_output_t,
    const at::Tensor& scan_temp,
    const at::Tensor& packed_records,
    const at::Tensor& active_rows_heavy_light,
    const at::Tensor& active_row_count,
    const at::Tensor& active_sort_rows_in,
    const at::Tensor& active_sort_rows_out,
    const at::Tensor& active_sort_keys_in,
    const at::Tensor& active_sort_keys_out,
    const at::Tensor& active_sort_temp,
    const at::Tensor& hardcap_score_max,
    const at::Tensor& hardcap_histogram,
    double base_ratio,
    double min_ratio,
    double max_ratio,
    double reference_heaviness,
    int64_t capacity,
    int64_t stats_threads,
    int64_t fill_threads,
    int64_t columnwise_source,
    int64_t rht_random_sign_mask_t,
    bool overlap_columnwise,
    bool auto_expand_capacity,
    bool emit_packed_records,
    bool build_active_schedule,
    bool emit_direct_split,
    int64_t direct_policy_mode,
    int64_t direct_param0,
    int64_t direct_param1,
    bool direct_use_bucket,
    bool direct_no_host_slice) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return adaptive_rowcol_quant_fast_out_impl(input,
                                             flat_indices,
                                             outlier_values,
                                             outlier_cols,
                                             row_counts,
                                             row_offsets,
                                             selection_masks,
                                             num_selected,
                                             overflow,
                                             stats_amax,
                                             rowwise_data,
                                             rowwise_scale,
                                             columnwise_data,
                                             columnwise_scale,
                                             columnwise_amax,
                                             rht_output_t,
                                             scan_temp,
                                             packed_records,
                                             active_rows_heavy_light,
                                             active_row_count,
                                             active_sort_rows_in,
                                             active_sort_rows_out,
                                             active_sort_keys_in,
                                             active_sort_keys_out,
                                             active_sort_temp,
                                             hardcap_score_max,
                                             hardcap_histogram,
                                             base_ratio,
                                             min_ratio,
                                             max_ratio,
                                             reference_heaviness,
                                             capacity,
                                             stats_threads,
                                             fill_threads,
                                             columnwise_source,
                                             rht_random_sign_mask_t,
                                             overlap_columnwise,
                                             auto_expand_capacity,
                                             emit_packed_records,
                                             build_active_schedule,
                                             emit_direct_split,
                                             direct_policy_mode,
                                             direct_param0,
                                             direct_param1,
                                             direct_use_bucket,
                                             direct_no_host_slice);
}

std::vector<at::Tensor> adaptive_rowcol_quant_fast_padded_out_cuda(
    const at::Tensor& input,
    const at::Tensor& padded_values,
    const at::Tensor& padded_cols,
    const at::Tensor& row_counts,
    const at::Tensor& selection_masks,
    const at::Tensor& num_selected,
    const at::Tensor& overflow,
    const at::Tensor& overflow_rows,
    const at::Tensor& overflow_cols,
    const at::Tensor& overflow_values,
    const at::Tensor& overflow_count,
    const at::Tensor& stats_amax,
    const at::Tensor& rowwise_data,
    const at::Tensor& rowwise_scale,
    const at::Tensor& columnwise_data,
    const at::Tensor& columnwise_scale,
    const at::Tensor& columnwise_amax,
    const at::Tensor& rht_output_t,
    const at::Tensor& active_rows_heavy_light,
    const at::Tensor& active_row_count,
    const at::Tensor& active_sort_rows_in,
    const at::Tensor& active_sort_rows_out,
    const at::Tensor& active_sort_keys_in,
    const at::Tensor& active_sort_keys_out,
    const at::Tensor& active_sort_temp,
    const at::Tensor& hardcap_score_max,
    const at::Tensor& hardcap_histogram,
    double base_ratio,
    double min_ratio,
    double max_ratio,
    double reference_heaviness,
    int64_t max_per_row,
    int64_t stats_threads,
    int64_t fill_threads,
    int64_t columnwise_source,
    int64_t rht_random_sign_mask_t,
    bool overlap_columnwise,
    bool direct_nomask) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return adaptive_rowcol_quant_fast_padded_out_impl(input,
                                                    padded_values,
                                                    padded_cols,
                                                    row_counts,
                                                    selection_masks,
                                                    num_selected,
                                                    overflow,
                                                    overflow_rows,
                                                    overflow_cols,
                                                    overflow_values,
                                                    overflow_count,
                                                    stats_amax,
                                                    rowwise_data,
                                                    rowwise_scale,
                                                    columnwise_data,
                                                    columnwise_scale,
                                                    columnwise_amax,
                                                    rht_output_t,
                                                    active_rows_heavy_light,
                                                    active_row_count,
                                                    active_sort_rows_in,
                                                    active_sort_rows_out,
                                                    active_sort_keys_in,
                                                    active_sort_keys_out,
                                                    active_sort_temp,
                                                    hardcap_score_max,
                                                    hardcap_histogram,
                                                    base_ratio,
                                                    min_ratio,
                                                    max_ratio,
                                                    reference_heaviness,
                                                    max_per_row,
                                                    stats_threads,
                                                    fill_threads,
                                                    columnwise_source,
                                                    rht_random_sign_mask_t,
                                                    overlap_columnwise,
                                                    direct_nomask);
}

std::vector<at::Tensor> adaptive_rowcol_loghist_quant_fast_out_cuda(
    const at::Tensor& input,
    const at::Tensor& flat_indices,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& row_counts,
    const at::Tensor& row_offsets,
    const at::Tensor& selection_masks,
    const at::Tensor& num_selected,
    const at::Tensor& overflow,
    const at::Tensor& stats_amax,
    const at::Tensor& rowwise_data,
    const at::Tensor& rowwise_scale,
    const at::Tensor& columnwise_data,
    const at::Tensor& columnwise_scale,
    const at::Tensor& columnwise_amax,
    const at::Tensor& rht_output_t,
    const at::Tensor& scan_temp,
    const at::Tensor& log_hist,
    const at::Tensor& log_params,
    double ratio,
    int64_t hist_bins,
    int64_t min_exp,
    int64_t seed,
    int64_t capacity,
    int64_t stats_threads,
    int64_t fill_threads,
    int64_t columnwise_source,
    int64_t rht_random_sign_mask_t,
    bool overlap_columnwise,
    bool auto_expand_capacity) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return adaptive_rowcol_loghist_quant_fast_out_impl(input,
                                                     flat_indices,
                                                     outlier_values,
                                                     outlier_cols,
                                                     row_counts,
                                                     row_offsets,
                                                     selection_masks,
                                                     num_selected,
                                                     overflow,
                                                     stats_amax,
                                                     rowwise_data,
                                                     rowwise_scale,
                                                     columnwise_data,
                                                     columnwise_scale,
                                                     columnwise_amax,
                                                     rht_output_t,
                                                     scan_temp,
                                                     log_hist,
                                                     log_params,
                                                     ratio,
                                                     hist_bins,
                                                     min_exp,
                                                     seed,
                                                     capacity,
                                                     stats_threads,
                                                     fill_threads,
                                                     columnwise_source,
                                                     rht_random_sign_mask_t,
                                                     overlap_columnwise,
                                                     auto_expand_capacity);
}

std::vector<at::Tensor> adaptive_rowcol_refill_csr_cuda(
    const at::Tensor& input,
    const at::Tensor& row_offsets,
    const at::Tensor& selection_masks,
    int64_t selected_nnz,
    int64_t fill_threads) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return adaptive_rowcol_refill_csr_impl(
      input, row_offsets, selection_masks, selected_nnz, fill_threads);
}

std::vector<at::Tensor> adaptive_rowcol_cap_split_packed_cuda(
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    int64_t rows,
    int64_t cols,
    int64_t cap) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(row_offsets));
  return adaptive_rowcol_cap_split_packed_impl(row_offsets, row_ks, row_values, rows, cols, cap);
}

std::vector<at::Tensor> adaptive_rowcol_policy_split_packed_cuda(
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    int64_t rows,
    int64_t cols,
    int64_t policy_mode,
    int64_t param0,
    int64_t param1) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(row_offsets));
  return adaptive_rowcol_policy_split_packed_impl(
      row_offsets, row_ks, row_values, rows, cols, policy_mode, param0, param1);
}

std::vector<at::Tensor> adaptive_rowcol_policy_bucket_split_packed_cuda(
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    int64_t rows,
    int64_t cols,
    int64_t policy_mode,
    int64_t param0,
    int64_t param1) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(row_offsets));
  return adaptive_rowcol_policy_bucket_split_packed_impl(
      row_offsets, row_ks, row_values, rows, cols, policy_mode, param0, param1);
}

std::vector<at::Tensor> adaptive_rowcol_direct_split_packed_cuda(
    const at::Tensor& input,
    const at::Tensor& row_offsets,
    const at::Tensor& selection_masks,
    int64_t selected_nnz,
    int64_t policy_mode,
    int64_t param0,
    int64_t param1,
    bool use_bucket,
    int64_t fill_threads) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  return adaptive_rowcol_direct_split_packed_impl(input,
                                                  row_offsets,
                                                  selection_masks,
                                                  selected_nnz,
                                                  policy_mode,
                                                  param0,
                                                  param1,
                                                  use_bucket,
                                                  fill_threads);
}
