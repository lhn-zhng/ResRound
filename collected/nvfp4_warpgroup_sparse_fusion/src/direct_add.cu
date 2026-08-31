#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kDenseWarps = 8;
constexpr int kDense16Warps = 16;
constexpr int kDenseThreads = kDenseWarps * kWarpSize;
constexpr int kDense16Threads = kDense16Warps * kWarpSize;
constexpr int kDenseWarpNTiles = 15;
constexpr int kDenseWarpN = kDenseWarpNTiles * 8;
constexpr int kDenseBlockM = kDenseWarps * 16;
constexpr int kDense16BlockM = kDense16Warps * 16;
constexpr int kDenseMmaK = 64;
constexpr int kDenseKBytes = kDenseMmaK / 2;
constexpr int kDenseBRowStrideBytes = 32;
constexpr int kDenseBStageBytes = kDenseWarpN * kDenseBRowStrideBytes;
static_assert(kDenseBRowStrideBytes >= kDenseKBytes,
              "dense B shared-memory stride must cover one K tile");

__device__ __forceinline__ uint32_t dense_load_u32(const uint8_t* ptr) {
  return *reinterpret_cast<const uint32_t*>(ptr);
}

__device__ __forceinline__ void dense_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
  const unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void dense_cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" : :);
}

__device__ __forceinline__ void dense_cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" : :);
}

__device__ __forceinline__ float dense_decode_e4m3fn(uint8_t byte) {
  const uint32_t sign_bit = static_cast<uint32_t>(byte & 0x80u) << 24;
  const uint32_t exp = (byte >> 3) & 0x0fu;
  const uint32_t mant = byte & 0x07u;
  if (exp == 0) {
    if (mant == 0) {
      return __uint_as_float(sign_bit);
    }
    const float value = static_cast<float>(mant) * 0.001953125f;
    return sign_bit ? -value : value;
  }
  const uint32_t fp32_exp = exp + 120u;
  return __uint_as_float(sign_bit | (fp32_exp << 23) | (mant << 20));
}

__device__ __forceinline__ float dense_decode_e2m1(uint8_t nibble) {
  const uint8_t mag = nibble & 0x7u;
  float value = 0.0f;
  switch (mag) {
    case 0:
      value = 0.0f;
      break;
    case 1:
      value = 0.5f;
      break;
    case 2:
      value = 1.0f;
      break;
    case 3:
      value = 1.5f;
      break;
    case 4:
      value = 2.0f;
      break;
    case 5:
      value = 3.0f;
      break;
    case 6:
      value = 4.0f;
      break;
    default:
      value = 6.0f;
      break;
  }
  return (nibble & 0x8u) ? -value : value;
}

__device__ __forceinline__ uint32_t dense_pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 h2 = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h2);
}

__device__ __forceinline__ void dense_native_mxf4nvf4_mma(float d[4],
                                                          const uint32_t a[4],
                                                          const uint32_t b[2],
                                                          uint32_t sfa,
                                                          uint32_t sfb) {
  static constexpr uint16_t tidA = 0;
  static constexpr uint16_t bidA = 0;
  static constexpr uint16_t tidB = 0;
  static constexpr uint16_t bidB = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13},"
      "{%14},"
      "{%15, %16},"
      "{%17},"
      "{%18, %19};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "r"(sfa), "h"(bidA),
        "h"(tidA), "r"(sfb), "h"(bidB), "h"(tidB));
}

__device__ __forceinline__ void bf16_mma_m16n8k16(float d[4],
                                                  const uint32_t a[4],
                                                  const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
}

template <int DenseWarpNTiles>
__device__ __forceinline__ void dense_issue_b_tile_load_ntiles(
    uint8_t* __restrict__ b_smem,
    uint32_t* __restrict__ sfb_smem,
    const uint8_t* __restrict__ b_data,
    const uint8_t* __restrict__ b_scale_inv,
    int64_t k_half,
    int64_t k_scale,
    int64_t n,
    int64_t n0,
    int64_t k0,
    int stage,
    int block_tid) {
  static_assert(DenseWarpNTiles > 0 && DenseWarpNTiles <= 16,
                "dense N tile count must fit one warp-local row owner");
  constexpr int DenseWarpN = DenseWarpNTiles * 8;
  constexpr int DenseBStageBytes = DenseWarpN * kDenseBRowStrideBytes;
  uint8_t* stage_b = b_smem + stage * DenseBStageBytes;
  uint32_t* stage_sfb = sfb_smem + stage * DenseWarpN;
  const int load_vecs = DenseWarpN * 2;
  if (block_tid < load_vecs) {
    const int local_col = block_tid >> 1;
    const int half = block_tid & 1;
    const int64_t col = n0 + local_col;
    uint8_t* dst = stage_b + local_col * kDenseBRowStrideBytes + half * 16;
    if (col < n) {
      const uint8_t* src = b_data + col * k_half + (k0 >> 1) + half * 16;
      dense_cp_async_cg_16(dst, src);
    } else {
      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (block_tid < DenseWarpN) {
    const int64_t col = n0 + block_tid;
    stage_sfb[block_tid] =
        col < n ? dense_load_u32(b_scale_inv + col * k_scale + (k0 >> 4)) : 0u;
  }
}

__device__ __forceinline__ void dense_issue_b_tile_load(
    uint8_t* __restrict__ b_smem,
    uint32_t* __restrict__ sfb_smem,
    const uint8_t* __restrict__ b_data,
    const uint8_t* __restrict__ b_scale_inv,
    int64_t k_half,
    int64_t k_scale,
    int64_t n,
    int64_t n0,
    int64_t k0,
    int stage,
    int block_tid) {
  dense_issue_b_tile_load_ntiles<kDenseWarpNTiles>(
      b_smem, sfb_smem, b_data, b_scale_inv, k_half, k_scale, n, n0, k0, stage,
      block_tid);
}

__device__ __forceinline__ float bf16_to_float(const c10::BFloat16 value) {
  const __nv_bfloat16 raw = *reinterpret_cast<const __nv_bfloat16*>(&value);
  return __bfloat162float(raw);
}

__device__ __forceinline__ c10::BFloat16 float_to_bf16(float value) {
  const __nv_bfloat16 raw = __float2bfloat16(value);
  return *reinterpret_cast<const c10::BFloat16*>(&raw);
}

__device__ __forceinline__ float bf16_bits_to_float(uint32_t bits) {
  return __uint_as_float((bits & 0xffffu) << 16);
}

__device__ __forceinline__ float bf16_bits_hi_to_float(uint32_t bits) {
  return __uint_as_float(bits & 0xffff0000u);
}

__device__ __forceinline__ uint32_t bf16_add2_u32(uint32_t a, uint32_t b) {
  __nv_bfloat162_raw a_raw;
  __nv_bfloat162_raw b_raw;
  a_raw.x = static_cast<unsigned short>(a & 0xffffu);
  a_raw.y = static_cast<unsigned short>(a >> 16);
  b_raw.x = static_cast<unsigned short>(b & 0xffffu);
  b_raw.y = static_cast<unsigned short>(b >> 16);
  const __nv_bfloat162 sum = __nv_bfloat162(a_raw) + __nv_bfloat162(b_raw);
  const __nv_bfloat162_raw sum_raw = static_cast<__nv_bfloat162_raw>(sum);
  return static_cast<uint32_t>(sum_raw.x) | (static_cast<uint32_t>(sum_raw.y) << 16);
}

__device__ __forceinline__ uint4 bf16_add_packed_u4(uint4 lhs, uint4 rhs) {
  return make_uint4(
      bf16_add2_u32(lhs.x, rhs.x),
      bf16_add2_u32(lhs.y, rhs.y),
      bf16_add2_u32(lhs.z, rhs.z),
      bf16_add2_u32(lhs.w, rhs.w));
}

__global__ __launch_bounds__(kDenseThreads, 2)
void nvfp4_dense_kernel(c10::BFloat16* __restrict__ output,
                        const uint8_t* __restrict__ a_data,
                        const uint8_t* __restrict__ a_scale_inv,
                        const uint8_t* __restrict__ b_data,
                        const uint8_t* __restrict__ b_scale_inv,
                        const float* __restrict__ a_amax,
                        const float* __restrict__ b_amax,
                        int64_t m,
                        int64_t k,
                        int64_t n,
                        const int32_t* __restrict__ m_tile_ids,
                        int64_t m_tile_count) {
  __shared__ __align__(16) uint8_t b_smem[2 * kDenseBStageBytes];
  __shared__ uint32_t sfb_smem[2 * kDenseWarpN];

  const int block_tid = threadIdx.x;
  const int tid = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t tiles_n = (n + kDenseWarpN - 1) / kDenseWarpN;
  const int64_t block_tile = static_cast<int64_t>(blockIdx.x);
  const int64_t logical_tile_m = block_tile / tiles_n;
  if (m_tile_ids != nullptr && logical_tile_m >= m_tile_count) {
    return;
  }
  const int64_t tile_m =
      m_tile_ids != nullptr ? static_cast<int64_t>(m_tile_ids[logical_tile_m]) : logical_tile_m;
  const int64_t tile_n = block_tile - logical_tile_m * tiles_n;
  const int64_t block_m0 = tile_m * kDenseBlockM;
  const int64_t m0 = block_m0 + warp * 16;
  const int64_t n0 = tile_n * kDenseWarpN;
  if (block_m0 >= m) {
    return;
  }

  const int64_t k_half = k >> 1;
  const int64_t k_scale = k >> 4;
  const float dense_alpha = a_amax[0] * b_amax[0] / (6.0f * 6.0f * 448.0f * 448.0f);
  const int t0_4 = tid & 3;
  const int t1_8 = tid >> 2;
  float d[kDenseWarpNTiles][4] = {};

  dense_issue_b_tile_load(
      b_smem, sfb_smem, b_data, b_scale_inv, k_half, k_scale, n, n0, 0, 0, block_tid);
  dense_cp_async_commit_group();
  dense_cp_async_wait_all();
  __syncthreads();

  for (int64_t k0 = 0; k0 < k; k0 += kDenseMmaK) {
    const int stage = static_cast<int>((k0 / kDenseMmaK) & 1);
    const int next_stage = stage ^ 1;
    const int64_t next_k = k0 + kDenseMmaK;
    if (next_k < k) {
      dense_issue_b_tile_load(b_smem,
                              sfb_smem,
                              b_data,
                              b_scale_inv,
                              k_half,
                              k_scale,
                              n,
                              n0,
                              next_k,
                              next_stage,
                              block_tid);
      dense_cp_async_commit_group();
    }

    uint32_t ar[4] = {0u, 0u, 0u, 0u};
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int64_t row = m0 + t1_8 + ((r & 1) ? 8 : 0);
      const int64_t kk = k0 + (r >> 1) * 32 + t0_4 * 8;
      ar[r] = row < m ? dense_load_u32(a_data + row * k_half + (kk >> 1)) : 0u;
    }
    const int sfa_row = ((tid & 1) << 3) + (tid >> 2);
    const int64_t arow = m0 + sfa_row;
    const uint32_t sfa =
        arow < m ? dense_load_u32(a_scale_inv + arow * k_scale + (k0 >> 4)) : 0u;

    #pragma unroll
    for (int nt = 0; nt < kDenseWarpNTiles; ++nt) {
      uint32_t br[2] = {0u, 0u};
      #pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int byte = r * 16 + t0_4 * 4;
        br[r] = dense_load_u32(
            b_smem + stage * kDenseBStageBytes +
            (nt * 8 + t1_8) * kDenseBRowStrideBytes + byte);
      }
      const uint32_t sfb = sfb_smem[stage * kDenseWarpN + nt * 8 + t1_8];
      dense_native_mxf4nvf4_mma(d[nt], ar, br, sfa, sfb);
    }

    if (next_k < k) {
      dense_cp_async_wait_all();
      __syncthreads();
    }
  }

  #pragma unroll
  for (int nt = 0; nt < kDenseWarpNTiles; ++nt) {
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int linear = t0_4 * 32 + t1_8 + p * 8;
      const int lm = linear & 15;
      const int ln = linear >> 4;
      const int64_t row = m0 + lm;
      const int64_t col = n0 + nt * 8 + ln;
      if (row >= m || col >= n) {
        continue;
      }
      const float out0 = d[nt][p * 2] * dense_alpha;
      const float out1 = d[nt][p * 2 + 1] * dense_alpha;
      if (col + 1 < n) {
        const uint32_t packed = dense_pack_bf16x2(out0, out1);
        *reinterpret_cast<uint32_t*>(output + row * n + col) = packed;
      } else {
        output[row * n + col] = static_cast<c10::BFloat16>(out0);
      }
    }
  }
}

template <int DenseWarpNTiles, bool ActiveRows, bool AccumulateOutput = false>
__global__ __launch_bounds__(kDense16Threads, 1)
void nvfp4_dense16_sparse_tail_delta_kernel(c10::BFloat16* __restrict__ output,
                                            c10::BFloat16* __restrict__ delta_output,
                                            const uint8_t* __restrict__ a_data,
                                            const uint8_t* __restrict__ a_scale_inv,
                                            const uint8_t* __restrict__ b_data,
                                            const uint8_t* __restrict__ b_scale_inv,
                                            const float* __restrict__ a_amax,
                                            const float* __restrict__ b_amax,
                                            const int32_t* __restrict__ row_offsets,
                                            const int32_t* __restrict__ row_ks,
                                            const c10::BFloat16* __restrict__ row_values,
                                            const int32_t* __restrict__ active_row_offsets,
                                            const int32_t* __restrict__ active_rows,
                                            const c10::BFloat16* __restrict__ b_comp,
                                            int64_t m,
                                            int64_t k,
                                            int64_t n) {
  static_assert(DenseWarpNTiles > 0 && DenseWarpNTiles <= 16,
                "dense16 tail N tile count must be in [1, 16]");
  constexpr int DenseWarpN = DenseWarpNTiles * 8;
  constexpr int DenseBStageBytes = DenseWarpN * kDenseBRowStrideBytes;
  __shared__ __align__(16) uint8_t b_smem[2 * DenseBStageBytes];
  __shared__ uint32_t sfb_smem[2 * DenseWarpN];

  const int block_tid = threadIdx.x;
  const int tid = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t tiles_n = (n + DenseWarpN - 1) / DenseWarpN;
  const int64_t block_tile = static_cast<int64_t>(blockIdx.x);
  const int64_t tile_m = block_tile / tiles_n;
  const int64_t tile_n = block_tile - tile_m * tiles_n;
  const int64_t block_m0 = tile_m * kDense16BlockM;
  const int64_t n0 = tile_n * DenseWarpN;
  if (block_m0 >= m) {
    return;
  }

  const int64_t k_half = k >> 1;
  const int64_t k_scale = k >> 4;
  const float dense_alpha = a_amax[0] * b_amax[0] / (6.0f * 6.0f * 448.0f * 448.0f);
  const int t0_4 = tid & 3;
  const int t1_8 = tid >> 2;

  {
    const int64_t m0 = block_m0 + warp * 16;
    float d[DenseWarpNTiles][4] = {};

    dense_issue_b_tile_load_ntiles<DenseWarpNTiles>(
        b_smem, sfb_smem, b_data, b_scale_inv, k_half, k_scale, n, n0, 0, 0, block_tid);
    dense_cp_async_commit_group();
    dense_cp_async_wait_all();
    __syncthreads();

    for (int64_t k0 = 0; k0 < k; k0 += kDenseMmaK) {
      const int stage = static_cast<int>((k0 / kDenseMmaK) & 1);
      const int next_stage = stage ^ 1;
      const int64_t next_k = k0 + kDenseMmaK;
      if (next_k < k) {
        dense_issue_b_tile_load_ntiles<DenseWarpNTiles>(b_smem,
                                                         sfb_smem,
                                                         b_data,
                                                         b_scale_inv,
                                                         k_half,
                                                         k_scale,
                                                         n,
                                                         n0,
                                                         next_k,
                                                         next_stage,
                                                         block_tid);
        dense_cp_async_commit_group();
      }

      uint32_t ar[4] = {0u, 0u, 0u, 0u};
      #pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int64_t row = m0 + t1_8 + ((r & 1) ? 8 : 0);
        const int64_t kk = k0 + (r >> 1) * 32 + t0_4 * 8;
        ar[r] = row < m ? dense_load_u32(a_data + row * k_half + (kk >> 1)) : 0u;
      }
      const int sfa_row = ((tid & 1) << 3) + (tid >> 2);
      const int64_t arow = m0 + sfa_row;
      const uint32_t sfa =
          arow < m ? dense_load_u32(a_scale_inv + arow * k_scale + (k0 >> 4)) : 0u;

      #pragma unroll
      for (int nt = 0; nt < DenseWarpNTiles; ++nt) {
        uint32_t br[2] = {0u, 0u};
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int byte = r * 16 + t0_4 * 4;
          br[r] = dense_load_u32(
              b_smem + stage * DenseBStageBytes +
              (nt * 8 + t1_8) * kDenseBRowStrideBytes + byte);
        }
        const uint32_t sfb = sfb_smem[stage * DenseWarpN + nt * 8 + t1_8];
        dense_native_mxf4nvf4_mma(d[nt], ar, br, sfa, sfb);
      }

      if (next_k < k) {
        dense_cp_async_wait_all();
        __syncthreads();
      }
    }

    #pragma unroll
    for (int nt = 0; nt < DenseWarpNTiles; ++nt) {
      #pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int linear = t0_4 * 32 + t1_8 + p * 8;
        const int lm = linear & 15;
        const int ln = linear >> 4;
        const int64_t row = m0 + lm;
        const int64_t col = n0 + nt * 8 + ln;
        if (row >= m || col >= n) {
          continue;
        }
        const float out0 = d[nt][p * 2] * dense_alpha;
        const float out1 = d[nt][p * 2 + 1] * dense_alpha;
        if (col + 1 < n) {
          const uint32_t packed = dense_pack_bf16x2(out0, out1);
          *reinterpret_cast<uint32_t*>(output + row * n + col) = packed;
        } else {
          output[row * n + col] = static_cast<c10::BFloat16>(out0);
        }
      }
    }
  }
  if constexpr (AccumulateOutput) {
    __syncthreads();
  }

  constexpr int kSparseVecN = 8;
  constexpr int col_groups = DenseWarpN / kSparseVecN;
  static_assert(DenseWarpN % kSparseVecN == 0, "tail sparse path expects vec8 N tiles");
  static_assert(col_groups <= 16, "half-warp row owner supports at most 16 col groups");
  const int half = tid >> 4;
  const int half_lane = tid & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;

  int active_start = 0;
  int active_count = kDense16BlockM;
  if constexpr (ActiveRows) {
    active_start = active_row_offsets[tile_m];
    active_count = active_row_offsets[tile_m + 1] - active_start;
  }

  for (int active_item = warp * 2 + half;
       active_item < active_count;
       active_item += kDense16Warps * 2) {
    int local_row = active_item;
    if constexpr (ActiveRows) {
      local_row = active_rows[active_start + active_item];
    }
    const int local_col0 = half_lane * kSparseVecN;
    const int64_t global_row = block_m0 + local_row;

    int start = 0;
    int end = 0;
    if (half_lane == 0 && global_row < m) {
      start = row_offsets[global_row];
      end = row_offsets[global_row + 1];
    }
    start = __shfl_sync(half_mask, start, half_base_lane);
    end = __shfl_sync(half_mask, end, half_base_lane);
    if (global_row >= m || start == end) {
      continue;
    }

    float acc[kSparseVecN] = {};
    const int64_t global_col0 = n0 + local_col0;
    const bool full_packed_tile = ((n & 7) == 0) && (n0 + DenseWarpN <= n);
    if (full_packed_tile) {
      int cur_gk = 0;
      float cur_av = 0.0f;
      if (half_lane == 0) {
        cur_gk = row_ks[start];
        cur_av = bf16_to_float(row_values[start]);
      }
      cur_gk = __shfl_sync(half_mask, cur_gk, half_base_lane);
      cur_av = __shfl_sync(half_mask, cur_av, half_base_lane);

      bool cur_valid = half_lane < col_groups && cur_gk >= 0 && cur_gk < k;
      uint4 cur_bv = make_uint4(0u, 0u, 0u, 0u);
      if (cur_valid) {
        cur_bv = __ldg(reinterpret_cast<const uint4*>(
            b_comp + static_cast<int64_t>(cur_gk) * n + global_col0));
      }

      for (int entry_idx = start; entry_idx < end; ++entry_idx) {
        int next_gk = 0;
        float next_av = 0.0f;
        const int next_idx = entry_idx + 1;
        if (half_lane == 0) {
          next_gk = next_idx < end ? row_ks[next_idx] : 0;
          next_av = next_idx < end ? bf16_to_float(row_values[next_idx]) : 0.0f;
        }
        next_gk = __shfl_sync(half_mask, next_gk, half_base_lane);
        next_av = __shfl_sync(half_mask, next_av, half_base_lane);

        const bool next_valid = half_lane < col_groups && next_gk >= 0 && next_gk < k &&
                                next_idx < end;
        uint4 next_bv = make_uint4(0u, 0u, 0u, 0u);
        if (next_valid) {
          next_bv = __ldg(reinterpret_cast<const uint4*>(
              b_comp + static_cast<int64_t>(next_gk) * n + global_col0));
        }

        if (cur_valid) {
          acc[0] += cur_av * bf16_bits_to_float(cur_bv.x);
          acc[1] += cur_av * bf16_bits_hi_to_float(cur_bv.x);
          acc[2] += cur_av * bf16_bits_to_float(cur_bv.y);
          acc[3] += cur_av * bf16_bits_hi_to_float(cur_bv.y);
          acc[4] += cur_av * bf16_bits_to_float(cur_bv.z);
          acc[5] += cur_av * bf16_bits_hi_to_float(cur_bv.z);
          acc[6] += cur_av * bf16_bits_to_float(cur_bv.w);
          acc[7] += cur_av * bf16_bits_hi_to_float(cur_bv.w);
        }

        cur_gk = next_gk;
        cur_av = next_av;
        cur_valid = next_valid;
        cur_bv = next_bv;
      }
    } else {
      for (int entry_idx = start; entry_idx < end; ++entry_idx) {
        int gk = 0;
        float av = 0.0f;
        if (half_lane == 0) {
          gk = row_ks[entry_idx];
          av = bf16_to_float(row_values[entry_idx]);
        }
        gk = __shfl_sync(half_mask, gk, half_base_lane);
        av = __shfl_sync(half_mask, av, half_base_lane);

        if (half_lane < col_groups && gk >= 0 && gk < k) {
          #pragma unroll
          for (int cc = 0; cc < kSparseVecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < n) {
              acc[cc] += av * bf16_to_float(b_comp[static_cast<int64_t>(gk) * n + global_col]);
            }
          }
        }
      }
    }

    if (half_lane < col_groups) {
      if (full_packed_tile) {
        const uint4 packed_delta = make_uint4(
            dense_pack_bf16x2(acc[0], acc[1]),
            dense_pack_bf16x2(acc[2], acc[3]),
            dense_pack_bf16x2(acc[4], acc[5]),
            dense_pack_bf16x2(acc[6], acc[7]));
        if constexpr (AccumulateOutput) {
          const uint4 out_v = *reinterpret_cast<const uint4*>(
              output + global_row * n + global_col0);
          *reinterpret_cast<uint4*>(output + global_row * n + global_col0) =
              bf16_add_packed_u4(out_v, packed_delta);
        } else {
          *reinterpret_cast<uint4*>(delta_output + global_row * n + global_col0) = packed_delta;
        }
      } else {
        #pragma unroll
        for (int cc = 0; cc < kSparseVecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < n) {
            if constexpr (AccumulateOutput) {
              const int64_t out_idx = global_row * n + global_col;
              const c10::BFloat16 delta_bf16 = static_cast<c10::BFloat16>(acc[cc]);
              output[out_idx] =
                  float_to_bf16(bf16_to_float(output[out_idx]) + bf16_to_float(delta_bf16));
            } else {
              delta_output[global_row * n + global_col] = static_cast<c10::BFloat16>(acc[cc]);
            }
          }
        }
      }
    }
  }
}

}  // namespace

at::Tensor preallocated_nvfp4_dense_cuda(const at::Tensor& output,
                                         const at::Tensor& a_data,
                                         const at::Tensor& a_scale_inv,
                                         const at::Tensor& b_data,
                                         const at::Tensor& b_scale_inv,
                                         const at::Tensor& a_amax,
                                         const at::Tensor& b_amax,
                                         int64_t m,
                                         int64_t k,
                                         int64_t n) {
  const int64_t tiles_m = (m + kDenseBlockM - 1) / kDenseBlockM;
  const int64_t tiles_n = (n + kDenseWarpN - 1) / kDenseWarpN;
  const dim3 grid(tiles_m * tiles_n);
  const dim3 block(kDenseThreads);
  nvfp4_dense_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      a_data.data_ptr<uint8_t>(),
      a_scale_inv.data_ptr<uint8_t>(),
      b_data.data_ptr<uint8_t>(),
      b_scale_inv.data_ptr<uint8_t>(),
      a_amax.data_ptr<float>(),
      b_amax.data_ptr<float>(),
      m,
      k,
      n,
      nullptr,
      0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <int DenseWarpNTiles>
void launch_dense16_sparse_tail_add_active(at::Tensor output,
                                           const at::Tensor& a_data,
                                           const at::Tensor& a_scale_inv,
                                           const at::Tensor& b_data,
                                           const at::Tensor& b_scale_inv,
                                           const at::Tensor& a_amax,
                                           const at::Tensor& b_amax,
                                           const at::Tensor& row_offsets,
                                           const at::Tensor& row_ks,
                                           const at::Tensor& row_values,
                                           const at::Tensor& active_row_offsets,
                                           const at::Tensor& active_rows,
                                           const at::Tensor& b_comp,
                                           int64_t m,
                                           int64_t k,
                                           int64_t n) {
  constexpr int DenseWarpN = DenseWarpNTiles * 8;
  const int64_t tiles_m = (m + kDense16BlockM - 1) / kDense16BlockM;
  const int64_t tiles_n = (n + DenseWarpN - 1) / DenseWarpN;
  const dim3 grid(tiles_m * tiles_n);
  const dim3 block(kDense16Threads);
  nvfp4_dense16_sparse_tail_delta_kernel<DenseWarpNTiles, true, true>
      <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
          output.data_ptr<c10::BFloat16>(),
          nullptr,
          a_data.data_ptr<uint8_t>(),
          a_scale_inv.data_ptr<uint8_t>(),
          b_data.data_ptr<uint8_t>(),
          b_scale_inv.data_ptr<uint8_t>(),
          a_amax.data_ptr<float>(),
          b_amax.data_ptr<float>(),
          row_offsets.data_ptr<int32_t>(),
          row_ks.data_ptr<int32_t>(),
          row_values.data_ptr<c10::BFloat16>(),
          active_row_offsets.data_ptr<int32_t>(),
          active_rows.data_ptr<int32_t>(),
          b_comp.data_ptr<c10::BFloat16>(),
          m,
          k,
          n);
}

at::Tensor preallocated_nvfp4_dense16_sparse_tail_add_active_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_inv,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_inv,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& active_row_offsets,
    const at::Tensor& active_rows,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t r,
    int64_t kb,
    int64_t c,
    int64_t dense_ntiles) {
  TORCH_CHECK(r == 8 && kb == 32 && c == 32,
              "dense16 active tail direct-add currently supports only R=8, KB=32, C=32");
  if (active_rows.numel() == 0) {
    return preallocated_nvfp4_dense_cuda(
        output, a_data, a_scale_inv, b_data, b_scale_inv, a_amax, b_amax, m, k, n);
  }
  if (dense_ntiles == 16) {
    launch_dense16_sparse_tail_add_active<16>(output,
                                              a_data,
                                              a_scale_inv,
                                              b_data,
                                              b_scale_inv,
                                              a_amax,
                                              b_amax,
                                              row_offsets,
                                              row_ks,
                                              row_values,
                                              active_row_offsets,
                                              active_rows,
                                              b_comp,
                                              m,
                                              k,
                                              n);
  } else if (dense_ntiles == kDenseWarpNTiles) {
    launch_dense16_sparse_tail_add_active<kDenseWarpNTiles>(output,
                                                            a_data,
                                                            a_scale_inv,
                                                            b_data,
                                                            b_scale_inv,
                                                            a_amax,
                                                            b_amax,
                                                            row_offsets,
                                                            row_ks,
                                                            row_values,
                                                            active_row_offsets,
                                                            active_rows,
                                                            b_comp,
                                                            m,
                                                            k,
                                                            n);
  } else {
    TORCH_CHECK(false, "dense_ntiles must be 15 or 16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
