#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <tuple>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "utils.hpp"

namespace fp4 {
#ifndef HANDWRITTEN_TMA_EPIN
#define HANDWRITTEN_TMA_EPIN 64
#endif
#ifndef HANDWRITTEN_TMA_GM
#define HANDWRITTEN_TMA_GM 16
#endif
#ifndef HANDWRITTEN_TMA_STAGES
#define HANDWRITTEN_TMA_STAGES 3
#endif
#ifndef HANDWRITTEN_TMA_PRODUCER_REGS
#define HANDWRITTEN_TMA_PRODUCER_REGS 80
#endif
#ifndef HANDWRITTEN_TMA_CONSUMER_REGS
#define HANDWRITTEN_TMA_CONSUMER_REGS 208
#endif
#ifndef HANDWRITTEN_TMA_FORCE4WG_LOW_REGS
#define HANDWRITTEN_TMA_FORCE4WG_LOW_REGS HANDWRITTEN_TMA_PRODUCER_REGS
#endif
#ifndef HANDWRITTEN_TMA_FORCE4WG_CONSUMER_REGS
#define HANDWRITTEN_TMA_FORCE4WG_CONSUMER_REGS 176
#endif
#ifndef HANDWRITTEN_TMA_FORCE4WG_SPARSE_REGS
#define HANDWRITTEN_TMA_FORCE4WG_SPARSE_REGS HANDWRITTEN_TMA_FORCE4WG_LOW_REGS
#endif
#ifndef HANDWRITTEN_TMA_FORCE5WG_PRODUCER_REGS
#define HANDWRITTEN_TMA_FORCE5WG_PRODUCER_REGS 64
#endif
#ifndef HANDWRITTEN_TMA_FORCE5WG_CONSUMER_REGS
#define HANDWRITTEN_TMA_FORCE5WG_CONSUMER_REGS 168
#endif
#ifndef HANDWRITTEN_TMA_FORCE5WG_SPARSE_REGS
#define HANDWRITTEN_TMA_FORCE5WG_SPARSE_REGS 40
#endif
#ifndef HANDWRITTEN_TMA_SPARSE_ACC_ROWS
#define HANDWRITTEN_TMA_SPARSE_ACC_ROWS 16
#endif
#ifndef HANDWRITTEN_TMA_COMPACT_CONSUMER_MAX_NNZ
#define HANDWRITTEN_TMA_COMPACT_CONSUMER_MAX_NNZ 4
#endif
#ifndef HANDWRITTEN_TMA_COMPACT_CONSUMER_STATIC_N
#define HANDWRITTEN_TMA_COMPACT_CONSUMER_STATIC_N 4096
#endif
#ifndef HANDWRITTEN_TMA_HEAVY_ROW_THRESHOLD
#define HANDWRITTEN_TMA_HEAVY_ROW_THRESHOLD 1000000000
#endif
#ifndef HANDWRITTEN_TMA_LOCAL_DELTA_SPLIT_K
#define HANDWRITTEN_TMA_LOCAL_DELTA_SPLIT_K 1
#endif
#ifndef HANDWRITTEN_TMA_LOCAL_DELTA_SKIP_ROW_NNZ_GE
#define HANDWRITTEN_TMA_LOCAL_DELTA_SKIP_ROW_NNZ_GE 1000000000
#endif
#ifndef HANDWRITTEN_TMA_LOCAL_DELTA_ADAPTIVE_ROW_NNZ_GE
#define HANDWRITTEN_TMA_LOCAL_DELTA_ADAPTIVE_ROW_NNZ_GE 384
#endif
#ifndef HANDWRITTEN_TMA_LOCAL_DELTA_ADAPTIVE_MIN_BLOCK_NNZ
#define HANDWRITTEN_TMA_LOCAL_DELTA_ADAPTIVE_MIN_BLOCK_NNZ 4096
#endif
#ifndef HANDWRITTEN_TMA_LOCAL_DELTA_USE_HOT_ROW_SCHEDULE
#define HANDWRITTEN_TMA_LOCAL_DELTA_USE_HOT_ROW_SCHEDULE 0
#endif
#ifndef HANDWRITTEN_TMA_LOCAL_DELTA_LIGHT_VEC16
#define HANDWRITTEN_TMA_LOCAL_DELTA_LIGHT_VEC16 1
#endif
#ifndef HANDWRITTEN_TMA_LOCAL_DELTA_STAGE_BUFFERS
#define HANDWRITTEN_TMA_LOCAL_DELTA_STAGE_BUFFERS 1
#endif
#ifndef HANDWRITTEN_TMA_LOCAL_DELTA_ONE_WARP_STAGE_ROWS
#define HANDWRITTEN_TMA_LOCAL_DELTA_ONE_WARP_STAGE_ROWS 0
#endif
#ifndef HANDWRITTEN_TMA_PERSISTENT_ROWBLOCK_CHUNK
#define HANDWRITTEN_TMA_PERSISTENT_ROWBLOCK_CHUNK 2
#endif
#ifndef HANDWRITTEN_TMA_PERSISTENT_ROWPAIR_NGROUP
#define HANDWRITTEN_TMA_PERSISTENT_ROWPAIR_NGROUP 4
#endif
#ifndef HANDWRITTEN_TMA_LOCAL_DELTA_HOT_COLSPLIT
#define HANDWRITTEN_TMA_LOCAL_DELTA_HOT_COLSPLIT 0
#endif
#ifndef HANDWRITTEN_TMA_LOCAL_DELTA_SUPERHOT_COLSPLIT_NNZ_GE
#define HANDWRITTEN_TMA_LOCAL_DELTA_SUPERHOT_COLSPLIT_NNZ_GE 1000000000
#endif
#ifndef HANDWRITTEN_TMA_LOCAL_DELTA_SUPERHOT_ATOMIC_NNZ_GE
#define HANDWRITTEN_TMA_LOCAL_DELTA_SUPERHOT_ATOMIC_NNZ_GE 1000000000
#endif
#ifndef HANDWRITTEN_TMA_PHASE_TRACE
#define HANDWRITTEN_TMA_PHASE_TRACE 0
#endif
#ifndef HANDWRITTEN_TMA_ALL_LANE_READY_WAIT
#define HANDWRITTEN_TMA_ALL_LANE_READY_WAIT 0
#endif
#ifndef HANDWRITTEN_TMA_ASSUME_DENSE_TILE_META
#define HANDWRITTEN_TMA_ASSUME_DENSE_TILE_META 0
#endif
#ifndef HANDWRITTEN_TMA_FAST_DENSE_TILE_META_SETUP
#define HANDWRITTEN_TMA_FAST_DENSE_TILE_META_SETUP 0
#endif
#ifndef HANDWRITTEN_TMA_WARP_UNIFORM_TILE_META_LOAD
#define HANDWRITTEN_TMA_WARP_UNIFORM_TILE_META_LOAD 0
#endif
#ifndef HANDWRITTEN_TMA_PREFETCH_TILE_META_BEFORE_TRACE
#define HANDWRITTEN_TMA_PREFETCH_TILE_META_BEFORE_TRACE 0
#endif
#ifndef HANDWRITTEN_TMA_STAGE_READY_PER_WARP
#define HANDWRITTEN_TMA_STAGE_READY_PER_WARP 0
#endif
#ifndef HANDWRITTEN_TMA_STAGE_READY_WORDS
#define HANDWRITTEN_TMA_STAGE_READY_WORDS 0
#endif
#ifndef HANDWRITTEN_TMA_SKIP_READY_CONSUMER_FENCE
#define HANDWRITTEN_TMA_SKIP_READY_CONSUMER_FENCE 0
#endif
#ifndef HANDWRITTEN_TMA_SKIP_READY_PRODUCER_FENCE
#define HANDWRITTEN_TMA_SKIP_READY_PRODUCER_FENCE 0
#endif
#ifndef HANDWRITTEN_TMA_SKIP_STAGE_CONSUMED_EPOCH
#define HANDWRITTEN_TMA_SKIP_STAGE_CONSUMED_EPOCH 0
#endif
#ifndef HANDWRITTEN_TMA_MERGE_EARLY_SKIP
#define HANDWRITTEN_TMA_MERGE_EARLY_SKIP 0
#endif
//constexpr int CtaM = 128;
//constexpr int CtaN = 128;
//constexpr int CtaK = 128;
//constexpr int EpiM = 128;
//constexpr int EpiN = 64;

constexpr int NumThreadsPerWarp = 32;
constexpr int NumThreadsPerWarpGroup = 128;
constexpr int Stages = HANDWRITTEN_TMA_STAGES;
// 2 * 4 warps
constexpr int WorkerRepM = 2;
constexpr int WorkerRepN = 4;
// m16n8k64 * 2 * 4
constexpr int AtomM = 32;
constexpr int AtomN = 32;
constexpr int AtomK = 64;
constexpr int AtomRegA = AtomM * AtomK / NumThreadsPerWarp / 8;
constexpr int AtomRegB = AtomN * AtomK / NumThreadsPerWarp / 8;
constexpr int AtomRegC = AtomM * AtomN / NumThreadsPerWarp;

//constexpr int AtomRepM = CtaM / WorkerRepM / AtomM;
//constexpr int AtomRepN = CtaN / WorkerRepN / AtomN;
//constexpr int AtomRepK = CtaK / AtomK;
//constexpr int WorkerM = CtaM / WorkerRepM;
//constexpr int WorkerN = CtaN / WorkerRepN;

template<int WorkerM>
DEVICE
static size_t worker_m_offset(uint32_t widx) {
  return (widx / WorkerRepN) * WorkerM;
}

template<int WorkerN>
DEVICE
size_t worker_n_offset(uint32_t widx) {
  return (widx % WorkerRepN) * WorkerN;
}

DEVICE
size_t lane_inner_scale_offset(uint32_t lane_idx) {
  return (lane_idx / 4) + (lane_idx & 3) * 8;
}

DEVICE
uint32_t generic_to_shared(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

DEVICE dim3 get_swizzled_blk_coord(dim3 origin) {
  constexpr int GM = HANDWRITTEN_TMA_GM;
  int blk_idx = origin.y * gridDim.x + origin.x;
  const int BlkPerGrpRow = GM * gridDim.x;
  int group_m = blk_idx / BlkPerGrpRow;
  int group_n = blk_idx % BlkPerGrpRow / GM;
  int row_in_group = blk_idx % GM;
  dim3 ret(group_n, group_m * GM + row_in_group, origin.z);
  return ret;
}

// get the address after swizzling
template <int SwizzleByte = 128>
class SmemPtrSw {
public:
  DEVICE 
  SmemPtrSw(void *smem_ptr = nullptr): smem_ptr_(reinterpret_cast<uint8_t*>(smem_ptr)),smem_int_ptr_(generic_to_shared(smem_ptr)) {}

  // 32 banks corespond to 128 bytes
  DEVICE
  uint32_t operator() (uint32_t index) const {
    index += smem_int_ptr_;
    uint32_t row = index / 128;
    uint32_t col = index % SwizzleByte;
    uint32_t swizzled_col = col ^ ((row % (SwizzleByte / 16)) * 16);
    return (index - col) + swizzled_col;
  }

  DEVICE
  void* operator + (size_t offset) const {
    offset += (size_t)smem_ptr_;
    size_t row = offset / 128;
    size_t col = offset % SwizzleByte;
    size_t swizzled_col = col ^ ((row % (SwizzleByte / 16)) * 16);
    return reinterpret_cast<void*>( 
      (offset - col) + swizzled_col
    );
  }

private:
  uint8_t *smem_ptr_;
  uint32_t smem_int_ptr_;
};


namespace ptx {


DEVICE
void
prefetch_tensormap(CUtensorMap const *tensormap_ptr)
{
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tensormap_ptr);
  asm volatile (
    "prefetch.tensormap [%0];"
    :
    : "l"(gmem_int_desc)
    : "memory");
}

template <uint32_t RegCount>
DEVICE
void warpgroup_reg_alloc(){
#ifndef HANDWRITTEN_TMA_DISABLE_SETMAXNREG
  asm volatile ("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
#endif
}

template <uint32_t RegCount>
DEVICE
void warpgroup_reg_dealloc(){
#ifndef HANDWRITTEN_TMA_DISABLE_SETMAXNREG
  asm volatile ("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
#endif
}

DEVICE
void fence_shared_async() {
  asm volatile ("fence.proxy.async.shared::cta;\n");
}

DEVICE
void bar_sync(uint32_t const &tag, uint32_t const &thread_count) {
  asm volatile ("bar.cta.sync %0, %1;\n" :: "r"(tag), "r"(thread_count));
}

DEVICE
void bar_arrive(uint32_t const &tag, uint32_t const &thread_count) {
  asm volatile ("bar.cta.arrive %0, %1;\n" :: "r"(tag), "r"(thread_count));
}

DEVICE
void mbarrier_init(
   uint64_t *mbar_ptr,
   int thread_count
) {
  uint32_t smem_int_ptr = generic_to_shared(mbar_ptr);
  asm volatile (
    "mbarrier.init.shared::cta.b64 [%0], %1;\n"
    :
    : "r"(smem_int_ptr),
      "r"(thread_count)
  );
}

DEVICE
void mbarrier_arrive(
  uint64_t *mbar_ptr
) {
  uint32_t smem_int_ptr = generic_to_shared(mbar_ptr);
  asm volatile(
    "{\n"
    ".reg .b64 state; \n"
    "mbarrier.arrive.shared::cta.b64   state, [%0];\n"
    "}\n"
    :
    : "r"(smem_int_ptr)
  );
}

DEVICE
void mbarrier_arrive_expect_tx(
  uint64_t *mbar_ptr,
  uint32_t const &bytes
) {
  uint32_t smem_int_ptr = generic_to_shared(mbar_ptr);
  asm volatile (
    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;\n"
    :
    : "r"(bytes), "r"(smem_int_ptr)
  );
} 

DEVICE
void mbarrier_arrive_cpasync(
  uint64_t *mbar_ptr
) {
  uint32_t smem_int_ptr = generic_to_shared(mbar_ptr);
  asm volatile (
    "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n"
    :: "r"(smem_int_ptr)
  );
}

DEVICE
uint32_t try_wait_barrier(
  uint64_t *mbar_ptr,
  uint32_t const &phase
) {
  uint32_t smem_int_ptr = generic_to_shared(mbar_ptr);
  uint32_t done;

  asm volatile (
    "{\n"
    ".reg .pred P1;\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
    "selp.b32 %0, 1, 0, P1;\n"
    "}\n"
    : "=r"(done)
    : "r"(smem_int_ptr),
      "r"(phase)
  );

  return done;
}

DEVICE
void wait_barrier(
  uint64_t *mbar_ptr,
  uint32_t const &phase,
  uint32_t const &done = 0
) {
  if (done) return;
  uint32_t smem_int_ptr = generic_to_shared(mbar_ptr);
  uint32_t ticks = 0x989680;
  asm volatile(
    "{\n"
    ".reg .pred                P1;\n"
    "LAB_WAIT:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
    "@P1                       bra DONE;\n"
    "bra                   LAB_WAIT;\n"
    "DONE:\n"
    "}\n"
    :
    : "r"(smem_int_ptr),
      "r"(phase),
      "r"(ticks)
  );
}

template <int bytes, typename T, typename U>
DEVICE
void cpasync(
  T *global,
  U *shared
) {
  uint32_t smem_int_ptr = generic_to_shared(shared);
  asm volatile (
    "cp.async.ca.shared.global [%0], [%1], %2; \n"
    :
    : "r"(smem_int_ptr), "l"(global), "n"(bytes)
  );
}

DEVICE
void tma_copy_tensor_2d(
  CUtensorMap const *tensormap_ptr,
  void const *smem_ptr,
  uint64_t *mbar_ptr,
  uint32_t const &crd0, 
  uint32_t const &crd1
) {
  uint32_t smem_int_ptr = generic_to_shared(smem_ptr);
  uint32_t mbar_int_ptr = generic_to_shared(mbar_ptr);
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tensormap_ptr);
  asm volatile (
    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
    " [%0], [%1, {%3, %4}], [%2], 256;\n"
    :
    : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(mbar_int_ptr),
      "r"(crd1), "r"(crd0)
    : "memory"
  );
}

DEVICE
void tma_store_2d( 
  CUtensorMap const *tensormap_ptr,
  void const *smem_ptr,
  int32_t const &crd0, 
  int32_t const &crd1
) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tensormap_ptr);
  uint32_t smem_int_ptr = generic_to_shared(smem_ptr);
  asm volatile (
    "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];\n"
    :
    : "l"(gmem_int_desc), "r"(smem_int_ptr),
      "r"(crd1), "r"(crd0)
    : "memory"
  );
}

DEVICE
void tma_store_commit() {
  asm volatile ("cp.async.bulk.commit_group;\n");
}

DEVICE
void tma_wait() {
  asm volatile ("cp.async.bulk.wait_group 0;\n");
}

template <int SwizzleByte>
DEVICE
void ldmatrix_a_b4_32x64(
  SmemPtrSw<SwizzleByte> const &smem_ptr,
  uint32_t reg[8],
  uint32_t const &row,
  uint32_t const &col,
  uint32_t const &bytes_per_row,
  int const &lane_idx
) {
  /*
  0 | 2
  1 | 3
  4 | 6
  5 | 7
  */
  uint32_t smem_int_ptr = smem_ptr((row + lane_idx % 16) * bytes_per_row + col / 2 + 16 * (lane_idx / 16));
  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
    : "r"(smem_int_ptr)
    : "memory"
  );
  smem_int_ptr = smem_ptr((16 + row + lane_idx % 16) * bytes_per_row + col / 2 + 16 * (lane_idx / 16));
  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(reg[4]), "=r"(reg[5]), "=r"(reg[6]), "=r"(reg[7])
    : "r"(smem_int_ptr)
    : "memory"
  );
}

template <int SwizzleByte>
DEVICE
void ldmatrix_b_b4_32x64(
  SmemPtrSw<SwizzleByte> const &smem_ptr,
  uint32_t reg[8],
  uint32_t const &row,
  uint32_t const &col,
  uint32_t const &bytes_per_row,
  int const &lane_idx
) {
  /*
  0 | 1
  2 | 3
  4 | 5
  6 | 7
  */
  uint32_t smem_int_ptr = smem_ptr((row + (lane_idx / 16) * 8 + lane_idx % 8) * bytes_per_row + col / 2 + (lane_idx % 16) / 8 * 16);
  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
    : "r"(smem_int_ptr)
    : "memory"
  );
  smem_int_ptr = smem_ptr((row + 16 + (lane_idx / 16) * 8 + lane_idx % 8) * bytes_per_row + col / 2 + (lane_idx % 16) / 8 * 16);
  asm volatile (
    "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(reg[4]), "=r"(reg[5]), "=r"(reg[6]), "=r"(reg[7])
    : "r"(smem_int_ptr)
    : "memory"
  );
}

DEVICE
void mma_nvfp4_16x8x64_(
  uint32_t const &a0, uint32_t const &a1, uint32_t const &a2, uint32_t const &a3,
  uint32_t const &b0, uint32_t const &b1,
  float          &c0, float          &c1, float          &c2, float          &c3,
  uint32_t const &scale_a,
  uint32_t const &scale_b,
  uint16_t const &thr_id_a,
  uint16_t const &thr_id_b
) {
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%0, %1, %2, %3},"
    "{%10},"
    "{%11, %12},"
    "{%13},"
    "{%14, %15};\n"
    :  "+f"(c0),  "+f"(c1),  "+f"(c2),  "+f"(c3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "r"(scale_a), "n"(0), "h"(thr_id_a),
        "r"(scale_b), "n"(0), "h"(thr_id_b)
  );
}

DEVICE
void mma_nvfp4_32x32x64(
  uint32_t const regA[8],
  uint32_t const regB[8],
  float regC[32],
  uint32_t const &scale_a,
  uint32_t const &scale_b
) {
  #pragma unroll
  for (uint16_t n = 0; n < 4; ++n) {
    #pragma unroll
    for (uint16_t m = 0; m < 2; ++m) {
      mma_nvfp4_16x8x64_(
        regA[0+m*4], regA[1+m*4], regA[2+m*4], regA[3+m*4],
        regB[0+n*2], regB[1+n*2],
        regC[0+m*4+n*8], regC[1+m*4+n*8], regC[2+m*4+n*8], regC[3+m*4+n*8], 
        scale_a, scale_b,
        m, n
      );
    }
  }
}

template <int SwizzleByte>
DEVICE
void stmatrix_b16_32x32(
  SmemPtrSw<SwizzleByte> const &smem_ptr,
  uint32_t reg[16],
  uint32_t const &row,
  uint32_t const &col,
  uint32_t const &bytes_per_row,
  int lane_idx
) {
  /*
  0 | 4 | 8  | 12
  1 | 5 | 9  | 13
  2 | 6 | 10 | 14
  3 | 7 | 11 | 15
  */
  #pragma unroll
  for (size_t n = 0; n < 4; ++n) {
    uint32_t smem_int_ptr = smem_ptr((row + lane_idx) * bytes_per_row + col * 2 + n * 16);
    asm volatile (
      "stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%4], {%0, %1, %2, %3}; \n"
      :
      : "r"(reg[0+n*4]), "r"(reg[1+n*4]), "r"(reg[2+n*4]), "r"(reg[3+n*4]), "r"(smem_int_ptr)
    );
  }
}

DEVICE
void clc_try_cancel(
  int4 *clc_ptr,
  uint64_t *mbar_ptr
) {
  uint32_t clc_int_ptr = generic_to_shared(clc_ptr);
  uint32_t mbar_int_ptr = generic_to_shared(mbar_ptr);
  asm volatile (
    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];\n" 
    :
    : "r"(clc_int_ptr),
      "r"(mbar_int_ptr)
  );
}

DEVICE
std::tuple<dim3, uint32_t> clc_query(
  int4 *clc_ptr
) {
  uint32_t valid = 0;
  dim3 result(0, 0, 0);
  uint32_t clc_int_ptr = generic_to_shared(clc_ptr);
  asm volatile(
    "{\n"
    ".reg .pred p1;\n\t"
    ".reg .b128 clc_result;\n\t"
    "ld.shared.b128 clc_result, [%4];\n\t"
    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;\n\t"
    "selp.u32 %3, 1, 0, p1;\n\t"
    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_result;\n\t"
    "}\n"
    : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(valid)
    : "r"(clc_int_ptr)
    : "memory"
  );
  ptx::fence_shared_async();
  return std::make_tuple(get_swizzled_blk_coord(result), valid);
}

DEVICE
uint32_t cvt_fp32_to_bf16x2(float const &a, float const &b) {
  uint32_t ret;
  asm volatile (
    "cvt.rn.bf16x2.f32 %0, %1, %2;\n"
    : "=r"(ret)
    : "f"(b), "f"(a)
  );
  return ret;
}

DEVICE
uint32_t cvt_fp32_to_fp16x2(float const &a, float const &b) {
  uint32_t ret;
  asm volatile (
    "cvt.rn.f16x2.f32 %0, %1, %2;\n"
    : "=r"(ret)
    : "f"(b), "f"(a)
  );
  return ret;
}


} // namespace ptx;


template <int Stage>
class Pipeline {
public:
  class State {
  public:
    DEVICE
    State (uint32_t const &init_phase = 0): phase_(init_phase), stage_(0) {}
    DEVICE
    void advance() {
      if (++stage_ == Stage) {
        stage_ = 0;
        phase_ ^= 1;
      }
    }
    DEVICE
    uint32_t const &phase() const { return phase_; }
    DEVICE
    uint32_t const &stage() const { return stage_; }
  private:
    uint32_t phase_;
    uint32_t stage_;
  };

  struct alignas(8) Storage {
    alignas(8) uint64_t full[Stage];
    alignas(8) uint64_t empty[Stage];
  };

  DEVICE
  Pipeline(Storage &storage, uint32_t num_producers, uint32_t num_consumers): storage_(storage) {
    if (threadIdx.x == 0) {
      #pragma unroll
      for (int i = 0; i < Stage; ++i) {
        ptx::mbarrier_init(&storage.full[i], num_producers);
        ptx::mbarrier_init(&storage.empty[i], num_consumers);
      }
    }
    __syncwarp();
  }

  DEVICE
  void producer_arrive_expect_tx(State const &state, uint32_t bytes) {
    ptx::mbarrier_arrive_expect_tx(
      &storage_.full[state.stage()], bytes
    );
  }

  DEVICE
  void producer_arrive_cpasync(State const &state) {
    ptx::mbarrier_arrive_cpasync(&storage_.full[state.stage()]);
  }

  DEVICE
  uint32_t producer_try_wait(State const &state) {
    return ptx::try_wait_barrier(
      &storage_.empty[state.stage()],
      state.phase()
    );
  }

  DEVICE
  void producer_wait(State const &state, uint32_t const &done = 0) {
    ptx::wait_barrier(
      &storage_.empty[state.stage()],
      state.phase(), done
    );
  }

  DEVICE
  void producer_commit(State const &state) {
    ptx::mbarrier_arrive(&storage_.full[state.stage()]);
  }

  DEVICE
  uint64_t *producer_get_mbar(State const &state) {
    return &storage_.full[state.stage()];
  }

  DEVICE
  void consumer_arrive(State const &state) {
    ptx::mbarrier_arrive(&storage_.empty[state.stage()]);
  }

  DEVICE
  uint32_t consumer_try_wait(State const &state) {
    return ptx::try_wait_barrier(&storage_.full[state.stage()], state.phase());
  }

  DEVICE
  void consumer_wait(State const &state, uint32_t const &done = 0) {
    ptx::wait_barrier(
      &storage_.full[state.stage()],
      state.phase(), done
    );
  }

private:
  Storage &storage_;
};

struct Params {
  CUtensorMap tensormap_A;
  CUtensorMap tensormap_B;
  CUtensorMap tensormap_D;
  uint8_t *inner_scale_A;
  uint8_t *inner_scale_B;
  const float *amax_A;
  const float *amax_B;
  int32_t *ready_flags;
  int32_t *ready_queue;
  int32_t *ready_slot_status;
  int32_t *ready_tail;
  int32_t *ready_m_queue;
  int32_t *ready_m_slot_status;
  int32_t *ready_m_tail;
  int32_t *ready_m_counts;
  int32_t ready_m_slices;
  bool scale_tile_major;
  void *D;
  int64_t m;
  int64_t n;
  int64_t k;
  const int32_t* direct_row_offsets;
  const int32_t* direct_row_ks;
  const c10::BFloat16* direct_row_values;
  const int32_t* direct_active_row_offsets;
  const int32_t* direct_active_rows;
  int32_t direct_active_row_count;
  const int32_t* direct_packed_tile_offsets;
  const int64_t* direct_packed_row_records;
  const int32_t* direct_packed_entry_records;
  const c10::BFloat16* direct_b_comp;
  const int32_t* direct_probe_active_mblocks;
  const int32_t* direct_kmajor_group_offsets;
  const int32_t* direct_kmajor_group_ks;
  const int32_t* direct_kmajor_entry_offsets;
  const int32_t* direct_kmajor_entry_rows;
  const c10::BFloat16* direct_kmajor_entry_values;
  const int32_t* direct_kmajor_tile_group_starts;
  const int32_t* direct_kmajor_tile_group_counts;
  const int64_t* direct_kmajor_tile_group_meta;
  c10::BFloat16* direct_delta_output;
  float* direct_probe_sink;
  int32_t* direct_probe_counter;
  int32_t direct_probe_warps;
  int32_t direct_probe_do_math;
  int32_t direct_probe_total_tiles;
  int32_t direct_probe_mixed_cta;
  int32_t direct_probe_kmajor;
  int32_t direct_delta_write_mode;
  int32_t direct_probe_dense_grid_y;
  int32_t direct_probe_active_mblock_count;
  int32_t direct_probe_persistent_cta_count;
  int32_t direct_probe_group_budget;
  int32_t direct_delta_chunk_limit;
  int32_t direct_smem_add;
  int32_t force_dense_4wg;
  uint64_t* phase_trace;
  int32_t phase_trace_stride;
  int32_t phase_trace_max_ctas;
  int32_t phase_trace_mode;
  int32_t direct_packed_payload_mode;
};

DEVICE bool direct_kmajor_has_tile_group_meta(Params const& params) {
  return params.direct_kmajor_tile_group_meta != nullptr ||
         (params.direct_kmajor_tile_group_starts != nullptr &&
          params.direct_kmajor_tile_group_counts != nullptr);
}

DEVICE uint64_t direct_kmajor_tile_group_meta_load_warp(Params const& params,
                                                        uint32_t blk_m,
                                                        int lane_idx) {
  uint32_t meta_lo = 0u;
  uint32_t meta_hi = 0u;
  if (lane_idx == 0) {
    const uint64_t meta =
        static_cast<uint64_t>(params.direct_kmajor_tile_group_meta[blk_m]);
    meta_lo = static_cast<uint32_t>(meta & 0xffffffffull);
    meta_hi = static_cast<uint32_t>(meta >> 32);
  }
  meta_lo = __shfl_sync(0xffffffffu, meta_lo, 0);
  meta_hi = __shfl_sync(0xffffffffu, meta_hi, 0);
  return static_cast<uint64_t>(meta_lo) |
         (static_cast<uint64_t>(meta_hi) << 32);
}

DEVICE uint64_t direct_kmajor_tile_group_meta_load(Params const& params,
                                                   uint32_t blk_m,
                                                   int lane_idx) {
#if HANDWRITTEN_TMA_WARP_UNIFORM_TILE_META_LOAD
  return direct_kmajor_tile_group_meta_load_warp(params, blk_m, lane_idx);
#else
  (void)lane_idx;
  return static_cast<uint64_t>(params.direct_kmajor_tile_group_meta[blk_m]);
#endif
}

DEVICE void direct_kmajor_tile_group_range(Params const& params,
                                           int active_m_idx,
                                           uint32_t blk_m,
                                           int& group_start,
                                           int& group_count) {
  if (params.direct_kmajor_tile_group_starts != nullptr &&
      params.direct_kmajor_tile_group_counts != nullptr) {
    group_start = params.direct_kmajor_tile_group_starts[blk_m];
    group_count = params.direct_kmajor_tile_group_counts[blk_m];
  } else if (params.direct_kmajor_tile_group_meta != nullptr) {
    const uint64_t meta =
        static_cast<uint64_t>(params.direct_kmajor_tile_group_meta[blk_m]);
    group_start = static_cast<int>(meta & 0xffffffffull);
    group_count = static_cast<int>(meta >> 32);
  } else if (active_m_idx >= 0 && params.direct_kmajor_group_offsets != nullptr) {
    group_start = params.direct_kmajor_group_offsets[active_m_idx];
    group_count = params.direct_kmajor_group_offsets[active_m_idx + 1] - group_start;
  } else {
    group_start = 0;
    group_count = 0;
  }
}

#if HANDWRITTEN_TMA_PHASE_TRACE
enum PhaseTraceSlot : int {
  PhaseTraceKernelEntry = 0,
  PhaseTraceProducerTmaWait = 1,
  PhaseTraceProducerTmaIssue = 2,
  PhaseTraceScaleAWait = 3,
  PhaseTraceScaleACopy = 4,
  PhaseTraceScaleBWait = 5,
  PhaseTraceScaleBCopy = 6,
  PhaseTraceDenseTotal = 7,
  PhaseTraceDenseWait = 8,
  PhaseTraceDenseScaleLoad = 9,
  PhaseTraceDenseLdmatrix = 10,
  PhaseTraceDenseMma = 11,
  PhaseTraceDenseAccum = 12,
  PhaseTraceSparseSideTotal = 13,
  PhaseTraceSparseSideSetup = 14,
  PhaseTraceSparseSideCompute = 15,
  PhaseTraceSparseSideSync = 16,
  PhaseTraceEpilogueTotal = 17,
  PhaseTraceEpilogueConvert = 18,
  PhaseTraceEpilogueStageWait = 19,
  PhaseTraceEpilogueLocalDeltaMerge = 20,
  PhaseTraceEpilogueOtherMerge = 21,
  PhaseTraceEpilogueTmaStore = 22,
  PhaseTraceKernelExit = 23,
  PhaseTraceTileM = 24,
  PhaseTraceTileN = 25,
  PhaseTraceLocalActiveRows = 26,
  PhaseTraceLocalHotGroups = 27,
  PhaseTraceMixedCta = 28,
  PhaseTraceProbeWarps = 29,
  PhaseTraceKStages = 30,
  PhaseTraceValid = 31,
  PhaseTraceSparseLightMeta = 32,
  PhaseTraceSparseLightBLoad = 33,
  PhaseTraceSparseLightFma = 34,
  PhaseTraceSparseLightStore = 35,
  PhaseTraceSparseLightEntries = 36,
  PhaseTraceSparseLightRows = 37,
  PhaseTraceSparseLightRowMeta = 38,
  PhaseTraceSparseLightEntryMeta = 39,
  PhaseTraceWg0ScaleASparseProbe = 40,
  PhaseTraceWg0ScaleBSparseProbe = 41,
  PhaseTraceWg0ScaleASparseProbeCalls = 42,
  PhaseTraceWg0ScaleBSparseProbeCalls = 43,
  PhaseTraceSparseStage0Compute = 44,
  PhaseTraceSparseStage1Compute = 45,
  PhaseTraceSparseStage0Sync = 46,
  PhaseTraceSparseStage1Sync = 47,
  PhaseTraceEpilogueStage0Wait = 48,
  PhaseTraceEpilogueStage1Wait = 49,
  PhaseTraceEpilogueStage0Merge = 50,
  PhaseTraceEpilogueStage1Merge = 51,
  PhaseTraceSparseSideSetupCore = 52,
  PhaseTraceSparseSetupActiveLookup = 53,
  PhaseTraceSparseSetupGroupCount = 54,
  PhaseTraceSparseSetupLocalRows = 55,
  PhaseTraceEpilogueReadySpin = 56,
  PhaseTraceEpilogueReadyBarrier = 57,
  PhaseTraceSideMergeScWait = 58,
  PhaseTraceSideMergeApply = 59,
  PhaseTraceDenseMergeDoneWait = 60,
  PhaseTraceSlotCount = 61
};

DEVICE uint64_t phase_trace_clock() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

DEVICE bool phase_trace_enabled(const Params& params) {
  return params.phase_trace != nullptr &&
         params.phase_trace_stride >= PhaseTraceSlotCount &&
         params.phase_trace_max_ctas > 0;
}

DEVICE int64_t phase_trace_cta_index() {
  return static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
}

DEVICE bool phase_trace_cta_enabled(const Params& params) {
  return phase_trace_enabled(params) &&
         phase_trace_cta_index() < params.phase_trace_max_ctas;
}

DEVICE uint64_t* phase_trace_row(const Params& params) {
  return params.phase_trace +
         phase_trace_cta_index() * static_cast<int64_t>(params.phase_trace_stride);
}

DEVICE void phase_trace_write(const Params& params, int slot, uint64_t value) {
  if (phase_trace_cta_enabled(params)) {
    phase_trace_row(params)[slot] = value;
  }
}

DEVICE void phase_trace_add(const Params& params, int slot, uint64_t value) {
  if (phase_trace_cta_enabled(params)) {
    phase_trace_row(params)[slot] += value;
  }
}

DEVICE bool phase_trace_producer_lane(const Params& params, int lane_idx) {
  return lane_idx == 0 && phase_trace_cta_enabled(params);
}

DEVICE bool phase_trace_dense_lane(const Params& params, int wg_id, int widx, int lane_idx) {
  return wg_id == 1 && widx == 0 && lane_idx == 0 &&
         phase_trace_cta_enabled(params);
}

DEVICE bool phase_trace_sparse_lane(const Params& params, int sparse_warp_rank, int lane_idx) {
  return sparse_warp_rank == 0 && lane_idx == 0 &&
         phase_trace_cta_enabled(params);
}
#endif

DEVICE float direct_bf16_to_float(const c10::BFloat16 value) {
  const __nv_bfloat16 raw = *reinterpret_cast<const __nv_bfloat16*>(&value);
  return __bfloat162float(raw);
}

DEVICE c10::BFloat16 direct_float_to_bf16(float value) {
  const __nv_bfloat16 raw = __float2bfloat16(value);
  return *reinterpret_cast<const c10::BFloat16*>(&raw);
}

DEVICE uint16_t direct_float_to_bf16_bits_u16(float value) {
  const __nv_bfloat16 raw = __float2bfloat16(value);
  return *reinterpret_cast<const uint16_t*>(&raw);
}

DEVICE void direct_atomic_add_bf16(c10::BFloat16* ptr, float value) {
  atomicAdd(reinterpret_cast<__nv_bfloat16*>(ptr), __float2bfloat16(value));
}

DEVICE float direct_bf16_bits_to_float(uint32_t bits) {
  return __uint_as_float((bits & 0xffffu) << 16);
}

DEVICE float direct_bf16_bits_hi_to_float(uint32_t bits) {
  return __uint_as_float(bits & 0xffff0000u);
}

DEVICE uint32_t direct_pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 h2 = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h2);
}

DEVICE uint32_t direct_bf16_add2_u32(uint32_t a, uint32_t b) {
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

DEVICE uint4 direct_bf16_add_packed_u4(uint4 lhs, uint4 rhs) {
  return make_uint4(direct_bf16_add2_u32(lhs.x, rhs.x),
                    direct_bf16_add2_u32(lhs.y, rhs.y),
                    direct_bf16_add2_u32(lhs.z, rhs.z),
                    direct_bf16_add2_u32(lhs.w, rhs.w));
}

DEVICE uint32_t direct_bf16_add2_u16_delta(
    uint32_t packed,
    uint16_t delta_lo,
    uint16_t delta_hi) {
  const uint32_t packed_delta =
      static_cast<uint32_t>(delta_lo) |
      (static_cast<uint32_t>(delta_hi) << 16);
  return direct_bf16_add2_u32(packed, packed_delta);
}

DEVICE uint32_t direct_reg_add_sparse_bf16x2(
    Params const& params,
    uint32_t packed,
    int64_t global_row,
    int64_t global_col0) {
  if (params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_row_values == nullptr ||
      params.direct_b_comp == nullptr ||
      global_row < 0 || global_row >= params.m ||
      global_col0 >= params.n) {
    return packed;
  }
  const int start = params.direct_row_offsets[global_row];
  const int end = params.direct_row_offsets[global_row + 1];
  if (start >= end) {
    return packed;
  }
  float lo = direct_bf16_bits_to_float(packed);
  float hi = direct_bf16_bits_hi_to_float(packed);
  float acc_lo = 0.0f;
  float acc_hi = 0.0f;
  const bool do_lo = global_col0 >= 0 && global_col0 < params.n;
  const bool do_hi = global_col0 + 1 >= 0 && global_col0 + 1 < params.n;
  for (int pos = start; pos < end; ++pos) {
    const int gk = params.direct_row_ks[pos];
    if (gk < 0 || gk >= params.k) {
      continue;
    }
    const float av = direct_bf16_to_float(params.direct_row_values[pos]);
    const int64_t base = static_cast<int64_t>(gk) * params.n;
    if (do_lo) {
      acc_lo = fmaf(
          av,
          direct_bf16_to_float(params.direct_b_comp[base + global_col0]),
          acc_lo);
    }
    if (do_hi) {
      acc_hi = fmaf(
          av,
          direct_bf16_to_float(params.direct_b_comp[base + global_col0 + 1]),
          acc_hi);
    }
  }
  if (do_lo) {
    lo += acc_lo;
  }
  if (do_hi) {
    hi += acc_hi;
  }
  return direct_pack_bf16x2(lo, hi);
}

template<int Rows, int MaxNnz>
struct alignas(128) SparseCompactInputStorage {
  // WG3 writes one fixed-size record per local M row.  Consumers only read
  // records for accumulator rows they own, so no dense-shaped correction ever
  // crosses the warpgroup boundary.
  alignas(128) uint32_t counts[Rows];
  alignas(128) uint32_t entries[Rows][MaxNnz];
};

DEVICE uint32_t direct_bf16_add3_u32(uint32_t a, uint32_t b, uint32_t c) {
  const float a0 = direct_bf16_bits_to_float(a);
  const float a1 = direct_bf16_bits_hi_to_float(a);
  const float b0 = direct_bf16_bits_to_float(b);
  const float b1 = direct_bf16_bits_hi_to_float(b);
  const float c0 = direct_bf16_bits_to_float(c);
  const float c1 = direct_bf16_bits_hi_to_float(c);
  return direct_pack_bf16x2(a0 + b0 + c0, a1 + b1 + c1);
}

DEVICE uint4 direct_bf16_add3_packed_u4(uint4 lhs, uint4 rhs0, uint4 rhs1) {
  return make_uint4(direct_bf16_add3_u32(lhs.x, rhs0.x, rhs1.x),
                    direct_bf16_add3_u32(lhs.y, rhs0.y, rhs1.y),
                    direct_bf16_add3_u32(lhs.z, rhs0.z, rhs1.z),
                    direct_bf16_add3_u32(lhs.w, rhs0.w, rhs1.w));
}

DEVICE float warp_reduce_sum_f32(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template<int CtaM, int CtaN>
DEVICE void apply_sparse_loadfma_probe_tile(Params const& params,
                                            uint32_t blk_m,
                                            uint32_t blk_n,
                                            int lane_idx,
                                            int probe_rank,
                                            int probe_count,
                                            int sink_rank_offset) {
  if (params.direct_probe_sink == nullptr ||
      params.direct_row_offsets == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = CtaN / VecN;
  static_assert(CtaN % VecN == 0, "load+FMA probe expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  constexpr unsigned ValidHalfMask =
      ColGroups >= 16 ? 0x0000ffffu : ((1u << ColGroups) - 1u);
  const unsigned half_mask = ValidHalfMask << half_base_lane;
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_count =
      params.direct_active_row_offsets[blk_m + 1] - active_start;
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
  const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);
  float lane_sum = 0.0f;

  for (int active_item = probe_rank * 2 + half; active_item < active_count;
       active_item += probe_count * 2) {
    const int local_row = params.direct_active_rows[active_start + active_item];
    const int64_t global_row = block_m0 + local_row;
    const int64_t global_col0 = n0 + half_lane * VecN;

    int start = 0;
    int end = 0;
    if (half_lane == 0 && global_row < params.m) {
      start = params.direct_row_offsets[global_row];
      end = params.direct_row_offsets[global_row + 1];
    }
    start = __shfl_sync(half_mask, start, half_base_lane);
    end = __shfl_sync(half_mask, end, half_base_lane);
    if (global_row >= params.m || start == end) {
      continue;
    }

    float acc[VecN] = {};
    if (full_packed_tile) {
      int cur_gk = 0;
      float cur_av = 0.0f;
      if (half_lane == 0) {
        cur_gk = params.direct_row_ks[start];
        cur_av = direct_bf16_to_float(params.direct_row_values[start]);
      }
      cur_gk = __shfl_sync(half_mask, cur_gk, half_base_lane);
      cur_av = __shfl_sync(half_mask, cur_av, half_base_lane);

      bool cur_valid = half_lane < ColGroups && cur_gk >= 0 && cur_gk < params.k;
      uint4 cur_bv = make_uint4(0u, 0u, 0u, 0u);
      if (cur_valid) {
        cur_bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(cur_gk) * params.n + global_col0));
      }

      for (int entry_idx = start; entry_idx < end; ++entry_idx) {
        int next_gk = 0;
        float next_av = 0.0f;
        const int next_idx = entry_idx + 1;
        if (half_lane == 0) {
          next_gk = next_idx < end ? params.direct_row_ks[next_idx] : 0;
          next_av =
              next_idx < end ? direct_bf16_to_float(params.direct_row_values[next_idx]) : 0.0f;
        }
        next_gk = __shfl_sync(half_mask, next_gk, half_base_lane);
        next_av = __shfl_sync(half_mask, next_av, half_base_lane);

        const bool next_valid =
            half_lane < ColGroups && next_gk >= 0 && next_gk < params.k && next_idx < end;
        uint4 next_bv = make_uint4(0u, 0u, 0u, 0u);
        if (next_valid) {
          next_bv = __ldg(reinterpret_cast<const uint4*>(
              params.direct_b_comp + static_cast<int64_t>(next_gk) * params.n + global_col0));
        }

        if (cur_valid) {
          acc[0] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.x), acc[0]);
          acc[1] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.x), acc[1]);
          acc[2] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.y), acc[2]);
          acc[3] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.y), acc[3]);
          acc[4] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.z), acc[4]);
          acc[5] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.z), acc[5]);
          acc[6] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.w), acc[6]);
          acc[7] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.w), acc[7]);
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
          gk = params.direct_row_ks[entry_idx];
          av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
        }
        gk = __shfl_sync(half_mask, gk, half_base_lane);
        av = __shfl_sync(half_mask, av, half_base_lane);
        if (half_lane < ColGroups && gk >= 0 && gk < params.k) {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < params.n) {
              acc[cc] = fmaf(
                  av,
                  direct_bf16_to_float(
                      params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                  acc[cc]);
            }
          }
        }
      }
    }

    if (half_lane < ColGroups) {
      if (params.direct_delta_output != nullptr) {
        c10::BFloat16* delta_output = params.direct_delta_output;
        if (full_packed_tile) {
          const uint4 packed_delta =
              make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                         direct_pack_bf16x2(acc[2], acc[3]),
                         direct_pack_bf16x2(acc[4], acc[5]),
                         direct_pack_bf16x2(acc[6], acc[7]));
          *reinterpret_cast<uint4*>(
              delta_output + global_row * params.n + global_col0) = packed_delta;
        } else {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < params.n) {
              delta_output[global_row * params.n + global_col] = direct_float_to_bf16(acc[cc]);
            }
          }
        }
      }
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        lane_sum += acc[cc];
      }
    }
  }

  lane_sum = warp_reduce_sum_f32(lane_sum);
  if (lane_idx == 0) {
    const int32_t tile_id =
        static_cast<int32_t>(blk_m * gridDim.x + blk_n);
    volatile float* sink = params.direct_probe_sink;
    sink[tile_id * params.direct_probe_warps + sink_rank_offset + probe_rank] = lane_sum;
  }
}

template <bool StoreDelta,
          bool CompactDelta = false,
          bool RecordSink = true,
          int SparseThreadsPerCta = NumThreadsPerWarpGroup,
          bool StaticN4096 = false>
DEVICE void apply_sparse_active_row_vec8_delta_worker(Params const& params,
                                                      int32_t worker_id,
                                                      int32_t worker_count,
                                                      int32_t local_thread_rank) {
  static_assert(SparseThreadsPerCta > 0, "sparse worker needs at least one thread");
  if (params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_row_values == nullptr ||
      params.direct_active_rows == nullptr ||
      params.direct_active_row_count <= 0 ||
      params.direct_b_comp == nullptr ||
      (StoreDelta && params.direct_delta_output == nullptr) ||
      worker_id < 0 ||
      worker_count <= 0 ||
      local_thread_rank < 0) {
    return;
  }
  if constexpr (RecordSink) {
    if (params.direct_probe_sink == nullptr) {
      return;
    }
  }

  constexpr int VecN = 8;
  constexpr int64_t StaticGroupsPerRow = 4096 / VecN;
  const int64_t groups_per_row =
      StaticN4096 ? StaticGroupsPerRow : static_cast<int64_t>(params.n) / VecN;
  if ((!StaticN4096 && (groups_per_row <= 0 || (params.n & 7) != 0)) ||
      (StaticN4096 && params.n != 4096)) {
    return;
  }
  const int64_t total_groups =
      static_cast<int64_t>(params.direct_active_row_count) * groups_per_row;
  const int64_t worker_threads =
      static_cast<int64_t>(worker_count) * SparseThreadsPerCta;
  int64_t group_idx =
      static_cast<int64_t>(worker_id) * SparseThreadsPerCta + local_thread_rank;
  float lane_sum = 0.0f;

  for (; group_idx < total_groups; group_idx += worker_threads) {
    const int64_t active_row_idx =
        StaticN4096 ? (group_idx >> 9) : (group_idx / groups_per_row);
    const int64_t row = static_cast<int64_t>(params.direct_active_rows[active_row_idx]);
    if (row < 0 || row >= params.m) {
      continue;
    }
    const int64_t col_group =
        StaticN4096 ? (group_idx & (StaticGroupsPerRow - 1))
                    : (group_idx - active_row_idx * groups_per_row);
    const int64_t base_col = col_group * VecN;
    const int32_t start = params.direct_row_offsets[row];
    const int32_t end = params.direct_row_offsets[row + 1];

    float acc[VecN] = {};
    for (int32_t pos = start; pos < end; ++pos) {
      const int32_t kk = params.direct_row_ks[pos];
      if (kk < 0 || kk >= params.k) {
        continue;
      }
      const float value = direct_bf16_to_float(params.direct_row_values[pos]);
      const uint4 packed_weight = __ldg(reinterpret_cast<const uint4*>(
          params.direct_b_comp +
          static_cast<int64_t>(kk) * (StaticN4096 ? 4096 : params.n) + base_col));
      acc[0] = fmaf(value, direct_bf16_bits_to_float(packed_weight.x), acc[0]);
      acc[1] = fmaf(value, direct_bf16_bits_hi_to_float(packed_weight.x), acc[1]);
      acc[2] = fmaf(value, direct_bf16_bits_to_float(packed_weight.y), acc[2]);
      acc[3] = fmaf(value, direct_bf16_bits_hi_to_float(packed_weight.y), acc[3]);
      acc[4] = fmaf(value, direct_bf16_bits_to_float(packed_weight.z), acc[4]);
      acc[5] = fmaf(value, direct_bf16_bits_hi_to_float(packed_weight.z), acc[5]);
      acc[6] = fmaf(value, direct_bf16_bits_to_float(packed_weight.w), acc[6]);
      acc[7] = fmaf(value, direct_bf16_bits_hi_to_float(packed_weight.w), acc[7]);
    }

    if constexpr (StoreDelta) {
      const int64_t delta_row = CompactDelta ? active_row_idx : row;
      const uint4 packed_delta =
          make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                     direct_pack_bf16x2(acc[2], acc[3]),
                     direct_pack_bf16x2(acc[4], acc[5]),
                     direct_pack_bf16x2(acc[6], acc[7]));
      *reinterpret_cast<uint4*>(
          params.direct_delta_output +
          delta_row * (StaticN4096 ? 4096 : params.n) + base_col) = packed_delta;
    }

    if constexpr (RecordSink) {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        lane_sum += acc[cc];
      }
    }
  }

  if constexpr (RecordSink) {
    if (local_thread_rank == 0) {
      params.direct_probe_sink[
          static_cast<int64_t>(worker_id) * params.direct_probe_warps] = lane_sum;
    }
  }
}

template <bool StoreDelta, bool CompactDelta = false>
DEVICE void apply_sparse_active_row_vec8_prefetch_delta_worker(Params const& params,
                                                               int32_t worker_id,
                                                               int32_t worker_count,
                                                               int32_t local_thread_rank) {
  if (params.direct_probe_sink == nullptr ||
      params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_row_values == nullptr ||
      params.direct_active_rows == nullptr ||
      params.direct_active_row_count <= 0 ||
      params.direct_b_comp == nullptr ||
      (StoreDelta && params.direct_delta_output == nullptr) ||
      worker_id < 0 ||
      worker_count <= 0 ||
      local_thread_rank < 0) {
    return;
  }

  constexpr int VecN = 8;
  const int64_t groups_per_row = static_cast<int64_t>(params.n) / VecN;
  if (groups_per_row <= 0 || (params.n & 7) != 0) {
    return;
  }
  const int64_t total_groups =
      static_cast<int64_t>(params.direct_active_row_count) * groups_per_row;
  const int64_t worker_threads =
      static_cast<int64_t>(worker_count) * NumThreadsPerWarpGroup;
  int64_t group_idx =
      static_cast<int64_t>(worker_id) * NumThreadsPerWarpGroup + local_thread_rank;
  float lane_sum = 0.0f;

  for (; group_idx < total_groups; group_idx += worker_threads) {
    const int64_t active_row_idx = group_idx / groups_per_row;
    const int64_t row = static_cast<int64_t>(params.direct_active_rows[active_row_idx]);
    if (row < 0 || row >= params.m) {
      continue;
    }
    const int64_t col_group = group_idx - active_row_idx * groups_per_row;
    const int64_t base_col = col_group * VecN;
    const int32_t start = params.direct_row_offsets[row];
    const int32_t end = params.direct_row_offsets[row + 1];
    if (start >= end) {
      continue;
    }

    float acc[VecN] = {};
    int32_t cur_kk = params.direct_row_ks[start];
    float cur_value = direct_bf16_to_float(params.direct_row_values[start]);
    bool cur_valid = cur_kk >= 0 && cur_kk < params.k;
    uint4 cur_b = make_uint4(0u, 0u, 0u, 0u);
    if (cur_valid) {
      cur_b = __ldg(reinterpret_cast<const uint4*>(
          params.direct_b_comp + static_cast<int64_t>(cur_kk) * params.n + base_col));
    }

    for (int32_t pos = start; pos < end; ++pos) {
      const int32_t next_pos = pos + 1;
      int32_t next_kk = 0;
      float next_value = 0.0f;
      bool next_valid = false;
      uint4 next_b = make_uint4(0u, 0u, 0u, 0u);
      if (next_pos < end) {
        next_kk = params.direct_row_ks[next_pos];
        next_value = direct_bf16_to_float(params.direct_row_values[next_pos]);
        next_valid = next_kk >= 0 && next_kk < params.k;
        if (next_valid) {
          next_b = __ldg(reinterpret_cast<const uint4*>(
              params.direct_b_comp + static_cast<int64_t>(next_kk) * params.n + base_col));
        }
      }

      if (cur_valid) {
        acc[0] = fmaf(cur_value, direct_bf16_bits_to_float(cur_b.x), acc[0]);
        acc[1] = fmaf(cur_value, direct_bf16_bits_hi_to_float(cur_b.x), acc[1]);
        acc[2] = fmaf(cur_value, direct_bf16_bits_to_float(cur_b.y), acc[2]);
        acc[3] = fmaf(cur_value, direct_bf16_bits_hi_to_float(cur_b.y), acc[3]);
        acc[4] = fmaf(cur_value, direct_bf16_bits_to_float(cur_b.z), acc[4]);
        acc[5] = fmaf(cur_value, direct_bf16_bits_hi_to_float(cur_b.z), acc[5]);
        acc[6] = fmaf(cur_value, direct_bf16_bits_to_float(cur_b.w), acc[6]);
        acc[7] = fmaf(cur_value, direct_bf16_bits_hi_to_float(cur_b.w), acc[7]);
      }

      cur_kk = next_kk;
      cur_value = next_value;
      cur_valid = next_valid;
      cur_b = next_b;
    }

    if constexpr (StoreDelta) {
      const int64_t delta_row = CompactDelta ? active_row_idx : row;
      const uint4 packed_delta =
          make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                     direct_pack_bf16x2(acc[2], acc[3]),
                     direct_pack_bf16x2(acc[4], acc[5]),
                     direct_pack_bf16x2(acc[6], acc[7]));
      *reinterpret_cast<uint4*>(
          params.direct_delta_output + delta_row * params.n + base_col) = packed_delta;
    }

    #pragma unroll
    for (int cc = 0; cc < VecN; ++cc) {
      lane_sum += acc[cc];
    }
  }

  if (local_thread_rank == 0) {
    params.direct_probe_sink[static_cast<int64_t>(worker_id) * params.direct_probe_warps] =
        lane_sum;
  }
}

template <bool StoreDelta>
DEVICE void apply_sparse_active_row_warp256_delta_worker(Params const& params,
                                                         int32_t worker_id,
                                                         int32_t worker_count,
                                                         int32_t local_thread_rank) {
  if (params.direct_probe_sink == nullptr ||
      params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_row_values == nullptr ||
      params.direct_active_rows == nullptr ||
      params.direct_active_row_count <= 0 ||
      params.direct_b_comp == nullptr ||
      (StoreDelta && params.direct_delta_output == nullptr) ||
      worker_id < 0 ||
      worker_count <= 0 ||
      local_thread_rank < 0) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int WarpCols = NumThreadsPerWarp * VecN;
  const int64_t n_tiles_per_row =
      (static_cast<int64_t>(params.n) + WarpCols - 1) / WarpCols;
  if (n_tiles_per_row <= 0 || (params.n & 7) != 0) {
    return;
  }

  const int32_t warp_in_worker = local_thread_rank / NumThreadsPerWarp;
  const int32_t lane_idx = local_thread_rank & (NumThreadsPerWarp - 1);
  const int64_t worker_warps =
      static_cast<int64_t>(worker_count) * (NumThreadsPerWarpGroup / NumThreadsPerWarp);
  int64_t task_idx =
      (static_cast<int64_t>(worker_id) * (NumThreadsPerWarpGroup / NumThreadsPerWarp)) +
      warp_in_worker;
  const int64_t total_tasks =
      static_cast<int64_t>(params.direct_active_row_count) * n_tiles_per_row;
  float lane_sum = 0.0f;

  for (; task_idx < total_tasks; task_idx += worker_warps) {
    const int64_t active_row_idx = task_idx / n_tiles_per_row;
    const int64_t n_tile = task_idx - active_row_idx * n_tiles_per_row;
    int32_t row_i32 = 0;
    int32_t start = 0;
    int32_t end = 0;

    if (lane_idx == 0) {
      row_i32 = params.direct_active_rows[active_row_idx];
      if (row_i32 >= 0 && static_cast<int64_t>(row_i32) < params.m) {
        start = params.direct_row_offsets[row_i32];
        end = params.direct_row_offsets[row_i32 + 1];
      }
    }
    row_i32 = __shfl_sync(0xffffffffu, row_i32, 0);
    start = __shfl_sync(0xffffffffu, start, 0);
    end = __shfl_sync(0xffffffffu, end, 0);
    const int64_t row = static_cast<int64_t>(row_i32);
    if (row < 0 || row >= params.m || start >= end) {
      continue;
    }

    const int64_t base_col = n_tile * WarpCols + lane_idx * VecN;
    const bool full_packed_vec = base_col + VecN <= params.n;
    float acc[VecN] = {};

    int32_t cur_kk = 0;
    float cur_value = 0.0f;
    if (lane_idx == 0) {
      cur_kk = params.direct_row_ks[start];
      cur_value = direct_bf16_to_float(params.direct_row_values[start]);
    }
    cur_kk = __shfl_sync(0xffffffffu, cur_kk, 0);
    cur_value = __shfl_sync(0xffffffffu, cur_value, 0);
    bool cur_valid = full_packed_vec && cur_kk >= 0 && cur_kk < params.k;
    uint4 cur_b = make_uint4(0u, 0u, 0u, 0u);
    if (cur_valid) {
      cur_b = __ldg(reinterpret_cast<const uint4*>(
          params.direct_b_comp + static_cast<int64_t>(cur_kk) * params.n + base_col));
    }

    for (int32_t pos = start; pos < end; ++pos) {
      const int32_t next_pos = pos + 1;
      int32_t next_kk = 0;
      float next_value = 0.0f;
      if (lane_idx == 0 && next_pos < end) {
        next_kk = params.direct_row_ks[next_pos];
        next_value = direct_bf16_to_float(params.direct_row_values[next_pos]);
      }
      next_kk = __shfl_sync(0xffffffffu, next_kk, 0);
      next_value = __shfl_sync(0xffffffffu, next_value, 0);
      const bool next_valid =
          full_packed_vec && next_pos < end && next_kk >= 0 && next_kk < params.k;
      uint4 next_b = make_uint4(0u, 0u, 0u, 0u);
      if (next_valid) {
        next_b = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(next_kk) * params.n + base_col));
      }

      if (cur_valid) {
        acc[0] = fmaf(cur_value, direct_bf16_bits_to_float(cur_b.x), acc[0]);
        acc[1] = fmaf(cur_value, direct_bf16_bits_hi_to_float(cur_b.x), acc[1]);
        acc[2] = fmaf(cur_value, direct_bf16_bits_to_float(cur_b.y), acc[2]);
        acc[3] = fmaf(cur_value, direct_bf16_bits_hi_to_float(cur_b.y), acc[3]);
        acc[4] = fmaf(cur_value, direct_bf16_bits_to_float(cur_b.z), acc[4]);
        acc[5] = fmaf(cur_value, direct_bf16_bits_hi_to_float(cur_b.z), acc[5]);
        acc[6] = fmaf(cur_value, direct_bf16_bits_to_float(cur_b.w), acc[6]);
        acc[7] = fmaf(cur_value, direct_bf16_bits_hi_to_float(cur_b.w), acc[7]);
      } else if (base_col < params.n && cur_kk >= 0 && cur_kk < params.k) {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t col = base_col + cc;
          if (col < params.n) {
            acc[cc] = fmaf(
                cur_value,
                direct_bf16_to_float(
                    params.direct_b_comp[static_cast<int64_t>(cur_kk) * params.n + col]),
                acc[cc]);
          }
        }
      }

      cur_kk = next_kk;
      cur_value = next_value;
      cur_valid = next_valid;
      cur_b = next_b;
    }

    if constexpr (StoreDelta) {
      if (full_packed_vec) {
        const uint4 packed_delta =
            make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                       direct_pack_bf16x2(acc[2], acc[3]),
                       direct_pack_bf16x2(acc[4], acc[5]),
                       direct_pack_bf16x2(acc[6], acc[7]));
        *reinterpret_cast<uint4*>(
            params.direct_delta_output + row * params.n + base_col) = packed_delta;
      } else if (base_col < params.n) {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t col = base_col + cc;
          if (col < params.n) {
            params.direct_delta_output[row * params.n + col] = direct_float_to_bf16(acc[cc]);
          }
        }
      }
    }

    #pragma unroll
    for (int cc = 0; cc < VecN; ++cc) {
      lane_sum += acc[cc];
    }
  }

  lane_sum = warp_reduce_sum_f32(lane_sum);
  if (lane_idx == 0) {
    params.direct_probe_sink[static_cast<int64_t>(worker_id) *
                                 params.direct_probe_warps +
                             warp_in_worker] = lane_sum;
  }
}

template<int CtaM, int CtaN>
DEVICE void apply_sparse_rowblock_nblock_delta_tasks(Params const& params,
                                                     uint32_t dense_blk_m,
                                                     uint32_t dense_blk_n,
                                                     uint32_t dense_grid_m,
                                                     uint32_t dense_grid_n,
                                                     int lane_idx,
                                                     int sparse_warp_rank,
                                                     int sparse_warps) {
  if (params.direct_probe_sink == nullptr ||
      params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_row_values == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int RowsPerTask = 8;
  constexpr int VecN = 8;
  constexpr int ColGroups = 16;
  constexpr int TaskN = ColGroups * VecN;
  static_assert(CtaN % TaskN == 0 || CtaN == TaskN,
                "rowblock side path assumes a 128-column sparse N task");

  if (sparse_warps != 4) {
    return;
  }

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  constexpr unsigned HalfMask = 0x0000ffffu;
  const unsigned half_mask = HalfMask << half_base_lane;
  const int local_row = sparse_warp_rank * 2 + half;
  const int64_t dense_cta_count =
      static_cast<int64_t>(dense_grid_m) * static_cast<int64_t>(dense_grid_n);
  const int64_t dense_linear_id =
      static_cast<int64_t>(dense_blk_m) * static_cast<int64_t>(dense_grid_n) +
      static_cast<int64_t>(dense_blk_n);
  const bool use_active_rowblocks =
      (params.direct_probe_mixed_cta == 39 ||
       params.direct_probe_mixed_cta == 43) &&
      params.direct_probe_active_mblocks != nullptr &&
      params.direct_probe_active_mblock_count > 0;
  const int64_t rowblock_count =
      use_active_rowblocks
          ? static_cast<int64_t>(params.direct_probe_active_mblock_count)
          : (static_cast<int64_t>(params.m) + RowsPerTask - 1) / RowsPerTask;
  const int64_t nblock_count = (static_cast<int64_t>(params.n) + TaskN - 1) / TaskN;
  const int64_t total_tasks = rowblock_count * nblock_count;
  float lane_sum = 0.0f;

  for (int64_t task_id = dense_linear_id; task_id < total_tasks; task_id += dense_cta_count) {
    const int64_t rowblock_task = task_id / nblock_count;
    const int64_t rowblock =
        use_active_rowblocks
            ? static_cast<int64_t>(params.direct_probe_active_mblocks[rowblock_task])
            : rowblock_task;
    const int64_t nblock = task_id - rowblock_task * nblock_count;
    const int64_t global_row = rowblock * RowsPerTask + local_row;
    const int64_t global_col0 = nblock * TaskN + half_lane * VecN;
    if (global_row >= params.m) {
      continue;
    }

    int start = 0;
    int end = 0;
    if (half_lane == 0) {
      start = params.direct_row_offsets[global_row];
      end = params.direct_row_offsets[global_row + 1];
    }
    start = __shfl_sync(half_mask, start, half_base_lane);
    end = __shfl_sync(half_mask, end, half_base_lane);
    if (start == end) {
      continue;
    }

    float acc[VecN] = {};
    const bool full_packed_tile = ((params.n & 7) == 0) && (global_col0 + VecN <= params.n);
    if (full_packed_tile) {
      int cur_gk = 0;
      float cur_av = 0.0f;
      if (half_lane == 0) {
        cur_gk = params.direct_row_ks[start];
        cur_av = direct_bf16_to_float(params.direct_row_values[start]);
      }
      cur_gk = __shfl_sync(half_mask, cur_gk, half_base_lane);
      cur_av = __shfl_sync(half_mask, cur_av, half_base_lane);

      bool cur_valid = cur_gk >= 0 && cur_gk < params.k;
      uint4 cur_bv = make_uint4(0u, 0u, 0u, 0u);
      if (cur_valid) {
        cur_bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(cur_gk) * params.n + global_col0));
      }

      for (int entry_idx = start; entry_idx < end; ++entry_idx) {
        int next_gk = 0;
        float next_av = 0.0f;
        const int next_idx = entry_idx + 1;
        if (half_lane == 0) {
          next_gk = next_idx < end ? params.direct_row_ks[next_idx] : 0;
          next_av =
              next_idx < end ? direct_bf16_to_float(params.direct_row_values[next_idx]) : 0.0f;
        }
        next_gk = __shfl_sync(half_mask, next_gk, half_base_lane);
        next_av = __shfl_sync(half_mask, next_av, half_base_lane);

        const bool next_valid = next_gk >= 0 && next_gk < params.k && next_idx < end;
        uint4 next_bv = make_uint4(0u, 0u, 0u, 0u);
        if (next_valid) {
          next_bv = __ldg(reinterpret_cast<const uint4*>(
              params.direct_b_comp + static_cast<int64_t>(next_gk) * params.n + global_col0));
        }

        if (cur_valid) {
          acc[0] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.x), acc[0]);
          acc[1] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.x), acc[1]);
          acc[2] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.y), acc[2]);
          acc[3] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.y), acc[3]);
          acc[4] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.z), acc[4]);
          acc[5] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.z), acc[5]);
          acc[6] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.w), acc[6]);
          acc[7] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.w), acc[7]);
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
          gk = params.direct_row_ks[entry_idx];
          av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
        }
        gk = __shfl_sync(half_mask, gk, half_base_lane);
        av = __shfl_sync(half_mask, av, half_base_lane);
        if (gk >= 0 && gk < params.k) {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < params.n) {
              acc[cc] = fmaf(
                  av,
                  direct_bf16_to_float(
                      params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                  acc[cc]);
            }
          }
        }
      }
    }

    if (params.direct_delta_output != nullptr) {
      c10::BFloat16* delta_output = params.direct_delta_output;
      if (full_packed_tile) {
        const uint4 packed_delta =
            make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                       direct_pack_bf16x2(acc[2], acc[3]),
                       direct_pack_bf16x2(acc[4], acc[5]),
                       direct_pack_bf16x2(acc[6], acc[7]));
        *reinterpret_cast<uint4*>(
            delta_output + global_row * params.n + global_col0) = packed_delta;
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            delta_output[global_row * params.n + global_col] = direct_float_to_bf16(acc[cc]);
          }
        }
      }
    }

    #pragma unroll
    for (int cc = 0; cc < VecN; ++cc) {
      lane_sum += acc[cc];
    }
  }

  lane_sum = warp_reduce_sum_f32(lane_sum);
  if (lane_idx == 0) {
    const int64_t sink_idx = dense_linear_id * sparse_warps + sparse_warp_rank;
    params.direct_probe_sink[sink_idx] = lane_sum;
  }
}

template<int CtaM, int CtaN>
DEVICE void apply_sparse_packed_rowblock_nblock_delta_tasks(Params const& params,
                                                            uint32_t dense_blk_m,
                                                            uint32_t dense_blk_n,
                                                            uint32_t dense_grid_m,
                                                            uint32_t dense_grid_n,
                                                            int lane_idx,
                                                            int sparse_warp_rank,
                                                            int sparse_warps) {
  if (params.direct_probe_sink == nullptr ||
      params.direct_probe_active_mblocks == nullptr ||
      params.direct_probe_active_mblock_count <= 0 ||
      params.direct_packed_tile_offsets == nullptr ||
      params.direct_packed_row_records == nullptr ||
      params.direct_packed_entry_records == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = 16;
  constexpr int TaskN = ColGroups * VecN;
  if (sparse_warps != 4) {
    return;
  }

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  constexpr unsigned HalfMask = 0x0000ffffu;
  const unsigned half_mask = HalfMask << half_base_lane;
  const int record_slot = sparse_warp_rank * 2 + half;
  const int64_t dense_cta_count =
      static_cast<int64_t>(dense_grid_m) * static_cast<int64_t>(dense_grid_n);
  const int64_t dense_linear_id =
      static_cast<int64_t>(dense_blk_m) * static_cast<int64_t>(dense_grid_n) +
      static_cast<int64_t>(dense_blk_n);
  const int64_t rowblock_count =
      static_cast<int64_t>(params.direct_probe_active_mblock_count);
  const int64_t nblock_count = (static_cast<int64_t>(params.n) + TaskN - 1) / TaskN;
  const int64_t rowblock_nblock_tasks = rowblock_count * nblock_count;
  const int64_t total_tasks = rowblock_nblock_tasks * 4;
  float lane_sum = 0.0f;

  for (int64_t task_id = dense_linear_id; task_id < total_tasks; task_id += dense_cta_count) {
    const int64_t rowblock_task = task_id / nblock_count;
    const int64_t rowblock =
        static_cast<int64_t>(params.direct_probe_active_mblocks[rowblock_task]);
    const int64_t nblock = task_id - rowblock_task * nblock_count;
    const int row_record_start = params.direct_packed_tile_offsets[rowblock_task];
    const int row_record_end = params.direct_packed_tile_offsets[rowblock_task + 1];
    const int record_idx = row_record_start + record_slot;
    if (record_idx >= row_record_end) {
      continue;
    }

    const uint64_t row_record =
        static_cast<uint64_t>(params.direct_packed_row_records[record_idx]);
    const int local_row = static_cast<int>(row_record & 0xffffull);
    const int row_nnz = static_cast<int>((row_record >> 16) & 0xffffull);
    const int start = static_cast<int>((row_record >> 32) & 0xffffffffull);
    if (row_nnz <= 0 || local_row < 0 || local_row >= 8) {
      continue;
    }

    const int64_t global_row = rowblock * 8 + local_row;
    const int64_t global_col0 = nblock * TaskN + half_lane * VecN;
    if (global_row >= params.m) {
      continue;
    }

    float acc[VecN] = {};
    const bool full_packed_tile = ((params.n & 7) == 0) && (global_col0 + VecN <= params.n);
    if (full_packed_tile) {
      for (int entry_idx = start; entry_idx < start + row_nnz; ++entry_idx) {
        const uint32_t packed_entry =
            static_cast<uint32_t>(params.direct_packed_entry_records[entry_idx]);
        const int gk = static_cast<int>((packed_entry >> 16) & 0xffffu);
        const float av = direct_bf16_bits_to_float(packed_entry);
        if (gk < 0 || gk >= params.k) {
          continue;
        }
        const uint4 bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
        acc[0] = fmaf(av, direct_bf16_bits_to_float(bv.x), acc[0]);
        acc[1] = fmaf(av, direct_bf16_bits_hi_to_float(bv.x), acc[1]);
        acc[2] = fmaf(av, direct_bf16_bits_to_float(bv.y), acc[2]);
        acc[3] = fmaf(av, direct_bf16_bits_hi_to_float(bv.y), acc[3]);
        acc[4] = fmaf(av, direct_bf16_bits_to_float(bv.z), acc[4]);
        acc[5] = fmaf(av, direct_bf16_bits_hi_to_float(bv.z), acc[5]);
        acc[6] = fmaf(av, direct_bf16_bits_to_float(bv.w), acc[6]);
        acc[7] = fmaf(av, direct_bf16_bits_hi_to_float(bv.w), acc[7]);
      }
    } else {
      for (int entry_idx = start; entry_idx < start + row_nnz; ++entry_idx) {
        const uint32_t packed_entry =
            static_cast<uint32_t>(params.direct_packed_entry_records[entry_idx]);
        const int gk = static_cast<int>((packed_entry >> 16) & 0xffffu);
        const float av = direct_bf16_bits_to_float(packed_entry);
        if (gk < 0 || gk >= params.k) {
          continue;
        }
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            acc[cc] = fmaf(
                av,
                direct_bf16_to_float(
                    params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                acc[cc]);
          }
        }
      }
    }

    if (params.direct_delta_output != nullptr) {
      c10::BFloat16* delta_output = params.direct_delta_output;
      if (full_packed_tile) {
        const uint4 packed_delta =
            make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                       direct_pack_bf16x2(acc[2], acc[3]),
                       direct_pack_bf16x2(acc[4], acc[5]),
                       direct_pack_bf16x2(acc[6], acc[7]));
        *reinterpret_cast<uint4*>(
            delta_output + global_row * params.n + global_col0) = packed_delta;
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            delta_output[global_row * params.n + global_col] = direct_float_to_bf16(acc[cc]);
          }
        }
      }
    }

    #pragma unroll
    for (int cc = 0; cc < VecN; ++cc) {
      lane_sum += acc[cc];
    }
  }

  lane_sum = warp_reduce_sum_f32(lane_sum);
  if (lane_idx == 0) {
    const int64_t sink_idx = dense_linear_id * sparse_warps + sparse_warp_rank;
    params.direct_probe_sink[sink_idx] = lane_sum;
  }
}

template<int CtaM, int CtaN, int LocalN>
DEVICE void apply_sparse_loadfma_local_delta_tile(Params const& params,
                                                 uint16_t* local_delta,
                                                 float* local_partials,
                                                 uint32_t blk_m,
                                                 uint32_t blk_n,
                                                 int active_m_idx,
                                                 int epi_st_n_idx,
                                                 int lane_idx,
                                                 int probe_rank,
                                                 int probe_count) {
  if (local_delta == nullptr ||
      local_partials == nullptr ||
      params.direct_row_offsets == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = LocalN / VecN;
  static_assert(LocalN % VecN == 0, "local delta producer expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  constexpr unsigned ValidHalfMask =
      ColGroups >= 16 ? 0x0000ffffu : ((1u << ColGroups) - 1u);
  const unsigned half_mask = ValidHalfMask << half_base_lane;
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_count =
      params.direct_active_row_offsets[blk_m + 1] - active_start;
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;

  if constexpr (LocalN == 64) {
    constexpr int HeavyRowThreshold = HANDWRITTEN_TMA_HEAVY_ROW_THRESHOLD;
    constexpr int SplitK = HANDWRITTEN_TMA_LOCAL_DELTA_SPLIT_K;
    constexpr int SkipRowNnzGe = HANDWRITTEN_TMA_LOCAL_DELTA_SKIP_ROW_NNZ_GE;
    constexpr int AdaptiveRowNnzGe = HANDWRITTEN_TMA_LOCAL_DELTA_ADAPTIVE_ROW_NNZ_GE;
    constexpr int AdaptiveMinBlockNnz =
        HANDWRITTEN_TMA_LOCAL_DELTA_ADAPTIVE_MIN_BLOCK_NNZ;
	    const bool use_hot_row_schedule =
	        (HANDWRITTEN_TMA_LOCAL_DELTA_USE_HOT_ROW_SCHEDULE != 0) ||
	        params.direct_probe_mixed_cta == 22 ||
	        params.direct_probe_mixed_cta == 23 ||
	        params.direct_probe_mixed_cta == 24 ||
	        params.direct_probe_mixed_cta == 25 ||
	        params.direct_probe_mixed_cta == 26 ||
	        params.direct_probe_mixed_cta == 27 ||
	        params.direct_probe_mixed_cta == 28 ||
	        params.direct_probe_mixed_cta == 29 ||
	        params.direct_probe_mixed_cta == 30 ||
	        params.direct_probe_mixed_cta == 31 ||
	        params.direct_probe_mixed_cta == 33;
    constexpr int SplitCols = ColGroups;
    static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4,
                  "local delta split-k must be 1, 2, or 4");
    static_assert(NumThreadsPerWarp % (SplitK * SplitCols) == 0,
                  "local delta split-k must evenly partition a warp");
    constexpr int LanesPerRow = SplitK * SplitCols;
    constexpr int RowsPerWarp = NumThreadsPerWarp / LanesPerRow;
    constexpr bool EnableHeavyRowSplit =
        HeavyRowThreshold < 1000000000 && SplitK == 4;
    constexpr bool EnableAdaptiveWarpHeavy = AdaptiveRowNnzGe < 1000000000;
    constexpr bool EnableLightVec16 =
        HANDWRITTEN_TMA_LOCAL_DELTA_LIGHT_VEC16 != 0 && SplitK == 1;
    constexpr unsigned warp_mask = 0xffffffffu;
    constexpr unsigned RowGroupMaskBase =
        LanesPerRow >= 32 ? 0xffffffffu : ((1u << LanesPerRow) - 1u);
    const int row_group = lane_idx / LanesPerRow;
    const int lane_in_row_group = lane_idx - row_group * LanesPerRow;
    const int row_base_lane = row_group * LanesPerRow;
    const unsigned row_mask = RowGroupMaskBase << row_base_lane;
    const int col_group = lane_in_row_group % SplitCols;
    const int split_rank = lane_in_row_group / SplitCols;
    const int local_col0 = col_group * VecN;
    const int64_t global_col0 = n0 + epi_st_n_idx + local_col0;
    const bool full_packed_tile =
        ((params.n & 7) == 0) && (n0 + epi_st_n_idx + LocalN <= params.n);
    const int coop_threads = probe_count * NumThreadsPerWarp;
    const int64_t block_row_end =
        block_m0 + CtaM < params.m ? block_m0 + CtaM : params.m;
    const int block_entry_count =
        block_m0 < params.m
            ? params.direct_row_offsets[block_row_end] - params.direct_row_offsets[block_m0]
            : 0;
    const bool adaptive_block_enabled =
        block_entry_count >= AdaptiveMinBlockNnz;
    int hot_start = 0;
    int hot_end = 0;
    int hot_count = 0;
    int adaptive_row_nnz_threshold = AdaptiveRowNnzGe;
    if (use_hot_row_schedule && active_m_idx >= 0 &&
        params.direct_kmajor_group_offsets != nullptr &&
        params.direct_kmajor_group_ks != nullptr) {
      hot_start = params.direct_kmajor_group_offsets[active_m_idx];
      hot_end = params.direct_kmajor_group_offsets[active_m_idx + 1];
      hot_count = hot_end - hot_start;
      if (hot_count > 0) {
        adaptive_row_nnz_threshold = params.direct_kmajor_group_ks[hot_start];
      }
    }
    const bool adaptive_heavy_enabled =
        adaptive_block_enabled &&
        (!use_hot_row_schedule || (active_m_idx >= 0 && hot_count > 0));
		    const bool use_payload_adaptive_threshold =
		        adaptive_row_nnz_threshold != AdaptiveRowNnzGe;
			    bool use_hot_colsplit_schedule = false;
			    bool use_superhot_colsplit_schedule = false;
		    bool use_superhot_atomic_schedule = false;
		    const bool use_heavy_pipeline = params.direct_probe_mixed_cta == 30;
		    constexpr int SuperHotColSplitNnzGe =
		        HANDWRITTEN_TMA_LOCAL_DELTA_SUPERHOT_COLSPLIT_NNZ_GE;
		    constexpr int SuperHotAtomicNnzGe =
		        HANDWRITTEN_TMA_LOCAL_DELTA_SUPERHOT_ATOMIC_NNZ_GE;
		    constexpr int SuperHotAtomicMaxRows = 4;
		    if constexpr (SuperHotColSplitNnzGe < 1000000000) {
		      use_superhot_colsplit_schedule =
		          params.direct_probe_mixed_cta == 28 &&
		          use_hot_row_schedule &&
		          probe_count == 4 &&
		          active_m_idx >= 0 &&
		          hot_count > 0 &&
		          params.direct_kmajor_entry_rows != nullptr;
		      if (use_superhot_colsplit_schedule) {
		        constexpr int HotVecN = 8;
		        constexpr int HotColsPerWarp = 16;
		        constexpr int HotColGroupsPerWarp = HotColsPerWarp / HotVecN;
		        constexpr int HotSplitK = NumThreadsPerWarp / HotColGroupsPerWarp;
		        static_assert(HotColsPerWarp * 4 == LocalN,
		                      "super-hot col-split path expects 4 side warps over LocalN");
		        static_assert(HotColGroupsPerWarp == 2,
		                      "super-hot col-split path expects two vec8 col groups per warp");
		        const int hot_col_group = lane_idx & (HotColGroupsPerWarp - 1);
		        const int hot_split_rank = lane_idx >> 1;
		        const int hot_local_col0 =
		            probe_rank * HotColsPerWarp + hot_col_group * HotVecN;
		        const int64_t hot_global_col0 = n0 + epi_st_n_idx + hot_local_col0;
		        const unsigned hot_reduce_mask =
		            hot_col_group == 0 ? 0x55555555u : 0xaaaaaaaau;

		        for (int hot_item = 0; hot_item < hot_count; ++hot_item) {
		          const int local_row =
		              params.direct_kmajor_entry_rows[hot_start + hot_item];
		          const int64_t global_row = block_m0 + local_row;
		          int start = 0;
		          int end = 0;
		          if (lane_idx == 0 && local_row >= 0 && local_row < CtaM &&
		              global_row >= 0 && global_row < params.m) {
		            start = params.direct_row_offsets[global_row];
		            end = params.direct_row_offsets[global_row + 1];
		          }
		          if (local_row < 0 || local_row >= CtaM ||
		              global_row < 0 || global_row >= params.m) {
		            continue;
		          }
		          start = __shfl_sync(warp_mask, start, 0);
		          end = __shfl_sync(warp_mask, end, 0);
		          if (end - start < SuperHotColSplitNnzGe) {
		            continue;
		          }

		          float acc[HotVecN] = {};
		          if (full_packed_tile) {
		            for (int entry_idx = start + hot_split_rank;
		                 entry_idx < end;
		                 entry_idx += HotSplitK) {
		              const int gk = params.direct_row_ks[entry_idx];
		              if (gk < 0 || gk >= params.k) {
		                continue;
		              }
		              const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
		              const uint4 bv = __ldg(reinterpret_cast<const uint4*>(
		                  params.direct_b_comp + static_cast<int64_t>(gk) * params.n +
		                  hot_global_col0));
		              acc[0] = fmaf(av, direct_bf16_bits_to_float(bv.x), acc[0]);
		              acc[1] = fmaf(av, direct_bf16_bits_hi_to_float(bv.x), acc[1]);
		              acc[2] = fmaf(av, direct_bf16_bits_to_float(bv.y), acc[2]);
		              acc[3] = fmaf(av, direct_bf16_bits_hi_to_float(bv.y), acc[3]);
		              acc[4] = fmaf(av, direct_bf16_bits_to_float(bv.z), acc[4]);
		              acc[5] = fmaf(av, direct_bf16_bits_hi_to_float(bv.z), acc[5]);
		              acc[6] = fmaf(av, direct_bf16_bits_to_float(bv.w), acc[6]);
		              acc[7] = fmaf(av, direct_bf16_bits_hi_to_float(bv.w), acc[7]);
		            }
		          } else {
		            for (int entry_idx = start + hot_split_rank;
		                 entry_idx < end;
		                 entry_idx += HotSplitK) {
		              const int gk = params.direct_row_ks[entry_idx];
		              if (gk < 0 || gk >= params.k) {
		                continue;
		              }
		              const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
		              #pragma unroll
		              for (int cc = 0; cc < HotVecN; ++cc) {
		                const int64_t global_col = hot_global_col0 + cc;
		                if (global_col < params.n) {
		                  acc[cc] = fmaf(
		                      av,
		                      direct_bf16_to_float(
		                          params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
		                      acc[cc]);
		                }
		              }
		            }
		          }

		          #pragma unroll
		          for (int cc = 0; cc < HotVecN; ++cc) {
		            float sum = acc[cc];
		            sum += __shfl_down_sync(hot_reduce_mask, sum, 2);
		            sum += __shfl_down_sync(hot_reduce_mask, sum, 4);
		            sum += __shfl_down_sync(hot_reduce_mask, sum, 8);
		            sum += __shfl_down_sync(hot_reduce_mask, sum, 16);
		            acc[cc] = sum;
		          }
		          if (hot_split_rank == 0) {
		            if (full_packed_tile) {
		              const uint4 packed_delta =
		                  make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
		                             direct_pack_bf16x2(acc[2], acc[3]),
		                             direct_pack_bf16x2(acc[4], acc[5]),
		                             direct_pack_bf16x2(acc[6], acc[7]));
		              *reinterpret_cast<uint4*>(
		                  local_delta + static_cast<int64_t>(local_row) * LocalN + hot_local_col0) =
		                  packed_delta;
		            } else {
		              #pragma unroll
		              for (int cc = 0; cc < HotVecN; ++cc) {
		                const int64_t global_col = hot_global_col0 + cc;
		                if (global_col < params.n) {
		                  local_delta[static_cast<int64_t>(local_row) * LocalN + hot_local_col0 + cc] =
		                      direct_float_to_bf16_bits_u16(acc[cc]);
		                }
		              }
		            }
		          }
		        }
		      }
		    }
		    if constexpr (SuperHotAtomicNnzGe < 1000000000) {
		      use_superhot_atomic_schedule =
		          params.direct_probe_mixed_cta == 29 &&
		          use_hot_row_schedule &&
		          probe_count == 4 &&
		          active_m_idx >= 0 &&
		          hot_count > 0 &&
		          params.direct_kmajor_entry_rows != nullptr;
		      if (use_superhot_atomic_schedule) {
		        constexpr int AtomicSplitK = 4;
		        constexpr int AtomicLanesPerRow = AtomicSplitK * ColGroups;
		        static_assert(AtomicLanesPerRow == NumThreadsPerWarp,
		                      "super-hot atomic path expects one row per warp");
		        static_assert(SuperHotAtomicMaxRows * LocalN <= 4 * LocalN,
		                      "super-hot atomic path exceeds local partial storage");
		        for (int idx = probe_rank * NumThreadsPerWarp + lane_idx;
		             idx < SuperHotAtomicMaxRows * LocalN;
		             idx += coop_threads) {
		          local_partials[idx] = 0.0f;
		        }
		        ptx::bar_sync(6, coop_threads);

		        const int atomic_col_group = lane_idx & (ColGroups - 1);
		        const int atomic_split_rank = lane_idx >> 3;
		        const int atomic_local_col0 = atomic_col_group * VecN;
		        const int64_t atomic_global_col0 =
		            n0 + epi_st_n_idx + atomic_local_col0;
		        const int atomic_split_offset =
		            probe_rank * AtomicSplitK + atomic_split_rank;
		        const int atomic_split_stride = probe_count * AtomicSplitK;

		        int superhot_slot = 0;
		        for (int hot_item = 0; hot_item < hot_count; ++hot_item) {
		          const int local_row =
		              params.direct_kmajor_entry_rows[hot_start + hot_item];
		          const int64_t global_row = block_m0 + local_row;
		          int start = 0;
		          int end = 0;
		          if (lane_idx == 0 && local_row >= 0 && local_row < CtaM &&
		              global_row >= 0 && global_row < params.m) {
		            start = params.direct_row_offsets[global_row];
		            end = params.direct_row_offsets[global_row + 1];
		          }
		          start = __shfl_sync(warp_mask, start, 0);
		          end = __shfl_sync(warp_mask, end, 0);
		          const int row_nnz = end - start;
		          if (local_row < 0 || local_row >= CtaM ||
		              global_row < 0 || global_row >= params.m ||
		              row_nnz < SuperHotAtomicNnzGe) {
		            continue;
		          }
		          const int slot = superhot_slot++;
		          if (slot >= SuperHotAtomicMaxRows) {
		            continue;
		          }

		          float acc[VecN] = {};
		          if (full_packed_tile) {
		            for (int entry_idx = start + atomic_split_offset;
		                 entry_idx < end;
		                 entry_idx += atomic_split_stride) {
		              const int gk = params.direct_row_ks[entry_idx];
		              if (gk < 0 || gk >= params.k) {
		                continue;
		              }
		              const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
		              const uint4 bv = __ldg(reinterpret_cast<const uint4*>(
		                  params.direct_b_comp + static_cast<int64_t>(gk) * params.n +
		                  atomic_global_col0));
		              acc[0] = fmaf(av, direct_bf16_bits_to_float(bv.x), acc[0]);
		              acc[1] = fmaf(av, direct_bf16_bits_hi_to_float(bv.x), acc[1]);
		              acc[2] = fmaf(av, direct_bf16_bits_to_float(bv.y), acc[2]);
		              acc[3] = fmaf(av, direct_bf16_bits_hi_to_float(bv.y), acc[3]);
		              acc[4] = fmaf(av, direct_bf16_bits_to_float(bv.z), acc[4]);
		              acc[5] = fmaf(av, direct_bf16_bits_hi_to_float(bv.z), acc[5]);
		              acc[6] = fmaf(av, direct_bf16_bits_to_float(bv.w), acc[6]);
		              acc[7] = fmaf(av, direct_bf16_bits_hi_to_float(bv.w), acc[7]);
		            }
		          } else {
		            for (int entry_idx = start + atomic_split_offset;
		                 entry_idx < end;
		                 entry_idx += atomic_split_stride) {
		              const int gk = params.direct_row_ks[entry_idx];
		              if (gk < 0 || gk >= params.k) {
		                continue;
		              }
		              const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
		              #pragma unroll
		              for (int cc = 0; cc < VecN; ++cc) {
		                const int64_t global_col = atomic_global_col0 + cc;
		                if (global_col < params.n) {
		                  acc[cc] = fmaf(
		                      av,
		                      direct_bf16_to_float(
		                          params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
		                      acc[cc]);
		                }
		              }
		            }
		          }

		          #pragma unroll
		          for (int cc = 0; cc < VecN; ++cc) {
		            const float part = acc[cc];
		            const float sum =
		                part +
		                __shfl_sync(warp_mask, part, ColGroups + atomic_col_group) +
		                __shfl_sync(warp_mask, part, ColGroups * 2 + atomic_col_group) +
		                __shfl_sync(warp_mask, part, ColGroups * 3 + atomic_col_group);
		            if (atomic_split_rank == 0) {
		              atomicAdd(
		                  local_partials + slot * LocalN + atomic_local_col0 + cc,
		                  sum);
		            }
		          }
		        }

		        ptx::bar_sync(6, coop_threads);
		        superhot_slot = 0;
		        for (int hot_item = 0; hot_item < hot_count; ++hot_item) {
		          const int local_row =
		              params.direct_kmajor_entry_rows[hot_start + hot_item];
		          const int64_t global_row = block_m0 + local_row;
		          int start = 0;
		          int end = 0;
		          if (lane_idx == 0 && local_row >= 0 && local_row < CtaM &&
		              global_row >= 0 && global_row < params.m) {
		            start = params.direct_row_offsets[global_row];
		            end = params.direct_row_offsets[global_row + 1];
		          }
		          start = __shfl_sync(warp_mask, start, 0);
		          end = __shfl_sync(warp_mask, end, 0);
		          const int row_nnz = end - start;
		          if (local_row < 0 || local_row >= CtaM ||
		              global_row < 0 || global_row >= params.m ||
		              row_nnz < SuperHotAtomicNnzGe) {
		            continue;
		          }
		          const int slot = superhot_slot++;
		          if (slot >= SuperHotAtomicMaxRows) {
		            continue;
		          }
		          for (int idx = probe_rank * NumThreadsPerWarp + lane_idx;
		               idx < LocalN;
		               idx += coop_threads) {
		            const int64_t global_col = n0 + epi_st_n_idx + idx;
		            if (global_col < params.n) {
		              local_delta[static_cast<int64_t>(local_row) * LocalN + idx] =
		                  direct_float_to_bf16_bits_u16(
		                      local_partials[slot * LocalN + idx]);
		            }
		          }
		        }
		        ptx::bar_sync(6, coop_threads);
		      }
		    }
		    constexpr bool EnableHotColSplit =
		        HANDWRITTEN_TMA_LOCAL_DELTA_HOT_COLSPLIT != 0;
	    if constexpr (EnableHotColSplit) {
	      use_hot_colsplit_schedule =
	          params.direct_probe_mixed_cta == 25 &&
	          use_hot_row_schedule &&
	          probe_count == 4 &&
	          active_m_idx >= 0 &&
	          hot_count > 0 &&
	          params.direct_kmajor_entry_rows != nullptr;
	    if (use_hot_colsplit_schedule) {
	      constexpr int HotVecN = 8;
	      constexpr int HotColsPerWarp = 16;
	      constexpr int HotColGroupsPerWarp = HotColsPerWarp / HotVecN;
	      constexpr int HotSplitK = NumThreadsPerWarp / HotColGroupsPerWarp;
	      static_assert(HotColsPerWarp * 4 == LocalN,
	                    "hot col-split path expects 4 side warps over LocalN");
	      static_assert(HotColGroupsPerWarp == 2,
	                    "hot col-split path expects two vec8 col groups per warp");
	      const int hot_col_group = lane_idx & (HotColGroupsPerWarp - 1);
	      const int hot_split_rank = lane_idx >> 1;
	      const int hot_local_col0 =
	          probe_rank * HotColsPerWarp + hot_col_group * HotVecN;
	      const int64_t hot_global_col0 = n0 + epi_st_n_idx + hot_local_col0;
	      const unsigned hot_reduce_mask =
	          hot_col_group == 0 ? 0x55555555u : 0xaaaaaaaau;

	      for (int hot_item = 0; hot_item < hot_count; ++hot_item) {
	        const int local_row =
	            params.direct_kmajor_entry_rows[hot_start + hot_item];
	        const int64_t global_row = block_m0 + local_row;
	        int start = 0;
	        int end = 0;
	        if (lane_idx == 0 && local_row >= 0 && local_row < CtaM &&
	            global_row >= 0 && global_row < params.m) {
	          start = params.direct_row_offsets[global_row];
	          end = params.direct_row_offsets[global_row + 1];
	        }
	        if (local_row < 0 || local_row >= CtaM ||
	            global_row < 0 || global_row >= params.m) {
	          continue;
	        }
	        start = __shfl_sync(warp_mask, start, 0);
	        end = __shfl_sync(warp_mask, end, 0);

	        float acc[HotVecN] = {};
	        if (full_packed_tile) {
	          for (int entry_idx = start + hot_split_rank;
	               entry_idx < end;
	               entry_idx += HotSplitK) {
	            const int gk = params.direct_row_ks[entry_idx];
	            if (gk < 0 || gk >= params.k) {
	              continue;
	            }
	            const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
	            const uint4 bv = __ldg(reinterpret_cast<const uint4*>(
	                params.direct_b_comp + static_cast<int64_t>(gk) * params.n +
	                hot_global_col0));
	            acc[0] = fmaf(av, direct_bf16_bits_to_float(bv.x), acc[0]);
	            acc[1] = fmaf(av, direct_bf16_bits_hi_to_float(bv.x), acc[1]);
	            acc[2] = fmaf(av, direct_bf16_bits_to_float(bv.y), acc[2]);
	            acc[3] = fmaf(av, direct_bf16_bits_hi_to_float(bv.y), acc[3]);
	            acc[4] = fmaf(av, direct_bf16_bits_to_float(bv.z), acc[4]);
	            acc[5] = fmaf(av, direct_bf16_bits_hi_to_float(bv.z), acc[5]);
	            acc[6] = fmaf(av, direct_bf16_bits_to_float(bv.w), acc[6]);
	            acc[7] = fmaf(av, direct_bf16_bits_hi_to_float(bv.w), acc[7]);
	          }
	        } else {
	          for (int entry_idx = start + hot_split_rank;
	               entry_idx < end;
	               entry_idx += HotSplitK) {
	            const int gk = params.direct_row_ks[entry_idx];
	            if (gk < 0 || gk >= params.k) {
	              continue;
	            }
	            const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
	            #pragma unroll
	            for (int cc = 0; cc < HotVecN; ++cc) {
	              const int64_t global_col = hot_global_col0 + cc;
	              if (global_col < params.n) {
	                acc[cc] = fmaf(
	                    av,
	                    direct_bf16_to_float(
	                        params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
	                    acc[cc]);
	              }
	            }
	          }
	        }

	        #pragma unroll
	        for (int cc = 0; cc < HotVecN; ++cc) {
	          float sum = acc[cc];
	          sum += __shfl_down_sync(hot_reduce_mask, sum, 2);
	          sum += __shfl_down_sync(hot_reduce_mask, sum, 4);
	          sum += __shfl_down_sync(hot_reduce_mask, sum, 8);
	          sum += __shfl_down_sync(hot_reduce_mask, sum, 16);
	          acc[cc] = sum;
	        }
	        if (hot_split_rank == 0) {
	          if (full_packed_tile) {
	            const uint4 packed_delta =
	                make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
	                           direct_pack_bf16x2(acc[2], acc[3]),
	                           direct_pack_bf16x2(acc[4], acc[5]),
	                           direct_pack_bf16x2(acc[6], acc[7]));
	            *reinterpret_cast<uint4*>(
	                local_delta + static_cast<int64_t>(local_row) * LocalN + hot_local_col0) =
	                packed_delta;
	          } else {
	            #pragma unroll
	            for (int cc = 0; cc < HotVecN; ++cc) {
	              const int64_t global_col = hot_global_col0 + cc;
	              if (global_col < params.n) {
	                local_delta[static_cast<int64_t>(local_row) * LocalN + hot_local_col0 + cc] =
	                    direct_float_to_bf16_bits_u16(acc[cc]);
	              }
	            }
	          }
	        }
	      }
	    }
	    }

	    if constexpr (EnableHeavyRowSplit) {
    if (probe_count > 1) {
      for (int active_item = 0; active_item < active_count; ++active_item) {
        const int local_row = params.direct_active_rows[active_start + active_item];
        const int64_t global_row = block_m0 + local_row;
        int start = 0;
        int end = 0;
        if (lane_idx == 0 && local_row >= 0 && local_row < CtaM &&
            global_row >= 0 && global_row < params.m) {
          start = params.direct_row_offsets[global_row];
          end = params.direct_row_offsets[global_row + 1];
        }
        if (local_row < 0 || local_row >= CtaM ||
            global_row < 0 || global_row >= params.m) {
          continue;
        }
        start = __shfl_sync(warp_mask, start, 0);
        end = __shfl_sync(warp_mask, end, 0);
        if constexpr (SkipRowNnzGe < 1000000000) {
          if (end - start >= SkipRowNnzGe) {
            continue;
          }
        }
        if (end - start < HeavyRowThreshold) {
          continue;
        }

        float acc[VecN] = {};
        const int split_offset = probe_rank * SplitK + split_rank;
        const int split_stride = probe_count * SplitK;
        if (full_packed_tile) {
          for (int entry_idx = start + split_offset; entry_idx < end;
               entry_idx += split_stride) {
            const int gk = params.direct_row_ks[entry_idx];
            if (gk < 0 || gk >= params.k) {
              continue;
            }
            const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
            const uint4 bv = __ldg(reinterpret_cast<const uint4*>(
                params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
            acc[0] = fmaf(av, direct_bf16_bits_to_float(bv.x), acc[0]);
            acc[1] = fmaf(av, direct_bf16_bits_hi_to_float(bv.x), acc[1]);
            acc[2] = fmaf(av, direct_bf16_bits_to_float(bv.y), acc[2]);
            acc[3] = fmaf(av, direct_bf16_bits_hi_to_float(bv.y), acc[3]);
            acc[4] = fmaf(av, direct_bf16_bits_to_float(bv.z), acc[4]);
            acc[5] = fmaf(av, direct_bf16_bits_hi_to_float(bv.z), acc[5]);
            acc[6] = fmaf(av, direct_bf16_bits_to_float(bv.w), acc[6]);
            acc[7] = fmaf(av, direct_bf16_bits_hi_to_float(bv.w), acc[7]);
          }
        } else {
          for (int entry_idx = start + split_offset; entry_idx < end;
               entry_idx += split_stride) {
            const int gk = params.direct_row_ks[entry_idx];
            if (gk < 0 || gk >= params.k) {
              continue;
            }
            const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
            #pragma unroll
            for (int cc = 0; cc < VecN; ++cc) {
              const int64_t global_col = global_col0 + cc;
              if (global_col < params.n) {
                acc[cc] = fmaf(
                    av,
                    direct_bf16_to_float(
                        params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                    acc[cc]);
              }
            }
          }
        }

        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const float part = acc[cc];
          acc[cc] = part +
                    __shfl_sync(warp_mask, part, SplitCols + col_group) +
                    __shfl_sync(warp_mask, part, SplitCols * 2 + col_group) +
                    __shfl_sync(warp_mask, part, SplitCols * 3 + col_group);
        }
        if (split_rank == 0) {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            local_partials[probe_rank * LocalN + local_col0 + cc] = acc[cc];
          }
        }
        ptx::bar_sync(6, coop_threads);

        if (probe_rank == 0 && split_rank == 0) {
          float total[VecN] = {};
          for (int pr = 0; pr < probe_count; ++pr) {
            #pragma unroll
            for (int cc = 0; cc < VecN; ++cc) {
              total[cc] += local_partials[pr * LocalN + local_col0 + cc];
            }
          }
          if (full_packed_tile) {
            const uint4 packed_delta =
                make_uint4(direct_pack_bf16x2(total[0], total[1]),
                           direct_pack_bf16x2(total[2], total[3]),
                           direct_pack_bf16x2(total[4], total[5]),
                           direct_pack_bf16x2(total[6], total[7]));
            *reinterpret_cast<uint4*>(
                local_delta + static_cast<int64_t>(local_row) * LocalN + local_col0) =
                packed_delta;
          } else {
            #pragma unroll
            for (int cc = 0; cc < VecN; ++cc) {
              const int64_t global_col = global_col0 + cc;
              if (global_col < params.n) {
                local_delta[static_cast<int64_t>(local_row) * LocalN + local_col0 + cc] =
                    direct_float_to_bf16_bits_u16(total[cc]);
              }
            }
          }
        }
        ptx::bar_sync(6, coop_threads);
      }
    }
    }

	    if constexpr (EnableAdaptiveWarpHeavy) {
	      if (adaptive_heavy_enabled && !use_hot_colsplit_schedule) {
      constexpr int HeavySplitK = 4;
      const int heavy_col_group = lane_idx & (SplitCols - 1);
      const int heavy_split_rank = lane_idx >> 3;
      const int heavy_local_col0 = heavy_col_group * VecN;
      const int64_t heavy_global_col0 = n0 + epi_st_n_idx + heavy_local_col0;
      const int heavy_loop_count = use_hot_row_schedule ? hot_count : active_count;
      for (int heavy_item = probe_rank; heavy_item < heavy_loop_count;
           heavy_item += probe_count) {
        const int local_row =
            use_hot_row_schedule
                ? params.direct_kmajor_entry_rows[hot_start + heavy_item]
                : params.direct_active_rows[active_start + heavy_item];
        const int64_t global_row = block_m0 + local_row;
        int start = 0;
        int end = 0;
        if (lane_idx == 0 && local_row >= 0 && local_row < CtaM &&
            global_row >= 0 && global_row < params.m) {
          start = params.direct_row_offsets[global_row];
          end = params.direct_row_offsets[global_row + 1];
        }
        if (local_row < 0 || local_row >= CtaM ||
            global_row < 0 || global_row >= params.m) {
          continue;
        }
        start = __shfl_sync(warp_mask, start, 0);
        end = __shfl_sync(warp_mask, end, 0);
	        if constexpr (SkipRowNnzGe < 1000000000) {
	          if (end - start >= SkipRowNnzGe) {
	            continue;
	          }
	        }
	        if (use_superhot_colsplit_schedule &&
	            end - start >= SuperHotColSplitNnzGe) {
	          continue;
	        }
	        if (use_superhot_atomic_schedule &&
	            end - start >= SuperHotAtomicNnzGe) {
	          int superhot_slot = 0;
	          for (int prev_hot = 0; prev_hot < heavy_item; ++prev_hot) {
	            const int prev_local_row =
	                params.direct_kmajor_entry_rows[hot_start + prev_hot];
	            const int64_t prev_global_row = block_m0 + prev_local_row;
	            if (prev_local_row >= 0 && prev_local_row < CtaM &&
	                prev_global_row >= 0 && prev_global_row < params.m) {
	              const int prev_start = params.direct_row_offsets[prev_global_row];
	              const int prev_end = params.direct_row_offsets[prev_global_row + 1];
	              if (prev_end - prev_start >= SuperHotAtomicNnzGe) {
	                ++superhot_slot;
	              }
	            }
	          }
	          if (superhot_slot < SuperHotAtomicMaxRows) {
	            continue;
	          }
	        }
	        if (use_payload_adaptive_threshold) {
	          if (end - start < adaptive_row_nnz_threshold) {
	            continue;
          }
        } else {
          if (end - start < AdaptiveRowNnzGe) {
            continue;
          }
        }

	        float acc[VecN] = {};
	        if (full_packed_tile) {
	          if (use_heavy_pipeline && start + heavy_split_rank < end) {
	            int cur_gk = params.direct_row_ks[start + heavy_split_rank];
	            float cur_av =
	                direct_bf16_to_float(params.direct_row_values[start + heavy_split_rank]);
	            bool cur_valid = cur_gk >= 0 && cur_gk < params.k;
	            uint4 cur_bv = make_uint4(0u, 0u, 0u, 0u);
	            if (cur_valid) {
	              cur_bv = __ldg(reinterpret_cast<const uint4*>(
	                  params.direct_b_comp + static_cast<int64_t>(cur_gk) * params.n +
	                  heavy_global_col0));
	            }
	            for (int entry_idx = start + heavy_split_rank; entry_idx < end;
	                 entry_idx += HeavySplitK) {
	              const int next_idx = entry_idx + HeavySplitK;
	              const bool next_in = next_idx < end;
	              const int next_gk = next_in ? params.direct_row_ks[next_idx] : 0;
	              const float next_av =
	                  next_in ? direct_bf16_to_float(params.direct_row_values[next_idx]) : 0.0f;
	              const bool next_valid = next_in && next_gk >= 0 && next_gk < params.k;
	              uint4 next_bv = make_uint4(0u, 0u, 0u, 0u);
	              if (next_valid) {
	                next_bv = __ldg(reinterpret_cast<const uint4*>(
	                    params.direct_b_comp + static_cast<int64_t>(next_gk) * params.n +
	                    heavy_global_col0));
	              }
	              if (cur_valid) {
	                acc[0] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.x), acc[0]);
	                acc[1] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.x), acc[1]);
	                acc[2] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.y), acc[2]);
	                acc[3] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.y), acc[3]);
	                acc[4] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.z), acc[4]);
	                acc[5] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.z), acc[5]);
	                acc[6] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.w), acc[6]);
	                acc[7] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.w), acc[7]);
	              }
	              cur_gk = next_gk;
	              cur_av = next_av;
	              cur_valid = next_valid;
	              cur_bv = next_bv;
	            }
	          } else {
	            for (int entry_idx = start + heavy_split_rank; entry_idx < end;
	                 entry_idx += HeavySplitK) {
	              const int gk = params.direct_row_ks[entry_idx];
	              if (gk < 0 || gk >= params.k) {
	                continue;
	              }
	              const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
	              const uint4 bv = __ldg(reinterpret_cast<const uint4*>(
	                  params.direct_b_comp + static_cast<int64_t>(gk) * params.n + heavy_global_col0));
	              acc[0] = fmaf(av, direct_bf16_bits_to_float(bv.x), acc[0]);
	              acc[1] = fmaf(av, direct_bf16_bits_hi_to_float(bv.x), acc[1]);
	              acc[2] = fmaf(av, direct_bf16_bits_to_float(bv.y), acc[2]);
	              acc[3] = fmaf(av, direct_bf16_bits_hi_to_float(bv.y), acc[3]);
	              acc[4] = fmaf(av, direct_bf16_bits_to_float(bv.z), acc[4]);
	              acc[5] = fmaf(av, direct_bf16_bits_hi_to_float(bv.z), acc[5]);
	              acc[6] = fmaf(av, direct_bf16_bits_to_float(bv.w), acc[6]);
	              acc[7] = fmaf(av, direct_bf16_bits_hi_to_float(bv.w), acc[7]);
	            }
	          }
        } else {
          for (int entry_idx = start + heavy_split_rank; entry_idx < end;
               entry_idx += HeavySplitK) {
            const int gk = params.direct_row_ks[entry_idx];
            if (gk < 0 || gk >= params.k) {
              continue;
            }
            const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
            #pragma unroll
            for (int cc = 0; cc < VecN; ++cc) {
              const int64_t global_col = heavy_global_col0 + cc;
              if (global_col < params.n) {
                acc[cc] = fmaf(
                    av,
                    direct_bf16_to_float(
                        params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                    acc[cc]);
              }
            }
          }
        }

        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const float part = acc[cc];
          acc[cc] = part +
                    __shfl_sync(warp_mask, part, SplitCols + heavy_col_group) +
                    __shfl_sync(warp_mask, part, SplitCols * 2 + heavy_col_group) +
                    __shfl_sync(warp_mask, part, SplitCols * 3 + heavy_col_group);
        }
        if (heavy_split_rank == 0) {
          if (full_packed_tile) {
            const uint4 packed_delta =
                make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                           direct_pack_bf16x2(acc[2], acc[3]),
                           direct_pack_bf16x2(acc[4], acc[5]),
                           direct_pack_bf16x2(acc[6], acc[7]));
            *reinterpret_cast<uint4*>(
                local_delta + static_cast<int64_t>(local_row) * LocalN + heavy_local_col0) =
                packed_delta;
          } else {
            #pragma unroll
            for (int cc = 0; cc < VecN; ++cc) {
              const int64_t global_col = heavy_global_col0 + cc;
              if (global_col < params.n) {
                local_delta[static_cast<int64_t>(local_row) * LocalN + heavy_local_col0 + cc] =
                    direct_float_to_bf16_bits_u16(acc[cc]);
              }
            }
          }
        }
      }
      }
    }

    if constexpr (EnableLightVec16) {
      constexpr int LightVecN = 16;
      constexpr int LightColGroups = LocalN / LightVecN;
      static_assert(LocalN % LightVecN == 0,
                    "vec16 light local-delta expects divisible LocalN");
      static_assert(NumThreadsPerWarp % LightColGroups == 0,
                    "vec16 light local-delta expects row groups to tile a warp");
      constexpr int LightRowsPerWarp = NumThreadsPerWarp / LightColGroups;
      constexpr unsigned LightRowGroupMaskBase =
          LightColGroups >= 32 ? 0xffffffffu : ((1u << LightColGroups) - 1u);
      const int light_row_group = lane_idx / LightColGroups;
      const int light_lane_in_row = lane_idx - light_row_group * LightColGroups;
      const int light_row_base_lane = light_row_group * LightColGroups;
      const unsigned light_row_mask = LightRowGroupMaskBase << light_row_base_lane;
      const int light_local_col0 = light_lane_in_row * LightVecN;
	      const int64_t light_global_col0 = n0 + epi_st_n_idx + light_local_col0;

	      const bool use_packed_light =
	          (params.direct_packed_payload_mode == 1 ||
	           params.direct_packed_payload_mode == 2 ||
	           params.direct_packed_payload_mode == 3) &&
	          params.direct_packed_tile_offsets != nullptr &&
	          params.direct_packed_row_records != nullptr &&
	          params.direct_packed_entry_records != nullptr;
	      if (use_packed_light) {
	        const bool packed_light_prefiltered =
	            params.direct_packed_payload_mode >= 2;
	        const bool packed_light_strict =
	            params.direct_packed_payload_mode == 3;
	        const int packed_row_start = params.direct_packed_tile_offsets[blk_m];
	        const int packed_row_end = params.direct_packed_tile_offsets[blk_m + 1];
	        const int packed_row_count = packed_row_end - packed_row_start;
	        for (int packed_item = probe_rank * LightRowsPerWarp + light_row_group;
	             packed_item < packed_row_count;
	             packed_item += probe_count * LightRowsPerWarp) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	          uint64_t phase_light_meta_start = 0;
	          if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	            phase_light_meta_start = phase_trace_clock();
	          }
#endif
	          const uint64_t row_record = static_cast<uint64_t>(
	              params.direct_packed_row_records[packed_row_start + packed_item]);
	          const int local_row = static_cast<int>(row_record & 0xffffull);
	          const int row_nnz = static_cast<int>((row_record >> 16) & 0xffffull);
	          const int start = static_cast<int>((row_record >> 32) & 0xffffffffull);
	          const int end = start + row_nnz;
	          if (local_row < 0 || local_row >= CtaM || row_nnz <= 0) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	            if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	              phase_trace_add(
	                  params,
	                  PhaseTraceSparseLightMeta,
	                  phase_trace_clock() - phase_light_meta_start);
	            }
#endif
	            continue;
	          }
	          if constexpr (SkipRowNnzGe < 1000000000) {
	            if (!packed_light_prefiltered && row_nnz >= SkipRowNnzGe) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	              if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                phase_trace_add(
	                    params,
	                    PhaseTraceSparseLightMeta,
	                    phase_trace_clock() - phase_light_meta_start);
	              }
#endif
	              continue;
	            }
	          }
	          if constexpr (EnableHeavyRowSplit) {
	            if (!packed_light_prefiltered &&
	                probe_count > 1 && row_nnz >= HeavyRowThreshold) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	              if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                phase_trace_add(
	                    params,
	                    PhaseTraceSparseLightMeta,
	                    phase_trace_clock() - phase_light_meta_start);
	              }
#endif
	              continue;
	            }
	          }
	          if constexpr (EnableAdaptiveWarpHeavy) {
	            if (!packed_light_prefiltered && adaptive_heavy_enabled) {
	              if (use_payload_adaptive_threshold) {
	                if (row_nnz >= adaptive_row_nnz_threshold) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	                  if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                    phase_trace_add(
	                        params,
	                        PhaseTraceSparseLightMeta,
	                        phase_trace_clock() - phase_light_meta_start);
	                  }
#endif
	                  continue;
	                }
	              } else {
	                if (row_nnz >= AdaptiveRowNnzGe) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	                  if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                    phase_trace_add(
	                        params,
	                        PhaseTraceSparseLightMeta,
	                        phase_trace_clock() - phase_light_meta_start);
	                  }
#endif
	                  continue;
	                }
	              }
	            }
	          }

#if HANDWRITTEN_TMA_PHASE_TRACE
	          if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	            const uint64_t phase_after_row_meta = phase_trace_clock();
	            phase_trace_add(
	                params,
	                PhaseTraceSparseLightMeta,
	                phase_after_row_meta - phase_light_meta_start);
	            phase_trace_add(
	                params,
	                PhaseTraceSparseLightRowMeta,
	                phase_after_row_meta - phase_light_meta_start);
	            phase_trace_add(params, PhaseTraceSparseLightRows, 1);
	          }
#endif
	          float acc[LightVecN] = {};
	          if (full_packed_tile) {
	            for (int entry_idx = start; entry_idx < end; ++entry_idx) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	              uint64_t phase_entry_start = 0;
	              if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                phase_entry_start = phase_trace_clock();
	              }
#endif
	              const uint32_t packed_entry = static_cast<uint32_t>(
	                  params.direct_packed_entry_records[entry_idx]);
	              const int gk = static_cast<int>((packed_entry >> 16) & 0xffffu);
	              const float av = direct_bf16_bits_to_float(packed_entry);
	              if (!packed_light_strict && (gk < 0 || gk >= params.k)) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	                if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                  const uint64_t phase_after_entry_meta = phase_trace_clock();
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSparseLightMeta,
	                      phase_after_entry_meta - phase_entry_start);
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSparseLightEntryMeta,
	                      phase_after_entry_meta - phase_entry_start);
	                }
#endif
	                continue;
	              }
#if HANDWRITTEN_TMA_PHASE_TRACE
	              if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                if (!packed_light_strict) {
	                  const uint64_t phase_after_meta = phase_trace_clock();
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSparseLightMeta,
	                      phase_after_meta - phase_entry_start);
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSparseLightEntryMeta,
	                      phase_after_meta - phase_entry_start);
	                  phase_entry_start = phase_after_meta;
	                }
	                phase_trace_add(params, PhaseTraceSparseLightEntries, 1);
	              }
#endif
	              const uint4 bv0 = __ldg(reinterpret_cast<const uint4*>(
	                  params.direct_b_comp + static_cast<int64_t>(gk) * params.n +
	                  light_global_col0));
	              const uint4 bv1 = __ldg(reinterpret_cast<const uint4*>(
	                  params.direct_b_comp + static_cast<int64_t>(gk) * params.n +
	                  light_global_col0 + 8));
#if HANDWRITTEN_TMA_PHASE_TRACE
	              if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                const uint64_t phase_after_b_load = phase_trace_clock();
	                phase_trace_add(
	                    params,
	                    PhaseTraceSparseLightBLoad,
	                    phase_after_b_load - phase_entry_start);
	                phase_entry_start = phase_after_b_load;
	              }
#endif
	              acc[0] = fmaf(av, direct_bf16_bits_to_float(bv0.x), acc[0]);
	              acc[1] = fmaf(av, direct_bf16_bits_hi_to_float(bv0.x), acc[1]);
	              acc[2] = fmaf(av, direct_bf16_bits_to_float(bv0.y), acc[2]);
	              acc[3] = fmaf(av, direct_bf16_bits_hi_to_float(bv0.y), acc[3]);
	              acc[4] = fmaf(av, direct_bf16_bits_to_float(bv0.z), acc[4]);
	              acc[5] = fmaf(av, direct_bf16_bits_hi_to_float(bv0.z), acc[5]);
	              acc[6] = fmaf(av, direct_bf16_bits_to_float(bv0.w), acc[6]);
	              acc[7] = fmaf(av, direct_bf16_bits_hi_to_float(bv0.w), acc[7]);
	              acc[8] = fmaf(av, direct_bf16_bits_to_float(bv1.x), acc[8]);
	              acc[9] = fmaf(av, direct_bf16_bits_hi_to_float(bv1.x), acc[9]);
	              acc[10] = fmaf(av, direct_bf16_bits_to_float(bv1.y), acc[10]);
	              acc[11] = fmaf(av, direct_bf16_bits_hi_to_float(bv1.y), acc[11]);
	              acc[12] = fmaf(av, direct_bf16_bits_to_float(bv1.z), acc[12]);
	              acc[13] = fmaf(av, direct_bf16_bits_hi_to_float(bv1.z), acc[13]);
	              acc[14] = fmaf(av, direct_bf16_bits_to_float(bv1.w), acc[14]);
	              acc[15] = fmaf(av, direct_bf16_bits_hi_to_float(bv1.w), acc[15]);
#if HANDWRITTEN_TMA_PHASE_TRACE
	              if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                phase_trace_add(
	                    params,
	                    PhaseTraceSparseLightFma,
	                    phase_trace_clock() - phase_entry_start);
	              }
#endif
	            }
	          } else {
	            for (int entry_idx = start; entry_idx < end; ++entry_idx) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	              uint64_t phase_entry_start = 0;
	              if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                phase_entry_start = phase_trace_clock();
	              }
#endif
	              const uint32_t packed_entry = static_cast<uint32_t>(
	                  params.direct_packed_entry_records[entry_idx]);
	              const int gk = static_cast<int>((packed_entry >> 16) & 0xffffu);
	              const float av = direct_bf16_bits_to_float(packed_entry);
	              if (!packed_light_strict && (gk < 0 || gk >= params.k)) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	                if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                  const uint64_t phase_after_entry_meta = phase_trace_clock();
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSparseLightMeta,
	                      phase_after_entry_meta - phase_entry_start);
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSparseLightEntryMeta,
	                      phase_after_entry_meta - phase_entry_start);
	                }
#endif
	                continue;
	              }
#if HANDWRITTEN_TMA_PHASE_TRACE
	              if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                if (!packed_light_strict) {
	                  const uint64_t phase_after_meta = phase_trace_clock();
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSparseLightMeta,
	                      phase_after_meta - phase_entry_start);
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSparseLightEntryMeta,
	                      phase_after_meta - phase_entry_start);
	                  phase_entry_start = phase_after_meta;
	                }
	                phase_trace_add(params, PhaseTraceSparseLightEntries, 1);
	              }
#endif
	              #pragma unroll
	              for (int cc = 0; cc < LightVecN; ++cc) {
	                const int64_t global_col = light_global_col0 + cc;
	                if (global_col < params.n) {
	                  acc[cc] = fmaf(
	                      av,
	                      direct_bf16_to_float(
	                          params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
	                      acc[cc]);
	                }
	              }
#if HANDWRITTEN_TMA_PHASE_TRACE
	              if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                phase_trace_add(
	                    params,
	                    PhaseTraceSparseLightFma,
	                    phase_trace_clock() - phase_entry_start);
	              }
#endif
	            }
	          }

#if HANDWRITTEN_TMA_PHASE_TRACE
	          uint64_t phase_light_store_start = 0;
	          if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	            phase_light_store_start = phase_trace_clock();
	          }
#endif
	          if (full_packed_tile) {
	            const uint4 packed0 =
	                make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
	                           direct_pack_bf16x2(acc[2], acc[3]),
	                           direct_pack_bf16x2(acc[4], acc[5]),
	                           direct_pack_bf16x2(acc[6], acc[7]));
	            const uint4 packed1 =
	                make_uint4(direct_pack_bf16x2(acc[8], acc[9]),
	                           direct_pack_bf16x2(acc[10], acc[11]),
	                           direct_pack_bf16x2(acc[12], acc[13]),
	                           direct_pack_bf16x2(acc[14], acc[15]));
	            *reinterpret_cast<uint4*>(
	                local_delta + static_cast<int64_t>(local_row) * LocalN + light_local_col0) =
	                packed0;
	            *reinterpret_cast<uint4*>(
	                local_delta + static_cast<int64_t>(local_row) * LocalN + light_local_col0 + 8) =
	                packed1;
	          } else {
	            #pragma unroll
	            for (int cc = 0; cc < LightVecN; ++cc) {
	              const int64_t global_col = light_global_col0 + cc;
	              if (global_col < params.n) {
	                local_delta[static_cast<int64_t>(local_row) * LocalN + light_local_col0 + cc] =
	                    direct_float_to_bf16_bits_u16(acc[cc]);
	              }
	            }
	          }
#if HANDWRITTEN_TMA_PHASE_TRACE
	          if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	            phase_trace_add(
	                params,
	                PhaseTraceSparseLightStore,
	                phase_trace_clock() - phase_light_store_start);
	          }
#endif
	        }
	        return;
	      }
	
	      for (int active_item = probe_rank * LightRowsPerWarp + light_row_group;
	           active_item < active_count;
	           active_item += probe_count * LightRowsPerWarp) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	        uint64_t phase_light_meta_start = 0;
	        if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	          phase_light_meta_start = phase_trace_clock();
	        }
#endif
	        const int local_row = params.direct_active_rows[active_start + active_item];
	        const int64_t global_row = block_m0 + local_row;
	        int start = 0;
        int end = 0;
        if (light_lane_in_row == 0 && local_row >= 0 && local_row < CtaM &&
            global_row >= 0 && global_row < params.m) {
          start = params.direct_row_offsets[global_row];
          end = params.direct_row_offsets[global_row + 1];
        }
        if (local_row < 0 || local_row >= CtaM ||
            global_row < 0 || global_row >= params.m) {
          continue;
        }
        start = __shfl_sync(light_row_mask, start, light_row_base_lane);
        end = __shfl_sync(light_row_mask, end, light_row_base_lane);
        if constexpr (SkipRowNnzGe < 1000000000) {
          if (end - start >= SkipRowNnzGe) {
            continue;
          }
        }
        if constexpr (EnableHeavyRowSplit) {
          if (probe_count > 1 && end - start >= HeavyRowThreshold) {
            continue;
          }
        }
        if constexpr (EnableAdaptiveWarpHeavy) {
          if (adaptive_heavy_enabled) {
            if (use_payload_adaptive_threshold) {
              if (end - start >= adaptive_row_nnz_threshold) {
                continue;
              }
            } else {
              if (end - start >= AdaptiveRowNnzGe) {
                continue;
	              }
	            }
	          }
	        }
	
#if HANDWRITTEN_TMA_PHASE_TRACE
	        if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	          const uint64_t phase_after_row_meta = phase_trace_clock();
	          phase_trace_add(
	              params,
	              PhaseTraceSparseLightMeta,
	              phase_after_row_meta - phase_light_meta_start);
	          phase_trace_add(
	              params,
	              PhaseTraceSparseLightRowMeta,
	              phase_after_row_meta - phase_light_meta_start);
	          phase_trace_add(params, PhaseTraceSparseLightRows, 1);
	        }
#endif
	        float acc[LightVecN] = {};
	        if (full_packed_tile) {
	          for (int entry_idx = start; entry_idx < end; ++entry_idx) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	            uint64_t phase_entry_start = 0;
	            if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	              phase_entry_start = phase_trace_clock();
	            }
#endif
	            const int gk = params.direct_row_ks[entry_idx];
	            if (gk < 0 || gk >= params.k) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	              if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	                const uint64_t phase_after_entry_meta = phase_trace_clock();
	                phase_trace_add(
	                    params,
	                    PhaseTraceSparseLightMeta,
	                    phase_after_entry_meta - phase_entry_start);
	                phase_trace_add(
	                    params,
	                    PhaseTraceSparseLightEntryMeta,
	                    phase_after_entry_meta - phase_entry_start);
	              }
#endif
	              continue;
	            }
	            const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
#if HANDWRITTEN_TMA_PHASE_TRACE
	            if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	              const uint64_t phase_after_meta = phase_trace_clock();
	              phase_trace_add(
	                  params,
	                  PhaseTraceSparseLightMeta,
	                  phase_after_meta - phase_entry_start);
	              phase_trace_add(
	                  params,
	                  PhaseTraceSparseLightEntryMeta,
	                  phase_after_meta - phase_entry_start);
	              phase_trace_add(params, PhaseTraceSparseLightEntries, 1);
	              phase_entry_start = phase_after_meta;
	            }
#endif
	            const uint4 bv0 = __ldg(reinterpret_cast<const uint4*>(
	                params.direct_b_comp + static_cast<int64_t>(gk) * params.n +
	                light_global_col0));
	            const uint4 bv1 = __ldg(reinterpret_cast<const uint4*>(
	                params.direct_b_comp + static_cast<int64_t>(gk) * params.n +
	                light_global_col0 + 8));
#if HANDWRITTEN_TMA_PHASE_TRACE
	            if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	              const uint64_t phase_after_b_load = phase_trace_clock();
	              phase_trace_add(
	                  params,
	                  PhaseTraceSparseLightBLoad,
	                  phase_after_b_load - phase_entry_start);
	              phase_entry_start = phase_after_b_load;
	            }
#endif
	            acc[0] = fmaf(av, direct_bf16_bits_to_float(bv0.x), acc[0]);
	            acc[1] = fmaf(av, direct_bf16_bits_hi_to_float(bv0.x), acc[1]);
	            acc[2] = fmaf(av, direct_bf16_bits_to_float(bv0.y), acc[2]);
            acc[3] = fmaf(av, direct_bf16_bits_hi_to_float(bv0.y), acc[3]);
            acc[4] = fmaf(av, direct_bf16_bits_to_float(bv0.z), acc[4]);
            acc[5] = fmaf(av, direct_bf16_bits_hi_to_float(bv0.z), acc[5]);
            acc[6] = fmaf(av, direct_bf16_bits_to_float(bv0.w), acc[6]);
            acc[7] = fmaf(av, direct_bf16_bits_hi_to_float(bv0.w), acc[7]);
            acc[8] = fmaf(av, direct_bf16_bits_to_float(bv1.x), acc[8]);
            acc[9] = fmaf(av, direct_bf16_bits_hi_to_float(bv1.x), acc[9]);
            acc[10] = fmaf(av, direct_bf16_bits_to_float(bv1.y), acc[10]);
            acc[11] = fmaf(av, direct_bf16_bits_hi_to_float(bv1.y), acc[11]);
            acc[12] = fmaf(av, direct_bf16_bits_to_float(bv1.z), acc[12]);
	            acc[13] = fmaf(av, direct_bf16_bits_hi_to_float(bv1.z), acc[13]);
	            acc[14] = fmaf(av, direct_bf16_bits_to_float(bv1.w), acc[14]);
	            acc[15] = fmaf(av, direct_bf16_bits_hi_to_float(bv1.w), acc[15]);
#if HANDWRITTEN_TMA_PHASE_TRACE
	            if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	              phase_trace_add(
	                  params,
	                  PhaseTraceSparseLightFma,
	                  phase_trace_clock() - phase_entry_start);
	            }
#endif
	          }
	        } else {
#if HANDWRITTEN_TMA_PHASE_TRACE
	          uint64_t phase_light_fma_start = 0;
	          if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	            phase_light_fma_start = phase_trace_clock();
	          }
#endif
	          for (int entry_idx = start; entry_idx < end; ++entry_idx) {
	            const int gk = params.direct_row_ks[entry_idx];
	            if (gk < 0 || gk >= params.k) {
              continue;
            }
            const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
            #pragma unroll
            for (int cc = 0; cc < LightVecN; ++cc) {
              const int64_t global_col = light_global_col0 + cc;
              if (global_col < params.n) {
                acc[cc] = fmaf(
                    av,
                    direct_bf16_to_float(
                        params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                    acc[cc]);
	              }
	            }
	          }
#if HANDWRITTEN_TMA_PHASE_TRACE
	          if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	            phase_trace_add(
	                params,
	                PhaseTraceSparseLightFma,
	                phase_trace_clock() - phase_light_fma_start);
	          }
#endif
	        }
	
#if HANDWRITTEN_TMA_PHASE_TRACE
	        uint64_t phase_light_store_start = 0;
	        if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	          phase_light_store_start = phase_trace_clock();
	        }
#endif
	        if (full_packed_tile) {
	          const uint4 packed0 =
	              make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                         direct_pack_bf16x2(acc[2], acc[3]),
                         direct_pack_bf16x2(acc[4], acc[5]),
                         direct_pack_bf16x2(acc[6], acc[7]));
          const uint4 packed1 =
              make_uint4(direct_pack_bf16x2(acc[8], acc[9]),
                         direct_pack_bf16x2(acc[10], acc[11]),
                         direct_pack_bf16x2(acc[12], acc[13]),
                         direct_pack_bf16x2(acc[14], acc[15]));
          *reinterpret_cast<uint4*>(
              local_delta + static_cast<int64_t>(local_row) * LocalN + light_local_col0) =
              packed0;
          *reinterpret_cast<uint4*>(
              local_delta + static_cast<int64_t>(local_row) * LocalN + light_local_col0 + 8) =
              packed1;
        } else {
          #pragma unroll
          for (int cc = 0; cc < LightVecN; ++cc) {
            const int64_t global_col = light_global_col0 + cc;
            if (global_col < params.n) {
	              local_delta[static_cast<int64_t>(local_row) * LocalN + light_local_col0 + cc] =
	                  direct_float_to_bf16_bits_u16(acc[cc]);
	            }
	          }
	        }
#if HANDWRITTEN_TMA_PHASE_TRACE
	        if (phase_trace_sparse_lane(params, probe_rank, lane_idx)) {
	          phase_trace_add(
	              params,
	              PhaseTraceSparseLightStore,
	              phase_trace_clock() - phase_light_store_start);
	        }
#endif
	      }
      return;
    }

    for (int active_item = probe_rank * RowsPerWarp + row_group;
         active_item < active_count;
         active_item += probe_count * RowsPerWarp) {
      const int local_row = params.direct_active_rows[active_start + active_item];
      const int64_t global_row = block_m0 + local_row;
      int start = 0;
      int end = 0;
      if (lane_in_row_group == 0 && local_row >= 0 && local_row < CtaM &&
          global_row >= 0 && global_row < params.m) {
        start = params.direct_row_offsets[global_row];
        end = params.direct_row_offsets[global_row + 1];
      }
      if (local_row < 0 || local_row >= CtaM ||
          global_row < 0 || global_row >= params.m) {
        continue;
      }
      start = __shfl_sync(row_mask, start, row_base_lane);
      end = __shfl_sync(row_mask, end, row_base_lane);
      if constexpr (SkipRowNnzGe < 1000000000) {
        if (end - start >= SkipRowNnzGe) {
          continue;
        }
      }
      if constexpr (EnableHeavyRowSplit) {
        if (probe_count > 1 && end - start >= HeavyRowThreshold) {
          continue;
        }
      }
      if constexpr (EnableAdaptiveWarpHeavy) {
        if (adaptive_heavy_enabled) {
          if (use_payload_adaptive_threshold) {
            if (end - start >= adaptive_row_nnz_threshold) {
              continue;
            }
          } else {
            if (end - start >= AdaptiveRowNnzGe) {
              continue;
            }
          }
        }
      }

      float acc[VecN] = {};
      if (full_packed_tile) {
        for (int entry_idx = start + split_rank; entry_idx < end; entry_idx += SplitK) {
          const int gk = params.direct_row_ks[entry_idx];
          if (gk < 0 || gk >= params.k) {
            continue;
          }
          const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
          const uint4 bv = __ldg(reinterpret_cast<const uint4*>(
              params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
          acc[0] = fmaf(av, direct_bf16_bits_to_float(bv.x), acc[0]);
          acc[1] = fmaf(av, direct_bf16_bits_hi_to_float(bv.x), acc[1]);
          acc[2] = fmaf(av, direct_bf16_bits_to_float(bv.y), acc[2]);
          acc[3] = fmaf(av, direct_bf16_bits_hi_to_float(bv.y), acc[3]);
          acc[4] = fmaf(av, direct_bf16_bits_to_float(bv.z), acc[4]);
          acc[5] = fmaf(av, direct_bf16_bits_hi_to_float(bv.z), acc[5]);
          acc[6] = fmaf(av, direct_bf16_bits_to_float(bv.w), acc[6]);
          acc[7] = fmaf(av, direct_bf16_bits_hi_to_float(bv.w), acc[7]);
        }
      } else {
        for (int entry_idx = start + split_rank; entry_idx < end; entry_idx += SplitK) {
          const int gk = params.direct_row_ks[entry_idx];
          if (gk < 0 || gk >= params.k) {
            continue;
          }
          const float av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < params.n) {
              acc[cc] = fmaf(
                  av,
                  direct_bf16_to_float(
                      params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                  acc[cc]);
            }
          }
        }
      }

      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const float part = acc[cc];
        float sum = part;
        if constexpr (SplitK >= 2) {
          sum += __shfl_sync(row_mask, part, row_base_lane + SplitCols + col_group);
        }
        if constexpr (SplitK >= 4) {
          sum += __shfl_sync(row_mask, part, row_base_lane + SplitCols * 2 + col_group);
          sum += __shfl_sync(row_mask, part, row_base_lane + SplitCols * 3 + col_group);
        }
        acc[cc] = sum;
      }
      if (split_rank == 0) {
        if (full_packed_tile) {
          const uint4 packed_delta =
              make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                         direct_pack_bf16x2(acc[2], acc[3]),
                         direct_pack_bf16x2(acc[4], acc[5]),
                         direct_pack_bf16x2(acc[6], acc[7]));
          *reinterpret_cast<uint4*>(
              local_delta + static_cast<int64_t>(local_row) * LocalN + local_col0) =
              packed_delta;
        } else {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < params.n) {
              local_delta[static_cast<int64_t>(local_row) * LocalN + local_col0 + cc] =
                  direct_float_to_bf16_bits_u16(acc[cc]);
            }
          }
        }
      }
    }
    return;
  }

  const int local_col0 = half_lane * VecN;
  const int64_t global_col0 = n0 + epi_st_n_idx + local_col0;
  const bool lane_valid = half_lane < ColGroups;
  const bool full_packed_tile =
      ((params.n & 7) == 0) && (n0 + epi_st_n_idx + LocalN <= params.n);

  for (int active_item = probe_rank * 2 + half; active_item < active_count;
       active_item += probe_count * 2) {
    const int local_row = params.direct_active_rows[active_start + active_item];
    const int64_t global_row = block_m0 + local_row;
    int start = 0;
    int end = 0;
    if (half_lane == 0 && local_row >= 0 && local_row < CtaM &&
        global_row >= 0 && global_row < params.m) {
      start = params.direct_row_offsets[global_row];
      end = params.direct_row_offsets[global_row + 1];
    }
    if (!lane_valid || local_row < 0 || local_row >= CtaM ||
        global_row < 0 || global_row >= params.m) {
      continue;
    }
    start = __shfl_sync(half_mask, start, half_base_lane);
    end = __shfl_sync(half_mask, end, half_base_lane);

    float acc[VecN] = {};
    if (full_packed_tile) {
      int cur_gk = 0;
      float cur_av = 0.0f;
      if (half_lane == 0 && start < end) {
        cur_gk = params.direct_row_ks[start];
        cur_av = direct_bf16_to_float(params.direct_row_values[start]);
      }
      cur_gk = __shfl_sync(half_mask, cur_gk, half_base_lane);
      cur_av = __shfl_sync(half_mask, cur_av, half_base_lane);

      bool cur_valid = start < end && cur_gk >= 0 && cur_gk < params.k;
      uint4 cur_bv = make_uint4(0u, 0u, 0u, 0u);
      if (cur_valid) {
        cur_bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(cur_gk) * params.n + global_col0));
      }

      for (int entry_idx = start; entry_idx < end; ++entry_idx) {
        int next_gk = 0;
        float next_av = 0.0f;
        const int next_idx = entry_idx + 1;
        if (half_lane == 0) {
          next_gk = next_idx < end ? params.direct_row_ks[next_idx] : 0;
          next_av =
              next_idx < end ? direct_bf16_to_float(params.direct_row_values[next_idx]) : 0.0f;
        }
        next_gk = __shfl_sync(half_mask, next_gk, half_base_lane);
        next_av = __shfl_sync(half_mask, next_av, half_base_lane);

        const bool next_valid = next_idx < end && next_gk >= 0 && next_gk < params.k;
        uint4 next_bv = make_uint4(0u, 0u, 0u, 0u);
        if (next_valid) {
          next_bv = __ldg(reinterpret_cast<const uint4*>(
              params.direct_b_comp + static_cast<int64_t>(next_gk) * params.n + global_col0));
        }

        if (cur_valid) {
          acc[0] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.x), acc[0]);
          acc[1] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.x), acc[1]);
          acc[2] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.y), acc[2]);
          acc[3] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.y), acc[3]);
          acc[4] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.z), acc[4]);
          acc[5] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.z), acc[5]);
          acc[6] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.w), acc[6]);
          acc[7] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.w), acc[7]);
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
          gk = params.direct_row_ks[entry_idx];
          av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
        }
        gk = __shfl_sync(half_mask, gk, half_base_lane);
        av = __shfl_sync(half_mask, av, half_base_lane);
        if (gk >= 0 && gk < params.k) {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < params.n) {
              acc[cc] = fmaf(
                  av,
                  direct_bf16_to_float(
                      params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                  acc[cc]);
            }
          }
        }
      }
    }

    if (full_packed_tile) {
      const uint4 packed_delta =
          make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                     direct_pack_bf16x2(acc[2], acc[3]),
                     direct_pack_bf16x2(acc[4], acc[5]),
                     direct_pack_bf16x2(acc[6], acc[7]));
      *reinterpret_cast<uint4*>(
          local_delta + static_cast<int64_t>(local_row) * LocalN + local_col0) = packed_delta;
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_col0 + cc;
        if (global_col < params.n) {
          local_delta[static_cast<int64_t>(local_row) * LocalN + local_col0 + cc] =
              direct_float_to_bf16_bits_u16(acc[cc]);
        }
      }
    }
  }
}

template<int CtaM, int CtaN, bool WriteDelta = false>
DEVICE void apply_sparse_loadfma_probe_kmajor_tile(Params const& params,
                                                   uint32_t active_m_idx,
                                                   uint32_t blk_m,
                                                   uint32_t blk_n,
                                                   int lane_idx,
                                                   int probe_rank,
                                                   int probe_count,
                                                   int sink_rank_offset) {
  if (params.direct_probe_sink == nullptr ||
      params.direct_kmajor_group_ks == nullptr ||
      params.direct_kmajor_entry_offsets == nullptr ||
      params.direct_kmajor_entry_rows == nullptr ||
      params.direct_kmajor_entry_values == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = CtaN / VecN;
  static_assert(CtaN % VecN == 0, "k-major load+FMA probe expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp k owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  int group_start = 0;
  int group_count = 0;
  direct_kmajor_tile_group_range(
      params, static_cast<int>(active_m_idx), blk_m, group_start, group_count);
  const int group_end = group_start + group_count;
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
  const int64_t global_col0 = n0 + half_lane * VecN;
  const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);
  float lane_sum = 0.0f;

  for (int group_idx = group_start + probe_rank * 2 + half; group_idx < group_end;
       group_idx += probe_count * 2) {
    int gk = 0;
    int entry_start = 0;
    int entry_end = 0;
    if (half_lane == 0) {
      gk = params.direct_kmajor_group_ks[group_idx];
      entry_start = params.direct_kmajor_entry_offsets[group_idx];
      entry_end = params.direct_kmajor_entry_offsets[group_idx + 1];
    }
    gk = __shfl_sync(half_mask, gk, half_base_lane);
    entry_start = __shfl_sync(half_mask, entry_start, half_base_lane);
    entry_end = __shfl_sync(half_mask, entry_end, half_base_lane);
    if (entry_start == entry_end || gk < 0 || gk >= params.k) {
      continue;
    }

    const bool lane_valid = half_lane < ColGroups;
    float bvals[VecN] = {};
    if (full_packed_tile) {
      uint4 bv = make_uint4(0u, 0u, 0u, 0u);
      if (lane_valid) {
        bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
      }
      bvals[0] = direct_bf16_bits_to_float(bv.x);
      bvals[1] = direct_bf16_bits_hi_to_float(bv.x);
      bvals[2] = direct_bf16_bits_to_float(bv.y);
      bvals[3] = direct_bf16_bits_hi_to_float(bv.y);
      bvals[4] = direct_bf16_bits_to_float(bv.z);
      bvals[5] = direct_bf16_bits_hi_to_float(bv.z);
      bvals[6] = direct_bf16_bits_to_float(bv.w);
      bvals[7] = direct_bf16_bits_hi_to_float(bv.w);
    } else if (lane_valid) {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_col0 + cc;
        if (global_col < params.n) {
          bvals[cc] = direct_bf16_to_float(
              params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]);
        }
      }
    }

    for (int entry_idx = entry_start; entry_idx < entry_end; ++entry_idx) {
      int local_row = 0;
      if (half_lane == 0) {
        local_row = params.direct_kmajor_entry_rows[entry_idx];
      }
      local_row = __shfl_sync(half_mask, local_row, half_base_lane);
      const int64_t global_row = static_cast<int64_t>(blk_m) * CtaM + local_row;
      const float av = direct_bf16_to_float(params.direct_kmajor_entry_values[entry_idx]);
      if (lane_valid && global_row < params.m) {
        if constexpr (WriteDelta) {
          const int64_t delta_row =
              params.direct_delta_write_mode == 4 ? static_cast<int64_t>(entry_idx) : global_row;
          if ((params.direct_delta_write_mode == 2 || params.direct_delta_write_mode == 4) &&
              full_packed_tile) {
            float delta[VecN];
            #pragma unroll
            for (int cc = 0; cc < VecN; ++cc) {
              delta[cc] = av * bvals[cc];
              lane_sum += delta[cc];
            }
            const uint4 packed_delta =
                make_uint4(direct_pack_bf16x2(delta[0], delta[1]),
                           direct_pack_bf16x2(delta[2], delta[3]),
                           direct_pack_bf16x2(delta[4], delta[5]),
                           direct_pack_bf16x2(delta[6], delta[7]));
            *reinterpret_cast<uint4*>(
                params.direct_delta_output + delta_row * params.n + global_col0) = packed_delta;
          } else {
            #pragma unroll
            for (int cc = 0; cc < VecN; ++cc) {
              const int64_t global_col = global_col0 + cc;
              if (global_col < params.n) {
                const float delta = av * bvals[cc];
                lane_sum += delta;
                c10::BFloat16* delta_ptr =
                    params.direct_delta_output + delta_row * params.n + global_col;
                if (params.direct_delta_write_mode == 2 || params.direct_delta_write_mode == 4) {
                  *delta_ptr = direct_float_to_bf16(delta);
                } else {
                  direct_atomic_add_bf16(delta_ptr, delta);
                }
              }
            }
          }
        } else {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            lane_sum = fmaf(av, bvals[cc], lane_sum);
          }
        }
      }
    }
  }

  lane_sum = warp_reduce_sum_f32(lane_sum);
  if (lane_idx == 0) {
    const int32_t tile_id =
        static_cast<int32_t>(blk_m * gridDim.x + blk_n);
    volatile float* sink = params.direct_probe_sink;
    sink[tile_id * params.direct_probe_warps + sink_rank_offset + probe_rank] = lane_sum;
  }
}

DEVICE int find_active_mblock_index(Params const& params, uint32_t blk_m) {
  if (params.direct_probe_active_mblocks == nullptr) {
    return static_cast<int>(blk_m);
  }
  int lo = 0;
  int hi = params.direct_probe_active_mblock_count;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    const int32_t mid_blk = params.direct_probe_active_mblocks[mid];
    if (mid_blk < static_cast<int32_t>(blk_m)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < params.direct_probe_active_mblock_count &&
      params.direct_probe_active_mblocks[lo] == static_cast<int32_t>(blk_m)) {
    return lo;
  }
  return -1;
}

template<int CtaM, int CtaN, bool WriteDelta = false>
DEVICE void apply_sparse_loadfma_kmajor_bounded_groups_tile(Params const& params,
                                                            uint32_t active_m_idx,
                                                            uint32_t blk_m,
                                                            uint32_t blk_n,
                                                            int lane_idx,
                                                            int probe_rank,
                                                            int probe_count,
                                                            int sink_rank_offset,
                                                            int stage_idx,
                                                            int max_groups,
                                                            int groups_per_call = 0) {
  if (params.direct_probe_sink == nullptr ||
      params.direct_kmajor_group_offsets == nullptr ||
      params.direct_kmajor_group_ks == nullptr ||
      params.direct_kmajor_entry_offsets == nullptr ||
      params.direct_kmajor_entry_rows == nullptr ||
      params.direct_kmajor_entry_values == nullptr ||
      params.direct_b_comp == nullptr ||
      max_groups <= 0) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = CtaN / VecN;
  static_assert(CtaN % VecN == 0, "bounded k-major expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp k owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  int group_start = 0;
  int group_count = 0;
  direct_kmajor_tile_group_range(
      params, static_cast<int>(active_m_idx), blk_m, group_start, group_count);
  const int group_end = group_start + group_count;
  const bool split_total_groups = stage_idx < 0;
  const int groups_per_rank =
      split_total_groups ? ((max_groups + probe_count - 1) / probe_count) : max_groups;
  const int local_begin =
      split_total_groups ? probe_rank * groups_per_rank
                         : groups_per_call > 0
                               ? (stage_idx * probe_count + probe_rank) * groups_per_call
                         : (stage_idx * probe_count + probe_rank) * max_groups;
  const int local_end =
      split_total_groups ? min(max_groups, local_begin + groups_per_rank)
                         : groups_per_call > 0
                               ? min(max_groups, local_begin + groups_per_call)
                         : local_begin + max_groups;
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
  const int64_t global_col0 = n0 + half_lane * VecN;
  const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);
  float lane_sum = 0.0f;

  for (int local_group = local_begin + half; local_group < local_end; local_group += 2) {
    if (local_group >= group_count) {
      continue;
    }
    const int group_idx = group_start + local_group;
    int gk = 0;
    int entry_start = 0;
    int entry_end = 0;
    if (half_lane == 0) {
      gk = params.direct_kmajor_group_ks[group_idx];
      entry_start = params.direct_kmajor_entry_offsets[group_idx];
      entry_end = params.direct_kmajor_entry_offsets[group_idx + 1];
    }
    gk = __shfl_sync(half_mask, gk, half_base_lane);
    entry_start = __shfl_sync(half_mask, entry_start, half_base_lane);
    entry_end = __shfl_sync(half_mask, entry_end, half_base_lane);
    if (entry_start == entry_end || gk < 0 || gk >= params.k) {
      continue;
    }

    const bool lane_valid = half_lane < ColGroups;
    float bvals[VecN] = {};
    if (full_packed_tile) {
      uint4 bv = make_uint4(0u, 0u, 0u, 0u);
      if (lane_valid) {
        bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
      }
      bvals[0] = direct_bf16_bits_to_float(bv.x);
      bvals[1] = direct_bf16_bits_hi_to_float(bv.x);
      bvals[2] = direct_bf16_bits_to_float(bv.y);
      bvals[3] = direct_bf16_bits_hi_to_float(bv.y);
      bvals[4] = direct_bf16_bits_to_float(bv.z);
      bvals[5] = direct_bf16_bits_hi_to_float(bv.z);
      bvals[6] = direct_bf16_bits_to_float(bv.w);
      bvals[7] = direct_bf16_bits_hi_to_float(bv.w);
    } else if (lane_valid) {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_col0 + cc;
        if (global_col < params.n) {
          bvals[cc] = direct_bf16_to_float(
              params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]);
        }
      }
    }

    for (int entry_idx = entry_start; entry_idx < entry_end; ++entry_idx) {
      int local_row = 0;
      if (half_lane == 0) {
        local_row = params.direct_kmajor_entry_rows[entry_idx];
      }
      local_row = __shfl_sync(half_mask, local_row, half_base_lane);
      const int64_t global_row = static_cast<int64_t>(blk_m) * CtaM + local_row;
      const float av = direct_bf16_to_float(params.direct_kmajor_entry_values[entry_idx]);
      if (lane_valid && global_row < params.m) {
        if constexpr (WriteDelta) {
          const int64_t delta_row =
              params.direct_delta_write_mode == 4 ? static_cast<int64_t>(entry_idx) : global_row;
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < params.n) {
              const float delta = av * bvals[cc];
              lane_sum += delta;
              c10::BFloat16* delta_ptr =
                  params.direct_delta_output + delta_row * params.n + global_col;
              if (params.direct_delta_write_mode == 2 || params.direct_delta_write_mode == 4) {
                *delta_ptr = direct_float_to_bf16(delta);
              } else {
                direct_atomic_add_bf16(delta_ptr, delta);
              }
            }
          }
        } else {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            lane_sum = fmaf(av, bvals[cc], lane_sum);
          }
        }
      }
    }
  }

  lane_sum = warp_reduce_sum_f32(lane_sum);
  if (lane_idx == 0) {
    const int32_t tile_id = static_cast<int32_t>(blk_m * gridDim.x + blk_n);
    volatile float* sink = params.direct_probe_sink;
    sink[tile_id * params.direct_probe_warps + sink_rank_offset + probe_rank] = lane_sum;
  }
}

template<int CtaM, int CtaN>
DEVICE void apply_sparse_loadfma_kmajor_bounded_groups_tile_noprobe(
    Params const& params,
    uint32_t active_m_idx,
    uint32_t blk_m,
    uint32_t blk_n,
    int lane_idx,
    int probe_rank,
    int probe_count,
    int max_groups) {
  if (params.direct_kmajor_group_offsets == nullptr ||
      params.direct_kmajor_group_ks == nullptr ||
      params.direct_kmajor_entry_offsets == nullptr ||
      params.direct_kmajor_entry_rows == nullptr ||
      params.direct_kmajor_entry_values == nullptr ||
      params.direct_b_comp == nullptr ||
      max_groups <= 0) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = CtaN / VecN;
  static_assert(CtaN % VecN == 0, "bounded k-major expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp k owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  const int group_start = params.direct_kmajor_group_offsets[active_m_idx];
  const int group_end = params.direct_kmajor_group_offsets[active_m_idx + 1];
  const int group_count = group_end - group_start;
  const int groups_per_rank = (max_groups + probe_count - 1) / probe_count;
  const int local_begin = probe_rank * groups_per_rank;
  const int local_end = min(max_groups, local_begin + groups_per_rank);
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
  const int64_t global_col0 = n0 + half_lane * VecN;
  const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  for (int local_group = local_begin + half; local_group < local_end; local_group += 2) {
    if (local_group >= group_count) {
      continue;
    }
    const int group_idx = group_start + local_group;
    int gk = 0;
    int entry_start = 0;
    int entry_end = 0;
    if (half_lane == 0) {
      gk = params.direct_kmajor_group_ks[group_idx];
      entry_start = params.direct_kmajor_entry_offsets[group_idx];
      entry_end = params.direct_kmajor_entry_offsets[group_idx + 1];
    }
    gk = __shfl_sync(half_mask, gk, half_base_lane);
    entry_start = __shfl_sync(half_mask, entry_start, half_base_lane);
    entry_end = __shfl_sync(half_mask, entry_end, half_base_lane);
    if (entry_start == entry_end || gk < 0 || gk >= params.k) {
      continue;
    }

    const bool lane_valid = half_lane < ColGroups;
    float bvals[VecN] = {};
    if (full_packed_tile) {
      uint4 bv = make_uint4(0u, 0u, 0u, 0u);
      if (lane_valid) {
        bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
      }
      bvals[0] = direct_bf16_bits_to_float(bv.x);
      bvals[1] = direct_bf16_bits_hi_to_float(bv.x);
      bvals[2] = direct_bf16_bits_to_float(bv.y);
      bvals[3] = direct_bf16_bits_hi_to_float(bv.y);
      bvals[4] = direct_bf16_bits_to_float(bv.z);
      bvals[5] = direct_bf16_bits_hi_to_float(bv.z);
      bvals[6] = direct_bf16_bits_to_float(bv.w);
      bvals[7] = direct_bf16_bits_hi_to_float(bv.w);
    } else if (lane_valid) {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_col0 + cc;
        if (global_col < params.n) {
          bvals[cc] = direct_bf16_to_float(
              params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]);
        }
      }
    }

    for (int entry_idx = entry_start; entry_idx < entry_end; ++entry_idx) {
      int local_row = 0;
      if (half_lane == 0) {
        local_row = params.direct_kmajor_entry_rows[entry_idx];
      }
      local_row = __shfl_sync(half_mask, local_row, half_base_lane);
      const int64_t global_row = static_cast<int64_t>(blk_m) * CtaM + local_row;
      const float av = direct_bf16_to_float(params.direct_kmajor_entry_values[entry_idx]);
      if (lane_valid && global_row < params.m) {
        acc0 = fmaf(av, bvals[0], acc0);
        acc1 = fmaf(av, bvals[1], acc1);
        acc2 = fmaf(av, bvals[2], acc2);
        acc3 = fmaf(av, bvals[3], acc3);
        acc0 = fmaf(av, bvals[4], acc0);
        acc1 = fmaf(av, bvals[5], acc1);
        acc2 = fmaf(av, bvals[6], acc2);
        acc3 = fmaf(av, bvals[7], acc3);
      }
    }
  }

  asm volatile("" :: "f"(acc0), "f"(acc1), "f"(acc2), "f"(acc3) : "memory");
}

template<int CtaM, int CtaN>
DEVICE void apply_sparse_loadfma_kmajor_bounded_groups_tile_write_noprobe(
    Params const& params,
    uint32_t active_m_idx,
    uint32_t blk_m,
    uint32_t blk_n,
    int lane_idx,
    int probe_rank,
    int probe_count,
    int max_groups) {
  if (params.direct_delta_output == nullptr ||
      params.direct_kmajor_group_offsets == nullptr ||
      params.direct_kmajor_group_ks == nullptr ||
      params.direct_kmajor_entry_offsets == nullptr ||
      params.direct_kmajor_entry_rows == nullptr ||
      params.direct_kmajor_entry_values == nullptr ||
      params.direct_b_comp == nullptr ||
      max_groups <= 0) {
    return;
  }
  if (params.direct_delta_chunk_limit > 0 &&
      blk_n >= static_cast<uint32_t>(params.direct_delta_chunk_limit)) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = CtaN / VecN;
  static_assert(CtaN % VecN == 0, "bounded k-major expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp k owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  const int group_start = params.direct_kmajor_group_offsets[active_m_idx];
  const int group_end = params.direct_kmajor_group_offsets[active_m_idx + 1];
  const int group_count = group_end - group_start;
  const int groups_per_rank = (max_groups + probe_count - 1) / probe_count;
  const int local_begin = probe_rank * groups_per_rank;
  const int local_end = min(max_groups, local_begin + groups_per_rank);
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
  const int64_t global_col0 = n0 + half_lane * VecN;
  const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);

  for (int local_group = local_begin + half; local_group < local_end; local_group += 2) {
    if (local_group >= group_count) {
      continue;
    }
    const int group_idx = group_start + local_group;
    int gk = 0;
    int entry_start = 0;
    int entry_end = 0;
    if (half_lane == 0) {
      gk = params.direct_kmajor_group_ks[group_idx];
      entry_start = params.direct_kmajor_entry_offsets[group_idx];
      entry_end = params.direct_kmajor_entry_offsets[group_idx + 1];
    }
    gk = __shfl_sync(half_mask, gk, half_base_lane);
    entry_start = __shfl_sync(half_mask, entry_start, half_base_lane);
    entry_end = __shfl_sync(half_mask, entry_end, half_base_lane);
    if (entry_start == entry_end || gk < 0 || gk >= params.k) {
      continue;
    }

    const bool lane_valid = half_lane < ColGroups;
    float bvals[VecN] = {};
    if (full_packed_tile) {
      uint4 bv = make_uint4(0u, 0u, 0u, 0u);
      if (lane_valid) {
        bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
      }
      bvals[0] = direct_bf16_bits_to_float(bv.x);
      bvals[1] = direct_bf16_bits_hi_to_float(bv.x);
      bvals[2] = direct_bf16_bits_to_float(bv.y);
      bvals[3] = direct_bf16_bits_hi_to_float(bv.y);
      bvals[4] = direct_bf16_bits_to_float(bv.z);
      bvals[5] = direct_bf16_bits_hi_to_float(bv.z);
      bvals[6] = direct_bf16_bits_to_float(bv.w);
      bvals[7] = direct_bf16_bits_hi_to_float(bv.w);
    } else if (lane_valid) {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_col0 + cc;
        if (global_col < params.n) {
          bvals[cc] = direct_bf16_to_float(
              params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]);
        }
      }
    }

    for (int entry_idx = entry_start; entry_idx < entry_end; ++entry_idx) {
      int local_row = 0;
      if (half_lane == 0) {
        local_row = params.direct_kmajor_entry_rows[entry_idx];
      }
      local_row = __shfl_sync(half_mask, local_row, half_base_lane);
      const int64_t global_row = static_cast<int64_t>(blk_m) * CtaM + local_row;
      const float av = direct_bf16_to_float(params.direct_kmajor_entry_values[entry_idx]);
      if (lane_valid && global_row < params.m) {
        const int64_t delta_row =
            params.direct_delta_write_mode == 4 ? static_cast<int64_t>(entry_idx) : global_row;
        if (params.direct_delta_write_mode == 4 && full_packed_tile) {
          const uint4 packed_delta =
              make_uint4(direct_pack_bf16x2(av * bvals[0], av * bvals[1]),
                         direct_pack_bf16x2(av * bvals[2], av * bvals[3]),
                         direct_pack_bf16x2(av * bvals[4], av * bvals[5]),
                         direct_pack_bf16x2(av * bvals[6], av * bvals[7]));
          *reinterpret_cast<uint4*>(
              params.direct_delta_output + delta_row * params.n + global_col0) =
              packed_delta;
        } else {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < params.n) {
              const float delta = av * bvals[cc];
              c10::BFloat16* delta_ptr =
                  params.direct_delta_output + delta_row * params.n + global_col;
              if (params.direct_delta_write_mode == 2 ||
                  params.direct_delta_write_mode == 4) {
                *delta_ptr = direct_float_to_bf16(delta);
              } else {
                direct_atomic_add_bf16(delta_ptr, delta);
              }
            }
          }
        }
      }
    }
  }
}

template<int CtaM, int CtaN, int LocalN>
DEVICE void apply_sparse_loadfma_kmajor_local_delta_tile(
    Params const& params,
    uint16_t* local_delta,
    uint32_t active_m_idx,
    uint32_t blk_m,
    uint32_t blk_n,
    int epi_st_n_idx,
    int lane_idx,
    int probe_rank,
    int probe_count,
    int precomputed_group_start = -1,
    int precomputed_group_count = -1) {
  if (local_delta == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr ||
      (params.direct_kmajor_group_offsets == nullptr &&
       !direct_kmajor_has_tile_group_meta(params)) ||
      params.direct_kmajor_group_ks == nullptr ||
      params.direct_kmajor_entry_offsets == nullptr ||
      params.direct_kmajor_entry_rows == nullptr ||
      params.direct_kmajor_entry_values == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = LocalN / VecN;
  static_assert(LocalN % VecN == 0, "k-major local delta expects vec8 N groups");
  static_assert(ColGroups <= 16, "single half-warp local delta writer supports at most 16 groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const unsigned half_mask = 0x0000ffffu;
  const bool lane_valid = half == 0 && half_lane < ColGroups;
  const int local_col0 = half_lane * VecN;
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
  const int64_t global_col0 = n0 + epi_st_n_idx + local_col0;
  const bool full_packed_tile =
      ((params.n & 7) == 0) && (n0 + epi_st_n_idx + LocalN <= params.n);

  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_end = params.direct_active_row_offsets[blk_m + 1];
  if (lane_valid) {
    for (int active_idx = active_start; active_idx < active_end; ++active_idx) {
      const int local_row = params.direct_active_rows[active_idx];
      if (local_row < 0 || local_row >= CtaM) {
        continue;
      }
      if (probe_count > 1 && (local_row % probe_count) != probe_rank) {
        continue;
      }
      if (full_packed_tile) {
        *reinterpret_cast<uint4*>(
            local_delta + static_cast<int64_t>(local_row) * LocalN + local_col0) =
            make_uint4(0u, 0u, 0u, 0u);
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            local_delta[static_cast<int64_t>(local_row) * LocalN + local_col0 + cc] = 0u;
          }
        }
      }
    }
  }
  __syncwarp(0xffffffffu);
  if (half != 0) {
    return;
  }

  int group_start = precomputed_group_start;
  int group_count = precomputed_group_count;
  if (group_count < 0) {
    direct_kmajor_tile_group_range(
        params, static_cast<int>(active_m_idx), blk_m, group_start, group_count);
  }
  const int group_end = group_start + group_count;
  for (int group_idx = group_start; group_idx < group_end; ++group_idx) {
    int gk = 0;
    int entry_start = 0;
    int entry_end = 0;
    if (half_lane == 0) {
      gk = params.direct_kmajor_group_ks[group_idx];
      entry_start = params.direct_kmajor_entry_offsets[group_idx];
      entry_end = params.direct_kmajor_entry_offsets[group_idx + 1];
    }
    gk = __shfl_sync(half_mask, gk, 0);
    entry_start = __shfl_sync(half_mask, entry_start, 0);
    entry_end = __shfl_sync(half_mask, entry_end, 0);
    if (entry_start == entry_end || gk < 0 || gk >= params.k) {
      continue;
    }

    float bvals[VecN] = {};
    if (full_packed_tile) {
      uint4 bv = make_uint4(0u, 0u, 0u, 0u);
      if (lane_valid) {
        bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
      }
      bvals[0] = direct_bf16_bits_to_float(bv.x);
      bvals[1] = direct_bf16_bits_hi_to_float(bv.x);
      bvals[2] = direct_bf16_bits_to_float(bv.y);
      bvals[3] = direct_bf16_bits_hi_to_float(bv.y);
      bvals[4] = direct_bf16_bits_to_float(bv.z);
      bvals[5] = direct_bf16_bits_hi_to_float(bv.z);
      bvals[6] = direct_bf16_bits_to_float(bv.w);
      bvals[7] = direct_bf16_bits_hi_to_float(bv.w);
    } else if (lane_valid) {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_col0 + cc;
        if (global_col < params.n) {
          bvals[cc] = direct_bf16_to_float(
              params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]);
        }
      }
    }

    for (int entry_idx = entry_start; entry_idx < entry_end; ++entry_idx) {
      int local_row = 0;
      float av = 0.0f;
      if (half_lane == 0) {
        local_row = params.direct_kmajor_entry_rows[entry_idx];
        av = direct_bf16_to_float(params.direct_kmajor_entry_values[entry_idx]);
      }
      local_row = __shfl_sync(half_mask, local_row, 0);
      av = __shfl_sync(half_mask, av, 0);
      const int64_t global_row = static_cast<int64_t>(blk_m) * CtaM + local_row;
      if (!lane_valid || local_row < 0 || local_row >= CtaM ||
          global_row < 0 || global_row >= params.m) {
        continue;
      }
      if (probe_count > 1 && (local_row % probe_count) != probe_rank) {
        continue;
      }
      if (full_packed_tile) {
        uint4 old_delta = *reinterpret_cast<const uint4*>(
            local_delta + static_cast<int64_t>(local_row) * LocalN + local_col0);
        const uint4 new_delta =
            make_uint4(direct_pack_bf16x2(
                           direct_bf16_bits_to_float(old_delta.x) + av * bvals[0],
                           direct_bf16_bits_hi_to_float(old_delta.x) + av * bvals[1]),
                       direct_pack_bf16x2(
                           direct_bf16_bits_to_float(old_delta.y) + av * bvals[2],
                           direct_bf16_bits_hi_to_float(old_delta.y) + av * bvals[3]),
                       direct_pack_bf16x2(
                           direct_bf16_bits_to_float(old_delta.z) + av * bvals[4],
                           direct_bf16_bits_hi_to_float(old_delta.z) + av * bvals[5]),
                       direct_pack_bf16x2(
                           direct_bf16_bits_to_float(old_delta.w) + av * bvals[6],
                           direct_bf16_bits_hi_to_float(old_delta.w) + av * bvals[7]));
        *reinterpret_cast<uint4*>(
            local_delta + static_cast<int64_t>(local_row) * LocalN + local_col0) = new_delta;
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            uint16_t* delta_ptr =
                local_delta + static_cast<int64_t>(local_row) * LocalN + local_col0 + cc;
            const float old_delta =
                direct_bf16_bits_to_float(static_cast<uint32_t>(*delta_ptr));
            *delta_ptr = direct_float_to_bf16_bits_u16(old_delta + av * bvals[cc]);
          }
        }
      }
    }
  }
}

template<int CtaM, int CtaN>
DEVICE void apply_sparse_kmajor_bounded_groups_tile_add_delta_noprobe(
    Params const& params,
    uint32_t active_m_idx,
    uint32_t blk_m,
    uint32_t blk_n,
    int lane_idx,
    int probe_rank,
    int probe_count,
    int max_groups) {
  if (params.direct_delta_output == nullptr ||
      params.direct_kmajor_group_offsets == nullptr ||
      params.direct_kmajor_entry_offsets == nullptr ||
      params.direct_kmajor_entry_rows == nullptr ||
      max_groups <= 0) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = CtaN / VecN;
  static_assert(CtaN % VecN == 0, "bounded k-major expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp k owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  const int group_start = params.direct_kmajor_group_offsets[active_m_idx];
  const int group_end = params.direct_kmajor_group_offsets[active_m_idx + 1];
  const int group_count = group_end - group_start;
  const int groups_per_rank = (max_groups + probe_count - 1) / probe_count;
  const int local_begin = probe_rank * groups_per_rank;
  const int local_end = min(max_groups, local_begin + groups_per_rank);
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
  const int64_t global_col0 = n0 + half_lane * VecN;
  const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);
  c10::BFloat16* output = static_cast<c10::BFloat16*>(params.D);

  for (int local_group = local_begin + half; local_group < local_end; local_group += 2) {
    if (local_group >= group_count) {
      continue;
    }
    const int group_idx = group_start + local_group;
    int entry_start = 0;
    int entry_end = 0;
    if (half_lane == 0) {
      entry_start = params.direct_kmajor_entry_offsets[group_idx];
      entry_end = params.direct_kmajor_entry_offsets[group_idx + 1];
    }
    entry_start = __shfl_sync(half_mask, entry_start, half_base_lane);
    entry_end = __shfl_sync(half_mask, entry_end, half_base_lane);
    if (entry_start == entry_end) {
      continue;
    }

    const bool lane_valid = half_lane < ColGroups;
    for (int entry_idx = entry_start; entry_idx < entry_end; ++entry_idx) {
      int local_row = 0;
      if (half_lane == 0) {
        local_row = params.direct_kmajor_entry_rows[entry_idx];
      }
      local_row = __shfl_sync(half_mask, local_row, half_base_lane);
      const int64_t global_row = static_cast<int64_t>(blk_m) * CtaM + local_row;
      if (!lane_valid || global_row >= params.m) {
        continue;
      }
      const int64_t delta_row =
          params.direct_delta_write_mode == 4 ? static_cast<int64_t>(entry_idx) : global_row;
      if (full_packed_tile) {
        const uint4 delta = *reinterpret_cast<const uint4*>(
            params.direct_delta_output + delta_row * params.n + global_col0);
        direct_atomic_add_bf16(
            output + global_row * params.n + global_col0 + 0,
            direct_bf16_bits_to_float(delta.x));
        direct_atomic_add_bf16(
            output + global_row * params.n + global_col0 + 1,
            direct_bf16_bits_hi_to_float(delta.x));
        direct_atomic_add_bf16(
            output + global_row * params.n + global_col0 + 2,
            direct_bf16_bits_to_float(delta.y));
        direct_atomic_add_bf16(
            output + global_row * params.n + global_col0 + 3,
            direct_bf16_bits_hi_to_float(delta.y));
        direct_atomic_add_bf16(
            output + global_row * params.n + global_col0 + 4,
            direct_bf16_bits_to_float(delta.z));
        direct_atomic_add_bf16(
            output + global_row * params.n + global_col0 + 5,
            direct_bf16_bits_hi_to_float(delta.z));
        direct_atomic_add_bf16(
            output + global_row * params.n + global_col0 + 6,
            direct_bf16_bits_to_float(delta.w));
        direct_atomic_add_bf16(
            output + global_row * params.n + global_col0 + 7,
            direct_bf16_bits_hi_to_float(delta.w));
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            const float delta = direct_bf16_to_float(
                params.direct_delta_output[delta_row * params.n + global_col]);
            direct_atomic_add_bf16(output + global_row * params.n + global_col, delta);
          }
        }
      }
    }
  }
}

template<int CtaM, int CtaN>
DEVICE void apply_sparse_kmajor_bounded_groups_tile_rowmerge_delta_noprobe(
    Params const& params,
    uint32_t active_m_idx,
    uint32_t blk_m,
    uint32_t blk_n,
    int lane_idx,
    int probe_rank,
    int probe_count,
    int max_groups) {
  if (params.direct_delta_output == nullptr ||
      params.direct_kmajor_group_offsets == nullptr ||
      params.direct_kmajor_entry_offsets == nullptr ||
      params.direct_kmajor_entry_rows == nullptr ||
      max_groups <= 0) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = CtaN / VecN;
  static_assert(CtaN % VecN == 0, "bounded k-major expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  const int group_start = params.direct_kmajor_group_offsets[active_m_idx];
  const int group_end = params.direct_kmajor_group_offsets[active_m_idx + 1];
  const int group_count = min(max_groups, group_end - group_start);
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
  const int64_t global_col0 = n0 + half_lane * VecN;
  const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);
  c10::BFloat16* output = static_cast<c10::BFloat16*>(params.D);

  for (int local_row = probe_rank * 2 + half; local_row < CtaM;
       local_row += probe_count * 2) {
    const int64_t global_row = static_cast<int64_t>(blk_m) * CtaM + local_row;
    if (global_row >= params.m || half_lane >= ColGroups) {
      continue;
    }

    float acc[VecN] = {};
    bool has_delta = false;
    for (int local_group = 0; local_group < group_count; ++local_group) {
      const int group_idx = group_start + local_group;
      int entry_start = 0;
      int entry_end = 0;
      if (half_lane == 0) {
        entry_start = params.direct_kmajor_entry_offsets[group_idx];
        entry_end = params.direct_kmajor_entry_offsets[group_idx + 1];
      }
      entry_start = __shfl_sync(half_mask, entry_start, half_base_lane);
      entry_end = __shfl_sync(half_mask, entry_end, half_base_lane);
      for (int entry_idx = entry_start; entry_idx < entry_end; ++entry_idx) {
        int entry_row = -1;
        if (half_lane == 0) {
          entry_row = params.direct_kmajor_entry_rows[entry_idx];
        }
        entry_row = __shfl_sync(half_mask, entry_row, half_base_lane);
        if (entry_row != local_row) {
          continue;
        }
        has_delta = true;
        const int64_t delta_row =
            params.direct_delta_write_mode == 4 ? static_cast<int64_t>(entry_idx) : global_row;
        if (full_packed_tile) {
          const uint4 delta = *reinterpret_cast<const uint4*>(
              params.direct_delta_output + delta_row * params.n + global_col0);
          acc[0] += direct_bf16_bits_to_float(delta.x);
          acc[1] += direct_bf16_bits_hi_to_float(delta.x);
          acc[2] += direct_bf16_bits_to_float(delta.y);
          acc[3] += direct_bf16_bits_hi_to_float(delta.y);
          acc[4] += direct_bf16_bits_to_float(delta.z);
          acc[5] += direct_bf16_bits_hi_to_float(delta.z);
          acc[6] += direct_bf16_bits_to_float(delta.w);
          acc[7] += direct_bf16_bits_hi_to_float(delta.w);
        } else {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < params.n) {
              acc[cc] += direct_bf16_to_float(
                  params.direct_delta_output[delta_row * params.n + global_col]);
            }
          }
        }
      }
    }

    if (!has_delta) {
      continue;
    }
    if (full_packed_tile) {
      const uint4 out = *reinterpret_cast<const uint4*>(
          output + global_row * params.n + global_col0);
      const uint4 delta =
          make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                     direct_pack_bf16x2(acc[2], acc[3]),
                     direct_pack_bf16x2(acc[4], acc[5]),
                     direct_pack_bf16x2(acc[6], acc[7]));
      *reinterpret_cast<uint4*>(output + global_row * params.n + global_col0) =
          direct_bf16_add_packed_u4(out, delta);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_col0 + cc;
        if (global_col < params.n) {
          const int64_t out_idx = global_row * params.n + global_col;
          output[out_idx] =
              direct_float_to_bf16(direct_bf16_to_float(output[out_idx]) + acc[cc]);
        }
      }
    }
  }
}

template<int CtaM, int CtaN>
DEVICE void apply_sparse_kmajor_bounded_groups_tile_meta_merge_delta_noprobe(
    Params const& params,
    uint32_t blk_m,
    uint32_t blk_n,
    int lane_idx,
    int probe_rank,
    int probe_count) {
  if (params.direct_delta_output == nullptr ||
      params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = CtaN / VecN;
  static_assert(CtaN % VecN == 0, "meta merge expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_end = params.direct_active_row_offsets[blk_m + 1];
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
  const int64_t global_col0 = n0 + half_lane * VecN;
  const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);
  c10::BFloat16* output = static_cast<c10::BFloat16*>(params.D);

  for (int active_idx = active_start + probe_rank * 2 + half; active_idx < active_end;
       active_idx += probe_count * 2) {
    int row = 0;
    int start = 0;
    int end = 0;
    if (half_lane == 0) {
      row = params.direct_active_rows[active_idx];
      start = (row >= 0 && row < params.m) ? params.direct_row_offsets[row] : 0;
      end = (row >= 0 && row < params.m) ? params.direct_row_offsets[row + 1] : 0;
    }
    row = __shfl_sync(half_mask, row, half_base_lane);
    start = __shfl_sync(half_mask, start, half_base_lane);
    end = __shfl_sync(half_mask, end, half_base_lane);
    if (row < 0 || row >= params.m || start == end || half_lane >= ColGroups) {
      continue;
    }

    float acc[VecN] = {};
    for (int pos = start; pos < end; ++pos) {
      int entry = 0;
#if 0
      if (half_lane == 0) {
        entry = params.direct_row_ks[pos];
      }
#endif
#if 0
      entry = __shfl_sync(half_mask, entry, half_base_lane);
#endif
      if (full_packed_tile) {
        const uint4 delta = *reinterpret_cast<const uint4*>(
            params.direct_delta_output + static_cast<int64_t>(entry) * params.n + global_col0);
        acc[0] += direct_bf16_bits_to_float(delta.x);
        acc[1] += direct_bf16_bits_hi_to_float(delta.x);
        acc[2] += direct_bf16_bits_to_float(delta.y);
        acc[3] += direct_bf16_bits_hi_to_float(delta.y);
        acc[4] += direct_bf16_bits_to_float(delta.z);
        acc[5] += direct_bf16_bits_hi_to_float(delta.z);
        acc[6] += direct_bf16_bits_to_float(delta.w);
        acc[7] += direct_bf16_bits_hi_to_float(delta.w);
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            acc[cc] += direct_bf16_to_float(
                params.direct_delta_output[static_cast<int64_t>(entry) * params.n + global_col]);
          }
        }
      }
    }

    if (full_packed_tile) {
      const uint4 out = *reinterpret_cast<const uint4*>(
          output + static_cast<int64_t>(row) * params.n + global_col0);
      const uint4 delta =
          make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                     direct_pack_bf16x2(acc[2], acc[3]),
                     direct_pack_bf16x2(acc[4], acc[5]),
                     direct_pack_bf16x2(acc[6], acc[7]));
      *reinterpret_cast<uint4*>(output + static_cast<int64_t>(row) * params.n + global_col0) =
          direct_bf16_add_packed_u4(out, delta);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_col0 + cc;
        if (global_col < params.n) {
          const int64_t out_idx = static_cast<int64_t>(row) * params.n + global_col;
          output[out_idx] =
              direct_float_to_bf16(direct_bf16_to_float(output[out_idx]) + acc[cc]);
        }
      }
    }
  }
}

template<int CtaM, int CtaN, int EpiN>
DEVICE void apply_sparse_entry_delta_smem_merge_noprobe(
    Params const& params,
    uint16_t* sC,
    uint32_t blk_m,
    uint32_t blk_n,
    int epi_st_n_idx,
    int widx,
    int lane_idx) {
  if (params.direct_delta_output == nullptr ||
      params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = EpiN / VecN;
  static_assert(EpiN % VecN == 0, "smem merge expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_end = params.direct_active_row_offsets[blk_m + 1];
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t global_n0 =
      static_cast<int64_t>(blk_n) * CtaN + epi_st_n_idx + half_lane * VecN;
  const int local_col0 = half_lane * VecN;
  const bool full_packed_tile = ((params.n & 7) == 0) && (global_n0 + VecN <= params.n);
  const bool lane_valid = half_lane < ColGroups;
  auto smem = SmemPtrSw(sC);
  if (params.direct_smem_add == 4) {
    return;
  }

  for (int active_idx = active_start + widx * 2 + half; active_idx < active_end;
       active_idx += WorkerRepM * WorkerRepN * 2) {
    int row = 0;
    int start = 0;
    int end = 0;
    if (half_lane == 0) {
      row = params.direct_active_rows[active_idx];
      start = (row >= 0 && row < params.m) ? params.direct_row_offsets[row] : 0;
      end = (row >= 0 && row < params.m) ? params.direct_row_offsets[row + 1] : 0;
    }
    row = __shfl_sync(half_mask, row, half_base_lane);
    start = __shfl_sync(half_mask, start, half_base_lane);
    end = __shfl_sync(half_mask, end, half_base_lane);
    const int local_row = static_cast<int>(static_cast<int64_t>(row) - block_m0);
    if (row < 0 || row >= params.m || local_row < 0 || local_row >= CtaM ||
        start == end || half_lane >= ColGroups) {
      continue;
    }

    float acc[VecN] = {};
    for (int pos = start; pos < end; ++pos) {
      const int entry = params.direct_row_ks[pos];
      if (entry < 0) {
        continue;
      }
      if (full_packed_tile) {
        const uint4 delta = *reinterpret_cast<const uint4*>(
            params.direct_delta_output + static_cast<int64_t>(entry) * params.n + global_n0);
        acc[0] += direct_bf16_bits_to_float(delta.x);
        acc[1] += direct_bf16_bits_hi_to_float(delta.x);
        acc[2] += direct_bf16_bits_to_float(delta.y);
        acc[3] += direct_bf16_bits_hi_to_float(delta.y);
        acc[4] += direct_bf16_bits_to_float(delta.z);
        acc[5] += direct_bf16_bits_hi_to_float(delta.z);
        acc[6] += direct_bf16_bits_to_float(delta.w);
        acc[7] += direct_bf16_bits_hi_to_float(delta.w);
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_n0 + cc;
          if (global_col < params.n) {
            acc[cc] += direct_bf16_to_float(
                params.direct_delta_output[static_cast<int64_t>(entry) * params.n + global_col]);
          }
        }
      }
    }

    #pragma unroll
    for (int cc = 0; cc < VecN; ++cc) {
      const int64_t global_col = global_n0 + cc;
      if (global_col < params.n) {
        uint16_t* out_ptr = reinterpret_cast<uint16_t*>(
            smem + (static_cast<size_t>(local_row) * EpiN + local_col0 + cc) *
                       sizeof(uint16_t));
        const float out = direct_bf16_bits_to_float(static_cast<uint32_t>(*out_ptr));
        *out_ptr = direct_float_to_bf16_bits_u16(out + acc[cc]);
      }
    }
  }
}

template<int CtaM, int CtaN, int EpiN>
DEVICE void apply_direct_smem_sparse_tile(
    Params const& params,
    uint16_t* sC,
    uint32_t blk_m,
    uint32_t blk_n,
    int epi_st_n_idx,
    int widx,
    int lane_idx) {
  if (params.direct_smem_add == 4) {
    return;
  }
  if (params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_row_values == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }
  constexpr int SkipRowNnzGe = HANDWRITTEN_TMA_LOCAL_DELTA_SKIP_ROW_NNZ_GE;

  constexpr int VecN = 8;
  constexpr int ColGroups = EpiN / VecN;
  static_assert(EpiN % VecN == 0, "smem direct correction expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_end = params.direct_active_row_offsets[blk_m + 1];
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t global_n0 =
      static_cast<int64_t>(blk_n) * CtaN + epi_st_n_idx + half_lane * VecN;
  const int local_col0 = half_lane * VecN;
  const bool full_packed_tile = ((params.n & 7) == 0) && (global_n0 + VecN <= params.n);
  const bool lane_valid = half_lane < ColGroups;
  auto smem = SmemPtrSw(sC);

  for (int active_idx = active_start + widx * 2 + half; active_idx < active_end;
       active_idx += WorkerRepM * WorkerRepN * 2) {
    int local_row = 0;
    int start = 0;
    int end = 0;
    if (half_lane == 0) {
      local_row = params.direct_active_rows[active_idx];
      const int64_t global_row = block_m0 + local_row;
      if (local_row >= 0 && local_row < CtaM &&
          global_row >= 0 && global_row < params.m) {
        start = params.direct_row_offsets[global_row];
        end = params.direct_row_offsets[global_row + 1];
      }
    }
    local_row = __shfl_sync(half_mask, local_row, half_base_lane);
    start = __shfl_sync(half_mask, start, half_base_lane);
    end = __shfl_sync(half_mask, end, half_base_lane);
    const int64_t global_row = block_m0 + local_row;
    const int row_nnz = end - start;
    if (local_row < 0 || local_row >= CtaM || global_row >= params.m ||
        row_nnz <= 0) {
      continue;
    }
    if (params.direct_smem_add == 13) {
      if constexpr (SkipRowNnzGe >= 1000000000) {
        continue;
      } else if (row_nnz < SkipRowNnzGe) {
        continue;
      }
    }
    if (params.direct_smem_add == 5) {
      const int keep_i = local_row + start + end;
      asm volatile("" : : "r"(keep_i) : "memory");
      continue;
    }
    if (params.direct_smem_add == 10) {
      const int keep_k = params.direct_row_ks[start];
      asm volatile("" : : "r"(keep_k) : "memory");
      continue;
    }

    float acc[VecN] = {};
    if (full_packed_tile) {
      int cur_gk = params.direct_row_ks[start];
      if (params.direct_smem_add == 8) {
        asm volatile("" : : "r"(cur_gk) : "memory");
        continue;
      }
      float cur_av = direct_bf16_to_float(params.direct_row_values[start]);
      if (params.direct_smem_add == 9) {
        asm volatile("" : : "f"(cur_av) : "memory");
        continue;
      }
      if (params.direct_smem_add == 6) {
        asm volatile("" : : "r"(cur_gk), "f"(cur_av) : "memory");
        continue;
      }

      bool cur_valid = cur_gk >= 0 && cur_gk < params.k;
      uint4 cur_bv = make_uint4(0u, 0u, 0u, 0u);
      if (lane_valid && cur_valid) {
        cur_bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(cur_gk) * params.n + global_n0));
      }
      if (params.direct_smem_add == 7) {
        const uint32_t keep = cur_bv.x ^ cur_bv.y ^ cur_bv.z ^ cur_bv.w;
        asm volatile("" : : "r"(keep) : "memory");
        continue;
      }

      for (int pos = start; pos < end; ++pos) {
        const int next_pos = pos + 1;
        const int next_gk = next_pos < end ? params.direct_row_ks[next_pos] : 0;
        const float next_av =
            next_pos < end ? direct_bf16_to_float(params.direct_row_values[next_pos]) : 0.0f;

        const bool next_valid = next_pos < end && next_gk >= 0 && next_gk < params.k;
        uint4 next_bv = make_uint4(0u, 0u, 0u, 0u);
        if (lane_valid && next_valid) {
          next_bv = __ldg(reinterpret_cast<const uint4*>(
              params.direct_b_comp + static_cast<int64_t>(next_gk) * params.n + global_n0));
        }

        if (lane_valid && cur_valid) {
          acc[0] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.x), acc[0]);
          acc[1] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.x), acc[1]);
          acc[2] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.y), acc[2]);
          acc[3] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.y), acc[3]);
          acc[4] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.z), acc[4]);
          acc[5] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.z), acc[5]);
          acc[6] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.w), acc[6]);
          acc[7] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.w), acc[7]);
        }

        cur_gk = next_gk;
        cur_av = next_av;
        cur_valid = next_valid;
        cur_bv = next_bv;
      }
    } else {
      for (int pos = start; pos < end; ++pos) {
        const int gk = params.direct_row_ks[pos];
        const float av = direct_bf16_to_float(params.direct_row_values[pos]);
        if (gk < 0 || gk >= params.k) {
          continue;
        }
        if (lane_valid) {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_n0 + cc;
            if (global_col < params.n) {
              acc[cc] = fmaf(
                  av,
                  direct_bf16_to_float(
                      params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                  acc[cc]);
            }
          }
        }
      }
    }

    if (params.direct_smem_add == 3) {
      float keep = 0.0f;
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        keep += acc[cc];
      }
      asm volatile("" : : "f"(keep) : "memory");
      continue;
    }

    if (lane_valid) {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_n0 + cc;
        if (global_col < params.n) {
          uint16_t* out_ptr = reinterpret_cast<uint16_t*>(
              smem + (static_cast<size_t>(local_row) * EpiN + local_col0 + cc) *
                         sizeof(uint16_t));
          const float out = direct_bf16_bits_to_float(static_cast<uint32_t>(*out_ptr));
          *out_ptr = direct_float_to_bf16_bits_u16(out + acc[cc]);
        }
      }
    }
  }
}

template<int CtaM, int CtaN, int EpiN>
DEVICE void apply_precomputed_delta_smem_tile(
    Params const& params,
    uint16_t* sC,
    uint32_t blk_m,
    uint32_t blk_n,
    int epi_st_n_idx,
    int widx,
    int lane_idx) {
  if (params.direct_delta_output == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = EpiN / VecN;
  static_assert(EpiN % VecN == 0, "precomputed smem delta expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
#if HANDWRITTEN_TMA_MERGE_EARLY_SKIP
  constexpr unsigned ValidHalfMaskBase =
      ColGroups >= 16 ? 0x0000ffffu : ((1u << ColGroups) - 1u);
  const unsigned half_mask = ValidHalfMaskBase << half_base_lane;
#else
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
#endif
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_end = params.direct_active_row_offsets[blk_m + 1];
#if HANDWRITTEN_TMA_MERGE_EARLY_SKIP
  const int active_count = active_end - active_start;
  if (active_count <= 0 || half_lane >= ColGroups || widx * 2 >= active_count) {
    return;
  }
#endif
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t global_n0 =
      static_cast<int64_t>(blk_n) * CtaN + epi_st_n_idx + half_lane * VecN;
  const int local_col0 = half_lane * VecN;
  const bool lane_valid = half_lane < ColGroups;
  const bool full_packed_tile = ((params.n & 7) == 0) && (global_n0 + VecN <= params.n);
  auto smem = SmemPtrSw(sC);

  for (int active_idx = active_start + widx * 2 + half; active_idx < active_end;
       active_idx += WorkerRepM * WorkerRepN * 2) {
    int local_row = 0;
    if (half_lane == 0) {
      local_row = params.direct_active_rows[active_idx];
    }
    local_row = __shfl_sync(half_mask, local_row, half_base_lane);
    const int64_t global_row = block_m0 + local_row;
    if (!lane_valid || local_row < 0 || local_row >= CtaM ||
        global_row < 0 || global_row >= params.m) {
      continue;
    }

    float delta[VecN] = {};
    if (full_packed_tile) {
      const uint4 packed_delta = __ldg(reinterpret_cast<const uint4*>(
          params.direct_delta_output + global_row * params.n + global_n0));
      delta[0] = direct_bf16_bits_to_float(packed_delta.x);
      delta[1] = direct_bf16_bits_hi_to_float(packed_delta.x);
      delta[2] = direct_bf16_bits_to_float(packed_delta.y);
      delta[3] = direct_bf16_bits_hi_to_float(packed_delta.y);
      delta[4] = direct_bf16_bits_to_float(packed_delta.z);
      delta[5] = direct_bf16_bits_hi_to_float(packed_delta.z);
      delta[6] = direct_bf16_bits_to_float(packed_delta.w);
      delta[7] = direct_bf16_bits_hi_to_float(packed_delta.w);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_n0 + cc;
        if (global_col < params.n) {
          delta[cc] =
              direct_bf16_to_float(params.direct_delta_output[global_row * params.n + global_col]);
        }
      }
    }

    #pragma unroll
    for (int cc = 0; cc < VecN; ++cc) {
      const int64_t global_col = global_n0 + cc;
      if (global_col < params.n) {
        uint16_t* out_ptr = reinterpret_cast<uint16_t*>(
            smem + (static_cast<size_t>(local_row) * EpiN + local_col0 + cc) *
                       sizeof(uint16_t));
        const float out = direct_bf16_bits_to_float(static_cast<uint32_t>(*out_ptr));
        *out_ptr = direct_float_to_bf16_bits_u16(out + delta[cc]);
      }
    }
  }
}

template<int CtaM, int CtaN, int EpiN>
DEVICE void apply_precomputed_delta_smem_tile_vec8(
    Params const& params,
    uint16_t* sC,
    uint32_t blk_m,
    uint32_t blk_n,
    int epi_st_n_idx,
    int widx,
    int lane_idx) {
  if (params.direct_delta_output == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = EpiN / VecN;
  static_assert(EpiN % VecN == 0, "precomputed smem delta expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_end = params.direct_active_row_offsets[blk_m + 1];
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t global_n0 =
      static_cast<int64_t>(blk_n) * CtaN + epi_st_n_idx + half_lane * VecN;
  const int local_col0 = half_lane * VecN;
  const bool lane_valid = half_lane < ColGroups;
  const bool full_packed_tile = ((params.n & 7) == 0) && (global_n0 + VecN <= params.n);
  auto smem = SmemPtrSw(sC);

  for (int active_idx = active_start + widx * 2 + half; active_idx < active_end;
       active_idx += WorkerRepM * WorkerRepN * 2) {
    int local_row = 0;
    if (half_lane == 0) {
      local_row = params.direct_active_rows[active_idx];
    }
    local_row = __shfl_sync(half_mask, local_row, half_base_lane);
    const int64_t global_row = block_m0 + local_row;
    if (local_row < 0 || local_row >= CtaM ||
        global_row < 0 || global_row >= params.m) {
      continue;
    }

    if (full_packed_tile) {
      const uint4 packed_delta = __ldg(reinterpret_cast<const uint4*>(
          params.direct_delta_output + global_row * params.n + global_n0));
      void* out_ptr = smem + (static_cast<size_t>(local_row) * EpiN + local_col0) *
                                 sizeof(uint16_t);
      const uint4 packed_out = *reinterpret_cast<const uint4*>(out_ptr);
      *reinterpret_cast<uint4*>(out_ptr) =
          direct_bf16_add_packed_u4(packed_out, packed_delta);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_n0 + cc;
        if (global_col < params.n) {
          uint16_t* out_ptr = reinterpret_cast<uint16_t*>(
              smem + (static_cast<size_t>(local_row) * EpiN + local_col0 + cc) *
                         sizeof(uint16_t));
          const float out = direct_bf16_bits_to_float(static_cast<uint32_t>(*out_ptr));
          const float delta =
              direct_bf16_to_float(params.direct_delta_output[global_row * params.n + global_col]);
          *out_ptr = direct_float_to_bf16_bits_u16(out + delta);
        }
      }
    }
  }
}

template<int CtaM, int CtaN, int EpiN>
DEVICE void apply_local_delta_smem_tile_vec8(
    Params const& params,
    const uint16_t* local_delta,
    uint16_t* sC,
    uint32_t blk_m,
    uint32_t blk_n,
    int epi_st_n_idx,
    int widx,
    int lane_idx) {
  if (local_delta == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr) {
    return;
  }
  constexpr int SkipRowNnzGe = HANDWRITTEN_TMA_LOCAL_DELTA_SKIP_ROW_NNZ_GE;
  if constexpr (SkipRowNnzGe < 1000000000) {
    if (params.direct_row_offsets == nullptr) {
      return;
    }
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = EpiN / VecN;
  static_assert(EpiN % VecN == 0, "local delta smem merge expects vec8 N groups");
  static_assert(CtaN % VecN == 0, "local delta tile expects vec8 N groups");
  static_assert(CtaN % EpiN == 0, "local delta staging expects epilogue-width stages");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
#if HANDWRITTEN_TMA_MERGE_EARLY_SKIP
  constexpr unsigned ValidHalfMaskBase =
      ColGroups >= 16 ? 0x0000ffffu : ((1u << ColGroups) - 1u);
  const unsigned half_mask = ValidHalfMaskBase << half_base_lane;
#else
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
#endif
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_end = params.direct_active_row_offsets[blk_m + 1];
#if HANDWRITTEN_TMA_MERGE_EARLY_SKIP
  const int active_count = active_end - active_start;
  if (active_count <= 0 || half_lane >= ColGroups || widx * 2 >= active_count) {
    return;
  }
#endif
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t global_n0 =
      static_cast<int64_t>(blk_n) * CtaN + epi_st_n_idx + half_lane * VecN;
  const int local_col0 = half_lane * VecN;
  const bool lane_valid = half_lane < ColGroups;
  const bool full_packed_tile = ((params.n & 7) == 0) && (global_n0 + VecN <= params.n);
  auto smem = SmemPtrSw(sC);

  for (int active_idx = active_start + widx * 2 + half; active_idx < active_end;
       active_idx += WorkerRepM * WorkerRepN * 2) {
    int local_row = 0;
    if (half_lane == 0) {
      local_row = params.direct_active_rows[active_idx];
    }
    local_row = __shfl_sync(half_mask, local_row, half_base_lane);
    const int64_t global_row = block_m0 + local_row;
    if (!lane_valid || local_row < 0 || local_row >= CtaM ||
        global_row < 0 || global_row >= params.m) {
      continue;
    }
    if constexpr (SkipRowNnzGe < 1000000000) {
      const int row_start = params.direct_row_offsets[global_row];
      const int row_end = params.direct_row_offsets[global_row + 1];
      if (row_end - row_start >= SkipRowNnzGe) {
        continue;
      }
    }

    if (full_packed_tile) {
      const uint4 packed_delta = *reinterpret_cast<const uint4*>(
          local_delta + static_cast<int64_t>(local_row) * EpiN + local_col0);
      void* out_ptr = smem + (static_cast<size_t>(local_row) * EpiN + local_col0) *
                                 sizeof(uint16_t);
      const uint4 packed_out = *reinterpret_cast<const uint4*>(out_ptr);
      *reinterpret_cast<uint4*>(out_ptr) =
          direct_bf16_add_packed_u4(packed_out, packed_delta);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_n0 + cc;
        if (global_col < params.n) {
          uint16_t* out_ptr = reinterpret_cast<uint16_t*>(
              smem + (static_cast<size_t>(local_row) * EpiN + local_col0 + cc) *
                         sizeof(uint16_t));
          const float out = direct_bf16_bits_to_float(static_cast<uint32_t>(*out_ptr));
          const float delta = direct_bf16_bits_to_float(static_cast<uint32_t>(
              local_delta[static_cast<int64_t>(local_row) * EpiN + local_col0 + cc]));
          *out_ptr = direct_float_to_bf16_bits_u16(out + delta);
        }
      }
    }
  }
}

template<int CtaM, int CtaN, int EpiN>
DEVICE void apply_local_delta_smem_tile_vec8_side(
    Params const& params,
    const uint16_t* local_delta,
    uint16_t* sC,
    uint32_t blk_m,
    uint32_t blk_n,
    int epi_st_n_idx,
    int probe_rank,
    int probe_count,
    int lane_idx) {
  if (local_delta == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr ||
      probe_count <= 0) {
    return;
  }
  constexpr int SkipRowNnzGe = HANDWRITTEN_TMA_LOCAL_DELTA_SKIP_ROW_NNZ_GE;
  if constexpr (SkipRowNnzGe < 1000000000) {
    if (params.direct_row_offsets == nullptr) {
      return;
    }
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = EpiN / VecN;
  static_assert(EpiN % VecN == 0, "side local delta smem merge expects vec8 N groups");
  static_assert(CtaN % VecN == 0, "side local delta tile expects vec8 N groups");
  static_assert(CtaN % EpiN == 0, "side local delta staging expects epilogue-width stages");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
#if HANDWRITTEN_TMA_MERGE_EARLY_SKIP
  constexpr unsigned ValidHalfMaskBase =
      ColGroups >= 16 ? 0x0000ffffu : ((1u << ColGroups) - 1u);
  const unsigned half_mask = ValidHalfMaskBase << half_base_lane;
#else
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
#endif
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_end = params.direct_active_row_offsets[blk_m + 1];
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t global_n0 =
      static_cast<int64_t>(blk_n) * CtaN + epi_st_n_idx + half_lane * VecN;
  const int local_col0 = half_lane * VecN;
  const bool lane_valid = half_lane < ColGroups;
  const bool full_packed_tile = ((params.n & 7) == 0) && (global_n0 + VecN <= params.n);
  auto smem = SmemPtrSw(sC);

  for (int active_idx = active_start + probe_rank * 2 + half; active_idx < active_end;
       active_idx += probe_count * 2) {
    int local_row = 0;
    if (half_lane == 0) {
      local_row = params.direct_active_rows[active_idx];
    }
    local_row = __shfl_sync(half_mask, local_row, half_base_lane);
    const int64_t global_row = block_m0 + local_row;
    if (!lane_valid || local_row < 0 || local_row >= CtaM ||
        global_row < 0 || global_row >= params.m) {
      continue;
    }
    if constexpr (SkipRowNnzGe < 1000000000) {
      const int row_start = params.direct_row_offsets[global_row];
      const int row_end = params.direct_row_offsets[global_row + 1];
      if (row_end - row_start >= SkipRowNnzGe) {
        continue;
      }
    }

    if (full_packed_tile) {
      const uint4 packed_delta = *reinterpret_cast<const uint4*>(
          local_delta + static_cast<int64_t>(local_row) * EpiN + local_col0);
      void* out_ptr = smem + (static_cast<size_t>(local_row) * EpiN + local_col0) *
                                 sizeof(uint16_t);
      const uint4 packed_out = *reinterpret_cast<const uint4*>(out_ptr);
      *reinterpret_cast<uint4*>(out_ptr) =
          direct_bf16_add_packed_u4(packed_out, packed_delta);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_n0 + cc;
        if (global_col < params.n) {
          uint16_t* out_ptr = reinterpret_cast<uint16_t*>(
              smem + (static_cast<size_t>(local_row) * EpiN + local_col0 + cc) *
                         sizeof(uint16_t));
          const float out = direct_bf16_bits_to_float(static_cast<uint32_t>(*out_ptr));
          const float delta = direct_bf16_bits_to_float(static_cast<uint32_t>(
              local_delta[static_cast<int64_t>(local_row) * EpiN + local_col0 + cc]));
          *out_ptr = direct_float_to_bf16_bits_u16(out + delta);
        }
      }
    }
  }
}

template<int CtaM, int CtaN, int EpiN>
DEVICE void apply_packed_local_delta_smem_tile_vec8(
    Params const& params,
    const uint16_t* local_delta,
    uint16_t* sC,
    uint32_t blk_m,
    uint32_t blk_n,
    int epi_st_n_idx,
    int widx,
    int lane_idx) {
  if (local_delta == nullptr ||
      params.direct_packed_tile_offsets == nullptr ||
      params.direct_packed_row_records == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = EpiN / VecN;
  static_assert(EpiN % VecN == 0, "packed local delta merge expects vec8 N groups");
  static_assert(CtaN % VecN == 0, "packed local delta tile expects vec8 N groups");
  static_assert(CtaN % EpiN == 0, "packed local delta staging expects EpiN stages");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
#if HANDWRITTEN_TMA_MERGE_EARLY_SKIP
  constexpr unsigned ValidHalfMaskBase =
      ColGroups >= 16 ? 0x0000ffffu : ((1u << ColGroups) - 1u);
  const unsigned half_mask = ValidHalfMaskBase << half_base_lane;
#else
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
#endif
  const int packed_start = params.direct_packed_tile_offsets[blk_m];
  const int packed_end = params.direct_packed_tile_offsets[blk_m + 1];
#if HANDWRITTEN_TMA_MERGE_EARLY_SKIP
  const int packed_count = packed_end - packed_start;
  if (packed_count <= 0 || half_lane >= ColGroups || widx * 2 >= packed_count) {
    return;
  }
#endif
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t global_n0 =
      static_cast<int64_t>(blk_n) * CtaN + epi_st_n_idx + half_lane * VecN;
  const int local_col0 = half_lane * VecN;
  const bool lane_valid = half_lane < ColGroups;
  const bool full_packed_tile = ((params.n & 7) == 0) && (global_n0 + VecN <= params.n);
  auto smem = SmemPtrSw(sC);

  for (int packed_idx = packed_start + widx * 2 + half; packed_idx < packed_end;
       packed_idx += WorkerRepM * WorkerRepN * 2) {
    int local_row = 0;
    if (half_lane == 0) {
      const uint64_t row_record =
          static_cast<uint64_t>(params.direct_packed_row_records[packed_idx]);
      local_row = static_cast<int>(row_record & 0xffffull);
    }
    local_row = __shfl_sync(half_mask, local_row, half_base_lane);
    const int64_t global_row = block_m0 + local_row;
    if (!lane_valid || local_row < 0 || local_row >= CtaM ||
        global_row < 0 || global_row >= params.m) {
      continue;
    }

    if (full_packed_tile) {
      const uint4 packed_delta = *reinterpret_cast<const uint4*>(
          local_delta + static_cast<int64_t>(local_row) * EpiN + local_col0);
      void* out_ptr = smem + (static_cast<size_t>(local_row) * EpiN + local_col0) *
                                 sizeof(uint16_t);
      const uint4 packed_out = *reinterpret_cast<const uint4*>(out_ptr);
      *reinterpret_cast<uint4*>(out_ptr) =
          direct_bf16_add_packed_u4(packed_out, packed_delta);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_n0 + cc;
        if (global_col < params.n) {
          uint16_t* out_ptr = reinterpret_cast<uint16_t*>(
              smem + (static_cast<size_t>(local_row) * EpiN + local_col0 + cc) *
                         sizeof(uint16_t));
          const float out = direct_bf16_bits_to_float(static_cast<uint32_t>(*out_ptr));
          const float delta = direct_bf16_bits_to_float(static_cast<uint32_t>(
              local_delta[static_cast<int64_t>(local_row) * EpiN + local_col0 + cc]));
          *out_ptr = direct_float_to_bf16_bits_u16(out + delta);
        }
      }
    }
  }
}

template<int CtaM, int CtaN>
DEVICE void apply_sparse_rowchunk_late_add_noprobe(
    Params const& params,
    uint32_t blk_m,
    uint32_t blk_n,
    int lane_idx,
    int probe_rank,
    int probe_count) {
  if (params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_row_values == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = CtaN / VecN;
  static_assert(CtaN % VecN == 0, "rowchunk late add expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_end = params.direct_active_row_offsets[blk_m + 1];
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
  const int64_t global_col0 = n0 + half_lane * VecN;
  const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);
  c10::BFloat16* output = static_cast<c10::BFloat16*>(params.D);

  for (int active_idx = active_start + probe_rank * 2 + half; active_idx < active_end;
       active_idx += probe_count * 2) {
    int row = 0;
    int start = 0;
    int end = 0;
    if (half_lane == 0) {
      row = params.direct_active_rows[active_idx];
      if (row >= 0 && row < params.m) {
        start = params.direct_row_offsets[row];
        end = params.direct_row_offsets[row + 1];
      }
    }
    row = __shfl_sync(half_mask, row, half_base_lane);
    start = __shfl_sync(half_mask, start, half_base_lane);
    end = __shfl_sync(half_mask, end, half_base_lane);
    if (row < 0 || row >= params.m || start == end || half_lane >= ColGroups) {
      continue;
    }

    float acc[VecN] = {};
    for (int pos = start; pos < end; ++pos) {
      int gk = 0;
      float av = 0.0f;
      if (half_lane == 0) {
        gk = params.direct_row_ks[pos];
        av = direct_bf16_to_float(params.direct_row_values[pos]);
      }
      gk = __shfl_sync(half_mask, gk, half_base_lane);
      av = __shfl_sync(half_mask, av, half_base_lane);
      if (gk < 0 || gk >= params.k) {
        continue;
      }

      if (full_packed_tile) {
        const uint4 bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
        acc[0] = fmaf(av, direct_bf16_bits_to_float(bv.x), acc[0]);
        acc[1] = fmaf(av, direct_bf16_bits_hi_to_float(bv.x), acc[1]);
        acc[2] = fmaf(av, direct_bf16_bits_to_float(bv.y), acc[2]);
        acc[3] = fmaf(av, direct_bf16_bits_hi_to_float(bv.y), acc[3]);
        acc[4] = fmaf(av, direct_bf16_bits_to_float(bv.z), acc[4]);
        acc[5] = fmaf(av, direct_bf16_bits_hi_to_float(bv.z), acc[5]);
        acc[6] = fmaf(av, direct_bf16_bits_to_float(bv.w), acc[6]);
        acc[7] = fmaf(av, direct_bf16_bits_hi_to_float(bv.w), acc[7]);
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            acc[cc] = fmaf(
                av,
                direct_bf16_to_float(
                    params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                acc[cc]);
          }
        }
      }
    }

    if (full_packed_tile) {
      const uint4 out = *reinterpret_cast<const uint4*>(
          output + static_cast<int64_t>(row) * params.n + global_col0);
      const uint4 delta =
          make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                     direct_pack_bf16x2(acc[2], acc[3]),
                     direct_pack_bf16x2(acc[4], acc[5]),
                     direct_pack_bf16x2(acc[6], acc[7]));
      *reinterpret_cast<uint4*>(output + static_cast<int64_t>(row) * params.n + global_col0) =
          direct_bf16_add_packed_u4(out, delta);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_col0 + cc;
        if (global_col < params.n) {
          const int64_t out_idx = static_cast<int64_t>(row) * params.n + global_col;
          output[out_idx] =
              direct_float_to_bf16(direct_bf16_to_float(output[out_idx]) + acc[cc]);
        }
      }
    }
  }
}

template<int CtaM, int CtaN>
DEVICE void apply_sparse_rowchunk_late_add_localrows_skip_noprobe(
    Params const& params,
    uint32_t blk_m,
    uint32_t blk_n,
    int lane_idx,
    int probe_rank,
    int probe_count) {
  constexpr int SkipRowNnzGe = HANDWRITTEN_TMA_LOCAL_DELTA_SKIP_ROW_NNZ_GE;
  if constexpr (SkipRowNnzGe >= 1000000000) {
    return;
  }
  if (params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_row_values == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = CtaN / VecN;
  static_assert(CtaN % VecN == 0, "local-row late add expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_end = params.direct_active_row_offsets[blk_m + 1];
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
  const int64_t global_col0 = n0 + half_lane * VecN;
  const bool lane_valid = half_lane < ColGroups;
  const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);
  c10::BFloat16* output = static_cast<c10::BFloat16*>(params.D);

  for (int active_idx = active_start + probe_rank * 2 + half; active_idx < active_end;
       active_idx += probe_count * 2) {
    int local_row = 0;
    int start = 0;
    int end = 0;
    if (half_lane == 0) {
      local_row = params.direct_active_rows[active_idx];
      const int64_t global_row = block_m0 + local_row;
      if (local_row >= 0 && local_row < CtaM &&
          global_row >= 0 && global_row < params.m) {
        start = params.direct_row_offsets[global_row];
        end = params.direct_row_offsets[global_row + 1];
      }
    }
    local_row = __shfl_sync(half_mask, local_row, half_base_lane);
    start = __shfl_sync(half_mask, start, half_base_lane);
    end = __shfl_sync(half_mask, end, half_base_lane);
    const int64_t global_row = block_m0 + local_row;
    const int row_nnz = end - start;
    if (!lane_valid || local_row < 0 || local_row >= CtaM ||
        global_row < 0 || global_row >= params.m ||
        row_nnz < SkipRowNnzGe) {
      continue;
    }

    float acc[VecN] = {};
    for (int pos = start; pos < end; ++pos) {
      int gk = 0;
      float av = 0.0f;
      if (half_lane == 0) {
        gk = params.direct_row_ks[pos];
        av = direct_bf16_to_float(params.direct_row_values[pos]);
      }
      gk = __shfl_sync(half_mask, gk, half_base_lane);
      av = __shfl_sync(half_mask, av, half_base_lane);
      if (gk < 0 || gk >= params.k) {
        continue;
      }

      if (full_packed_tile) {
        const uint4 bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
        acc[0] = fmaf(av, direct_bf16_bits_to_float(bv.x), acc[0]);
        acc[1] = fmaf(av, direct_bf16_bits_hi_to_float(bv.x), acc[1]);
        acc[2] = fmaf(av, direct_bf16_bits_to_float(bv.y), acc[2]);
        acc[3] = fmaf(av, direct_bf16_bits_hi_to_float(bv.y), acc[3]);
        acc[4] = fmaf(av, direct_bf16_bits_to_float(bv.z), acc[4]);
        acc[5] = fmaf(av, direct_bf16_bits_hi_to_float(bv.z), acc[5]);
        acc[6] = fmaf(av, direct_bf16_bits_to_float(bv.w), acc[6]);
        acc[7] = fmaf(av, direct_bf16_bits_hi_to_float(bv.w), acc[7]);
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            acc[cc] = fmaf(
                av,
                direct_bf16_to_float(
                    params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                acc[cc]);
          }
        }
      }
    }

    if (full_packed_tile) {
      const uint4 out = *reinterpret_cast<const uint4*>(
          output + global_row * params.n + global_col0);
      const uint4 delta =
          make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                     direct_pack_bf16x2(acc[2], acc[3]),
                     direct_pack_bf16x2(acc[4], acc[5]),
                     direct_pack_bf16x2(acc[6], acc[7]));
      *reinterpret_cast<uint4*>(output + global_row * params.n + global_col0) =
          direct_bf16_add_packed_u4(out, delta);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_col0 + cc;
        if (global_col < params.n) {
          const int64_t out_idx = global_row * params.n + global_col;
          output[out_idx] =
              direct_float_to_bf16(direct_bf16_to_float(output[out_idx]) + acc[cc]);
        }
      }
    }
  }
}

template<int CtaM, int CtaN, int EpiN>
DEVICE void apply_sparse_rowchunk_smem_add_localrows_skip_noprobe(
    Params const& params,
    uint16_t* sC,
    uint32_t blk_m,
    uint32_t blk_n,
    int epi_st_n_idx,
    int lane_idx,
    int probe_rank,
    int probe_count) {
  constexpr int SkipRowNnzGe = HANDWRITTEN_TMA_LOCAL_DELTA_SKIP_ROW_NNZ_GE;
  if constexpr (SkipRowNnzGe >= 1000000000) {
    return;
  }
  if (sC == nullptr ||
      params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_row_values == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = EpiN / VecN;
  static_assert(EpiN % VecN == 0, "side smem add expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_end = params.direct_active_row_offsets[blk_m + 1];
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t global_n0 =
      static_cast<int64_t>(blk_n) * CtaN + epi_st_n_idx + half_lane * VecN;
  const int local_col0 = half_lane * VecN;
  const bool lane_valid = half_lane < ColGroups;
  const bool full_packed_tile = ((params.n & 7) == 0) && (global_n0 + VecN <= params.n);
  auto smem = SmemPtrSw(sC);

  for (int active_idx = active_start + probe_rank * 2 + half;
       active_idx < active_end;
       active_idx += probe_count * 2) {
    int local_row = 0;
    int start = 0;
    int end = 0;
    if (half_lane == 0) {
      local_row = params.direct_active_rows[active_idx];
      const int64_t global_row = block_m0 + local_row;
      if (local_row >= 0 && local_row < CtaM &&
          global_row >= 0 && global_row < params.m) {
        start = params.direct_row_offsets[global_row];
        end = params.direct_row_offsets[global_row + 1];
      }
    }
    local_row = __shfl_sync(half_mask, local_row, half_base_lane);
    start = __shfl_sync(half_mask, start, half_base_lane);
    end = __shfl_sync(half_mask, end, half_base_lane);
    const int64_t global_row = block_m0 + local_row;
    const int row_nnz = end - start;
    if (!lane_valid || local_row < 0 || local_row >= CtaM ||
        global_row < 0 || global_row >= params.m ||
        row_nnz < SkipRowNnzGe) {
      continue;
    }

    float acc[VecN] = {};
    if (full_packed_tile) {
      for (int pos = start; pos < end; ++pos) {
        const int gk = params.direct_row_ks[pos];
        if (gk < 0 || gk >= params.k) {
          continue;
        }
        const float av = direct_bf16_to_float(params.direct_row_values[pos]);
        const uint4 bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_n0));
        acc[0] = fmaf(av, direct_bf16_bits_to_float(bv.x), acc[0]);
        acc[1] = fmaf(av, direct_bf16_bits_hi_to_float(bv.x), acc[1]);
        acc[2] = fmaf(av, direct_bf16_bits_to_float(bv.y), acc[2]);
        acc[3] = fmaf(av, direct_bf16_bits_hi_to_float(bv.y), acc[3]);
        acc[4] = fmaf(av, direct_bf16_bits_to_float(bv.z), acc[4]);
        acc[5] = fmaf(av, direct_bf16_bits_hi_to_float(bv.z), acc[5]);
        acc[6] = fmaf(av, direct_bf16_bits_to_float(bv.w), acc[6]);
        acc[7] = fmaf(av, direct_bf16_bits_hi_to_float(bv.w), acc[7]);
      }
    } else {
      for (int pos = start; pos < end; ++pos) {
        const int gk = params.direct_row_ks[pos];
        if (gk < 0 || gk >= params.k) {
          continue;
        }
        const float av = direct_bf16_to_float(params.direct_row_values[pos]);
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_n0 + cc;
          if (global_col < params.n) {
            acc[cc] = fmaf(
                av,
                direct_bf16_to_float(
                    params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                acc[cc]);
          }
        }
      }
    }

    #pragma unroll
    for (int cc = 0; cc < VecN; ++cc) {
      const int64_t global_col = global_n0 + cc;
      if (global_col < params.n) {
        uint16_t* out_ptr = reinterpret_cast<uint16_t*>(
            smem + (static_cast<size_t>(local_row) * EpiN + local_col0 + cc) *
                       sizeof(uint16_t));
        const float out = direct_bf16_bits_to_float(static_cast<uint32_t>(*out_ptr));
        *out_ptr = direct_float_to_bf16_bits_u16(out + acc[cc]);
      }
    }
  }
}

template<int CtaM, int CtaN>
DEVICE void apply_sparse_rowchunk_chunked_atomic_add_noprobe(
    Params const& params,
    uint32_t blk_m,
    uint32_t blk_n,
    int lane_idx,
    int probe_rank,
    int probe_count) {
  if (params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_row_values == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = CtaN / VecN;
  constexpr int ChunkK = 8;
  static_assert(CtaN % VecN == 0, "rowchunk atomic add expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_rank = probe_rank * 2 + half;
  const int half_count = probe_count * 2;
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_end = params.direct_active_row_offsets[blk_m + 1];
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
  const int64_t global_col0 = n0 + half_lane * VecN;
  const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);
  c10::BFloat16* output = static_cast<c10::BFloat16*>(params.D);

  int task = 0;
  for (int active_idx = active_start; active_idx < active_end; ++active_idx) {
    const int row = params.direct_active_rows[active_idx];
    if (row < 0 || row >= params.m) {
      continue;
    }
    const int start = params.direct_row_offsets[row];
    const int end = params.direct_row_offsets[row + 1];
    const int nnz = end - start;
    if (nnz <= 0) {
      continue;
    }
    const int chunks = (nnz + ChunkK - 1) / ChunkK;
    for (int chunk = 0; chunk < chunks; ++chunk, ++task) {
      if ((task % half_count) != half_rank || half_lane >= ColGroups) {
        continue;
      }
      const int chunk_start = start + chunk * ChunkK;
      const int chunk_end = min(chunk_start + ChunkK, end);
      float acc[VecN] = {};
      for (int pos = chunk_start; pos < chunk_end; ++pos) {
        const int gk = params.direct_row_ks[pos];
        const float av = direct_bf16_to_float(params.direct_row_values[pos]);
        if (gk < 0 || gk >= params.k) {
          continue;
        }
        if (full_packed_tile) {
          const uint4 bv = __ldg(reinterpret_cast<const uint4*>(
              params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
          acc[0] = fmaf(av, direct_bf16_bits_to_float(bv.x), acc[0]);
          acc[1] = fmaf(av, direct_bf16_bits_hi_to_float(bv.x), acc[1]);
          acc[2] = fmaf(av, direct_bf16_bits_to_float(bv.y), acc[2]);
          acc[3] = fmaf(av, direct_bf16_bits_hi_to_float(bv.y), acc[3]);
          acc[4] = fmaf(av, direct_bf16_bits_to_float(bv.z), acc[4]);
          acc[5] = fmaf(av, direct_bf16_bits_hi_to_float(bv.z), acc[5]);
          acc[6] = fmaf(av, direct_bf16_bits_to_float(bv.w), acc[6]);
          acc[7] = fmaf(av, direct_bf16_bits_hi_to_float(bv.w), acc[7]);
        } else {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < params.n) {
              acc[cc] = fmaf(
                  av,
                  direct_bf16_to_float(
                      params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                  acc[cc]);
            }
          }
        }
      }

      if (chunks > 1) {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            direct_atomic_add_bf16(
                output + static_cast<int64_t>(row) * params.n + global_col, acc[cc]);
          }
        }
      } else if (full_packed_tile) {
        const uint4 out = *reinterpret_cast<const uint4*>(
            output + static_cast<int64_t>(row) * params.n + global_col0);
        const uint4 delta =
            make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                       direct_pack_bf16x2(acc[2], acc[3]),
                       direct_pack_bf16x2(acc[4], acc[5]),
                       direct_pack_bf16x2(acc[6], acc[7]));
        *reinterpret_cast<uint4*>(output + static_cast<int64_t>(row) * params.n + global_col0) =
            direct_bf16_add_packed_u4(out, delta);
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            const int64_t out_idx = static_cast<int64_t>(row) * params.n + global_col;
            output[out_idx] =
                direct_float_to_bf16(direct_bf16_to_float(output[out_idx]) + acc[cc]);
          }
        }
      }
    }
  }
}

template<int Rows, int CtaM, int CtaN>
struct SparseAccStorage;

template<int CtaM, int CtaN, int AccRows>
DEVICE void apply_sparse_loadfma_kmajor_sharedacc_tile(
    Params const& params,
    SparseAccStorage<AccRows, CtaM, CtaN>& sparse_acc,
    uint32_t active_m_idx,
    uint32_t blk_m,
    uint32_t blk_n,
    int lane_idx,
    int probe_rank,
    int probe_count,
    int sink_rank_offset) {
  if (params.direct_probe_sink == nullptr ||
      params.direct_delta_output == nullptr ||
      params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr ||
      params.direct_kmajor_group_offsets == nullptr ||
      params.direct_kmajor_group_ks == nullptr ||
      params.direct_kmajor_entry_offsets == nullptr ||
      params.direct_kmajor_entry_rows == nullptr ||
      params.direct_kmajor_entry_values == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = CtaN / VecN;
  static_assert(CtaN % VecN == 0, "sharedacc k-major expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp k owner supports at most 16 col groups");

  const int producer_tid = probe_rank * NumThreadsPerWarp + lane_idx;
  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  const int group_start = params.direct_kmajor_group_offsets[active_m_idx];
  const int group_end = params.direct_kmajor_group_offsets[active_m_idx + 1];
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_count =
      params.direct_active_row_offsets[blk_m + 1] - active_start;
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
  const int64_t global_col0 = n0 + half_lane * VecN;
  const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);
  float lane_sum = 0.0f;

  for (int chunk_start = 0; chunk_start < active_count; chunk_start += AccRows) {
    const int rows_this_chunk = min(AccRows, active_count - chunk_start);

    for (int idx = producer_tid; idx < CtaM; idx += probe_count * NumThreadsPerWarp) {
      sparse_acc.row_to_slot[idx] = -1;
    }
    for (int idx = producer_tid; idx < AccRows * CtaN; idx += probe_count * NumThreadsPerWarp) {
      reinterpret_cast<float*>(sparse_acc.acc)[idx] = 0.0f;
    }
    for (int slot = producer_tid; slot < rows_this_chunk; slot += probe_count * NumThreadsPerWarp) {
      const int local_row = params.direct_active_rows[active_start + chunk_start + slot];
      sparse_acc.slot_rows[slot] = local_row;
      if (local_row >= 0 && local_row < CtaM) {
        sparse_acc.row_to_slot[local_row] = slot;
      }
    }
    ptx::bar_sync(1, probe_count * NumThreadsPerWarp);

    for (int group_idx = group_start + probe_rank * 2 + half; group_idx < group_end;
         group_idx += probe_count * 2) {
      int gk = 0;
      int entry_start = 0;
      int entry_end = 0;
      if (half_lane == 0) {
        gk = params.direct_kmajor_group_ks[group_idx];
        entry_start = params.direct_kmajor_entry_offsets[group_idx];
        entry_end = params.direct_kmajor_entry_offsets[group_idx + 1];
      }
      gk = __shfl_sync(half_mask, gk, half_base_lane);
      entry_start = __shfl_sync(half_mask, entry_start, half_base_lane);
      entry_end = __shfl_sync(half_mask, entry_end, half_base_lane);
      if (entry_start == entry_end || gk < 0 || gk >= params.k) {
        continue;
      }

      const bool lane_valid = half_lane < ColGroups;
      float bvals[VecN] = {};
      if (full_packed_tile) {
        uint4 bv = make_uint4(0u, 0u, 0u, 0u);
        if (lane_valid) {
          bv = __ldg(reinterpret_cast<const uint4*>(
              params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
        }
        bvals[0] = direct_bf16_bits_to_float(bv.x);
        bvals[1] = direct_bf16_bits_hi_to_float(bv.x);
        bvals[2] = direct_bf16_bits_to_float(bv.y);
        bvals[3] = direct_bf16_bits_hi_to_float(bv.y);
        bvals[4] = direct_bf16_bits_to_float(bv.z);
        bvals[5] = direct_bf16_bits_hi_to_float(bv.z);
        bvals[6] = direct_bf16_bits_to_float(bv.w);
        bvals[7] = direct_bf16_bits_hi_to_float(bv.w);
      } else if (lane_valid) {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            bvals[cc] = direct_bf16_to_float(
                params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]);
          }
        }
      }

      for (int entry_idx = entry_start; entry_idx < entry_end; ++entry_idx) {
        int local_row = 0;
        if (half_lane == 0) {
          local_row = params.direct_kmajor_entry_rows[entry_idx];
        }
        local_row = __shfl_sync(half_mask, local_row, half_base_lane);
        const int slot =
            (local_row >= 0 && local_row < CtaM) ? sparse_acc.row_to_slot[local_row] : -1;
        const float av = direct_bf16_to_float(params.direct_kmajor_entry_values[entry_idx]);
        if (lane_valid && slot >= 0) {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int col = half_lane * VecN + cc;
            if (col < CtaN && n0 + col < params.n) {
              const float delta = av * bvals[cc];
              lane_sum += delta;
              atomicAdd(&sparse_acc.acc[slot][col], delta);
            }
          }
        }
      }
    }
    ptx::bar_sync(1, probe_count * NumThreadsPerWarp);

    for (int slot = probe_rank * 2 + half; slot < rows_this_chunk; slot += probe_count * 2) {
      const int local_row = sparse_acc.slot_rows[slot];
      const int64_t global_row = block_m0 + local_row;
      if (global_row >= params.m || half_lane >= ColGroups) {
        continue;
      }
      if (full_packed_tile) {
        const uint4 packed_delta =
            make_uint4(direct_pack_bf16x2(sparse_acc.acc[slot][half_lane * VecN + 0],
                                          sparse_acc.acc[slot][half_lane * VecN + 1]),
                       direct_pack_bf16x2(sparse_acc.acc[slot][half_lane * VecN + 2],
                                          sparse_acc.acc[slot][half_lane * VecN + 3]),
                       direct_pack_bf16x2(sparse_acc.acc[slot][half_lane * VecN + 4],
                                          sparse_acc.acc[slot][half_lane * VecN + 5]),
                       direct_pack_bf16x2(sparse_acc.acc[slot][half_lane * VecN + 6],
                                          sparse_acc.acc[slot][half_lane * VecN + 7]));
        *reinterpret_cast<uint4*>(
            params.direct_delta_output + global_row * params.n + global_col0) =
            packed_delta;
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            params.direct_delta_output[global_row * params.n + global_col] =
                direct_float_to_bf16(sparse_acc.acc[slot][half_lane * VecN + cc]);
          }
        }
      }
    }
    ptx::bar_sync(1, probe_count * NumThreadsPerWarp);
  }

  lane_sum = warp_reduce_sum_f32(lane_sum);
  if (lane_idx == 0) {
    const int32_t tile_id =
        static_cast<int32_t>(blk_m * gridDim.x + blk_n);
    volatile float* sink = params.direct_probe_sink;
    sink[tile_id * params.direct_probe_warps + sink_rank_offset + probe_rank] = lane_sum;
  }
}

template<int CtaM, int CtaN, int AccRows>
DEVICE void apply_sparse_loadfma_kmajor_sharedacc_fill_tile(
    Params const& params,
    SparseAccStorage<AccRows, CtaM, CtaN>& sparse_acc,
    uint32_t active_m_idx,
    uint32_t blk_m,
    uint32_t blk_n,
    int lane_idx,
    int probe_rank,
    int probe_count,
    int max_groups) {
  if constexpr (AccRows <= 0) {
    return;
  } else {
    if (params.direct_active_row_offsets == nullptr ||
        params.direct_active_rows == nullptr ||
        params.direct_kmajor_group_offsets == nullptr ||
        params.direct_kmajor_group_ks == nullptr ||
        params.direct_kmajor_entry_offsets == nullptr ||
        params.direct_kmajor_entry_rows == nullptr ||
        params.direct_kmajor_entry_values == nullptr ||
        params.direct_b_comp == nullptr ||
        max_groups <= 0) {
      return;
    }

    constexpr int VecN = 8;
    constexpr int ColGroups = CtaN / VecN;
    static_assert(CtaN % VecN == 0, "sharedacc smem expects vec8 N groups");
    static_assert(ColGroups <= 16, "half-warp k owner supports at most 16 col groups");

    const int producer_tid = probe_rank * NumThreadsPerWarp + lane_idx;
    const int half = lane_idx >> 4;
    const int half_lane = lane_idx & 15;
    const int half_base_lane = half << 4;
    const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
    const int group_start = params.direct_kmajor_group_offsets[active_m_idx];
    const int group_end = params.direct_kmajor_group_offsets[active_m_idx + 1];
    const int group_count = group_end - group_start;
    const int groups_per_rank = (max_groups + probe_count - 1) / probe_count;
    const int local_begin = probe_rank * groups_per_rank;
    const int local_end = min(max_groups, local_begin + groups_per_rank);
    const int active_start = params.direct_active_row_offsets[blk_m];
    const int active_count =
        params.direct_active_row_offsets[blk_m + 1] - active_start;
    const int rows_this_chunk = min(AccRows, active_count);
    const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
    const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;
    const int64_t global_col0 = n0 + half_lane * VecN;
    const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);

    for (int idx = producer_tid; idx < CtaM; idx += probe_count * NumThreadsPerWarp) {
      sparse_acc.row_to_slot[idx] = -1;
    }
    for (int idx = producer_tid; idx < AccRows * CtaN;
         idx += probe_count * NumThreadsPerWarp) {
      reinterpret_cast<float*>(sparse_acc.acc)[idx] = 0.0f;
    }
    for (int slot = producer_tid; slot < rows_this_chunk;
         slot += probe_count * NumThreadsPerWarp) {
      const int row_value = params.direct_active_rows[active_start + slot];
      const int local_row =
          row_value >= CtaM ? static_cast<int>(static_cast<int64_t>(row_value) - block_m0)
                            : row_value;
      sparse_acc.slot_rows[slot] = local_row;
      if (local_row >= 0 && local_row < CtaM) {
        sparse_acc.row_to_slot[local_row] = slot;
      }
    }
    ptx::bar_sync(5, probe_count * NumThreadsPerWarp);

    for (int local_group = local_begin + half; local_group < local_end;
         local_group += 2) {
      if (local_group >= group_count) {
        continue;
      }
      const int group_idx = group_start + local_group;
      int gk = 0;
      int entry_start = 0;
      int entry_end = 0;
      if (half_lane == 0) {
        gk = params.direct_kmajor_group_ks[group_idx];
        entry_start = params.direct_kmajor_entry_offsets[group_idx];
        entry_end = params.direct_kmajor_entry_offsets[group_idx + 1];
      }
      gk = __shfl_sync(half_mask, gk, half_base_lane);
      entry_start = __shfl_sync(half_mask, entry_start, half_base_lane);
      entry_end = __shfl_sync(half_mask, entry_end, half_base_lane);
      if (entry_start == entry_end || gk < 0 || gk >= params.k) {
        continue;
      }

      const bool lane_valid = half_lane < ColGroups;
      float bvals[VecN] = {};
      if (full_packed_tile) {
        uint4 bv = make_uint4(0u, 0u, 0u, 0u);
        if (lane_valid) {
          bv = __ldg(reinterpret_cast<const uint4*>(
              params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
        }
        bvals[0] = direct_bf16_bits_to_float(bv.x);
        bvals[1] = direct_bf16_bits_hi_to_float(bv.x);
        bvals[2] = direct_bf16_bits_to_float(bv.y);
        bvals[3] = direct_bf16_bits_hi_to_float(bv.y);
        bvals[4] = direct_bf16_bits_to_float(bv.z);
        bvals[5] = direct_bf16_bits_hi_to_float(bv.z);
        bvals[6] = direct_bf16_bits_to_float(bv.w);
        bvals[7] = direct_bf16_bits_hi_to_float(bv.w);
      } else if (lane_valid) {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            bvals[cc] = direct_bf16_to_float(
                params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]);
          }
        }
      }

      for (int entry_idx = entry_start; entry_idx < entry_end; ++entry_idx) {
        int local_row = 0;
        if (half_lane == 0) {
          local_row = params.direct_kmajor_entry_rows[entry_idx];
        }
        local_row = __shfl_sync(half_mask, local_row, half_base_lane);
        const int slot =
            (local_row >= 0 && local_row < CtaM) ? sparse_acc.row_to_slot[local_row] : -1;
        const float av = direct_bf16_to_float(params.direct_kmajor_entry_values[entry_idx]);
        if (lane_valid && slot >= 0 && slot < rows_this_chunk) {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int col = half_lane * VecN + cc;
            if (col < CtaN && n0 + col < params.n) {
              atomicAdd(&sparse_acc.acc[slot][col], av * bvals[cc]);
            }
          }
        }
      }
    }
    ptx::bar_sync(5, probe_count * NumThreadsPerWarp);
  }
}

template<int CtaM, int CtaN, int EpiN, int AccRows>
DEVICE void apply_sparse_acc_smem_merge_tile(
    Params const& params,
    SparseAccStorage<AccRows, CtaM, CtaN>& sparse_acc,
    uint16_t* sC,
    uint32_t blk_m,
    uint32_t blk_n,
    int epi_st_n_idx,
    int widx,
    int lane_idx) {
  if constexpr (AccRows <= 0) {
    return;
  } else {
    if (params.direct_active_row_offsets == nullptr ||
        params.direct_active_rows == nullptr) {
      return;
    }

    constexpr int VecN = 8;
    constexpr int ColGroups = EpiN / VecN;
    static_assert(EpiN % VecN == 0, "sharedacc smem merge expects vec8 N groups");
    static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

    const int half = lane_idx >> 4;
    const int half_lane = lane_idx & 15;
    const int half_base_lane = half << 4;
    const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
    const int active_start = params.direct_active_row_offsets[blk_m];
    const int active_end = params.direct_active_row_offsets[blk_m + 1];
    const int active_count = active_end - active_start;
    const int rows_this_chunk = min(AccRows, active_count);
    const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
    const int64_t global_n0 =
        static_cast<int64_t>(blk_n) * CtaN + epi_st_n_idx + half_lane * VecN;
    const int local_col0 = half_lane * VecN;
    const bool lane_valid = half_lane < ColGroups;
    auto smem = SmemPtrSw(sC);

    for (int slot = widx * 2 + half; slot < rows_this_chunk;
         slot += WorkerRepM * WorkerRepN * 2) {
      int local_row = 0;
      if (half_lane == 0) {
        const int row_value = params.direct_active_rows[active_start + slot];
        local_row =
            row_value >= CtaM ? static_cast<int>(static_cast<int64_t>(row_value) - block_m0)
                              : row_value;
      }
      local_row = __shfl_sync(half_mask, local_row, half_base_lane);
      const int64_t global_row = block_m0 + local_row;
      if (!lane_valid || local_row < 0 || local_row >= CtaM ||
          global_row < 0 || global_row >= params.m) {
        continue;
      }

      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_n0 + cc;
        const int acc_col = epi_st_n_idx + local_col0 + cc;
        if (global_col < params.n && acc_col < CtaN) {
          uint16_t* out_ptr = reinterpret_cast<uint16_t*>(
              smem + (static_cast<size_t>(local_row) * EpiN + local_col0 + cc) *
                         sizeof(uint16_t));
          const float out = direct_bf16_bits_to_float(static_cast<uint32_t>(*out_ptr));
          *out_ptr = direct_float_to_bf16_bits_u16(out + sparse_acc.acc[slot][acc_col]);
        }
      }
    }
  }
}

template<int CtaM, int CtaN, int SubN, int StorageRows>
DEVICE void apply_sparse_loadfma_kmajor_subacc32_stage_tile(
    Params const& params,
    SparseAccStorage<StorageRows, CtaM, CtaN>& sparse_acc,
    uint32_t active_m_idx,
    bool has_groups,
    uint32_t blk_m,
    uint32_t blk_n,
    int sub_n_idx,
    int lane_idx,
    int probe_rank,
    int probe_count,
    int max_groups) {
  if constexpr (StorageRows * CtaN < CtaM * SubN) {
    return;
  } else {
    if (params.direct_kmajor_group_offsets == nullptr ||
        params.direct_kmajor_group_ks == nullptr ||
        params.direct_kmajor_entry_offsets == nullptr ||
        params.direct_kmajor_entry_rows == nullptr ||
        params.direct_kmajor_entry_values == nullptr ||
        params.direct_b_comp == nullptr ||
        max_groups <= 0) {
      has_groups = false;
    }

    constexpr int VecN = 8;
    constexpr int ColGroups = SubN / VecN;
    static_assert(SubN == 32, "subacc staged path currently expects 32 columns");
    static_assert(SubN % VecN == 0, "subacc staged path expects vec8 N groups");
    static_assert(ColGroups <= 16, "half-warp subacc owner supports at most 16 col groups");

    float* acc = reinterpret_cast<float*>(sparse_acc.acc);
    const int producer_tid = probe_rank * NumThreadsPerWarp + lane_idx;
    const int half = lane_idx >> 4;
    const int half_lane = lane_idx & 15;
    const int half_base_lane = half << 4;
    const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
    const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN + sub_n_idx;
    const int64_t global_col0 = n0 + half_lane * VecN;
    const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + SubN <= params.n);

    for (int idx = producer_tid; idx < CtaM * SubN;
         idx += probe_count * NumThreadsPerWarp) {
      acc[idx] = 0.0f;
    }
    ptx::bar_sync(5, probe_count * NumThreadsPerWarp);

    if (has_groups) {
      const int group_start = params.direct_kmajor_group_offsets[active_m_idx];
      const int group_end = params.direct_kmajor_group_offsets[active_m_idx + 1];
      const int group_count = group_end - group_start;
      const int effective_groups = min(max_groups, group_count);
      const int groups_per_rank = (effective_groups + probe_count - 1) / probe_count;
      const int local_begin = probe_rank * groups_per_rank;
      const int local_end = min(effective_groups, local_begin + groups_per_rank);

      for (int local_group = local_begin + half; local_group < local_end;
           local_group += 2) {
        const int group_idx = group_start + local_group;
        int gk = 0;
        int entry_start = 0;
        int entry_end = 0;
        if (half_lane == 0) {
          gk = params.direct_kmajor_group_ks[group_idx];
          entry_start = params.direct_kmajor_entry_offsets[group_idx];
          entry_end = params.direct_kmajor_entry_offsets[group_idx + 1];
        }
        gk = __shfl_sync(half_mask, gk, half_base_lane);
        entry_start = __shfl_sync(half_mask, entry_start, half_base_lane);
        entry_end = __shfl_sync(half_mask, entry_end, half_base_lane);
        if (entry_start == entry_end || gk < 0 || gk >= params.k) {
          continue;
        }

        const bool lane_valid = half_lane < ColGroups;
        float bvals[VecN] = {};
        if (full_packed_tile) {
          uint4 bv = make_uint4(0u, 0u, 0u, 0u);
          if (lane_valid) {
            bv = __ldg(reinterpret_cast<const uint4*>(
                params.direct_b_comp + static_cast<int64_t>(gk) * params.n + global_col0));
          }
          bvals[0] = direct_bf16_bits_to_float(bv.x);
          bvals[1] = direct_bf16_bits_hi_to_float(bv.x);
          bvals[2] = direct_bf16_bits_to_float(bv.y);
          bvals[3] = direct_bf16_bits_hi_to_float(bv.y);
          bvals[4] = direct_bf16_bits_to_float(bv.z);
          bvals[5] = direct_bf16_bits_hi_to_float(bv.z);
          bvals[6] = direct_bf16_bits_to_float(bv.w);
          bvals[7] = direct_bf16_bits_hi_to_float(bv.w);
        } else if (lane_valid) {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < params.n) {
              bvals[cc] = direct_bf16_to_float(
                  params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]);
            }
          }
        }

        for (int entry_idx = entry_start; entry_idx < entry_end; ++entry_idx) {
          int local_row = 0;
          if (half_lane == 0) {
            local_row = params.direct_kmajor_entry_rows[entry_idx];
          }
          local_row = __shfl_sync(half_mask, local_row, half_base_lane);
          const float av = direct_bf16_to_float(params.direct_kmajor_entry_values[entry_idx]);
          if (lane_valid && local_row >= 0 && local_row < CtaM) {
            #pragma unroll
            for (int cc = 0; cc < VecN; ++cc) {
              const int col = half_lane * VecN + cc;
              const int64_t global_col = global_col0 + cc;
              if (col < SubN && global_col < params.n) {
                atomicAdd(&acc[local_row * SubN + col], av * bvals[cc]);
              }
            }
          }
        }
      }
    }
    ptx::bar_sync(5, probe_count * NumThreadsPerWarp);
  }
}

template<int CtaM, int CtaN, int EpiN, int SubN, int StorageRows>
DEVICE void apply_sparse_subacc32_smem_merge_tile(
    Params const& params,
    SparseAccStorage<StorageRows, CtaM, CtaN>& sparse_acc,
    uint16_t* sC,
    uint32_t blk_m,
    uint32_t blk_n,
    int epi_st_n_idx,
    int sub_n_in_epi,
    int widx,
    int lane_idx) {
  if constexpr (StorageRows * CtaN < CtaM * SubN) {
    return;
  } else {
    constexpr int VecN = 8;
    constexpr int ColGroups = SubN / VecN;
    static_assert(SubN == 32, "subacc smem merge currently expects 32 columns");
    static_assert(SubN % VecN == 0, "subacc smem merge expects vec8 N groups");
    static_assert(EpiN % SubN == 0, "subacc smem merge expects EpiN divisible by SubN");

    float* acc = reinterpret_cast<float*>(sparse_acc.acc);
    const int half = lane_idx >> 4;
    const int half_lane = lane_idx & 15;
    const int half_base_lane = half << 4;
    const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
    const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
    const int64_t global_n0 =
        static_cast<int64_t>(blk_n) * CtaN + epi_st_n_idx + sub_n_in_epi + half_lane * VecN;
    const int local_col0 = sub_n_in_epi + half_lane * VecN;
    const int acc_col0 = half_lane * VecN;
    const bool lane_valid = half_lane < ColGroups;
    auto smem = SmemPtrSw(sC);

    for (int local_row = widx * 2 + half; local_row < CtaM;
         local_row += WorkerRepM * WorkerRepN * 2) {
      const int64_t global_row = block_m0 + local_row;
      if (!lane_valid || global_row < 0 || global_row >= params.m) {
        continue;
      }
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_n0 + cc;
        const int local_col = local_col0 + cc;
        const int acc_col = acc_col0 + cc;
        if (global_col < params.n && local_col < EpiN && acc_col < SubN) {
          uint16_t* out_ptr = reinterpret_cast<uint16_t*>(
              smem + (static_cast<size_t>(local_row) * EpiN + local_col) *
                         sizeof(uint16_t));
          const float out = direct_bf16_bits_to_float(static_cast<uint32_t>(*out_ptr));
          *out_ptr = direct_float_to_bf16_bits_u16(out + acc[local_row * SubN + acc_col]);
        }
      }
    }
  }
}

template<int CtaM, int CtaN>
DEVICE void apply_incta_sparse_probe_for_tile(Params const& params,
                                              SparseAccStorage<HANDWRITTEN_TMA_SPARSE_ACC_ROWS, CtaM, CtaN>& sparse_acc,
                                              uint32_t real_grid_m,
                                              uint32_t blk_m,
                                              uint32_t blk_n,
                                              int lane_idx,
                                              int probe_rank,
                                              int probe_count,
                                              int row_sink_offset,
                                              bool count_tile) {
  int active_m_idx = -1;
  if (lane_idx == 0 && params.direct_probe_active_mblocks != nullptr) {
    int lo = 0;
    int hi = params.direct_probe_active_mblock_count;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      const int32_t mid_blk = params.direct_probe_active_mblocks[mid];
      if (mid_blk < static_cast<int32_t>(blk_m)) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    if (lo < params.direct_probe_active_mblock_count &&
        params.direct_probe_active_mblocks[lo] == static_cast<int32_t>(blk_m)) {
      active_m_idx = lo;
    }
  } else if (params.direct_probe_active_mblocks == nullptr) {
    active_m_idx = static_cast<int>(blk_m);
  }
  active_m_idx = __shfl_sync(0xffffffffu, active_m_idx, 0);
  if (blk_m >= real_grid_m || active_m_idx < 0 || params.direct_probe_do_math == 0) {
    return;
  }
  if (params.direct_probe_kmajor == 2) {
    if (params.direct_delta_output != nullptr && params.direct_delta_write_mode == 3) {
      apply_sparse_loadfma_kmajor_sharedacc_tile<
          CtaM,
          CtaN,
          HANDWRITTEN_TMA_SPARSE_ACC_ROWS>(
          params,
          sparse_acc,
          static_cast<uint32_t>(active_m_idx),
          blk_m,
          blk_n,
          lane_idx,
          probe_rank,
          probe_count,
          0);
    } else if (params.direct_delta_output != nullptr) {
      apply_sparse_loadfma_probe_kmajor_tile<CtaM, CtaN, true>(
          params,
          static_cast<uint32_t>(active_m_idx),
          blk_m,
          blk_n,
          lane_idx,
          probe_rank,
          probe_count,
          0);
    } else {
      apply_sparse_loadfma_probe_kmajor_tile<CtaM, CtaN, false>(
          params,
          static_cast<uint32_t>(active_m_idx),
          blk_m,
          blk_n,
          lane_idx,
          probe_rank,
          probe_count,
          0);
    }
    if (!(params.direct_delta_output != nullptr && params.direct_delta_write_mode == 3)) {
      apply_sparse_loadfma_probe_tile<CtaM, CtaN>(
          params,
          blk_m,
          blk_n,
          lane_idx,
          probe_rank,
          probe_count,
          row_sink_offset);
    }
  } else if (params.direct_probe_kmajor != 0) {
    if (params.direct_delta_output != nullptr) {
      apply_sparse_loadfma_probe_kmajor_tile<CtaM, CtaN, true>(
          params,
          static_cast<uint32_t>(active_m_idx),
          blk_m,
          blk_n,
          lane_idx,
          probe_rank,
          probe_count,
          0);
    } else {
      apply_sparse_loadfma_probe_kmajor_tile<CtaM, CtaN, false>(
          params,
          static_cast<uint32_t>(active_m_idx),
          blk_m,
          blk_n,
          lane_idx,
          probe_rank,
          probe_count,
          0);
    }
  } else {
    apply_sparse_loadfma_probe_tile<CtaM, CtaN>(
        params,
        blk_m,
        blk_n,
        lane_idx,
        probe_rank,
        probe_count,
        0);
  }
  if (count_tile && lane_idx == 0 && params.direct_probe_counter != nullptr) {
    atomicAdd(params.direct_probe_counter, 1);
  }
}

template<int CtaM, int CtaN>
DEVICE void apply_direct_add_sparse_tile(Params const& params,
                                         uint32_t blk_m,
                                         uint32_t blk_n,
                                         int widx,
                                         int lane_idx,
                                         int worker_warps = WorkerRepM * WorkerRepN) {
  if (params.direct_row_offsets == nullptr || params.direct_active_row_offsets == nullptr ||
      params.direct_active_rows == nullptr || params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int VecN = 8;
  constexpr int ColGroups = CtaN / VecN;
  static_assert(CtaN % VecN == 0, "direct-add sparse path expects vec8 N groups");
  static_assert(ColGroups <= 16, "half-warp row owner supports at most 16 col groups");

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
  const int active_start = params.direct_active_row_offsets[blk_m];
  const int active_count =
      params.direct_active_row_offsets[blk_m + 1] - active_start;
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t n0 = static_cast<int64_t>(blk_n) * CtaN;

  for (int active_item = widx * 2 + half; active_item < active_count;
       active_item += worker_warps * 2) {
    const int local_row = params.direct_active_rows[active_start + active_item];
    const int64_t global_row = block_m0 + local_row;
    const int64_t global_col0 = n0 + half_lane * VecN;

    int start = 0;
    int end = 0;
    if (half_lane == 0 && global_row < params.m) {
      start = params.direct_row_offsets[global_row];
      end = params.direct_row_offsets[global_row + 1];
    }
    start = __shfl_sync(half_mask, start, half_base_lane);
    end = __shfl_sync(half_mask, end, half_base_lane);
    if (global_row >= params.m || start == end) {
      continue;
    }

    float acc[VecN] = {};
    const bool full_packed_tile = ((params.n & 7) == 0) && (n0 + CtaN <= params.n);
    if (full_packed_tile) {
      int cur_gk = 0;
      float cur_av = 0.0f;
      if (half_lane == 0) {
        cur_gk = params.direct_row_ks[start];
        cur_av = direct_bf16_to_float(params.direct_row_values[start]);
      }
      cur_gk = __shfl_sync(half_mask, cur_gk, half_base_lane);
      cur_av = __shfl_sync(half_mask, cur_av, half_base_lane);

      bool cur_valid = half_lane < ColGroups && cur_gk >= 0 && cur_gk < params.k;
      uint4 cur_bv = make_uint4(0u, 0u, 0u, 0u);
      if (cur_valid) {
        cur_bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(cur_gk) * params.n + global_col0));
      }

      for (int entry_idx = start; entry_idx < end; ++entry_idx) {
        int next_gk = 0;
        float next_av = 0.0f;
        const int next_idx = entry_idx + 1;
        if (half_lane == 0) {
          next_gk = next_idx < end ? params.direct_row_ks[next_idx] : 0;
          next_av =
              next_idx < end ? direct_bf16_to_float(params.direct_row_values[next_idx]) : 0.0f;
        }
        next_gk = __shfl_sync(half_mask, next_gk, half_base_lane);
        next_av = __shfl_sync(half_mask, next_av, half_base_lane);

        const bool next_valid =
            half_lane < ColGroups && next_gk >= 0 && next_gk < params.k && next_idx < end;
        uint4 next_bv = make_uint4(0u, 0u, 0u, 0u);
        if (next_valid) {
          next_bv = __ldg(reinterpret_cast<const uint4*>(
              params.direct_b_comp + static_cast<int64_t>(next_gk) * params.n + global_col0));
        }

        if (cur_valid) {
          acc[0] += cur_av * direct_bf16_bits_to_float(cur_bv.x);
          acc[1] += cur_av * direct_bf16_bits_hi_to_float(cur_bv.x);
          acc[2] += cur_av * direct_bf16_bits_to_float(cur_bv.y);
          acc[3] += cur_av * direct_bf16_bits_hi_to_float(cur_bv.y);
          acc[4] += cur_av * direct_bf16_bits_to_float(cur_bv.z);
          acc[5] += cur_av * direct_bf16_bits_hi_to_float(cur_bv.z);
          acc[6] += cur_av * direct_bf16_bits_to_float(cur_bv.w);
          acc[7] += cur_av * direct_bf16_bits_hi_to_float(cur_bv.w);
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
          gk = params.direct_row_ks[entry_idx];
          av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
        }
        gk = __shfl_sync(half_mask, gk, half_base_lane);
        av = __shfl_sync(half_mask, av, half_base_lane);
        if (half_lane < ColGroups && gk >= 0 && gk < params.k) {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < params.n) {
              acc[cc] += av * direct_bf16_to_float(
                                  params.direct_b_comp[static_cast<int64_t>(gk) * params.n +
                                                       global_col]);
            }
          }
        }
      }
    }

    if (half_lane < ColGroups) {
      if (full_packed_tile) {
        const uint4 packed_delta =
            make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                       direct_pack_bf16x2(acc[2], acc[3]),
                       direct_pack_bf16x2(acc[4], acc[5]),
                       direct_pack_bf16x2(acc[6], acc[7]));
        const uint4 out_v = *reinterpret_cast<const uint4*>(
            static_cast<c10::BFloat16*>(params.D) + global_row * params.n + global_col0);
        *reinterpret_cast<uint4*>(
            static_cast<c10::BFloat16*>(params.D) + global_row * params.n + global_col0) =
            direct_bf16_add_packed_u4(out_v, packed_delta);
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            const int64_t out_idx = global_row * params.n + global_col;
            c10::BFloat16* output = static_cast<c10::BFloat16*>(params.D);
            const c10::BFloat16 delta_bf16 = direct_float_to_bf16(acc[cc]);
            output[out_idx] =
                direct_float_to_bf16(direct_bf16_to_float(output[out_idx]) +
                                     direct_bf16_to_float(delta_bf16));
          }
        }
      }
    }
  }
}

enum class WarpGroup : uint32_t {
  Producer = 0,
  Consumer0 = 1,
  Consumer1 = 2
};

enum class ProducerRole : uint32_t {
  ScaleA = 0,
  TMA = 1,
  ScaleB = 2,
  Scheduler = 3,
  None = 4
};

template <class StageData_, int Stages_, int Alignment = 128>
class DataPipe {
public:
  static constexpr int Stages = Stages_;
  using StageData = StageData_;
  using Pipeline = Pipeline<Stages>;
  struct alignas(Alignment) Storage {
    StageData stages[Stages];
  };
  DEVICE
  DataPipe(Storage &storage, Pipeline &&pipe, typename Pipeline::State &&state): storage_(storage), pipe(std::move(pipe)), state(std::move(state)) {}
  DEVICE
  StageData &data() { return storage_.stages[state.stage()]; }
  Pipeline pipe;
  typename Pipeline::State state;
private:
  Storage &storage_;
};

template<int CtaM, int CtaN, int CtaK, int EpiM, int EpiN, int AtomRepK>
struct alignas(128) MainloopStageData {
  alignas(128) uint8_t sA[CtaM * CtaK / 2];
  alignas(128) uint8_t sB[CtaN * CtaK / 2];
  alignas(4) uint32_t sSFA[CtaM * AtomRepK];
  alignas(4) uint32_t sSFB[CtaN * AtomRepK];
  float sOuterSFA[CtaM * CtaK / 128];
  float sOuterSFB[CtaN * CtaK / 128];
};

//using MainloopPipe = DataPipe<MainloopStageData, Stages, 16>;

struct alignas(16) CLCloopStageData {
  alignas(16) int4 clc_ret;
};

using CLCloopPipe = DataPipe<CLCloopStageData, 1, 16>;

template<int EpiM, int EpiN>
struct alignas(128) EpilogueStorage {
  alignas(128) uint16_t sC[EpiM * EpiN];
};

struct alignas(16) SchedulerScratch {
  alignas(8) uint64_t mbar;
  alignas(16) int4 clc_ret;
};

struct alignas(16) SparseProbeState {
  alignas(8) uint64_t compact_sync_mbar;
  uint32_t blk_m;
  uint32_t blk_n;
  uint32_t valid;
  uint32_t epoch;
  uint32_t done_count;
  uint32_t sc_ready;
  uint32_t merge_done;
  uint32_t stage_ready[4];
};

template<int CtaM, int MaxNnz>
DEVICE void stage_sparse_compact_inputs_for_tile(
    Params const& params,
    SparseCompactInputStorage<CtaM, MaxNnz>& compact,
    SparseProbeState& state,
    uint32_t blk_m,
    int sparse_thread_rank) {
  static_assert(MaxNnz > 0 && MaxNnz <= 16,
                "compact consumer expects a small compile-time row cap");
  const int local_row = sparse_thread_rank;
  if (local_row < CtaM) {
    const int64_t global_row = static_cast<int64_t>(blk_m) * CtaM + local_row;
    uint32_t count = 0;
    if (global_row < params.m && params.direct_row_offsets != nullptr &&
        params.direct_row_ks != nullptr && params.direct_row_values != nullptr) {
      const int32_t start = params.direct_row_offsets[global_row];
      const int32_t end = params.direct_row_offsets[global_row + 1];
      count = static_cast<uint32_t>(end > start ? end - start : 0);
      // The strict fast-path host gate guarantees count <= MaxNnz.  The
      // explicit tail-fallback entry point may pass a longer full payload, so
      // clamp its same-kernel prefix to the compiled record capacity.
      count = count > static_cast<uint32_t>(MaxNnz)
                  ? static_cast<uint32_t>(MaxNnz)
                  : count;
      #pragma unroll
      for (int item = 0; item < MaxNnz; ++item) {
        if (item < static_cast<int>(count)) {
          const int32_t pos = start + item;
          const uint32_t gk = static_cast<uint32_t>(params.direct_row_ks[pos]);
          const uint16_t value_bits =
              *reinterpret_cast<const uint16_t*>(params.direct_row_values + pos);
          compact.entries[local_row][item] =
              ((gk & 0xffffu) << 16) | static_cast<uint32_t>(value_bits);
        }
      }
    }
    compact.counts[local_row] = count;
  }

  // Only WG3 participates in barrier 5.  Once all 128 row records are visible,
  // one lane publishes a single ready flag consumed after the dense mainloop.
  ptx::bar_sync(5, NumThreadsPerWarpGroup);
  if (sparse_thread_rank == 0) {
    __threadfence_block();
    atomicExch(reinterpret_cast<unsigned int*>(&state.done_count), 1u);
  }
}

DEVICE void compact_cross_wg_barrier(
    uint64_t* mbar,
    uint32_t& phase,
    int lane_idx) {
  // One arrival per warp is enough once the warp has converged here.  The
  // second syncwarp keeps non-leaders parked until their leader observes all
  // twelve participating warps (WG1 + WG2 + WG3).
  __syncwarp();
  if (lane_idx == 0) {
    ptx::mbarrier_arrive(mbar);
    ptx::wait_barrier(mbar, phase);
    phase ^= 1;
  }
  __syncwarp();
}

template<int Rows, int CtaM, int CtaN>
struct alignas(128) SparseAccStorage {
  alignas(128) float acc[Rows][CtaN];
  alignas(16) int32_t row_to_slot[CtaM];
  alignas(16) int32_t slot_rows[Rows];
};

template<int CtaM, int CtaN, bool Enable>
struct alignas(128) SparseLocalDeltaStorage {
  static constexpr int StageBuffers = HANDWRITTEN_TMA_LOCAL_DELTA_STAGE_BUFFERS;
  alignas(128) uint16_t tile[Enable ? StageBuffers * CtaM * CtaN : 1];
  alignas(128) float partials[Enable ? 4 * CtaN : 1];
  alignas(16) uint32_t row_mask[Enable ? ((CtaM + 31) / 32) : 1];
};

template<int CtaM, int CtaN, int CtaK, int EpiM, int EpiN, int AtomRepK, bool EnableLocalDelta = false>
struct SmemStorageT {
  using MainloopPipe = DataPipe<MainloopStageData<CtaM, CtaN, CtaK, EpiM, EpiN, AtomRepK>, Stages, 16>;
  struct alignas(16) PipelineStorage {
    alignas(8) typename MainloopPipe::Pipeline::Storage mainloop;
    alignas(8) CLCloopPipe::Pipeline::Storage clcloop;
  };
  struct DataStorage {
    alignas(128) typename MainloopPipe::Storage mainloop;
    alignas(128) CLCloopPipe::Storage clcloop;
    alignas(128) EpilogueStorage<EpiM, EpiN> epilogue;
  };
  alignas(16) PipelineStorage pipeline;
  alignas(128) DataStorage data;
  alignas(16) SchedulerScratch scheduler;
  alignas(16) SparseProbeState sparse_probe;
  union alignas(128) {
    SparseAccStorage<HANDWRITTEN_TMA_SPARSE_ACC_ROWS, CtaM, CtaN> sparse_acc;
    SparseLocalDeltaStorage<CtaM, EpiN, EnableLocalDelta> sparse_local_delta;
    SparseCompactInputStorage<CtaM, HANDWRITTEN_TMA_COMPACT_CONSUMER_MAX_NNZ>
        sparse_compact_input[2];
  };
};

template<int Items, class CompactStorage>
DEVICE void apply_compact_consumer_row(
    CompactStorage const& compact,
    Params const& params,
    uint32_t compact_count,
    int local_row,
    int global_col_base,
    int row_fragment,
    float alpha,
    float const* reg_c,
    uint32_t* reg_d) {
  static_assert(Items >= 1 && Items <= 16,
                "compact consumer row helper supports caps in [1, 16]");
  constexpr int StaticN = HANDWRITTEN_TMA_COMPACT_CONSUMER_STATIC_N;
  uint32_t compact_entries[Items];
  float compact_values[Items];
  #pragma unroll
  for (int item = 0; item < Items; ++item) {
    uint32_t packed = 0;
    if (item < static_cast<int>(compact_count)) {
      packed = compact.entries[local_row][item];
    }
    compact_entries[item] = packed;
    compact_values[item] = direct_bf16_bits_to_float(packed);
  }
  #pragma unroll
  for (int col_fragment = 0; col_fragment < 4; ++col_fragment) {
    const int k_pair = row_fragment + col_fragment * 4;
    const int global_col0 = global_col_base + col_fragment * 8;
    float delta_lo = 0.0f;
    float delta_hi = 0.0f;
    #pragma unroll
    for (int item = 0; item < Items; ++item) {
      if (item < static_cast<int>(compact_count)) {
        const uint32_t gk = compact_entries[item] >> 16;
        const uint32_t b_pair =
            *reinterpret_cast<const uint32_t*>(
                params.direct_b_comp +
                static_cast<int64_t>(gk) * StaticN + global_col0);
        delta_lo = fmaf(
            compact_values[item],
            direct_bf16_bits_to_float(b_pair),
            delta_lo);
        delta_hi = fmaf(
            compact_values[item],
            direct_bf16_bits_hi_to_float(b_pair),
            delta_hi);
      }
    }
    reg_d[k_pair] = ptx::cvt_fp32_to_bf16x2(
        alpha * reg_c[k_pair * 2] + delta_lo,
        alpha * reg_c[k_pair * 2 + 1] + delta_hi);
  }
}

template<int CtaM, int CtaN>
DEVICE void apply_sparse_rowblock_nblock_delta_persistent_tasks(Params const& params,
                                                                SparseProbeState& state,
                                                                int lane_idx,
                                                                int sparse_warp_rank,
                                                                int sparse_warps) {
  if (params.direct_probe_sink == nullptr ||
      params.direct_probe_counter == nullptr ||
      params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_row_values == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int RowsPerTask = 8;
  constexpr int VecN = 8;
  constexpr int ColGroups = 16;
  constexpr int TaskN = ColGroups * VecN;
  static_assert(CtaN % TaskN == 0 || CtaN == TaskN,
                "persistent rowblock side path assumes a 128-column sparse N task");

  if (sparse_warps != 4) {
    return;
  }

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  constexpr unsigned HalfMask = 0x0000ffffu;
  const unsigned half_mask = HalfMask << half_base_lane;
  const int local_row = sparse_warp_rank * 2 + half;
  const bool use_active_rowblocks =
      params.direct_probe_active_mblocks != nullptr &&
      params.direct_probe_active_mblock_count > 0;
  const int64_t rowblock_count =
      use_active_rowblocks
          ? static_cast<int64_t>(params.direct_probe_active_mblock_count)
          : (static_cast<int64_t>(params.m) + RowsPerTask - 1) / RowsPerTask;
  const int64_t nblock_count = (static_cast<int64_t>(params.n) + TaskN - 1) / TaskN;
  const int64_t total_tasks = rowblock_count * nblock_count;
  float lane_sum = 0.0f;

  while (true) {
    if (sparse_warp_rank == 0 && lane_idx == 0) {
      const int32_t task = atomicAdd(params.direct_probe_counter, 1);
      state.blk_m = static_cast<uint32_t>(task);
    }
    ptx::bar_sync(5, sparse_warps * NumThreadsPerWarp);

    const int64_t task_id = static_cast<int64_t>(state.blk_m);
    if (task_id >= total_tasks) {
      break;
    }

    const int64_t base_task_id = task_id >> 2;
    const int row_pair_for_task = static_cast<int>(task_id & 3);
    const int64_t rowblock_task = base_task_id / nblock_count;
    const int64_t rowblock =
        use_active_rowblocks
            ? static_cast<int64_t>(params.direct_probe_active_mblocks[rowblock_task])
            : rowblock_task;
    const int64_t nblock = task_id - rowblock_task * nblock_count;
    const int64_t global_row = rowblock * RowsPerTask + local_row;
    const int64_t global_col0 = nblock * TaskN + half_lane * VecN;
    if (global_row >= params.m) {
      ptx::bar_sync(5, sparse_warps * NumThreadsPerWarp);
      continue;
    }

    int start = 0;
    int end = 0;
    if (half_lane == 0) {
      start = params.direct_row_offsets[global_row];
      end = params.direct_row_offsets[global_row + 1];
    }
    start = __shfl_sync(half_mask, start, half_base_lane);
    end = __shfl_sync(half_mask, end, half_base_lane);
    if (start == end) {
      ptx::bar_sync(5, sparse_warps * NumThreadsPerWarp);
      continue;
    }

    float acc[VecN] = {};
    const bool full_packed_tile = ((params.n & 7) == 0) && (global_col0 + VecN <= params.n);
    if (full_packed_tile) {
      int cur_gk = 0;
      float cur_av = 0.0f;
      if (half_lane == 0) {
        cur_gk = params.direct_row_ks[start];
        cur_av = direct_bf16_to_float(params.direct_row_values[start]);
      }
      cur_gk = __shfl_sync(half_mask, cur_gk, half_base_lane);
      cur_av = __shfl_sync(half_mask, cur_av, half_base_lane);

      bool cur_valid = cur_gk >= 0 && cur_gk < params.k;
      uint4 cur_bv = make_uint4(0u, 0u, 0u, 0u);
      if (cur_valid) {
        cur_bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(cur_gk) * params.n + global_col0));
      }

      for (int entry_idx = start; entry_idx < end; ++entry_idx) {
        int next_gk = 0;
        float next_av = 0.0f;
        const int next_idx = entry_idx + 1;
        if (half_lane == 0) {
          next_gk = next_idx < end ? params.direct_row_ks[next_idx] : 0;
          next_av =
              next_idx < end ? direct_bf16_to_float(params.direct_row_values[next_idx]) : 0.0f;
        }
        next_gk = __shfl_sync(half_mask, next_gk, half_base_lane);
        next_av = __shfl_sync(half_mask, next_av, half_base_lane);

        const bool next_valid = next_gk >= 0 && next_gk < params.k && next_idx < end;
        uint4 next_bv = make_uint4(0u, 0u, 0u, 0u);
        if (next_valid) {
          next_bv = __ldg(reinterpret_cast<const uint4*>(
              params.direct_b_comp + static_cast<int64_t>(next_gk) * params.n + global_col0));
        }

        if (cur_valid) {
          acc[0] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.x), acc[0]);
          acc[1] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.x), acc[1]);
          acc[2] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.y), acc[2]);
          acc[3] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.y), acc[3]);
          acc[4] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.z), acc[4]);
          acc[5] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.z), acc[5]);
          acc[6] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.w), acc[6]);
          acc[7] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.w), acc[7]);
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
          gk = params.direct_row_ks[entry_idx];
          av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
        }
        gk = __shfl_sync(half_mask, gk, half_base_lane);
        av = __shfl_sync(half_mask, av, half_base_lane);
        if (gk >= 0 && gk < params.k) {
          #pragma unroll
          for (int cc = 0; cc < VecN; ++cc) {
            const int64_t global_col = global_col0 + cc;
            if (global_col < params.n) {
              acc[cc] = fmaf(
                  av,
                  direct_bf16_to_float(
                      params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                  acc[cc]);
            }
          }
        }
      }
    }

    if (params.direct_delta_output != nullptr) {
      c10::BFloat16* delta_output = params.direct_delta_output;
      if (full_packed_tile) {
        const uint4 packed_delta =
            make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                       direct_pack_bf16x2(acc[2], acc[3]),
                       direct_pack_bf16x2(acc[4], acc[5]),
                       direct_pack_bf16x2(acc[6], acc[7]));
        *reinterpret_cast<uint4*>(
            delta_output + global_row * params.n + global_col0) = packed_delta;
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            delta_output[global_row * params.n + global_col] = direct_float_to_bf16(acc[cc]);
          }
        }
      }
    }

    #pragma unroll
    for (int cc = 0; cc < VecN; ++cc) {
      lane_sum += acc[cc];
    }
    ptx::bar_sync(5, sparse_warps * NumThreadsPerWarp);
  }

  lane_sum = warp_reduce_sum_f32(lane_sum);
  if (lane_idx == 0) {
    atomicAdd(params.direct_probe_sink + sparse_warp_rank, lane_sum);
  }
}

DEVICE float apply_sparse_row_nblock_delta_known_range(Params const& params,
                                                       int half_lane,
                                                       int half_base_lane,
                                                       unsigned half_mask,
                                                       int start,
                                                       int end,
                                                       int64_t global_row,
                                                       int64_t global_col0) {
  constexpr int VecN = 8;
  float acc[VecN] = {};
  const bool full_packed_tile = ((params.n & 7) == 0) && (global_col0 + VecN <= params.n);
  if (full_packed_tile) {
    int cur_gk = 0;
    float cur_av = 0.0f;
    if (half_lane == 0) {
      cur_gk = params.direct_row_ks[start];
      cur_av = direct_bf16_to_float(params.direct_row_values[start]);
    }
    cur_gk = __shfl_sync(half_mask, cur_gk, half_base_lane);
    cur_av = __shfl_sync(half_mask, cur_av, half_base_lane);

    bool cur_valid = cur_gk >= 0 && cur_gk < params.k;
    uint4 cur_bv = make_uint4(0u, 0u, 0u, 0u);
    if (cur_valid) {
      cur_bv = __ldg(reinterpret_cast<const uint4*>(
          params.direct_b_comp + static_cast<int64_t>(cur_gk) * params.n + global_col0));
    }

    for (int entry_idx = start; entry_idx < end; ++entry_idx) {
      int next_gk = 0;
      float next_av = 0.0f;
      const int next_idx = entry_idx + 1;
      if (half_lane == 0) {
        next_gk = next_idx < end ? params.direct_row_ks[next_idx] : 0;
        next_av =
            next_idx < end ? direct_bf16_to_float(params.direct_row_values[next_idx]) : 0.0f;
      }
      next_gk = __shfl_sync(half_mask, next_gk, half_base_lane);
      next_av = __shfl_sync(half_mask, next_av, half_base_lane);

      const bool next_valid = next_gk >= 0 && next_gk < params.k && next_idx < end;
      uint4 next_bv = make_uint4(0u, 0u, 0u, 0u);
      if (next_valid) {
        next_bv = __ldg(reinterpret_cast<const uint4*>(
            params.direct_b_comp + static_cast<int64_t>(next_gk) * params.n + global_col0));
      }

      if (cur_valid) {
        acc[0] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.x), acc[0]);
        acc[1] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.x), acc[1]);
        acc[2] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.y), acc[2]);
        acc[3] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.y), acc[3]);
        acc[4] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.z), acc[4]);
        acc[5] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.z), acc[5]);
        acc[6] = fmaf(cur_av, direct_bf16_bits_to_float(cur_bv.w), acc[6]);
        acc[7] = fmaf(cur_av, direct_bf16_bits_hi_to_float(cur_bv.w), acc[7]);
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
        gk = params.direct_row_ks[entry_idx];
        av = direct_bf16_to_float(params.direct_row_values[entry_idx]);
      }
      gk = __shfl_sync(half_mask, gk, half_base_lane);
      av = __shfl_sync(half_mask, av, half_base_lane);
      if (gk >= 0 && gk < params.k) {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t global_col = global_col0 + cc;
          if (global_col < params.n) {
            acc[cc] = fmaf(
                av,
                direct_bf16_to_float(
                    params.direct_b_comp[static_cast<int64_t>(gk) * params.n + global_col]),
                acc[cc]);
          }
        }
      }
    }
  }

  if (params.direct_delta_output != nullptr) {
    c10::BFloat16* delta_output = params.direct_delta_output;
    if (full_packed_tile) {
      const uint4 packed_delta =
          make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                     direct_pack_bf16x2(acc[2], acc[3]),
                     direct_pack_bf16x2(acc[4], acc[5]),
                     direct_pack_bf16x2(acc[6], acc[7]));
      *reinterpret_cast<uint4*>(
          delta_output + global_row * params.n + global_col0) = packed_delta;
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t global_col = global_col0 + cc;
        if (global_col < params.n) {
          delta_output[global_row * params.n + global_col] = direct_float_to_bf16(acc[cc]);
        }
      }
    }
  }

  float lane_sum = 0.0f;
  #pragma unroll
  for (int cc = 0; cc < VecN; ++cc) {
    lane_sum += acc[cc];
  }
  return lane_sum;
}

template<int CtaM, int CtaN>
DEVICE void apply_sparse_rowblock_nblock_delta_persistent_warp_tasks(Params const& params,
                                                                     SparseProbeState& state,
                                                                     int lane_idx,
                                                                     int sparse_warp_rank,
                                                                     int sparse_warps) {
  if (params.direct_probe_sink == nullptr ||
      params.direct_probe_counter == nullptr ||
      params.direct_row_offsets == nullptr ||
      params.direct_row_ks == nullptr ||
      params.direct_row_values == nullptr ||
      params.direct_b_comp == nullptr) {
    return;
  }

  constexpr int RowsPerTask = 8;
  constexpr int VecN = 8;
  constexpr int ColGroups = 16;
  constexpr int TaskN = ColGroups * VecN;
  constexpr int TaskChunk = HANDWRITTEN_TMA_PERSISTENT_ROWBLOCK_CHUNK;
  constexpr int NGroup = HANDWRITTEN_TMA_PERSISTENT_ROWPAIR_NGROUP;
  static_assert(TaskChunk > 0, "persistent rowblock chunk must be positive");
  static_assert(NGroup > 0, "persistent row-pair N group must be positive");
  static_assert(CtaN % TaskN == 0 || CtaN == TaskN,
                "persistent warp rowblock side path assumes a 128-column sparse N task");

  if (sparse_warps != 4) {
    return;
  }

  const int half = lane_idx >> 4;
  const int half_lane = lane_idx & 15;
  const int half_base_lane = half << 4;
  constexpr unsigned HalfMask = 0x0000ffffu;
  const unsigned half_mask = HalfMask << half_base_lane;
  const int64_t max_rowblock_count =
      (static_cast<int64_t>(params.m) + RowsPerTask - 1) / RowsPerTask;
  const bool use_task_records =
      params.direct_probe_active_mblocks != nullptr &&
      params.direct_probe_active_mblock_count > max_rowblock_count;
  const bool use_active_rowblocks =
      !use_task_records &&
      params.direct_probe_active_mblocks != nullptr &&
      params.direct_probe_active_mblock_count > 0;
  const int64_t rowblock_count =
      use_active_rowblocks
          ? static_cast<int64_t>(params.direct_probe_active_mblock_count)
          : max_rowblock_count;
  const int64_t nblock_count = (static_cast<int64_t>(params.n) + TaskN - 1) / TaskN;
  const int64_t rowblock_nblock_tasks = rowblock_count * nblock_count;
  const int64_t nblock_group_count =
      (nblock_count + static_cast<int64_t>(NGroup) - 1) / static_cast<int64_t>(NGroup);
  const int64_t total_tasks =
      use_task_records
          ? static_cast<int64_t>(params.direct_probe_active_mblock_count)
          : rowblock_count * nblock_group_count * 4;
  (void)state;
  float lane_sum = 0.0f;

  while (true) {
    uint32_t task = 0u;
    if (lane_idx == 0) {
      task = static_cast<uint32_t>(atomicAdd(params.direct_probe_counter, TaskChunk));
    }
    task = __shfl_sync(0xffffffffu, task, 0);
    const int64_t task_id = static_cast<int64_t>(task);
    if (task_id >= total_tasks) {
      break;
    }

    const int64_t chunk_end =
        min(task_id + static_cast<int64_t>(TaskChunk), total_tasks);
    for (int64_t subtask_id = task_id; subtask_id < chunk_end; ++subtask_id) {
      int row_pair_for_task = 0;
      int64_t rowblock = 0;
      int64_t nblock_begin = 0;
      int64_t nblock_end = 0;
      if (use_task_records) {
        const uint32_t record =
            static_cast<uint32_t>(params.direct_probe_active_mblocks[subtask_id]);
        row_pair_for_task = static_cast<int>(record & 3u);
        nblock_begin = static_cast<int64_t>((record >> 2) & 0x0fffu);
        nblock_end = min(nblock_begin + 1, nblock_count);
        rowblock = static_cast<int64_t>(record >> 14);
      } else {
        const uint32_t logical_task_id = static_cast<uint32_t>(subtask_id >> 2);
        row_pair_for_task = static_cast<int>(subtask_id & 3);
        uint32_t rowblock_task = 0u;
        uint32_t nblock_group = 0u;
        if (nblock_group_count == 8) {
          rowblock_task = logical_task_id >> 3;
          nblock_group = logical_task_id & 7u;
        } else if (nblock_group_count == 4) {
          rowblock_task = logical_task_id >> 2;
          nblock_group = logical_task_id & 3u;
        } else if (nblock_group_count == 16) {
          rowblock_task = logical_task_id >> 4;
          nblock_group = logical_task_id & 15u;
        } else {
          rowblock_task = static_cast<uint32_t>(
              static_cast<int64_t>(logical_task_id) / nblock_group_count);
          nblock_group = static_cast<uint32_t>(
              static_cast<int64_t>(logical_task_id) -
              static_cast<int64_t>(rowblock_task) * nblock_group_count);
        }
        rowblock =
            use_active_rowblocks
                ? static_cast<int64_t>(params.direct_probe_active_mblocks[rowblock_task])
                : static_cast<int64_t>(rowblock_task);
        nblock_begin = static_cast<int64_t>(nblock_group) * static_cast<int64_t>(NGroup);
        nblock_end = min(nblock_begin + static_cast<int64_t>(NGroup), nblock_count);
      }

      #pragma unroll
      for (int row_pair = row_pair_for_task; row_pair < 4; row_pair += 4) {
        const int local_row = row_pair * 2 + half;
        const int64_t global_row = rowblock * RowsPerTask + local_row;
        if (global_row >= params.m) {
          continue;
        }

        int start = 0;
        int end = 0;
        if (half_lane == 0) {
          start = params.direct_row_offsets[global_row];
          end = params.direct_row_offsets[global_row + 1];
        }
        start = __shfl_sync(half_mask, start, half_base_lane);
        end = __shfl_sync(half_mask, end, half_base_lane);
        if (start == end) {
          continue;
        }

        for (int64_t nblock = nblock_begin; nblock < nblock_end; ++nblock) {
          const int64_t global_col0 = nblock * TaskN + half_lane * VecN;
          lane_sum += apply_sparse_row_nblock_delta_known_range(
              params,
              half_lane,
              half_base_lane,
              half_mask,
              start,
              end,
              global_row,
              global_col0);
        }
      }
    }

  }

  lane_sum = warp_reduce_sum_f32(lane_sum);
  if (lane_idx == 0) {
    atomicAdd(params.direct_probe_sink + sparse_warp_rank, lane_sum);
  }
}

struct WorkTileInfo {
  uint32_t blk_m;
  uint32_t blk_n;
  uint32_t valid;
};

template <int bytes, typename T, typename U>
DEVICE void load(T* dst, const U* src) {
  if constexpr (bytes == 4) {
    *(float*)dst = *(const float*)src;
  } 
  else if constexpr (bytes == 8) {
    *(float2*)dst = *(const float2*)src;
  } 
  else if constexpr (bytes == 16) {
    *(float4*)dst = *(const float4*)src;
  } 
  else {
    static_assert(bytes == 4, "Unsupported byte size for load!");
  }
}

DEVICE void update_work_info(dim3 coord, bool valid, WorkTileInfo& work_info) {
  work_info.blk_m = coord.y;
  work_info.blk_n = coord.x;
  work_info.valid = valid;
}

template<
    int CtaM,
    int CtaN,
    int CtaK,
    int EpiM,
    int EpiN,
    int LaunchThreads,
    bool EnableLocalDelta = false,
    bool Mode49Specialized = false,
    bool Mode49StaticN4096 = false,
    bool CompactConsumerSpecialized = false>
__launch_bounds__(LaunchThreads, 1)
__global__ void nvfp4_gemm(
  __grid_constant__ const Params params,
  __grid_constant__ const dim3 real_grid_dim
) {
  extern __shared__ uint8_t smem[];

  // 2 * 4 warps
  // m16n8k64 * 2 * 4
  constexpr int AtomRepM = CtaM / WorkerRepM / AtomM;
  constexpr int AtomRepN = CtaN / WorkerRepN / AtomN;
  constexpr int AtomRepK = CtaK / AtomK;
  constexpr int WorkerM = CtaM / WorkerRepM;
  constexpr int WorkerN = CtaN / WorkerRepN;

  // role
  int tidx = threadIdx.x;
  int lane_idx = threadIdx.x % NumThreadsPerWarp;
  int widx = threadIdx.x / NumThreadsPerWarp - 4;
  int wg_id = tidx / NumThreadsPerWarpGroup;
  ProducerRole role = static_cast<ProducerRole>(std::min(tidx / NumThreadsPerWarp, 4));

  // storage
  using SmemStorage = SmemStorageT<CtaM, CtaN, CtaK, EpiM, EpiN, AtomRepK, EnableLocalDelta>;
  using MainloopPipe = typename SmemStorage::MainloopPipe;
  SmemStorage& storage = *reinterpret_cast<SmemStorage*>(smem);

  // prefetch tensormap
  if (tidx == 0) {
    ptx::prefetch_tensormap(&params.tensormap_A);
    ptx::prefetch_tensormap(&params.tensormap_B);
    ptx::mbarrier_init(&storage.scheduler.mbar, 1);
    if constexpr (CompactConsumerSpecialized) {
      ptx::mbarrier_init(
          &storage.sparse_probe.compact_sync_mbar,
          3 * NumThreadsPerWarpGroup / NumThreadsPerWarp);
    }
  }

  __syncthreads();

  if constexpr (!Mode49Specialized) {
    if (params.direct_probe_mixed_cta != 0 &&
        blockIdx.y >= static_cast<uint32_t>(params.direct_probe_dense_grid_y)) {
    const int warp_id = threadIdx.x / NumThreadsPerWarp;
    while (true) {
      if (threadIdx.x == 0) {
        storage.sparse_probe.blk_m =
            static_cast<uint32_t>(atomicAdd(params.direct_probe_counter, 1));
      }
      __syncthreads();
      const int32_t tile_id = static_cast<int32_t>(storage.sparse_probe.blk_m);
      if (tile_id >= params.direct_probe_total_tiles) {
        break;
      }
      const uint32_t active_m_idx = static_cast<uint32_t>(tile_id / real_grid_dim.x);
      const uint32_t probe_blk_n = static_cast<uint32_t>(tile_id - active_m_idx * real_grid_dim.x);
      const uint32_t probe_blk_m =
          params.direct_probe_active_mblocks != nullptr
              ? static_cast<uint32_t>(params.direct_probe_active_mblocks[active_m_idx])
              : active_m_idx;
      if (params.direct_probe_do_math != 0) {
        if (params.direct_probe_kmajor == 2) {
          if (params.direct_delta_output != nullptr) {
            apply_sparse_loadfma_probe_kmajor_tile<CtaM, CtaN, true>(
                params,
                active_m_idx,
                probe_blk_m,
                probe_blk_n,
                lane_idx,
                warp_id,
                12,
                0);
          } else {
            apply_sparse_loadfma_probe_kmajor_tile<CtaM, CtaN, false>(
                params,
                active_m_idx,
                probe_blk_m,
                probe_blk_n,
                lane_idx,
                warp_id,
                12,
                0);
          }
          apply_sparse_loadfma_probe_tile<CtaM, CtaN>(
              params,
              probe_blk_m,
              probe_blk_n,
              lane_idx,
              warp_id,
              12,
              12);
        } else if (params.direct_probe_kmajor != 0) {
          if (params.direct_delta_output != nullptr) {
            apply_sparse_loadfma_probe_kmajor_tile<CtaM, CtaN, true>(
                params,
                active_m_idx,
                probe_blk_m,
                probe_blk_n,
                lane_idx,
                warp_id,
                12,
                0);
          } else {
            apply_sparse_loadfma_probe_kmajor_tile<CtaM, CtaN, false>(
                params,
                active_m_idx,
                probe_blk_m,
                probe_blk_n,
                lane_idx,
                warp_id,
                12,
                0);
          }
        } else {
          apply_sparse_loadfma_probe_tile<CtaM, CtaN>(
              params,
              probe_blk_m,
              probe_blk_n,
              lane_idx,
              warp_id,
              12,
              0);
        }
      } else if (lane_idx == 0) {
        params.direct_probe_sink[static_cast<int64_t>(tile_id) * 12 + warp_id] = 1.0f;
      }
      __syncthreads();
    }
      return;
    }
  }

  WorkTileInfo work_info;
  update_work_info(get_swizzled_blk_coord(blockIdx), 1, work_info);
  if constexpr (!Mode49Specialized || CompactConsumerSpecialized) {
    if (tidx == 0 &&
        (CompactConsumerSpecialized || params.direct_probe_sink != nullptr ||
         (params.force_dense_4wg != 0 && params.direct_row_offsets != nullptr &&
          params.direct_active_row_offsets != nullptr && params.direct_active_rows != nullptr &&
          params.direct_b_comp != nullptr))) {
      atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.blk_m),
                 static_cast<unsigned int>(work_info.blk_m));
      atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.blk_n),
                 static_cast<unsigned int>(work_info.blk_n));
      atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.valid),
                 work_info.blk_m < real_grid_dim.y ? 1u : 0u);
	      atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.done_count), 0u);
	      atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.epoch), 0u);
	      atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.sc_ready), 0u);
	      atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.merge_done), 0u);
	      atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.stage_ready[0]), 0u);
	      atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.stage_ready[1]), 0u);
	      atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.stage_ready[2]), 0u);
	      atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.stage_ready[3]), 0u);
	      __threadfence_block();
	    }
	    __syncthreads();
  }
	  if (work_info.blk_m >= real_grid_dim.y) return;

#if HANDWRITTEN_TMA_PHASE_TRACE
	  if (tidx == 0 && phase_trace_cta_enabled(params)) {
	    phase_trace_write(params, PhaseTraceKernelEntry, phase_trace_clock());
	    phase_trace_write(params, PhaseTraceTileM, static_cast<uint64_t>(work_info.blk_m));
	    phase_trace_write(params, PhaseTraceTileN, static_cast<uint64_t>(work_info.blk_n));
	    phase_trace_write(params, PhaseTraceMixedCta, static_cast<uint64_t>(params.direct_probe_mixed_cta));
	    phase_trace_write(params, PhaseTraceProbeWarps, static_cast<uint64_t>(params.direct_probe_warps));
	    phase_trace_write(params, PhaseTraceKStages, static_cast<uint64_t>(params.k / CtaK));
	    phase_trace_write(params, PhaseTraceValid, 1);
	  }
#endif

	  const bool force_dense_4wg_mode = params.force_dense_4wg != 0;
  const bool direct_tailassist_4wg =
      force_dense_4wg_mode && params.direct_probe_sink == nullptr &&
      params.direct_row_offsets != nullptr && params.direct_active_row_offsets != nullptr &&
      params.direct_active_rows != nullptr && params.direct_b_comp != nullptr;
  const bool four_wg_reg_mode =
      Mode49Specialized || force_dense_4wg_mode ||
      params.direct_probe_mixed_cta == 4 ||
      params.direct_probe_mixed_cta == 9 ||
      params.direct_probe_mixed_cta == 10 ||
      params.direct_probe_mixed_cta == 11 ||
      params.direct_probe_mixed_cta == 12 ||
      params.direct_probe_mixed_cta == 13 ||
      params.direct_probe_mixed_cta == 14 ||
      params.direct_probe_mixed_cta == 15 ||
      params.direct_probe_mixed_cta == 16 ||
      params.direct_probe_mixed_cta == 17 ||
	      params.direct_probe_mixed_cta == 18 ||
	      params.direct_probe_mixed_cta == 19 ||
	      params.direct_probe_mixed_cta == 20 ||
	      params.direct_probe_mixed_cta == 21 ||
	      params.direct_probe_mixed_cta == 22 ||
			      params.direct_probe_mixed_cta == 23 ||
			      params.direct_probe_mixed_cta == 24 ||
			      params.direct_probe_mixed_cta == 25 ||
			      params.direct_probe_mixed_cta == 26 ||
			      params.direct_probe_mixed_cta == 27 ||
			      params.direct_probe_mixed_cta == 28 ||
			      params.direct_probe_mixed_cta == 29 ||
			      params.direct_probe_mixed_cta == 30 ||
			      params.direct_probe_mixed_cta == 31 ||
				      params.direct_probe_mixed_cta == 32 ||
				      params.direct_probe_mixed_cta == 33 ||
				      params.direct_probe_mixed_cta == 34 ||
				      params.direct_probe_mixed_cta == 35 ||
				      params.direct_probe_mixed_cta == 36 ||
				      params.direct_probe_mixed_cta == 37 ||
				      params.direct_probe_mixed_cta == 38 ||
				      params.direct_probe_mixed_cta == 39 ||
      params.direct_probe_mixed_cta == 40 ||
      params.direct_probe_mixed_cta == 41 ||
      params.direct_probe_mixed_cta == 42 ||
      params.direct_probe_mixed_cta == 43 ||
      params.direct_probe_mixed_cta == 44 ||
      params.direct_probe_mixed_cta == 45 ||
      params.direct_probe_mixed_cta == 46 ||
      params.direct_probe_mixed_cta == 47 ||
      params.direct_probe_mixed_cta == 48 ||
      params.direct_probe_mixed_cta == 49;
  if constexpr (LaunchThreads > NumThreadsPerWarpGroup * 3) {
    if (four_wg_reg_mode) {
      if constexpr (LaunchThreads == NumThreadsPerWarpGroup * 5) {
        if (wg_id == 0) {
          ptx::warpgroup_reg_dealloc<HANDWRITTEN_TMA_FORCE5WG_PRODUCER_REGS>();
        } else if (wg_id >= 3) {
          ptx::warpgroup_reg_dealloc<HANDWRITTEN_TMA_FORCE5WG_SPARSE_REGS>();
        }
      } else if (wg_id == 0) {
        ptx::warpgroup_reg_dealloc<HANDWRITTEN_TMA_FORCE4WG_LOW_REGS>();
      } else if (wg_id >= 3) {
        ptx::warpgroup_reg_dealloc<HANDWRITTEN_TMA_FORCE4WG_SPARSE_REGS>();
      }
      ptx::bar_sync(4, LaunchThreads);
      if (wg_id >= 3 && force_dense_4wg_mode && !direct_tailassist_4wg) {
        return;
      }
    }
  }

  if constexpr (Mode49Specialized) {
    if (wg_id >= 3) {
      if constexpr (!CompactConsumerSpecialized) {
        const int32_t sparse_worker_count = params.direct_probe_persistent_cta_count;
        const int32_t sparse_worker_id =
            static_cast<int32_t>(blockIdx.y * gridDim.x + blockIdx.x);
        if (sparse_worker_id >= sparse_worker_count) {
          return;
        }
        const int32_t sparse_thread_rank =
            static_cast<int32_t>(threadIdx.x) - 3 * NumThreadsPerWarpGroup;
        constexpr int SparseThreadsPerCta =
            LaunchThreads - 3 * NumThreadsPerWarpGroup;
        if constexpr (Mode49StaticN4096) {
          apply_sparse_active_row_vec8_delta_worker<
              true, true, false, SparseThreadsPerCta, true>(
              params,
              sparse_worker_id,
              sparse_worker_count,
              sparse_thread_rank);
        } else {
          apply_sparse_active_row_vec8_delta_worker<
              true, true, false, SparseThreadsPerCta, false>(
              params,
              sparse_worker_id,
              sparse_worker_count,
              sparse_thread_rank);
        }
        return;
      }
    }
  } else {
  if (wg_id >= 3 && direct_tailassist_4wg) {
    volatile uint32_t* done =
        reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.done_count);
    while (*done == 0u) {
    }
    __threadfence_block();
    apply_direct_add_sparse_tile<CtaM, CtaN>(
        params,
        work_info.blk_m,
        work_info.blk_n,
        widx,
        lane_idx,
        12);
    return;
  }

  if (wg_id >= 3 && params.direct_probe_sink != nullptr) {
    if (!four_wg_reg_mode) {
#ifndef HANDWRITTEN_TMA_SPARSE_WG_NO_SETMAXNREG
      ptx::warpgroup_reg_dealloc<HANDWRITTEN_TMA_PRODUCER_REGS>();
#endif
    }
    const int sparse_warp_rank = threadIdx.x / NumThreadsPerWarp - 12;
    if ((params.direct_probe_mixed_cta == 4 ||
         params.direct_probe_mixed_cta == 9 ||
         params.direct_probe_mixed_cta == 11 ||
         params.direct_probe_mixed_cta == 12 ||
         params.direct_probe_mixed_cta == 13 ||
         params.direct_probe_mixed_cta == 14 ||
         params.direct_probe_mixed_cta == 15 ||
         params.direct_probe_mixed_cta == 16 ||
         params.direct_probe_mixed_cta == 17 ||
	         params.direct_probe_mixed_cta == 18 ||
	         params.direct_probe_mixed_cta == 19 ||
	         params.direct_probe_mixed_cta == 20 ||
	         params.direct_probe_mixed_cta == 21 ||
	         params.direct_probe_mixed_cta == 22 ||
			         params.direct_probe_mixed_cta == 23 ||
			         params.direct_probe_mixed_cta == 24 ||
			         params.direct_probe_mixed_cta == 25 ||
			         params.direct_probe_mixed_cta == 26 ||
			         params.direct_probe_mixed_cta == 27 ||
			         params.direct_probe_mixed_cta == 28 ||
			         params.direct_probe_mixed_cta == 29 ||
			         params.direct_probe_mixed_cta == 30 ||
			         params.direct_probe_mixed_cta == 31 ||
				         params.direct_probe_mixed_cta == 32 ||
				         params.direct_probe_mixed_cta == 33 ||
				         params.direct_probe_mixed_cta == 34 ||
				         params.direct_probe_mixed_cta == 35 ||
				         params.direct_probe_mixed_cta == 36 ||
				         params.direct_probe_mixed_cta == 37 ||
				         params.direct_probe_mixed_cta == 38 ||
				         params.direct_probe_mixed_cta == 39 ||
		         params.direct_probe_mixed_cta == 40 ||
		         params.direct_probe_mixed_cta == 41 ||
		         params.direct_probe_mixed_cta == 42 ||
		         params.direct_probe_mixed_cta == 43 ||
		         params.direct_probe_mixed_cta == 44 ||
		         params.direct_probe_mixed_cta == 45 ||
		         params.direct_probe_mixed_cta == 46 ||
		         params.direct_probe_mixed_cta == 47 ||
		         params.direct_probe_mixed_cta == 48 ||
		         params.direct_probe_mixed_cta == 49) &&
	        sparse_warp_rank >= params.direct_probe_warps) {
      return;
    }
    if (params.direct_probe_mixed_cta == 10) {
      return;
    }
    if (params.direct_probe_mixed_cta == 41) {
      const int32_t tile_id =
          static_cast<int32_t>(work_info.blk_m * real_grid_dim.x + work_info.blk_n);
      if (params.ready_flags != nullptr) {
        while (atomicAdd(params.ready_flags + tile_id, 0) == 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
          __nanosleep(64);
#endif
        }
        __threadfence();
      } else {
        volatile uint32_t* done =
            reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.done_count);
        while (*done == 0u) {
        }
        __threadfence_block();
      }
      if (params.direct_probe_do_math != 0) {
        apply_direct_add_sparse_tile<CtaM, CtaN>(
            params,
            work_info.blk_m,
            work_info.blk_n,
            sparse_warp_rank,
            lane_idx,
            params.direct_probe_warps);
      } else if (lane_idx == 0) {
        params.direct_probe_sink[static_cast<int64_t>(tile_id) *
                                     params.direct_probe_warps +
                                 sparse_warp_rank] = 1.0f;
      }
      return;
    }
    if (params.direct_probe_mixed_cta == 42) {
      if (params.direct_probe_do_math != 0) {
        apply_sparse_rowblock_nblock_delta_persistent_warp_tasks<CtaM, CtaN>(
            params,
            storage.sparse_probe,
            lane_idx,
            sparse_warp_rank,
            params.direct_probe_warps);
      } else if (lane_idx == 0) {
        params.direct_probe_sink[sparse_warp_rank] = 1.0f;
      }
      return;
    }
    if (params.direct_probe_mixed_cta == 43 ||
        params.direct_probe_mixed_cta == 44 ||
        params.direct_probe_mixed_cta == 45 ||
        params.direct_probe_mixed_cta == 46 ||
        params.direct_probe_mixed_cta == 47 ||
        params.direct_probe_mixed_cta == 48 ||
        params.direct_probe_mixed_cta == 49) {
      const int32_t sparse_worker_count =
          params.direct_probe_persistent_cta_count > 0
              ? params.direct_probe_persistent_cta_count
              : static_cast<int32_t>(gridDim.x * gridDim.y);
      const int32_t sparse_worker_id =
          static_cast<int32_t>(blockIdx.y * gridDim.x + blockIdx.x);
      if (sparse_worker_id >= sparse_worker_count) {
        return;
      }
      if (params.direct_probe_do_math != 0) {
        const int32_t sparse_thread_rank =
            static_cast<int32_t>(threadIdx.x) - 3 * NumThreadsPerWarpGroup;
        if (params.direct_probe_mixed_cta == 44) {
          apply_sparse_active_row_vec8_delta_worker<false>(
              params,
              sparse_worker_id,
              sparse_worker_count,
              sparse_thread_rank);
        } else if (params.direct_probe_mixed_cta == 45) {
          apply_sparse_active_row_warp256_delta_worker<true>(
              params,
              sparse_worker_id,
              sparse_worker_count,
              sparse_thread_rank);
        } else if (params.direct_probe_mixed_cta == 46) {
          apply_sparse_active_row_warp256_delta_worker<false>(
              params,
              sparse_worker_id,
              sparse_worker_count,
              sparse_thread_rank);
        } else if (params.direct_probe_mixed_cta == 47) {
          apply_sparse_active_row_vec8_prefetch_delta_worker<true>(
              params,
              sparse_worker_id,
              sparse_worker_count,
              sparse_thread_rank);
        } else if (params.direct_probe_mixed_cta == 48) {
          apply_sparse_active_row_vec8_prefetch_delta_worker<false>(
              params,
              sparse_worker_id,
              sparse_worker_count,
              sparse_thread_rank);
        } else if (params.direct_probe_mixed_cta == 49) {
          constexpr int SparseThreadsPerCta =
              LaunchThreads > 3 * NumThreadsPerWarpGroup
                  ? LaunchThreads - 3 * NumThreadsPerWarpGroup
                  : NumThreadsPerWarpGroup;
          if (params.n == 4096) {
            apply_sparse_active_row_vec8_delta_worker<
                true, true, false, SparseThreadsPerCta, true>(
                params,
                sparse_worker_id,
                sparse_worker_count,
                sparse_thread_rank);
          } else {
            apply_sparse_active_row_vec8_delta_worker<
                true, true, false, SparseThreadsPerCta, false>(
                params,
                sparse_worker_id,
                sparse_worker_count,
                sparse_thread_rank);
          }
        } else {
          apply_sparse_active_row_vec8_delta_worker<true>(
              params,
              sparse_worker_id,
              sparse_worker_count,
              sparse_thread_rank);
        }
      } else if (lane_idx == 0) {
        params.direct_probe_sink[static_cast<int64_t>(sparse_worker_id) *
                                     params.direct_probe_warps +
                                 sparse_warp_rank] = 1.0f;
      }
      return;
    }
    if (params.direct_probe_mixed_cta == 40) {
      if (params.direct_probe_do_math != 0) {
        apply_sparse_packed_rowblock_nblock_delta_tasks<CtaM, CtaN>(
            params,
            work_info.blk_m,
            work_info.blk_n,
            real_grid_dim.y,
            real_grid_dim.x,
            lane_idx,
            sparse_warp_rank,
            params.direct_probe_warps);
      } else if (lane_idx == 0) {
        const int32_t tile_id =
            static_cast<int32_t>(work_info.blk_m * real_grid_dim.x + work_info.blk_n);
        params.direct_probe_sink[static_cast<int64_t>(tile_id) *
                                     params.direct_probe_warps +
                                 sparse_warp_rank] = 1.0f;
      }
      return;
    }
    if (params.direct_probe_mixed_cta == 38 || params.direct_probe_mixed_cta == 39) {
      if (params.direct_probe_do_math != 0) {
        apply_sparse_rowblock_nblock_delta_tasks<CtaM, CtaN>(
            params,
            work_info.blk_m,
            work_info.blk_n,
            real_grid_dim.y,
            real_grid_dim.x,
            lane_idx,
            sparse_warp_rank,
            params.direct_probe_warps);
      } else if (lane_idx == 0) {
        const int32_t tile_id =
            static_cast<int32_t>(work_info.blk_m * real_grid_dim.x + work_info.blk_n);
        params.direct_probe_sink[static_cast<int64_t>(tile_id) *
                                     params.direct_probe_warps +
                                 sparse_warp_rank] = 1.0f;
      }
      return;
    }
#if HANDWRITTEN_TMA_PREFETCH_TILE_META_BEFORE_TRACE
    const bool prefetch_tile_meta_stage_mode =
        params.direct_probe_mixed_cta == 32 ||
        params.direct_probe_mixed_cta == 34 ||
        params.direct_probe_mixed_cta == 35 ||
        params.direct_probe_mixed_cta == 36 ||
        params.direct_probe_mixed_cta == 37;
    const bool prefetched_tile_meta_valid =
        prefetch_tile_meta_stage_mode &&
        params.direct_kmajor_tile_group_meta != nullptr;
    uint64_t prefetched_tile_group_meta = 0ull;
    if (prefetched_tile_meta_valid) {
      prefetched_tile_group_meta =
          direct_kmajor_tile_group_meta_load(params, work_info.blk_m, lane_idx);
    }
#endif
    if (params.direct_probe_mixed_cta == 9 ||
        params.direct_probe_mixed_cta == 11 ||
        params.direct_probe_mixed_cta == 12 ||
        params.direct_probe_mixed_cta == 13 ||
        params.direct_probe_mixed_cta == 14 ||
        params.direct_probe_mixed_cta == 15 ||
        params.direct_probe_mixed_cta == 16 ||
        params.direct_probe_mixed_cta == 17 ||
	        params.direct_probe_mixed_cta == 18 ||
	        params.direct_probe_mixed_cta == 19 ||
	        params.direct_probe_mixed_cta == 20 ||
	        params.direct_probe_mixed_cta == 21 ||
	        params.direct_probe_mixed_cta == 22 ||
			        params.direct_probe_mixed_cta == 23 ||
			        params.direct_probe_mixed_cta == 24 ||
			        params.direct_probe_mixed_cta == 25 ||
			        params.direct_probe_mixed_cta == 26 ||
			        params.direct_probe_mixed_cta == 27 ||
			        params.direct_probe_mixed_cta == 28 ||
				        params.direct_probe_mixed_cta == 29 ||
				        params.direct_probe_mixed_cta == 30 ||
				        params.direct_probe_mixed_cta == 31 ||
				        params.direct_probe_mixed_cta == 32 ||
				        params.direct_probe_mixed_cta == 33 ||
				        params.direct_probe_mixed_cta == 34 ||
				        params.direct_probe_mixed_cta == 35 ||
				        params.direct_probe_mixed_cta == 36 ||
				        params.direct_probe_mixed_cta == 37) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	      uint64_t phase_sparse_total_start = 0;
	      uint64_t phase_sparse_setup_cursor = 0;
	      if (phase_trace_sparse_lane(params, sparse_warp_rank, lane_idx)) {
	        phase_sparse_total_start = phase_trace_clock();
	        phase_sparse_setup_cursor = phase_sparse_total_start;
	      }
#endif
	      if (params.direct_probe_do_math != 0) {
        const bool kmajor_stage_mode =
            params.direct_probe_mixed_cta == 32 ||
            params.direct_probe_mixed_cta == 34 ||
            params.direct_probe_mixed_cta == 35 ||
            params.direct_probe_mixed_cta == 36 ||
            params.direct_probe_mixed_cta == 37;
#if HANDWRITTEN_TMA_ASSUME_DENSE_TILE_META
        const bool kmajor_tile_meta_payload =
            kmajor_stage_mode && params.direct_kmajor_tile_group_meta != nullptr;
        int active_m_idx = static_cast<int>(work_info.blk_m);
#else
        const bool kmajor_tile_meta_payload =
            kmajor_stage_mode && direct_kmajor_has_tile_group_meta(params);
        int active_m_idx = -1;
        if (kmajor_tile_meta_payload) {
          active_m_idx = static_cast<int>(work_info.blk_m);
	        } else {
	          if (lane_idx == 0) {
	            if (params.direct_probe_active_mblocks != nullptr &&
	                params.direct_probe_active_mblock_count ==
	                    static_cast<int32_t>(real_grid_dim.y) &&
	                params.direct_probe_active_mblocks[work_info.blk_m] ==
	                    static_cast<int32_t>(work_info.blk_m)) {
	              active_m_idx = static_cast<int>(work_info.blk_m);
	            } else {
	              active_m_idx = find_active_mblock_index(params, work_info.blk_m);
	            }
	          }
          active_m_idx = __shfl_sync(0xffffffffu, active_m_idx, 0);
        }
#endif
        const bool fast_dense_tile_meta_setup =
            HANDWRITTEN_TMA_ASSUME_DENSE_TILE_META &&
            HANDWRITTEN_TMA_FAST_DENSE_TILE_META_SETUP &&
            kmajor_tile_meta_payload;
#if HANDWRITTEN_TMA_PHASE_TRACE
        if (!fast_dense_tile_meta_setup &&
            phase_trace_sparse_lane(params, sparse_warp_rank, lane_idx)) {
          const uint64_t phase_after_active_lookup = phase_trace_clock();
	          phase_trace_add(
	              params,
	              PhaseTraceSparseSetupActiveLookup,
	              phase_after_active_lookup - phase_sparse_setup_cursor);
	          phase_sparse_setup_cursor = phase_after_active_lookup;
	        }
#endif
	        if (params.direct_probe_mixed_cta == 21 ||
	            params.direct_probe_mixed_cta == 22 ||
			            params.direct_probe_mixed_cta == 23 ||
			            params.direct_probe_mixed_cta == 24 ||
			            params.direct_probe_mixed_cta == 25 ||
			            params.direct_probe_mixed_cta == 26 ||
			            params.direct_probe_mixed_cta == 27 ||
			            params.direct_probe_mixed_cta == 28 ||
			            params.direct_probe_mixed_cta == 29 ||
			            params.direct_probe_mixed_cta == 30 ||
			            params.direct_probe_mixed_cta == 31 ||
			            params.direct_probe_mixed_cta == 32 ||
			            params.direct_probe_mixed_cta == 33 ||
			            params.direct_probe_mixed_cta == 34 ||
			            params.direct_probe_mixed_cta == 35 ||
			            params.direct_probe_mixed_cta == 36 ||
			            params.direct_probe_mixed_cta == 37) {
	          const bool kmajor_stage_payload =
	              kmajor_stage_mode &&
	              active_m_idx >= 0 &&
	              (params.direct_kmajor_group_offsets != nullptr ||
	               kmajor_tile_meta_payload);
          int local_delta_reg_hot_count = 0;
          int kmajor_stage_group_start = 0;
          int kmajor_stage_hot_count = 0;
          if ((params.direct_probe_mixed_cta == 23 || kmajor_stage_payload) &&
              active_m_idx >= 0 &&
              (params.direct_kmajor_group_offsets != nullptr ||
               kmajor_tile_meta_payload)) {
            int group_count = 0;
#if HANDWRITTEN_TMA_ASSUME_DENSE_TILE_META
            if (fast_dense_tile_meta_setup) {
              uint64_t meta =
                  direct_kmajor_tile_group_meta_load(params, work_info.blk_m, lane_idx);
#if HANDWRITTEN_TMA_PREFETCH_TILE_META_BEFORE_TRACE
              if (prefetched_tile_meta_valid) {
                meta = prefetched_tile_group_meta;
              }
#endif
              kmajor_stage_group_start = static_cast<int>(meta & 0xffffffffull);
              group_count = static_cast<int>(meta >> 32);
            } else if (kmajor_tile_meta_payload) {
              const uint64_t meta =
                  direct_kmajor_tile_group_meta_load(params, work_info.blk_m, lane_idx);
              kmajor_stage_group_start = static_cast<int>(meta & 0xffffffffull);
              group_count = static_cast<int>(meta >> 32);
            } else
#endif
            direct_kmajor_tile_group_range(
                params,
                active_m_idx,
                work_info.blk_m,
                kmajor_stage_group_start,
                group_count);
            if (params.direct_probe_mixed_cta == 23) {
              local_delta_reg_hot_count = group_count;
            }
            if (kmajor_stage_payload) {
              kmajor_stage_hot_count = group_count;
	            }
	          }
#if HANDWRITTEN_TMA_PHASE_TRACE
	          if (!fast_dense_tile_meta_setup &&
	              phase_trace_sparse_lane(params, sparse_warp_rank, lane_idx)) {
	            const uint64_t phase_after_group_count = phase_trace_clock();
	            phase_trace_add(
	                params,
	                PhaseTraceSparseSetupGroupCount,
	                phase_after_group_count - phase_sparse_setup_cursor);
	            phase_sparse_setup_cursor = phase_after_group_count;
	          }
#endif
	          const bool local_delta_reg_block =
	              params.direct_probe_mixed_cta == 23 &&
	              local_delta_reg_hot_count > 0;
	          const bool prestore_local_delta_stage =
	              params.direct_probe_mixed_cta == 35;
	          const bool packed_local_delta_payload =
	              params.direct_packed_payload_mode != 0 &&
	              params.direct_packed_tile_offsets != nullptr &&
	              params.direct_packed_row_records != nullptr;
		          const int packed_local_delta_count =
		              packed_local_delta_payload
		                  ? params.direct_packed_tile_offsets[work_info.blk_m + 1] -
		                        params.direct_packed_tile_offsets[work_info.blk_m]
		                  : 0;
		          const bool local_delta_has_rows =
		              kmajor_stage_payload
		                  ? kmajor_stage_hot_count > 0
		              : packed_local_delta_payload
		                  ? packed_local_delta_count > 0
		                  : (params.direct_active_row_offsets != nullptr &&
			                     params.direct_active_row_offsets[work_info.blk_m + 1] >
			                         params.direct_active_row_offsets[work_info.blk_m]);
		          if (!local_delta_has_rows) {
#if HANDWRITTEN_TMA_PHASE_TRACE
		            if (phase_trace_sparse_lane(params, sparse_warp_rank, lane_idx)) {
		              phase_trace_write(params, PhaseTraceLocalActiveRows, 0);
		              phase_trace_add(
		                  params,
		                  PhaseTraceSparseSideTotal,
		                  phase_trace_clock() - phase_sparse_total_start);
		            }
#endif
		            return;
		          }
	          constexpr int LocalStages = CtaN / EpiN;
	          static_assert(CtaN % EpiN == 0, "local delta staging expects EpiN stages");
	          constexpr bool LocalDeltaStageBuffered =
	              HANDWRITTEN_TMA_LOCAL_DELTA_STAGE_BUFFERS >= LocalStages;
	          const int coop_threads = params.direct_probe_warps * NumThreadsPerWarp;
	          const int local_delta_active_start =
	              (kmajor_stage_payload || packed_local_delta_payload ||
	               params.direct_active_row_offsets == nullptr)
	                  ? 0
	                  : params.direct_active_row_offsets[work_info.blk_m];
			          const int local_delta_active_count =
			              kmajor_stage_payload
			                  ? kmajor_stage_hot_count
			              : packed_local_delta_payload
		                  ? packed_local_delta_count
			                  : params.direct_active_row_offsets[work_info.blk_m + 1] -
			                        local_delta_active_start;
#if HANDWRITTEN_TMA_PHASE_TRACE
	          if (!fast_dense_tile_meta_setup &&
	              phase_trace_sparse_lane(params, sparse_warp_rank, lane_idx)) {
	            const uint64_t phase_after_local_rows = phase_trace_clock();
	            phase_trace_add(
	                params,
	                PhaseTraceSparseSetupLocalRows,
	                phase_after_local_rows - phase_sparse_setup_cursor);
	            phase_sparse_setup_cursor = phase_after_local_rows;
	          }
#endif
#if HANDWRITTEN_TMA_PHASE_TRACE
	          if (phase_trace_sparse_lane(params, sparse_warp_rank, lane_idx)) {
	            phase_trace_add(
	                params,
	                PhaseTraceSparseSideSetupCore,
	                phase_trace_clock() - phase_sparse_total_start);
	            phase_trace_write(
	                params,
	                PhaseTraceLocalActiveRows,
	                static_cast<uint64_t>(local_delta_active_count));
	            phase_trace_write(
	                params,
	                PhaseTraceLocalHotGroups,
	                kmajor_tile_meta_payload
	                    ? static_cast<uint64_t>(kmajor_stage_hot_count)
	                    : active_m_idx >= 0 && params.direct_kmajor_group_offsets != nullptr
	                        ? static_cast<uint64_t>(
	                              params.direct_kmajor_group_offsets[active_m_idx + 1] -
	                              params.direct_kmajor_group_offsets[active_m_idx])
	                    : 0);
	          }
#endif
	          const int rowmask_active_start =
	              params.direct_active_row_offsets != nullptr
	                  ? params.direct_active_row_offsets[work_info.blk_m]
	                  : local_delta_active_start;
	          const int rowmask_active_count =
	              params.direct_active_row_offsets != nullptr
	                  ? (params.direct_active_row_offsets[work_info.blk_m + 1] -
	                     rowmask_active_start)
	                  : local_delta_active_count;
	          const bool build_local_delta_row_mask =
	              (local_delta_reg_block || prestore_local_delta_stage) &&
	              rowmask_active_count < CtaM;
          if (build_local_delta_row_mask) {
            constexpr int RowMaskWords = (CtaM + 31) / 32;
            for (int idx = sparse_warp_rank * NumThreadsPerWarp + lane_idx;
                 idx < RowMaskWords;
                 idx += coop_threads) {
              storage.sparse_local_delta.row_mask[idx] = 0u;
            }
            ptx::bar_sync(5, coop_threads);
            for (int idx = sparse_warp_rank * NumThreadsPerWarp + lane_idx;
                 idx < rowmask_active_count;
                 idx += coop_threads) {
              const int local_row =
                  params.direct_active_rows[rowmask_active_start + idx];
              if (local_row >= 0 && local_row < CtaM) {
                atomicOr(
                    reinterpret_cast<unsigned int*>(
                        &storage.sparse_local_delta.row_mask[local_row >> 5]),
                    1u << (local_row & 31));
              }
            }
	            ptx::bar_sync(5, coop_threads);
	          }
#if HANDWRITTEN_TMA_PHASE_TRACE
	          if (phase_trace_sparse_lane(params, sparse_warp_rank, lane_idx)) {
	            phase_trace_add(
	                params,
	                PhaseTraceSparseSideSetup,
	                phase_trace_clock() - phase_sparse_total_start);
	          }
#endif
	          if ((params.direct_probe_mixed_cta == 33 ||
	               params.direct_probe_mixed_cta == 34 ||
	               params.direct_probe_mixed_cta == 35 ||
	               params.direct_probe_mixed_cta == 36 ||
	               params.direct_probe_mixed_cta == 37) &&
	              LocalDeltaStageBuffered &&
	              params.direct_probe_warps >= LocalStages &&
	              (params.direct_probe_warps % LocalStages) == 0) {
	            const bool one_warp_per_stage =
	                packed_local_delta_payload &&
	                packed_local_delta_count <=
	                    HANDWRITTEN_TMA_LOCAL_DELTA_ONE_WARP_STAGE_ROWS;
	            const int stage_warps =
	                one_warp_per_stage ? 1 : params.direct_probe_warps / LocalStages;
	            const int stage = sparse_warp_rank / stage_warps;
	            const int stage_probe_rank = sparse_warp_rank - stage * stage_warps;
	            if (stage < LocalStages) {
	              uint16_t* local_delta_stage_tile =
	                  storage.sparse_local_delta.tile +
	                  static_cast<int64_t>(stage) * CtaM * EpiN;
#if HANDWRITTEN_TMA_PHASE_TRACE
	              const bool trace_stage_lane =
	                  stage_probe_rank == 0 && lane_idx == 0 &&
	                  phase_trace_cta_enabled(params);
	              uint64_t phase_stage_total_start = 0;
	              uint64_t phase_sparse_compute_start = 0;
	              if (trace_stage_lane) {
	                phase_stage_total_start = phase_trace_clock();
	                phase_sparse_compute_start = phase_stage_total_start;
	              }
#endif
	              if ((params.direct_probe_mixed_cta == 34 ||
	                   params.direct_probe_mixed_cta == 35 ||
	                   params.direct_probe_mixed_cta == 36 ||
	                   params.direct_probe_mixed_cta == 37) &&
	                  active_m_idx >= 0) {
                apply_sparse_loadfma_kmajor_local_delta_tile<CtaM, CtaN, EpiN>(
                    params,
                    local_delta_stage_tile,
                    static_cast<uint32_t>(active_m_idx),
                    work_info.blk_m,
	                    work_info.blk_n,
                    stage * EpiN,
                    lane_idx,
                    stage_probe_rank,
                    stage_warps,
                    kmajor_stage_group_start,
                    kmajor_stage_hot_count);
	              } else {
	                apply_sparse_loadfma_local_delta_tile<CtaM, CtaN, EpiN>(
	                    params,
	                    local_delta_stage_tile,
	                    storage.sparse_local_delta.partials,
	                    work_info.blk_m,
	                    work_info.blk_n,
	                    active_m_idx,
	                    stage * EpiN,
	                    lane_idx,
	                    stage_probe_rank,
	                    stage_warps);
	              }
#if HANDWRITTEN_TMA_PHASE_TRACE
	              if (trace_stage_lane) {
	                const uint64_t phase_sparse_compute_elapsed =
	                    phase_trace_clock() - phase_sparse_compute_start;
	                if (stage == 0) {
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSparseStage0Compute,
	                      phase_sparse_compute_elapsed);
	                } else if (stage == 1) {
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSparseStage1Compute,
	                      phase_sparse_compute_elapsed);
	                }
	              }
	              uint64_t phase_sparse_sync_start = 0;
	              if (trace_stage_lane) {
	                phase_sparse_sync_start = phase_trace_clock();
	              }
#endif
		              const bool per_warp_stage_ready =
		                  HANDWRITTEN_TMA_STAGE_READY_PER_WARP != 0 &&
		                  (params.direct_probe_mixed_cta == 34 ||
		                   params.direct_probe_mixed_cta == 37);
		              const bool stage_ready_words =
		                  HANDWRITTEN_TMA_STAGE_READY_WORDS != 0 &&
		                  (params.direct_probe_mixed_cta == 34 ||
		                   params.direct_probe_mixed_cta == 37);
		              if (stage_ready_words) {
		                ptx::bar_sync(static_cast<uint32_t>(7 + stage),
		                              stage_warps * NumThreadsPerWarp);
		                if (stage_probe_rank == 0 && lane_idx == 0) {
#if !HANDWRITTEN_TMA_SKIP_READY_PRODUCER_FENCE
		                  __threadfence_block();
#endif
		                  volatile uint32_t* stage_ready =
		                      reinterpret_cast<volatile uint32_t*>(
		                          &storage.sparse_probe.stage_ready[stage]);
		                  *stage_ready = 1u;
		                }
		              } else if (per_warp_stage_ready) {
		                __syncwarp(0xffffffffu);
		                if (lane_idx == 0) {
#if !HANDWRITTEN_TMA_SKIP_READY_PRODUCER_FENCE
		                  __threadfence_block();
#endif
		                  const uint32_t ready_bit =
		                      1u << (stage * stage_warps + stage_probe_rank);
		                  atomicOr(
		                      reinterpret_cast<unsigned int*>(&storage.sparse_probe.done_count),
		                      ready_bit);
		                }
		              } else {
		                ptx::bar_sync(static_cast<uint32_t>(7 + stage),
		                              stage_warps * NumThreadsPerWarp);
		                if (stage_probe_rank == 0 && lane_idx == 0) {
#if !HANDWRITTEN_TMA_SKIP_READY_PRODUCER_FENCE
		                  __threadfence_block();
#endif
		                  atomicOr(
		                      reinterpret_cast<unsigned int*>(&storage.sparse_probe.done_count),
		                      1u << stage);
		                }
		              }
	              if (params.direct_probe_mixed_cta == 36) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	                uint64_t phase_side_sc_wait_start = 0;
	                if (trace_stage_lane) {
	                  phase_side_sc_wait_start = phase_trace_clock();
	                }
#endif
	                volatile uint32_t* sc_ready =
	                    reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.sc_ready);
	                const uint32_t stage_bit = 1u << stage;
	                while ((*sc_ready & stage_bit) == 0u) {
	                }
	                __threadfence_block();
#if HANDWRITTEN_TMA_PHASE_TRACE
	                if (trace_stage_lane) {
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSideMergeScWait,
	                      phase_trace_clock() - phase_side_sc_wait_start);
	                }
	                uint64_t phase_side_merge_start = 0;
	                if (trace_stage_lane) {
	                  phase_side_merge_start = phase_trace_clock();
	                }
#endif
	                apply_local_delta_smem_tile_vec8_side<CtaM, CtaN, EpiN>(
	                    params,
	                    local_delta_stage_tile,
	                    storage.data.epilogue.sC,
	                    work_info.blk_m,
	                    work_info.blk_n,
	                    stage * EpiN,
	                    stage_probe_rank,
	                    stage_warps,
	                    lane_idx);
	                ptx::bar_sync(static_cast<uint32_t>(7 + stage),
	                              stage_warps * NumThreadsPerWarp);
	                if (stage_probe_rank == 0 && lane_idx == 0) {
	                  __threadfence_block();
	                  atomicOr(
	                      reinterpret_cast<unsigned int*>(&storage.sparse_probe.merge_done),
	                      stage_bit);
	                }
#if HANDWRITTEN_TMA_PHASE_TRACE
	                if (trace_stage_lane) {
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSideMergeApply,
	                      phase_trace_clock() - phase_side_merge_start);
	                }
#endif
	              }
#if HANDWRITTEN_TMA_PHASE_TRACE
	              if (trace_stage_lane) {
	                const uint64_t phase_sparse_sync_elapsed =
	                    phase_trace_clock() - phase_sparse_sync_start;
	                if (stage == 0) {
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSparseStage0Sync,
	                      phase_sparse_sync_elapsed);
	                } else if (stage == 1) {
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSparseStage1Sync,
	                      phase_sparse_sync_elapsed);
	                  phase_trace_add(
	                      params,
	                      PhaseTraceSparseSideTotal,
	                      phase_trace_clock() - phase_stage_total_start);
	                }
	              }
#endif
	            }
	            return;
	          }
	          volatile uint32_t* consumed =
	              reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.epoch);
	          for (int stage = 0; stage < LocalStages; ++stage) {
            uint16_t* local_delta_stage_tile =
                storage.sparse_local_delta.tile +
	                ((params.direct_probe_mixed_cta == 24 ||
	                  params.direct_probe_mixed_cta == 25 ||
	                  params.direct_probe_mixed_cta == 26 ||
	                  params.direct_probe_mixed_cta == 27 ||
	                  params.direct_probe_mixed_cta == 28 ||
	                  params.direct_probe_mixed_cta == 29 ||
	                  params.direct_probe_mixed_cta == 30 ||
	                  params.direct_probe_mixed_cta == 31 ||
		                  params.direct_probe_mixed_cta == 32 ||
		                  params.direct_probe_mixed_cta == 33 ||
		                  params.direct_probe_mixed_cta == 34 ||
		                  params.direct_probe_mixed_cta == 35 ||
		                  params.direct_probe_mixed_cta == 36 ||
		                  params.direct_probe_mixed_cta == 37) && LocalDeltaStageBuffered
		                     ? static_cast<int64_t>(stage) * CtaM * EpiN
		                     : 0);
#if HANDWRITTEN_TMA_PHASE_TRACE
	            uint64_t phase_sparse_compute_start = 0;
	            if (phase_trace_sparse_lane(params, sparse_warp_rank, lane_idx)) {
	              phase_sparse_compute_start = phase_trace_clock();
	            }
#endif
	            if (params.direct_probe_mixed_cta == 32 && active_m_idx >= 0) {
	              apply_sparse_loadfma_kmajor_local_delta_tile<CtaM, CtaN, EpiN>(
                  params,
                  local_delta_stage_tile,
                  static_cast<uint32_t>(active_m_idx),
                  work_info.blk_m,
                  work_info.blk_n,
                  stage * EpiN,
                  lane_idx,
                  sparse_warp_rank,
                  params.direct_probe_warps);
            } else {
              apply_sparse_loadfma_local_delta_tile<CtaM, CtaN, EpiN>(
                  params,
                  local_delta_stage_tile,
                  storage.sparse_local_delta.partials,
                  work_info.blk_m,
                  work_info.blk_n,
                  active_m_idx,
                  stage * EpiN,
                  lane_idx,
	                  sparse_warp_rank,
	                  params.direct_probe_warps);
	            }
#if HANDWRITTEN_TMA_PHASE_TRACE
	            if (phase_trace_sparse_lane(params, sparse_warp_rank, lane_idx)) {
	              const uint64_t phase_sparse_compute_elapsed =
	                  phase_trace_clock() - phase_sparse_compute_start;
	              phase_trace_add(
	                  params,
	                  PhaseTraceSparseSideCompute,
	                  phase_sparse_compute_elapsed);
	              if (stage == 0) {
	                phase_trace_add(
	                    params,
	                    PhaseTraceSparseStage0Compute,
	                    phase_sparse_compute_elapsed);
	              } else if (stage == 1) {
	                phase_trace_add(
	                    params,
	                    PhaseTraceSparseStage1Compute,
	                    phase_sparse_compute_elapsed);
	              }
	            }
	            uint64_t phase_sparse_sync_start = 0;
	            if (phase_trace_sparse_lane(params, sparse_warp_rank, lane_idx)) {
	              phase_sparse_sync_start = phase_trace_clock();
	            }
#endif
	            ptx::bar_sync(5, coop_threads);
	            if (sparse_warp_rank == 0 && lane_idx == 0) {
	              __threadfence_block();
              atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.done_count),
                         static_cast<unsigned int>(stage + 1));
            }
		            if ((params.direct_probe_mixed_cta != 24 &&
		                 params.direct_probe_mixed_cta != 25 &&
		                 params.direct_probe_mixed_cta != 26 &&
		                 params.direct_probe_mixed_cta != 27 &&
		                 params.direct_probe_mixed_cta != 28 &&
		                 params.direct_probe_mixed_cta != 29 &&
		                 params.direct_probe_mixed_cta != 30 &&
		                 params.direct_probe_mixed_cta != 31 &&
		                 params.direct_probe_mixed_cta != 32 &&
		                 params.direct_probe_mixed_cta != 33) ||
		                !LocalDeltaStageBuffered) {
              while (*consumed < static_cast<uint32_t>(stage + 1)) {
              }
              __threadfence_block();
			              ptx::bar_sync(5, coop_threads);
		            }
#if HANDWRITTEN_TMA_PHASE_TRACE
	            if (phase_trace_sparse_lane(params, sparse_warp_rank, lane_idx)) {
	              const uint64_t phase_sparse_sync_elapsed =
	                  phase_trace_clock() - phase_sparse_sync_start;
	              phase_trace_add(
	                  params,
	                  PhaseTraceSparseSideSync,
	                  phase_sparse_sync_elapsed);
	              if (stage == 0) {
	                phase_trace_add(
	                    params,
	                    PhaseTraceSparseStage0Sync,
	                    phase_sparse_sync_elapsed);
	              } else if (stage == 1) {
	                phase_trace_add(
	                    params,
	                    PhaseTraceSparseStage1Sync,
	                    phase_sparse_sync_elapsed);
	              }
	            }
#endif
		          }
		          if (params.direct_probe_mixed_cta == 26) {
		            volatile uint32_t* ready =
		                reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.done_count);
		            while (*ready < static_cast<uint32_t>(LocalStages + 1)) {
		            }
		            __threadfence_block();
		            apply_sparse_rowchunk_late_add_localrows_skip_noprobe<CtaM, CtaN>(
		                params,
		                work_info.blk_m,
		                work_info.blk_n,
		                lane_idx,
		                sparse_warp_rank,
		                params.direct_probe_warps);
		          } else if (params.direct_probe_mixed_cta == 31) {
		            for (int stage = 0; stage < LocalStages; ++stage) {
		              while (*consumed < static_cast<uint32_t>(stage + 1)) {
		              }
#if !HANDWRITTEN_TMA_SKIP_READY_CONSUMER_FENCE
		              __threadfence_block();
#endif
		              apply_sparse_rowchunk_smem_add_localrows_skip_noprobe<CtaM, CtaN, EpiN>(
		                  params,
		                  storage.data.epilogue.sC,
		                  work_info.blk_m,
		                  work_info.blk_n,
		                  stage * EpiN,
		                  lane_idx,
		                  sparse_warp_rank,
		                  params.direct_probe_warps);
		              ptx::bar_sync(5, coop_threads);
		              if (sparse_warp_rank == 0 && lane_idx == 0) {
#if !HANDWRITTEN_TMA_SKIP_READY_CONSUMER_FENCE
		                __threadfence_block();
#endif
		                atomicExch(
		                    reinterpret_cast<unsigned int*>(&storage.sparse_probe.done_count),
		                    static_cast<unsigned int>(LocalStages + stage + 1));
		              }
		            }
		          }
	        } else if (params.direct_probe_mixed_cta == 20) {
          constexpr int SubN = 32;
          constexpr int SubStages = CtaN / SubN;
          static_assert(CtaN % SubN == 0, "subacc staged path expects CtaN divisible by 32");
          const bool has_groups = active_m_idx >= 0;
          const uint32_t safe_active_m_idx =
              has_groups ? static_cast<uint32_t>(active_m_idx) : 0u;
          const int coop_threads = params.direct_probe_warps * NumThreadsPerWarp;
          volatile uint32_t* ready =
              reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.done_count);
          volatile uint32_t* consumed =
              reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.epoch);
          for (int stage = 0; stage < SubStages; ++stage) {
            apply_sparse_loadfma_kmajor_subacc32_stage_tile<
                CtaM,
                CtaN,
                SubN,
                HANDWRITTEN_TMA_SPARSE_ACC_ROWS>(
                params,
                storage.sparse_acc,
                safe_active_m_idx,
                has_groups,
                work_info.blk_m,
                work_info.blk_n,
                stage * SubN,
                lane_idx,
                sparse_warp_rank,
                params.direct_probe_warps,
                params.direct_probe_group_budget);
            if (sparse_warp_rank == 0 && lane_idx == 0) {
#if !HANDWRITTEN_TMA_SKIP_READY_CONSUMER_FENCE
			              __threadfence_block();
#endif
              atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.done_count),
                         static_cast<unsigned int>(stage + 1));
            }
            ptx::bar_sync(5, coop_threads);
            if (lane_idx == 0) {
              while (*consumed < static_cast<uint32_t>(stage + 1)) {
              }
              __threadfence_block();
            }
            ptx::bar_sync(5, coop_threads);
          }
        } else if (active_m_idx >= 0) {
	          if (params.direct_probe_mixed_cta == 19) {
	            apply_sparse_loadfma_kmajor_sharedacc_fill_tile<
	                CtaM,
                CtaN,
                HANDWRITTEN_TMA_SPARSE_ACC_ROWS>(
                params,
                storage.sparse_acc,
                static_cast<uint32_t>(active_m_idx),
                work_info.blk_m,
                work_info.blk_n,
                lane_idx,
	                sparse_warp_rank,
	                params.direct_probe_warps,
	                params.direct_probe_group_budget);
	          } else if (params.direct_probe_mixed_cta == 16 ||
	              params.direct_probe_mixed_cta == 17) {
            volatile uint32_t* done =
                reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.done_count);
            while (*done == 0u) {
            }
            __threadfence_block();
            if (params.direct_probe_mixed_cta == 16) {
              apply_sparse_rowchunk_late_add_noprobe<CtaM, CtaN>(
                  params,
                  work_info.blk_m,
                  work_info.blk_n,
                  lane_idx,
                  sparse_warp_rank,
                  params.direct_probe_warps);
            } else {
              apply_sparse_rowchunk_chunked_atomic_add_noprobe<CtaM, CtaN>(
                  params,
                  work_info.blk_m,
                  work_info.blk_n,
                  lane_idx,
                  sparse_warp_rank,
                  params.direct_probe_warps);
            }
          } else if ((params.direct_probe_mixed_cta == 12 ||
               params.direct_probe_mixed_cta == 13 ||
               params.direct_probe_mixed_cta == 14 ||
               params.direct_probe_mixed_cta == 15 ||
               params.direct_probe_mixed_cta == 18) &&
              params.direct_delta_output != nullptr) {
            apply_sparse_loadfma_kmajor_bounded_groups_tile_write_noprobe<CtaM, CtaN>(
                params,
                static_cast<uint32_t>(active_m_idx),
                work_info.blk_m,
                work_info.blk_n,
                lane_idx,
                sparse_warp_rank,
                params.direct_probe_warps,
                params.direct_probe_group_budget);
            if (params.direct_probe_mixed_cta == 13 ||
                params.direct_probe_mixed_cta == 14 ||
                params.direct_probe_mixed_cta == 15) {
              ptx::bar_sync(5, params.direct_probe_warps * NumThreadsPerWarp);
              volatile uint32_t* done =
                  reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.done_count);
              while (*done == 0u) {
              }
              __threadfence_block();
              if (params.direct_probe_mixed_cta == 15) {
                apply_sparse_kmajor_bounded_groups_tile_meta_merge_delta_noprobe<CtaM, CtaN>(
                    params,
                    work_info.blk_m,
                    work_info.blk_n,
                    lane_idx,
                    sparse_warp_rank,
                    params.direct_probe_warps);
              } else if (params.direct_probe_mixed_cta == 14) {
                apply_sparse_kmajor_bounded_groups_tile_rowmerge_delta_noprobe<CtaM, CtaN>(
                    params,
                    static_cast<uint32_t>(active_m_idx),
                    work_info.blk_m,
                    work_info.blk_n,
                    lane_idx,
                    sparse_warp_rank,
                    params.direct_probe_warps,
                    params.direct_probe_group_budget);
              } else {
                apply_sparse_kmajor_bounded_groups_tile_add_delta_noprobe<CtaM, CtaN>(
                    params,
                    static_cast<uint32_t>(active_m_idx),
                    work_info.blk_m,
                    work_info.blk_n,
                    lane_idx,
                    sparse_warp_rank,
                    params.direct_probe_warps,
                params.direct_probe_group_budget);
              }
            }
          } else if (params.direct_probe_mixed_cta == 11 ||
                     params.direct_probe_mixed_cta == 12 ||
                     params.direct_probe_mixed_cta == 13 ||
                     params.direct_probe_mixed_cta == 14 ||
                     params.direct_probe_mixed_cta == 15 ||
                     params.direct_probe_mixed_cta == 16 ||
                     params.direct_probe_mixed_cta == 17 ||
                     params.direct_probe_mixed_cta == 18) {
            apply_sparse_loadfma_kmajor_bounded_groups_tile_noprobe<CtaM, CtaN>(
                params,
                static_cast<uint32_t>(active_m_idx),
                work_info.blk_m,
                work_info.blk_n,
                lane_idx,
                sparse_warp_rank,
                params.direct_probe_warps,
                params.direct_probe_group_budget);
          } else if (params.direct_delta_output != nullptr) {
            apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, true>(
                params,
                static_cast<uint32_t>(active_m_idx),
                work_info.blk_m,
                work_info.blk_n,
                lane_idx,
                sparse_warp_rank,
                params.direct_probe_warps,
                0,
                -1,
                params.direct_probe_group_budget);
          } else {
            apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, false>(
                params,
                static_cast<uint32_t>(active_m_idx),
                work_info.blk_m,
                work_info.blk_n,
                lane_idx,
                sparse_warp_rank,
                params.direct_probe_warps,
                0,
                -1,
                params.direct_probe_group_budget);
          }
        }
        if (((params.direct_probe_mixed_cta == 18 &&
              params.direct_delta_output != nullptr &&
              params.direct_delta_write_mode == 4) ||
	             params.direct_probe_mixed_cta == 19) &&
	             params.direct_probe_mixed_cta != 21 &&
	            lane_idx == 0) {
	          __threadfence_block();
	          atomicAdd(reinterpret_cast<unsigned int*>(&storage.sparse_probe.done_count), 1u);
	        }
	      } else if (params.direct_probe_mixed_cta == 9 && lane_idx == 0) {
	        const int32_t tile_id =
	            static_cast<int32_t>(work_info.blk_m * gridDim.x + work_info.blk_n);
	        params.direct_probe_sink[static_cast<int64_t>(tile_id) *
	                                     params.direct_probe_warps +
	                                 sparse_warp_rank] = 1.0f;
	      }
#if HANDWRITTEN_TMA_PHASE_TRACE
	      if (phase_trace_sparse_lane(params, sparse_warp_rank, lane_idx)) {
	        phase_trace_add(
	            params,
	            PhaseTraceSparseSideTotal,
	            phase_trace_clock() - phase_sparse_total_start);
	      }
#endif
	      return;
	    }
    if (params.direct_probe_mixed_cta == 4) {
      const uint32_t probe_blk_m = storage.sparse_probe.blk_m;
      const uint32_t probe_blk_n = storage.sparse_probe.blk_n;
      const uint32_t probe_valid = storage.sparse_probe.valid;
      if (probe_valid != 0 && params.direct_probe_do_math != 0) {
        apply_sparse_loadfma_probe_tile<CtaM, CtaN>(
            params,
            probe_blk_m,
            probe_blk_n,
            lane_idx,
            sparse_warp_rank,
            params.direct_probe_warps,
            0);
      } else if (probe_valid != 0 && lane_idx == 0) {
        const int32_t tile_id =
            static_cast<int32_t>(probe_blk_m * real_grid_dim.x + probe_blk_n);
        params.direct_probe_sink[static_cast<int64_t>(tile_id) *
                                     params.direct_probe_warps +
                                 sparse_warp_rank] = 1.0f;
      }
      return;
    }
    if (params.direct_probe_do_math == 0) {
      if (lane_idx == 0) {
        params.direct_probe_sink[sparse_warp_rank] = 1.0f;
      }
      return;
    }
    if (params.direct_probe_kmajor == 2 &&
        params.direct_delta_output != nullptr &&
        params.direct_delta_write_mode == 3) {
      const int coop_warps = params.direct_probe_warps;
      const int coop_threads = coop_warps * NumThreadsPerWarp;
      while (true) {
        if (sparse_warp_rank == 0 && lane_idx == 0) {
          storage.sparse_probe.blk_m =
              static_cast<uint32_t>(atomicAdd(params.direct_probe_counter, 1));
        }
        ptx::bar_sync(1, coop_threads);
        const int32_t tile_id = static_cast<int32_t>(storage.sparse_probe.blk_m);
        if (tile_id >= params.direct_probe_total_tiles) {
          ptx::bar_sync(1, coop_threads);
          break;
        }
        const uint32_t active_m_idx = static_cast<uint32_t>(tile_id / real_grid_dim.x);
        const uint32_t probe_blk_n =
            static_cast<uint32_t>(tile_id - active_m_idx * real_grid_dim.x);
        const uint32_t probe_blk_m =
            params.direct_probe_active_mblocks != nullptr
                ? static_cast<uint32_t>(params.direct_probe_active_mblocks[active_m_idx])
                : active_m_idx;
        apply_sparse_loadfma_kmajor_sharedacc_tile<
            CtaM,
            CtaN,
            HANDWRITTEN_TMA_SPARSE_ACC_ROWS>(
            params,
            storage.sparse_acc,
            active_m_idx,
            probe_blk_m,
            probe_blk_n,
            lane_idx,
            sparse_warp_rank,
            coop_warps,
            0);
        ptx::bar_sync(1, coop_threads);
      }
      return;
    }
    while (true) {
      int32_t tile_id = 0;
      if (lane_idx == 0) {
        tile_id = atomicAdd(params.direct_probe_counter, 1);
      }
      tile_id = __shfl_sync(0xffffffffu, tile_id, 0);
      if (tile_id >= params.direct_probe_total_tiles) {
        break;
      }
      const uint32_t active_m_idx = static_cast<uint32_t>(tile_id / real_grid_dim.x);
      const uint32_t probe_blk_n =
          static_cast<uint32_t>(tile_id - active_m_idx * real_grid_dim.x);
      const uint32_t probe_blk_m =
          params.direct_probe_active_mblocks != nullptr
              ? static_cast<uint32_t>(params.direct_probe_active_mblocks[active_m_idx])
              : active_m_idx;
      if (params.direct_probe_do_math != 0) {
        apply_sparse_loadfma_probe_tile<CtaM, CtaN>(
            params,
            probe_blk_m,
            probe_blk_n,
            lane_idx,
            0,
            1,
            0);
      } else if (lane_idx == 0) {
        params.direct_probe_sink[tile_id * params.direct_probe_warps + sparse_warp_rank] =
            1.0f;
      }
    }
    return;
  }
  }

  // init dense pipelines only on the original producer/consumer warpgroups
  auto create_mainloop = [&] {
    typename MainloopPipe::Pipeline mainloop_pipe(storage.pipeline.mainloop, NumThreadsPerWarp * 2 + 1, NumThreadsPerWarp * WorkerRepM * WorkerRepN);
    typename MainloopPipe::Pipeline::State mainloop_state(wg_id == 0 ? 1 : 0);
    return MainloopPipe(storage.data.mainloop, std::move(mainloop_pipe), std::move(mainloop_state));
  };

  auto create_clcloop = [&] {
    constexpr int CompactSchedulerConsumers =
        CompactConsumerSpecialized ? 1 : 0;
    CLCloopPipe::Pipeline clc_pipe(
        storage.pipeline.clcloop,
        1,
        NumThreadsPerWarp * 2 + 1 +
            NumThreadsPerWarp * WorkerRepM * WorkerRepN +
            CompactSchedulerConsumers);
    CLCloopPipe::Pipeline::State clc_state;
    return CLCloopPipe(storage.data.clcloop, std::move(clc_pipe), std::move(clc_state));
  };

  auto mainloop = create_mainloop();
  auto clcloop = create_clcloop();

  if constexpr (CompactConsumerSpecialized) {
    static_assert(LaunchThreads == NumThreadsPerWarpGroup * 4,
                  "compact consumer path is a specialized 4WG kernel");
    if (wg_id >= 3) {
      const int32_t sparse_thread_rank =
          static_cast<int32_t>(threadIdx.x) - 3 * NumThreadsPerWarpGroup;
      uint32_t compact_sync_phase = 0;
      int compact_stage = 0;
      while (work_info.valid) {
        stage_sparse_compact_inputs_for_tile<
            CtaM, HANDWRITTEN_TMA_COMPACT_CONSUMER_MAX_NNZ>(
            params,
            storage.sparse_compact_input[compact_stage],
            storage.sparse_probe,
            work_info.blk_m,
            sparse_thread_rank);

        // Publish this tile's compact input.  The other buffer can be filled
        // immediately after CLC returns the next tile; by the time a buffer is
        // reused two iterations later, both consumers have passed the next
        // ready barrier and can no longer be reading the older generation.
        compact_cross_wg_barrier(
            &storage.sparse_probe.compact_sync_mbar,
            compact_sync_phase,
            lane_idx);

        // One WG3 lane joins the CLC pipeline.  Broadcast the returned tile to
        // the rest of WG3 instead of adding 128 scheduler barrier arrivals.
        if (sparse_thread_rank == 0) {
          clcloop.pipe.consumer_arrive(clcloop.state);
          int4* clc_ret_ptr = &clcloop.data().clc_ret;
          clcloop.pipe.consumer_wait(clcloop.state);
          auto [ret, valid] = ptx::clc_query(clc_ret_ptr);
          update_work_info(ret, valid, work_info);
          storage.sparse_probe.blk_m = work_info.blk_m;
          storage.sparse_probe.blk_n = work_info.blk_n;
          storage.sparse_probe.valid = work_info.valid;
          __threadfence_block();
          clcloop.state.advance();
        }
        ptx::bar_sync(5, NumThreadsPerWarpGroup);
        if (sparse_thread_rank != 0) {
          work_info.blk_m = storage.sparse_probe.blk_m;
          work_info.blk_n = storage.sparse_probe.blk_n;
          work_info.valid = storage.sparse_probe.valid;
        }
        compact_stage ^= 1;
      }
      return;
    }
  }

  if constexpr (LaunchThreads > NumThreadsPerWarpGroup * 3) {
    ptx::bar_sync(3, NumThreadsPerWarpGroup * 3);
  }

  const bool keep_dense_persistent_extra_wg =
      Mode49Specialized ||
      params.direct_probe_mixed_cta == 43 || params.direct_probe_mixed_cta == 44 ||
      params.direct_probe_mixed_cta == 45 || params.direct_probe_mixed_cta == 46 ||
      params.direct_probe_mixed_cta == 47 || params.direct_probe_mixed_cta == 48 ||
      params.direct_probe_mixed_cta == 49;
  const bool single_tile_cta =
      (!Mode49Specialized &&
       params.direct_probe_mixed_cta != 0 && !keep_dense_persistent_extra_wg) ||
      force_dense_4wg_mode;
  int64_t kstages = params.k / CtaK;

  if (wg_id == 0) {
    if (!four_wg_reg_mode) {
      ptx::warpgroup_reg_dealloc<HANDWRITTEN_TMA_PRODUCER_REGS>();
    }

    if (role == ProducerRole::TMA) {
	      if (lane_idx == 0) {
	        while (work_info.valid) {
	          #pragma unroll 1
	          for (int k = 0; k < kstages; ++k) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	            uint64_t phase_tma_start = 0;
	            if (phase_trace_producer_lane(params, lane_idx)) {
	              phase_tma_start = phase_trace_clock();
	            }
#endif
	            mainloop.pipe.producer_wait(mainloop.state);
#if HANDWRITTEN_TMA_PHASE_TRACE
	            if (phase_trace_producer_lane(params, lane_idx)) {
	              const uint64_t phase_after_wait = phase_trace_clock();
	              phase_trace_add(
	                  params,
	                  PhaseTraceProducerTmaWait,
	                  phase_after_wait - phase_tma_start);
	              phase_tma_start = phase_after_wait;
	            }
#endif
	            mainloop.pipe.producer_arrive_expect_tx(
	              mainloop.state,
	              (CtaM + CtaN) * CtaK / 2 // sA + sB
            );
            ptx::tma_copy_tensor_2d(
              &params.tensormap_A,
              mainloop.data().sA,
              mainloop.pipe.producer_get_mbar(mainloop.state),
              work_info.blk_m * CtaM,
              k * CtaK / 2
            );
            ptx::tma_copy_tensor_2d(
              &params.tensormap_B,
              mainloop.data().sB,
              mainloop.pipe.producer_get_mbar(mainloop.state),
	              work_info.blk_n * CtaN,
	              k * CtaK / 2
	            );
#if HANDWRITTEN_TMA_PHASE_TRACE
	            if (phase_trace_producer_lane(params, lane_idx)) {
	              phase_trace_add(
	                  params,
	                  PhaseTraceProducerTmaIssue,
	                  phase_trace_clock() - phase_tma_start);
	            }
#endif
	            mainloop.state.advance();
	          }
          if (single_tile_cta) {
            work_info.valid = false;
          } else {
            clcloop.pipe.consumer_arrive(clcloop.state);
            {
              int4 *clc_ret_ptr = &clcloop.data().clc_ret;
              clcloop.pipe.consumer_wait(clcloop.state);
              auto [ret, valid] = ptx::clc_query(clc_ret_ptr);
              update_work_info(ret, valid, work_info);
            }
            clcloop.state.advance();
          }
        }
      }
      if (!Mode49Specialized && params.direct_probe_mixed_cta == 3) {
        apply_incta_sparse_probe_for_tile<CtaM, CtaN>(
            params,
            storage.sparse_acc,
            real_grid_dim.y,
            work_info.blk_m,
            work_info.blk_n,
            lane_idx,
            static_cast<int>(ProducerRole::TMA),
            4,
            4,
            false);
      }
    } else if (role == ProducerRole::ScaleA) {
      while (work_info.valid) {
        auto load_scale_A = [&] __attribute__((always_inline))(const int& k_blk) {
          #pragma unroll
          for (int i = lane_idx; i < CtaM; i += NumThreadsPerWarp) {
            const uint8_t* scale_src = params.scale_tile_major
                ? params.inner_scale_A + ((params.k / 16) * work_info.blk_m + k_blk * CtaK / 16) * CtaM + i * CtaK / 16
                : params.inner_scale_A + (work_info.blk_m * CtaM + i) * (params.k / 16) + k_blk * CtaK / 16;
            ptx::cpasync<CtaK / 16>(
              scale_src,
              &mainloop.data().sSFA[i * AtomRepK]
            );
          }
        };
	        #pragma unroll 1
	        for (int k = 0; k < kstages; ++k) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	          uint64_t phase_scale_start = 0;
	          if (phase_trace_producer_lane(params, lane_idx)) {
	            phase_scale_start = phase_trace_clock();
	          }
#endif
	          const uint32_t producer_done = mainloop.pipe.producer_try_wait(mainloop.state);
          if (!Mode49Specialized &&
              params.direct_probe_mixed_cta == 5 && params.direct_probe_do_math != 0 &&
              producer_done == 0) {
            int active_m_idx = -1;
            if (lane_idx == 0) {
              active_m_idx = find_active_mblock_index(params, work_info.blk_m);
            }
            active_m_idx = __shfl_sync(0xffffffffu, active_m_idx, 0);
            if (active_m_idx >= 0) {
#if HANDWRITTEN_TMA_PHASE_TRACE
              uint64_t phase_probe_start = 0;
              if (phase_trace_producer_lane(params, lane_idx)) {
                phase_probe_start = phase_trace_clock();
              }
#endif
              apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN>(
                  params,
                  static_cast<uint32_t>(active_m_idx),
                  work_info.blk_m,
                  work_info.blk_n,
                  lane_idx,
                  0,
                  2,
                  0,
                  k,
                  params.direct_probe_group_budget);
#if HANDWRITTEN_TMA_PHASE_TRACE
              if (phase_trace_producer_lane(params, lane_idx)) {
                const uint64_t phase_probe_done = phase_trace_clock();
                phase_trace_add(
                    params,
                    PhaseTraceWg0ScaleASparseProbe,
                    phase_probe_done - phase_probe_start);
                phase_trace_add(params, PhaseTraceWg0ScaleASparseProbeCalls, 1);
                phase_scale_start = phase_probe_done;
              }
#endif
            }
          }
          if (!Mode49Specialized &&
              params.direct_probe_mixed_cta == 8 && params.direct_probe_do_math != 0 &&
              k * 2 < params.direct_probe_group_budget) {
            int active_m_idx = -1;
            if (lane_idx == 0) {
              active_m_idx = find_active_mblock_index(params, work_info.blk_m);
            }
            active_m_idx = __shfl_sync(0xffffffffu, active_m_idx, 0);
            if (active_m_idx >= 0) {
#if HANDWRITTEN_TMA_PHASE_TRACE
              uint64_t phase_probe_start = 0;
              if (phase_trace_producer_lane(params, lane_idx)) {
                phase_probe_start = phase_trace_clock();
              }
#endif
              if (params.direct_delta_output != nullptr) {
                apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, true>(
                    params,
                    static_cast<uint32_t>(active_m_idx),
                    work_info.blk_m,
                    work_info.blk_n,
                    lane_idx,
                    0,
                    2,
                    0,
                    k,
                    params.direct_probe_group_budget,
                    1);
              } else {
                apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, false>(
                    params,
                    static_cast<uint32_t>(active_m_idx),
                    work_info.blk_m,
                    work_info.blk_n,
                    lane_idx,
                    0,
                    2,
                    0,
                    k,
                    params.direct_probe_group_budget,
                    1);
              }
#if HANDWRITTEN_TMA_PHASE_TRACE
              if (phase_trace_producer_lane(params, lane_idx)) {
                const uint64_t phase_probe_done = phase_trace_clock();
                phase_trace_add(
                    params,
                    PhaseTraceWg0ScaleASparseProbe,
                    phase_probe_done - phase_probe_start);
                phase_trace_add(params, PhaseTraceWg0ScaleASparseProbeCalls, 1);
                phase_scale_start = phase_probe_done;
              }
#endif
            }
          }
	          mainloop.pipe.producer_wait(mainloop.state, producer_done);
#if HANDWRITTEN_TMA_PHASE_TRACE
	          if (phase_trace_producer_lane(params, lane_idx)) {
	            const uint64_t phase_after_wait = phase_trace_clock();
	            phase_trace_add(
	                params,
	                PhaseTraceScaleAWait,
	                phase_after_wait - phase_scale_start);
	            phase_scale_start = phase_after_wait;
	          }
#endif
	          load_scale_A(k);
	          mainloop.pipe.producer_arrive_cpasync(mainloop.state);
#if HANDWRITTEN_TMA_PHASE_TRACE
	          if (phase_trace_producer_lane(params, lane_idx)) {
	            phase_trace_add(
	                params,
	                PhaseTraceScaleACopy,
	                phase_trace_clock() - phase_scale_start);
	          }
#endif
	          mainloop.state.advance();
	        }

        if (single_tile_cta) {
          work_info.valid = false;
        } else {
          clcloop.pipe.consumer_arrive(clcloop.state);
          {
            int4 *clc_ret_ptr = &clcloop.data().clc_ret;
            clcloop.pipe.consumer_wait(clcloop.state);
            auto [ret, valid] = ptx::clc_query(clc_ret_ptr);
            update_work_info(ret, valid, work_info);
          }
          clcloop.state.advance();
        }
      }
      if (!Mode49Specialized && params.direct_probe_mixed_cta == 3) {
        apply_incta_sparse_probe_for_tile<CtaM, CtaN>(
            params,
            storage.sparse_acc,
            real_grid_dim.y,
            work_info.blk_m,
            work_info.blk_n,
            lane_idx,
            static_cast<int>(ProducerRole::ScaleA),
            4,
            4,
            true);
      }
      if (!Mode49Specialized &&
          params.direct_probe_mixed_cta == 7 && params.direct_probe_do_math != 0 &&
          params.direct_probe_warps > 1) {
        int active_m_idx = -1;
        if (lane_idx == 0) {
          active_m_idx = find_active_mblock_index(params, work_info.blk_m);
        }
        active_m_idx = __shfl_sync(0xffffffffu, active_m_idx, 0);
        if (active_m_idx >= 0) {
          if (params.direct_delta_output != nullptr) {
            apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, true>(
                params,
                static_cast<uint32_t>(active_m_idx),
                work_info.blk_m,
                work_info.blk_n,
                lane_idx,
                1,
                params.direct_probe_warps,
                0,
                -1,
                params.direct_probe_group_budget);
          } else {
            apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, false>(
                params,
                static_cast<uint32_t>(active_m_idx),
                work_info.blk_m,
                work_info.blk_n,
                lane_idx,
                1,
                params.direct_probe_warps,
                0,
                -1,
                params.direct_probe_group_budget);
          }
        }
      }

    } else if (role == ProducerRole::ScaleB) {
      while (work_info.valid) {
        auto load_scale_B = [&] __attribute__((always_inline))(const int& k_blk) {
          #pragma unroll
          for (int i = lane_idx; i < CtaN; i += NumThreadsPerWarp) {
            const uint8_t* scale_src = params.scale_tile_major
                ? params.inner_scale_B + ((params.k / 16) * work_info.blk_n + k_blk * CtaK / 16) * CtaN + i * CtaK / 16
                : params.inner_scale_B + (work_info.blk_n * CtaN + i) * (params.k / 16) + k_blk * CtaK / 16;
            ptx::cpasync<CtaK / 16>(
              scale_src,
              &mainloop.data().sSFB[i * AtomRepK]
            );
          }
        };
	        #pragma unroll 1
	        for (int k = 0; k < kstages; ++k) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	          uint64_t phase_scale_start = 0;
	          if (phase_trace_producer_lane(params, lane_idx)) {
	            phase_scale_start = phase_trace_clock();
	          }
#endif
	          const uint32_t producer_done = mainloop.pipe.producer_try_wait(mainloop.state);
          if (!Mode49Specialized &&
              params.direct_probe_mixed_cta == 5 && params.direct_probe_do_math != 0 &&
              producer_done == 0) {
            int active_m_idx = -1;
            if (lane_idx == 0) {
              active_m_idx = find_active_mblock_index(params, work_info.blk_m);
            }
            active_m_idx = __shfl_sync(0xffffffffu, active_m_idx, 0);
            if (active_m_idx >= 0) {
#if HANDWRITTEN_TMA_PHASE_TRACE
              uint64_t phase_probe_start = 0;
              if (phase_trace_producer_lane(params, lane_idx)) {
                phase_probe_start = phase_trace_clock();
              }
#endif
              apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN>(
                  params,
                  static_cast<uint32_t>(active_m_idx),
                  work_info.blk_m,
                  work_info.blk_n,
                  lane_idx,
                  1,
                  2,
                  0,
                  k,
                  params.direct_probe_group_budget);
#if HANDWRITTEN_TMA_PHASE_TRACE
              if (phase_trace_producer_lane(params, lane_idx)) {
                const uint64_t phase_probe_done = phase_trace_clock();
                phase_trace_add(
                    params,
                    PhaseTraceWg0ScaleBSparseProbe,
                    phase_probe_done - phase_probe_start);
                phase_trace_add(params, PhaseTraceWg0ScaleBSparseProbeCalls, 1);
                phase_scale_start = phase_probe_done;
              }
#endif
            }
          }
          if (!Mode49Specialized &&
              params.direct_probe_mixed_cta == 8 && params.direct_probe_do_math != 0 &&
              k * 2 + 1 < params.direct_probe_group_budget) {
            int active_m_idx = -1;
            if (lane_idx == 0) {
              active_m_idx = find_active_mblock_index(params, work_info.blk_m);
            }
            active_m_idx = __shfl_sync(0xffffffffu, active_m_idx, 0);
            if (active_m_idx >= 0) {
#if HANDWRITTEN_TMA_PHASE_TRACE
              uint64_t phase_probe_start = 0;
              if (phase_trace_producer_lane(params, lane_idx)) {
                phase_probe_start = phase_trace_clock();
              }
#endif
              if (params.direct_delta_output != nullptr) {
                apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, true>(
                    params,
                    static_cast<uint32_t>(active_m_idx),
                    work_info.blk_m,
                    work_info.blk_n,
                    lane_idx,
                    1,
                    2,
                    0,
                    k,
                    params.direct_probe_group_budget,
                    1);
              } else {
                apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, false>(
                    params,
                    static_cast<uint32_t>(active_m_idx),
                    work_info.blk_m,
                    work_info.blk_n,
                    lane_idx,
                    1,
                    2,
                    0,
                    k,
                    params.direct_probe_group_budget,
                    1);
              }
#if HANDWRITTEN_TMA_PHASE_TRACE
              if (phase_trace_producer_lane(params, lane_idx)) {
                const uint64_t phase_probe_done = phase_trace_clock();
                phase_trace_add(
                    params,
                    PhaseTraceWg0ScaleBSparseProbe,
                    phase_probe_done - phase_probe_start);
                phase_trace_add(params, PhaseTraceWg0ScaleBSparseProbeCalls, 1);
                phase_scale_start = phase_probe_done;
              }
#endif
            }
          }
	          mainloop.pipe.producer_wait(mainloop.state, producer_done);
#if HANDWRITTEN_TMA_PHASE_TRACE
	          if (phase_trace_producer_lane(params, lane_idx)) {
	            const uint64_t phase_after_wait = phase_trace_clock();
	            phase_trace_add(
	                params,
	                PhaseTraceScaleBWait,
	                phase_after_wait - phase_scale_start);
	            phase_scale_start = phase_after_wait;
	          }
#endif
	          load_scale_B(k);
	          mainloop.pipe.producer_arrive_cpasync(mainloop.state);
#if HANDWRITTEN_TMA_PHASE_TRACE
	          if (phase_trace_producer_lane(params, lane_idx)) {
	            phase_trace_add(
	                params,
	                PhaseTraceScaleBCopy,
	                phase_trace_clock() - phase_scale_start);
	          }
#endif
	          mainloop.state.advance();
	        }

        if (single_tile_cta) {
          work_info.valid = false;
        } else {
          clcloop.pipe.consumer_arrive(clcloop.state);
          {
            int4 *clc_ret_ptr = &clcloop.data().clc_ret;
            clcloop.pipe.consumer_wait(clcloop.state);
            auto [ret, valid] = ptx::clc_query(clc_ret_ptr);
            update_work_info(ret, valid, work_info);
          }
          clcloop.state.advance();
        }

      }
      if (!Mode49Specialized && params.direct_probe_mixed_cta == 3) {
        apply_incta_sparse_probe_for_tile<CtaM, CtaN>(
            params,
            storage.sparse_acc,
            real_grid_dim.y,
            work_info.blk_m,
            work_info.blk_n,
            lane_idx,
            static_cast<int>(ProducerRole::ScaleB),
            4,
            4,
            false);
      }
      if (!Mode49Specialized &&
          params.direct_probe_mixed_cta == 7 && params.direct_probe_do_math != 0 &&
          params.direct_probe_warps > 2) {
        int active_m_idx = -1;
        if (lane_idx == 0) {
          active_m_idx = find_active_mblock_index(params, work_info.blk_m);
        }
        active_m_idx = __shfl_sync(0xffffffffu, active_m_idx, 0);
        if (active_m_idx >= 0) {
          if (params.direct_delta_output != nullptr) {
            apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, true>(
                params,
                static_cast<uint32_t>(active_m_idx),
                work_info.blk_m,
                work_info.blk_n,
                lane_idx,
                2,
                params.direct_probe_warps,
                0,
                -1,
                params.direct_probe_group_budget);
          } else {
            apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, false>(
                params,
                static_cast<uint32_t>(active_m_idx),
                work_info.blk_m,
                work_info.blk_n,
                lane_idx,
                2,
                params.direct_probe_warps,
                0,
                -1,
                params.direct_probe_group_budget);
          }
        }
      }
    } else if (role == ProducerRole::Scheduler) {
      if (!Mode49Specialized &&
          (params.direct_probe_mixed_cta == 2 || params.direct_probe_mixed_cta == 3)) {
        const int probe_count = params.direct_probe_mixed_cta == 3 ? 4 : 1;
        const int row_sink_offset = params.direct_probe_mixed_cta == 3 ? 4 : 1;
        apply_incta_sparse_probe_for_tile<CtaM, CtaN>(
            params,
            storage.sparse_acc,
            real_grid_dim.y,
            work_info.blk_m,
            work_info.blk_n,
            lane_idx,
            params.direct_probe_mixed_cta == 3 ? static_cast<int>(ProducerRole::Scheduler) : 0,
            probe_count,
            row_sink_offset,
            params.direct_probe_mixed_cta == 2);
        return;
      }
      if (!Mode49Specialized && params.direct_probe_mixed_cta == 6) {
        if (params.direct_probe_do_math != 0) {
          int active_m_idx = -1;
          if (lane_idx == 0) {
            active_m_idx = find_active_mblock_index(params, work_info.blk_m);
          }
          active_m_idx = __shfl_sync(0xffffffffu, active_m_idx, 0);
          if (active_m_idx >= 0) {
            if (params.direct_delta_output != nullptr) {
              apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, true>(
                  params,
                  static_cast<uint32_t>(active_m_idx),
                  work_info.blk_m,
                  work_info.blk_n,
                  lane_idx,
                  0,
                  1,
                  0,
                  0,
                  params.direct_probe_group_budget);
            } else {
              apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, false>(
                  params,
                  static_cast<uint32_t>(active_m_idx),
                  work_info.blk_m,
                  work_info.blk_n,
                  lane_idx,
                  0,
                  1,
                  0,
                  0,
                  params.direct_probe_group_budget);
            }
          }
        } else if (lane_idx == 0 && params.direct_probe_sink != nullptr) {
          const int32_t tile_id =
              static_cast<int32_t>(work_info.blk_m * gridDim.x + work_info.blk_n);
          params.direct_probe_sink[tile_id] = 1.0f;
        }
        return;
      }
      if (!Mode49Specialized && params.direct_probe_mixed_cta == 7) {
        if (params.direct_probe_do_math != 0) {
          int active_m_idx = -1;
          if (lane_idx == 0) {
            active_m_idx = find_active_mblock_index(params, work_info.blk_m);
          }
          active_m_idx = __shfl_sync(0xffffffffu, active_m_idx, 0);
          if (active_m_idx >= 0) {
            if (params.direct_delta_output != nullptr) {
              apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, true>(
                  params,
                  static_cast<uint32_t>(active_m_idx),
                  work_info.blk_m,
                  work_info.blk_n,
                  lane_idx,
                  0,
                  params.direct_probe_warps,
                  0,
                  -1,
                  params.direct_probe_group_budget);
            } else {
              apply_sparse_loadfma_kmajor_bounded_groups_tile<CtaM, CtaN, false>(
                  params,
                  static_cast<uint32_t>(active_m_idx),
                  work_info.blk_m,
                  work_info.blk_n,
                  lane_idx,
                  0,
                  params.direct_probe_warps,
                  0,
                  -1,
                  params.direct_probe_group_budget);
            }
          }
        } else if (lane_idx == 0 && params.direct_probe_sink != nullptr) {
          const int32_t tile_id =
              static_cast<int32_t>(work_info.blk_m * gridDim.x + work_info.blk_n);
          params.direct_probe_sink[tile_id * params.direct_probe_warps] = 1.0f;
        }
        return;
      }
      if (!Mode49Specialized && params.direct_probe_mixed_cta == 8) {
        return;
      }
      if (lane_idx == 0) {
        if (single_tile_cta) {
          return;
        }
        uint32_t scratch_phase = 0;
        SchedulerScratch* scratch = &storage.scheduler;
        while (work_info.valid) {
          clcloop.pipe.producer_wait(clcloop.state);
          bool found_valid_task = false;
          uint32_t next_blk_n = 0;
          uint32_t next_blk_m = 0;
          uint32_t next_valid = 0;
          while (!found_valid_task) {
            ptx::mbarrier_arrive_expect_tx(&scratch->mbar, 16);
            ptx::clc_try_cancel(&scratch->clc_ret, &scratch->mbar);
            ptx::wait_barrier(&scratch->mbar, scratch_phase);
            scratch_phase ^= 1;
            auto [ret, valid] = ptx::clc_query(&scratch->clc_ret);
            if (!valid) {
              found_valid_task = true;
              next_valid = 0;
            } else if (ret.y < real_grid_dim.y) {
              found_valid_task = true;
              next_valid = 1;
              next_blk_n = ret.x;
              next_blk_m = ret.y;
            }
          }

          clcloop.data().clc_ret = scratch->clc_ret;
          clcloop.pipe.producer_commit(clcloop.state);
          clcloop.state.advance();
          work_info.blk_n = next_blk_n;
          work_info.blk_m = next_blk_m;
          work_info.valid = next_valid;
        }
      }
    }
  } else if (wg_id == 1 || wg_id == 2) {
    if (four_wg_reg_mode) {
      if constexpr (LaunchThreads == NumThreadsPerWarpGroup * 5) {
        ptx::warpgroup_reg_alloc<HANDWRITTEN_TMA_FORCE5WG_CONSUMER_REGS>();
      } else {
        ptx::warpgroup_reg_alloc<HANDWRITTEN_TMA_FORCE4WG_CONSUMER_REGS>();
      }
    } else {
      ptx::warpgroup_reg_alloc<HANDWRITTEN_TMA_CONSUMER_REGS>();
    }
    size_t m_offset = worker_m_offset<WorkerM>(widx);
    size_t n_offset = worker_n_offset<WorkerN>(widx);

    // each atom needs 4 bytes sfa & 4 bytes sfb
    uint32_t sfa[AtomRepM][AtomRepK], sfb[AtomRepN][AtomRepK];

    uint32_t regA[AtomRepK][AtomRepM][AtomRegA];
    uint32_t regB[AtomRepK][AtomRepN][AtomRegB];
    float regC[AtomRepM][AtomRepN][AtomRegC];
    float reg_tmp[AtomRepM][AtomRepN][AtomRegC];

    constexpr int AtomThrdFragRow = AtomM / 8;
    constexpr int AtomThrdFragCol = AtomN / 8 * 2;
    float outer_scale_A[AtomRepM][AtomThrdFragRow];
    float outer_scale_B[AtomRepN][AtomThrdFragCol];

    auto ldmatrix = [&] __attribute__((always_inline))(uint32_t const &k_pipe) {
      #pragma unroll
      for (int i = 0; i < AtomRepM; ++i)
        ptx::ldmatrix_a_b4_32x64(
          SmemPtrSw<64>(mainloop.data().sA),
          regA[k_pipe][i],
          m_offset + i * AtomM,
          k_pipe * AtomK,
          CtaK / 2,
          lane_idx
        );
      #pragma unroll
      for (int j = 0; j < AtomRepN; ++j)
        ptx::ldmatrix_b_b4_32x64(
          SmemPtrSw<64>(mainloop.data().sB),
          regB[k_pipe][j],
          n_offset + j * AtomN,
          k_pipe * AtomK,
          CtaK / 2,
          lane_idx
        );
    };

    auto mma = [&] __attribute__((always_inline))(uint32_t const &k_pipe) {
      #pragma unroll
      for (int i = 0; i < AtomRepM; ++i) {
        #pragma unroll
        for (int j = 0; j < AtomRepN; ++j) {
          ptx::mma_nvfp4_32x32x64(
            regA[k_pipe][i],
            regB[k_pipe][j],
            reg_tmp[i][j],
            sfa[i][k_pipe],
            sfb[j][k_pipe]
          );
        }
      }
	    };
	
	    uint32_t compact_sync_phase = 0;
	    int compact_stage = 0;
	    while (work_info.valid) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	      uint64_t phase_dense_total_start = 0;
	      if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	        phase_dense_total_start = phase_trace_clock();
	      }
#endif
	      // init reg
	      #pragma unroll
	      for (int i = 0; i < AtomRepM; ++i)
        #pragma unroll
        for (int j = 0; j < AtomRepN; ++j)
          #pragma unroll
          for (int k = 0; k < AtomRegC; ++k)
            regC[i][j][k] = reg_tmp[i][j][k] = 0.0f;
	      #pragma unroll 1
	      for (int k_blk = 0; k_blk < kstages; ++k_blk) {
#if HANDWRITTEN_TMA_PHASE_TRACE
	        uint64_t phase_dense_start = 0;
	        if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	          phase_dense_start = phase_trace_clock();
	        }
#endif
	        auto token = mainloop.pipe.consumer_try_wait(mainloop.state);
	        mainloop.pipe.consumer_wait(mainloop.state, token);
#if HANDWRITTEN_TMA_PHASE_TRACE
	        if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	          const uint64_t phase_after_wait = phase_trace_clock();
	          phase_trace_add(
	              params,
	              PhaseTraceDenseWait,
	              phase_after_wait - phase_dense_start);
	          phase_dense_start = phase_after_wait;
	        }
#endif
	
	        // load inner scale
	        constexpr int AtomISBytes = AtomK / 16 * sizeof(uint8_t);
	        #pragma unroll
        for (int i = 0; i < AtomRepM; ++i)
          load<AtomRepK * AtomISBytes>
            (sfa[i], mainloop.data().sSFA + (m_offset + AtomM * i + lane_inner_scale_offset(lane_idx)) * AtomRepK);
	        #pragma unroll
	        for (int j = 0; j < AtomRepN; ++j)
	          load<AtomRepK * AtomISBytes>
	            (sfb[j], mainloop.data().sSFB + (n_offset + AtomN * j + lane_inner_scale_offset(lane_idx)) * AtomRepK);
#if HANDWRITTEN_TMA_PHASE_TRACE
	        if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	          const uint64_t phase_after_scale = phase_trace_clock();
	          phase_trace_add(
	              params,
	              PhaseTraceDenseScaleLoad,
	              phase_after_scale - phase_dense_start);
	          phase_dense_start = phase_after_scale;
	        }
#endif
	        
	        // load data & calc
	        ldmatrix(0);
#if HANDWRITTEN_TMA_PHASE_TRACE
	        if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	          const uint64_t phase_after_ldmatrix = phase_trace_clock();
	          phase_trace_add(
	              params,
	              PhaseTraceDenseLdmatrix,
	              phase_after_ldmatrix - phase_dense_start);
	          phase_dense_start = phase_after_ldmatrix;
	        }
#endif
	        #pragma unroll
	        for (uint32_t k_pipe = 0; k_pipe < AtomRepK; ++k_pipe) {
	          if (k_pipe < AtomRepK - 1) {
	            ldmatrix(k_pipe + 1);
#if HANDWRITTEN_TMA_PHASE_TRACE
	            if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	              const uint64_t phase_after_ldmatrix = phase_trace_clock();
	              phase_trace_add(
	                  params,
	                  PhaseTraceDenseLdmatrix,
	                  phase_after_ldmatrix - phase_dense_start);
	              phase_dense_start = phase_after_ldmatrix;
	            }
#endif
	            if (k_pipe == AtomRepK - 2) {
	              // last atom has been loaded into registers
	              mainloop.pipe.consumer_arrive(mainloop.state);
	            }
	          }
	          mma(k_pipe);
#if HANDWRITTEN_TMA_PHASE_TRACE
	          if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	            const uint64_t phase_after_mma = phase_trace_clock();
	            phase_trace_add(
	                params,
	                PhaseTraceDenseMma,
	                phase_after_mma - phase_dense_start);
	            phase_dense_start = phase_after_mma;
	          }
#endif
	        }
	
#if HANDWRITTEN_TMA_PHASE_TRACE
	        if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	          phase_dense_start = phase_trace_clock();
	        }
#endif
	        #pragma unroll
	        for (int i = 0; i < AtomRepM; i++) {
	          #pragma unroll
	          for (int j = 0; j < AtomRepN; j++) {
            #pragma unroll
            for (int k = 0; k < AtomRegC; k++) {
              regC[i][j][k] += reg_tmp[i][j][k];
	              reg_tmp[i][j][k] = 0.0f;
	            }
	          }
	        }
#if HANDWRITTEN_TMA_PHASE_TRACE
	        if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	          phase_trace_add(
	              params,
	              PhaseTraceDenseAccum,
	              phase_trace_clock() - phase_dense_start);
	        }
#endif
	
	        mainloop.state.advance();
	      }
#if HANDWRITTEN_TMA_PHASE_TRACE
	      if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	        phase_trace_add(
	            params,
	            PhaseTraceDenseTotal,
	            phase_trace_clock() - phase_dense_total_start);
	      }
#endif
		      if (!single_tile_cta) {
		        clcloop.pipe.consumer_arrive(clcloop.state);
		      }
		      if constexpr (CompactConsumerSpecialized) {
		        // MMA and mma_tail are complete.  Join WG3 only now, so compact
		        // staging overlaps the dense mainloop and accumulator lanes remain
		        // untouched until their A/B temporaries are dead.
		        compact_cross_wg_barrier(
		            &storage.sparse_probe.compact_sync_mbar,
		            compact_sync_phase,
		            lane_idx);
		      }
		      const uint32_t epi_bar_threads = 256u;
	      const bool smem_merge_delta =
	          !Mode49Specialized && params.direct_probe_mixed_cta == 18 &&
	          params.direct_delta_output != nullptr &&
	          params.direct_delta_write_mode == 4;
	      const bool smem_sharedacc_delta =
	          !Mode49Specialized && params.direct_probe_mixed_cta == 19;
		      const bool smem_subacc32_delta =
		          !Mode49Specialized && params.direct_probe_mixed_cta == 20;
			      const bool reg_local_delta_mode =
			          !Mode49Specialized && params.direct_probe_mixed_cta == 23;
			      const bool prestore_local_delta_mode =
			          !Mode49Specialized && params.direct_probe_mixed_cta == 35;
			      const bool side_merge_local_delta_mode =
			          !Mode49Specialized && params.direct_probe_mixed_cta == 36;
			      const bool all_lane_ready_wait_mode =
			          !Mode49Specialized && params.direct_probe_mixed_cta == 37;
				      const bool db_local_delta_mode =
				          !Mode49Specialized &&
				          (params.direct_probe_mixed_cta == 24 ||
				          params.direct_probe_mixed_cta == 25 ||
				          params.direct_probe_mixed_cta == 26 ||
				          params.direct_probe_mixed_cta == 27 ||
				          params.direct_probe_mixed_cta == 28 ||
				          params.direct_probe_mixed_cta == 29 ||
				          params.direct_probe_mixed_cta == 30 ||
				          params.direct_probe_mixed_cta == 31 ||
				          params.direct_probe_mixed_cta == 32 ||
				          params.direct_probe_mixed_cta == 33 ||
				          params.direct_probe_mixed_cta == 34 ||
				          params.direct_probe_mixed_cta == 37 ||
				          prestore_local_delta_mode ||
				          side_merge_local_delta_mode);
		      int reg_local_delta_active_m_idx = -1;
		      if (reg_local_delta_mode && lane_idx == 0) {
		        if (params.direct_probe_active_mblocks != nullptr &&
		            params.direct_probe_active_mblock_count == static_cast<int32_t>(real_grid_dim.y) &&
		            params.direct_probe_active_mblocks[work_info.blk_m] == static_cast<int32_t>(work_info.blk_m)) {
		          reg_local_delta_active_m_idx = static_cast<int>(work_info.blk_m);
		        } else {
		          reg_local_delta_active_m_idx =
		              find_active_mblock_index(params, work_info.blk_m);
		        }
		      }
		      reg_local_delta_active_m_idx =
		          __shfl_sync(0xffffffffu, reg_local_delta_active_m_idx, 0);
		      int reg_local_delta_hot_count = 0;
		      if (reg_local_delta_mode &&
		          reg_local_delta_active_m_idx >= 0 &&
		          params.direct_kmajor_group_offsets != nullptr) {
		        reg_local_delta_hot_count =
		            params.direct_kmajor_group_offsets[reg_local_delta_active_m_idx + 1] -
		            params.direct_kmajor_group_offsets[reg_local_delta_active_m_idx];
		      }
		      const bool reg_local_delta =
		          reg_local_delta_mode && reg_local_delta_hot_count > 0;
			      const bool smem_local_delta =
			          !Mode49Specialized &&
			          (params.direct_probe_mixed_cta == 21 ||
			          params.direct_probe_mixed_cta == 22 ||
			          reg_local_delta_mode ||
			          db_local_delta_mode);
		      const bool packed_local_delta_payload =
		          db_local_delta_mode &&
		          params.direct_packed_payload_mode != 0 &&
		          params.direct_packed_tile_offsets != nullptr &&
		          params.direct_packed_row_records != nullptr;
		      const int packed_local_delta_count =
		          packed_local_delta_payload
		              ? params.direct_packed_tile_offsets[work_info.blk_m + 1] -
		                    params.direct_packed_tile_offsets[work_info.blk_m]
		              : 0;
		      const bool local_delta_has_rows =
		          smem_local_delta &&
		          (packed_local_delta_payload
		               ? packed_local_delta_count > 0
		               : (params.direct_active_row_offsets != nullptr &&
		                  params.direct_active_row_offsets[work_info.blk_m + 1] >
		                      params.direct_active_row_offsets[work_info.blk_m]));
		      const int local_delta_active_count =
		          local_delta_has_rows
		              ? (packed_local_delta_payload
		                     ? packed_local_delta_count
		                     : params.direct_active_row_offsets[work_info.blk_m + 1] -
		                           params.direct_active_row_offsets[work_info.blk_m])
		              : 0;
			      const bool local_delta_full_rows =
			          local_delta_active_count >= CtaM;
			      constexpr int LocalDeltaConsumerStages = CtaN / EpiN;
			      constexpr bool LocalDeltaStageBuffered =
			          HANDWRITTEN_TMA_LOCAL_DELTA_STAGE_BUFFERS >= LocalDeltaConsumerStages;
				      const bool direct_reg_epilogue_sparse =
				          !Mode49Specialized && params.direct_smem_add == 14 &&
			          params.direct_active_row_offsets != nullptr &&
			          params.direct_active_rows != nullptr;
			      if (direct_reg_epilogue_sparse) {
			        #pragma unroll
			        for (int idx = widx * NumThreadsPerWarp + lane_idx;
			             idx < (CtaM + 31) / 32;
			             idx += NumThreadsPerWarp * WorkerRepM * WorkerRepN) {
			          storage.sparse_local_delta.row_mask[idx] = 0u;
			        }
			        ptx::bar_sync(2, epi_bar_threads);
			        const int active_start = params.direct_active_row_offsets[work_info.blk_m];
			        const int active_end = params.direct_active_row_offsets[work_info.blk_m + 1];
			        for (int active_idx = active_start + widx * NumThreadsPerWarp + lane_idx;
			             active_idx < active_end;
			             active_idx += NumThreadsPerWarp * WorkerRepM * WorkerRepN) {
			          const int local_row = params.direct_active_rows[active_idx];
			          if (local_row >= 0 && local_row < CtaM) {
			            atomicOr(
			                reinterpret_cast<unsigned int*>(
			                    &storage.sparse_local_delta.row_mask[local_row >> 5]),
			                1u << (local_row & 31));
			          }
			        }
			        ptx::bar_sync(2, epi_bar_threads);
			      }
			      if (smem_merge_delta || smem_sharedacc_delta) {
			        if (widx == 0 && lane_idx == 0) {
			          volatile uint32_t* done =
	              reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.done_count);
	          while (*done < static_cast<uint32_t>(params.direct_probe_warps)) {
	          }
	          __threadfence_block();
	        }
	        ptx::bar_sync(2, epi_bar_threads);
	      }
	      
	      // epilogue
	      {
#if HANDWRITTEN_TMA_PHASE_TRACE
	        uint64_t phase_epilogue_total_start = 0;
	        uint64_t phase_epilogue_start = 0;
	        if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	          phase_epilogue_total_start = phase_trace_clock();
	          phase_epilogue_start = phase_epilogue_total_start;
	        }
#endif
	        const float alpha =
	          params.amax_A[0] * params.amax_B[0] /
	          (6.0f * 6.0f * 448.0f * 448.0f);
        auto sC = SmemPtrSw(storage.data.epilogue.sC);
        uint32_t regD[AtomRepM][AtomRepN][AtomRegC / 2];
	        if constexpr (CompactConsumerSpecialized) {
	          static_assert(Mode49StaticN4096,
	                        "compact consumer currently requires compile-time N=4096");
	          constexpr int MaxNnz = HANDWRITTEN_TMA_COMPACT_CONSUMER_MAX_NNZ;
	          static_assert(MaxNnz >= 1 && MaxNnz <= 16,
	                        "compact consumer supports caps in [1, 16]");
	          #pragma unroll
	          for (int i = 0; i < AtomRepM; ++i) {
	            #pragma unroll
	            for (int j = 0; j < AtomRepN; ++j) {
	              // The MMA fragment maps k_pair as row_fragment + 4*col_fragment.
	              // Load one row record, reuse it for four output column pairs,
	              // then release those temporaries before moving to the next row.
	              #pragma unroll
	              for (int row_fragment = 0; row_fragment < 4; ++row_fragment) {
	                const int local_row =
	                    static_cast<int>(m_offset) + i * AtomM +
	                    (lane_idx >> 2) + row_fragment * 8;
	                const uint32_t compact_count =
	                    storage.sparse_compact_input[compact_stage].counts[local_row];
	                const int global_col_base =
	                    static_cast<int>(work_info.blk_n) * CtaN +
	                    static_cast<int>(n_offset) + j * AtomN +
	                    ((lane_idx & 3) << 1);
	                if constexpr (MaxNnz <= 4) {
	                  apply_compact_consumer_row<MaxNnz>(
	                      storage.sparse_compact_input[compact_stage],
	                      params,
	                      compact_count,
	                      local_row,
	                      global_col_base,
	                      row_fragment,
	                      alpha,
	                      regC[i][j],
	                      regD[i][j]);
	                } else if constexpr (MaxNnz <= 8) {
	                  if (compact_count <= 4) {
	                    apply_compact_consumer_row<4>(
	                        storage.sparse_compact_input[compact_stage],
	                        params,
	                        compact_count,
	                        local_row,
	                        global_col_base,
	                        row_fragment,
	                        alpha,
	                        regC[i][j],
	                        regD[i][j]);
	                  } else {
	                    apply_compact_consumer_row<MaxNnz>(
	                        storage.sparse_compact_input[compact_stage],
	                        params,
	                        compact_count,
	                        local_row,
	                        global_col_base,
	                        row_fragment,
	                        alpha,
	                        regC[i][j],
	                        regD[i][j]);
	                  }
	                } else {
	                  if (compact_count <= 4) {
	                    apply_compact_consumer_row<4>(
	                        storage.sparse_compact_input[compact_stage],
	                        params,
	                        compact_count,
	                        local_row,
	                        global_col_base,
	                        row_fragment,
	                        alpha,
	                        regC[i][j],
	                        regD[i][j]);
	                  } else if (compact_count <= 8) {
	                    apply_compact_consumer_row<8>(
	                        storage.sparse_compact_input[compact_stage],
	                        params,
	                        compact_count,
	                        local_row,
	                        global_col_base,
	                        row_fragment,
	                        alpha,
	                        regC[i][j],
	                        regD[i][j]);
	                  } else {
	                    apply_compact_consumer_row<MaxNnz>(
	                        storage.sparse_compact_input[compact_stage],
	                        params,
	                        compact_count,
	                        local_row,
	                        global_col_base,
	                        row_fragment,
	                        alpha,
	                        regC[i][j],
	                        regD[i][j]);
	                  }
	                }
	              }
	            }
	          }
	        } else {
	          #pragma unroll
	          for (int i = 0; i < AtomRepM; ++i) {
	            #pragma unroll
	            for (int j = 0; j < AtomRepN; ++j) {
	              #pragma unroll
	              for (int k = 0; k < AtomRegC / 2; ++k) {
	                regD[i][j][k] = ptx::cvt_fp32_to_bf16x2(
	                  alpha * regC[i][j][k * 2],
	                  alpha * regC[i][j][k * 2 + 1]
	                );
	              }
	            }
	          }
	        }
	        if constexpr (CompactConsumerSpecialized) {
	          // All reads from this generation are complete; switch to the
	          // buffer WG3 has been allowed to prepare in parallel.
	          compact_stage ^= 1;
	        }
#if HANDWRITTEN_TMA_PHASE_TRACE
	        if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	          const uint64_t phase_after_convert = phase_trace_clock();
	          phase_trace_add(
	              params,
	              PhaseTraceEpilogueConvert,
	              phase_after_convert - phase_epilogue_start);
	        }
#endif

	        for (int epi_m = 0; epi_m < CtaM / EpiM; ++epi_m) {
	          for (int epi_n = 0; epi_n < CtaN / EpiN; ++epi_n) {
	            int32_t epi_st_m_idx = epi_m * EpiM;
	            int32_t epi_ed_m_idx = epi_st_m_idx + EpiM;
	            int32_t epi_st_n_idx = epi_n * EpiN;
	            int32_t epi_ed_n_idx = epi_st_n_idx + EpiN;
	            if ((reg_local_delta || prestore_local_delta_mode) && local_delta_has_rows) {
	              const int stage_id = epi_st_n_idx / EpiN + 1;
#if HANDWRITTEN_TMA_PHASE_TRACE
	              uint64_t phase_prestore_stage_start = 0;
	              if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	                phase_prestore_stage_start = phase_trace_clock();
	              }
#endif
	              volatile uint32_t* ready =
	                  reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.done_count);
#if HANDWRITTEN_TMA_ALL_LANE_READY_WAIT
	              if (prestore_local_delta_mode) {
	                const uint32_t ready_bit = 1u << (stage_id - 1);
	                while ((*ready & ready_bit) == 0u) {
	                }
	              } else {
	                while (*ready < static_cast<uint32_t>(stage_id)) {
	                }
	              }
	              __threadfence_block();
#else
	              if (widx == 0 && lane_idx == 0) {
	                if (prestore_local_delta_mode) {
	                  const uint32_t ready_bit = 1u << (stage_id - 1);
	                  while ((*ready & ready_bit) == 0u) {
	                  }
	                } else {
	                  while (*ready < static_cast<uint32_t>(stage_id)) {
	                  }
	                }
	                __threadfence_block();
	              }
#endif
#if HANDWRITTEN_TMA_PHASE_TRACE
	              uint64_t phase_after_prestore_spin = 0;
	              if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	                phase_after_prestore_spin = phase_trace_clock();
	                phase_trace_add(
	                    params,
	                    PhaseTraceEpilogueReadySpin,
	                    phase_after_prestore_spin - phase_prestore_stage_start);
	              }
#endif
#if !HANDWRITTEN_TMA_ALL_LANE_READY_WAIT
	              ptx::bar_sync(2, epi_bar_threads);
#endif
#if HANDWRITTEN_TMA_PHASE_TRACE
	              if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	                const uint64_t phase_after_prestore_wait = phase_trace_clock();
#if !HANDWRITTEN_TMA_ALL_LANE_READY_WAIT
	                phase_trace_add(
	                    params,
	                    PhaseTraceEpilogueReadyBarrier,
	                    phase_after_prestore_wait - phase_after_prestore_spin);
#endif
	                const uint64_t phase_prestore_wait_elapsed =
	                    phase_after_prestore_wait - phase_prestore_stage_start;
	                phase_trace_add(
	                    params,
	                    PhaseTraceEpilogueStageWait,
	                    phase_prestore_wait_elapsed);
	                if (stage_id == 1) {
	                  phase_trace_add(
	                      params,
	                      PhaseTraceEpilogueStage0Wait,
	                      phase_prestore_wait_elapsed);
	                } else if (stage_id == 2) {
	                  phase_trace_add(
	                      params,
	                      PhaseTraceEpilogueStage1Wait,
	                      phase_prestore_wait_elapsed);
	                }
	              }
#endif
	            }
		            if (m_offset >= epi_st_m_idx && m_offset < epi_ed_m_idx &&
		                n_offset >= epi_st_n_idx && n_offset < epi_ed_n_idx) {
	              int32_t epi_m_offset = m_offset - epi_st_m_idx;
	              int32_t epi_n_offset = n_offset - epi_st_n_idx;
		              #pragma unroll
		              for (int i = 0; i < AtomRepM; ++i) {
		                #pragma unroll
		                for (int j = 0; j < AtomRepN; ++j) {
		                  if (direct_reg_epilogue_sparse) {
		                    #pragma unroll
		                    for (int k = 0; k < AtomRegC / 2; ++k) {
		                      const int atom_row =
		                          (lane_idx >> 2) + ((k & 3) << 3);
		                      const int atom_col =
		                          ((lane_idx & 3) << 1) + ((k >> 2) << 3);
		                      const int local_row =
		                          static_cast<int>(i * AtomM + epi_m_offset + atom_row);
		                      const int local_col =
		                          static_cast<int>(j * AtomN + epi_n_offset + atom_col);
		                      const int64_t global_row =
		                          static_cast<int64_t>(work_info.blk_m) * CtaM + local_row;
		                      const int64_t global_col =
		                          static_cast<int64_t>(work_info.blk_n) * CtaN +
		                          epi_st_n_idx + local_col;
		                      bool row_has_sparse = false;
		                      if (local_row >= 0 && local_row < CtaM) {
		                        const uint32_t row_word =
		                            storage.sparse_local_delta.row_mask[local_row >> 5];
		                        row_has_sparse =
		                            (row_word & (1u << (local_row & 31))) != 0u;
		                      }
		                      if (row_has_sparse) {
		                        regD[i][j][k] = direct_reg_add_sparse_bf16x2(
		                            params,
		                            regD[i][j][k],
		                            global_row,
		                            global_col);
		                      }
		                    }
		                  }
		                  if ((reg_local_delta || prestore_local_delta_mode) && local_delta_has_rows) {
#if HANDWRITTEN_TMA_PHASE_TRACE
		                    uint64_t phase_prestore_merge_start = 0;
		                    if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	                      phase_prestore_merge_start = phase_trace_clock();
	                    }
#endif
	                    const uint16_t* prestage_delta_tile =
	                        storage.sparse_local_delta.tile +
	                        (prestore_local_delta_mode && LocalDeltaStageBuffered
	                             ? static_cast<int64_t>(epi_st_n_idx / EpiN) * CtaM * EpiN
	                             : 0);
	                    #pragma unroll
	                    for (int k = 0; k < AtomRegC / 2; ++k) {
	                      const int atom_row =
	                          (lane_idx >> 2) + ((k & 3) << 3);
	                      const int atom_col =
	                          ((lane_idx & 3) << 1) + ((k >> 2) << 3);
	                      const int local_row =
	                          static_cast<int>(i * AtomM + epi_m_offset + atom_row);
	                      const int local_col =
	                          static_cast<int>(j * AtomN + epi_n_offset + atom_col);
	                      const int64_t global_row =
	                          static_cast<int64_t>(work_info.blk_m) * CtaM + local_row;
	                      const int64_t global_col =
	                          static_cast<int64_t>(work_info.blk_n) * CtaN +
	                          epi_st_n_idx + local_col;
	                      uint16_t delta_lo = 0;
	                      uint16_t delta_hi = 0;
	                      bool row_has_delta = local_delta_full_rows;
	                      if (!row_has_delta && local_row >= 0 && local_row < CtaM) {
	                        const uint32_t row_word =
	                            storage.sparse_local_delta.row_mask[local_row >> 5];
	                        row_has_delta =
	                            (row_word & (1u << (local_row & 31))) != 0u;
	                      }
	                      if (global_row >= 0 && global_row < params.m &&
	                          local_row >= 0 && local_row < CtaM &&
	                          local_col >= 0 && local_col + 1 < EpiN &&
	                          row_has_delta) {
	                        if (global_col < params.n) {
	                          delta_lo =
	                              prestage_delta_tile[
	                                  static_cast<int64_t>(local_row) * EpiN + local_col];
	                        }
	                        if (global_col + 1 < params.n) {
	                          delta_hi =
	                              prestage_delta_tile[
	                                  static_cast<int64_t>(local_row) * EpiN + local_col + 1];
	                        }
	                      }
	                      regD[i][j][k] =
	                          direct_bf16_add2_u16_delta(regD[i][j][k], delta_lo, delta_hi);
	                    }
#if HANDWRITTEN_TMA_PHASE_TRACE
	                    if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	                      const uint64_t phase_prestore_merge_elapsed =
	                          phase_trace_clock() - phase_prestore_merge_start;
	                      const int stage_id = epi_st_n_idx / EpiN + 1;
	                      phase_trace_add(
	                          params,
	                          PhaseTraceEpilogueLocalDeltaMerge,
	                          phase_prestore_merge_elapsed);
	                      if (stage_id == 1) {
	                        phase_trace_add(
	                            params,
	                            PhaseTraceEpilogueStage0Merge,
	                            phase_prestore_merge_elapsed);
	                      } else if (stage_id == 2) {
	                        phase_trace_add(
	                            params,
	                            PhaseTraceEpilogueStage1Merge,
	                            phase_prestore_merge_elapsed);
	                      }
	                    }
#endif
	                  }
		                  ptx::stmatrix_b16_32x32(
	                    sC,
	                    regD[i][j],
                    i * AtomM + epi_m_offset,
                    j * AtomN + epi_n_offset,
                    EpiN * sizeof(uint16_t),
                    lane_idx
                  );
                }
              }
              ptx::fence_shared_async();
            }
            ptx::bar_sync(2, epi_bar_threads);
            if (smem_merge_delta) {
              apply_sparse_entry_delta_smem_merge_noprobe<CtaM, CtaN, EpiN>(
                  params,
                  storage.data.epilogue.sC,
                  work_info.blk_m,
                  work_info.blk_n,
                  epi_st_n_idx,
                  widx,
                  lane_idx);
              ptx::bar_sync(2, epi_bar_threads);
            } else if (smem_sharedacc_delta) {
              apply_sparse_acc_smem_merge_tile<
                  CtaM,
                  CtaN,
                  EpiN,
                  HANDWRITTEN_TMA_SPARSE_ACC_ROWS>(
                  params,
                  storage.sparse_acc,
                  storage.data.epilogue.sC,
                  work_info.blk_m,
                  work_info.blk_n,
                  epi_st_n_idx,
	              widx,
	                  lane_idx);
	              ptx::bar_sync(2, epi_bar_threads);
			            } else if (!reg_local_delta && !prestore_local_delta_mode &&
			                       smem_local_delta && local_delta_has_rows) {
			              const int stage_id = epi_st_n_idx / EpiN + 1;
			              const uint16_t* local_delta_stage_tile =
			                  storage.sparse_local_delta.tile +
			                  (db_local_delta_mode && LocalDeltaStageBuffered
			                       ? static_cast<int64_t>(stage_id - 1) * CtaM * EpiN
			                       : 0);
			              if (side_merge_local_delta_mode) {
			                if (widx == 0 && lane_idx == 0) {
			                  __threadfence_block();
			                  atomicOr(
			                      reinterpret_cast<unsigned int*>(&storage.sparse_probe.sc_ready),
			                      1u << (stage_id - 1));
			                }
			              } else {
#if HANDWRITTEN_TMA_PHASE_TRACE
			              uint64_t phase_epilogue_stage_start = 0;
			              if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
		                phase_epilogue_stage_start = phase_trace_clock();
		              }
#endif
		              volatile uint32_t* ready =
		                  reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.done_count);
#if HANDWRITTEN_TMA_ALL_LANE_READY_WAIT
			              const bool stage_ready_words =
			                  HANDWRITTEN_TMA_STAGE_READY_WORDS != 0 &&
			                  (params.direct_probe_mixed_cta == 34 ||
			                   all_lane_ready_wait_mode) &&
			                  LocalDeltaStageBuffered &&
			                  params.direct_probe_warps >= LocalDeltaConsumerStages &&
			                  (params.direct_probe_warps % LocalDeltaConsumerStages) == 0;
			              const bool per_warp_stage_ready =
			                  HANDWRITTEN_TMA_STAGE_READY_PER_WARP != 0 &&
			                  (params.direct_probe_mixed_cta == 34 ||
			                   all_lane_ready_wait_mode) &&
			                  LocalDeltaStageBuffered &&
			                  params.direct_probe_warps >= LocalDeltaConsumerStages &&
			                  (params.direct_probe_warps % LocalDeltaConsumerStages) == 0;
			              if (stage_ready_words) {
			                volatile uint32_t* stage_ready =
			                    reinterpret_cast<volatile uint32_t*>(
			                        &storage.sparse_probe.stage_ready[stage_id - 1]);
			                while (*stage_ready == 0u) {
			                }
			              } else if (per_warp_stage_ready) {
			                const int stage_warps =
			                    params.direct_probe_warps / LocalDeltaConsumerStages;
			                const uint32_t ready_mask =
			                    ((1u << stage_warps) - 1u) << ((stage_id - 1) * stage_warps);
			                while ((*ready & ready_mask) != ready_mask) {
			                }
			              } else if (params.direct_probe_mixed_cta == 33 ||
			                         params.direct_probe_mixed_cta == 34 ||
			                         all_lane_ready_wait_mode) {
			                const uint32_t ready_bit = 1u << (stage_id - 1);
			                while ((*ready & ready_bit) == 0u) {
			                }
			              } else {
		                while (*ready < static_cast<uint32_t>(stage_id)) {
		                }
		              }
		              __threadfence_block();
#else
		              if (all_lane_ready_wait_mode) {
		                const uint32_t ready_bit = 1u << (stage_id - 1);
		                while ((*ready & ready_bit) == 0u) {
		                }
#if !HANDWRITTEN_TMA_SKIP_READY_CONSUMER_FENCE
		                __threadfence_block();
#endif
		              } else if (widx == 0 && lane_idx == 0) {
		                if (params.direct_probe_mixed_cta == 33 ||
		                    params.direct_probe_mixed_cta == 34) {
		                  const uint32_t ready_bit = 1u << (stage_id - 1);
		                  while ((*ready & ready_bit) == 0u) {
		                  }
		                } else {
		                  while (*ready < static_cast<uint32_t>(stage_id)) {
		                  }
		                }
		                __threadfence_block();
		              }
#endif
#if HANDWRITTEN_TMA_PHASE_TRACE
		              uint64_t phase_after_ready_spin = 0;
		              if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
		                phase_after_ready_spin = phase_trace_clock();
		                phase_trace_add(
		                    params,
		                    PhaseTraceEpilogueReadySpin,
		                    phase_after_ready_spin - phase_epilogue_stage_start);
		              }
#endif
#if !HANDWRITTEN_TMA_ALL_LANE_READY_WAIT
		              if (!all_lane_ready_wait_mode) {
		                ptx::bar_sync(2, epi_bar_threads);
		              }
#endif
#if HANDWRITTEN_TMA_PHASE_TRACE
		              if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
		                const uint64_t phase_after_stage_wait = phase_trace_clock();
#if !HANDWRITTEN_TMA_ALL_LANE_READY_WAIT
		                if (!all_lane_ready_wait_mode) {
		                  phase_trace_add(
		                      params,
		                      PhaseTraceEpilogueReadyBarrier,
		                      phase_after_stage_wait - phase_after_ready_spin);
		                }
#endif
		                const uint64_t phase_stage_wait_elapsed =
		                    phase_after_stage_wait - phase_epilogue_stage_start;
		                phase_trace_add(
		                    params,
		                    PhaseTraceEpilogueStageWait,
		                    phase_stage_wait_elapsed);
		                if (stage_id == 1) {
		                  phase_trace_add(
		                      params,
		                      PhaseTraceEpilogueStage0Wait,
		                      phase_stage_wait_elapsed);
		                } else if (stage_id == 2) {
		                  phase_trace_add(
		                      params,
		                      PhaseTraceEpilogueStage1Wait,
		                      phase_stage_wait_elapsed);
		                }
		                phase_epilogue_stage_start = phase_after_stage_wait;
		              }
#endif
		              const bool use_packed_local_delta_merge =
		                  db_local_delta_mode &&
		                  params.direct_packed_payload_mode != 0 &&
		                  params.direct_packed_tile_offsets != nullptr &&
		                  params.direct_packed_row_records != nullptr;
		              if (use_packed_local_delta_merge) {
		                apply_packed_local_delta_smem_tile_vec8<CtaM, CtaN, EpiN>(
		                    params,
		                    local_delta_stage_tile,
		                    storage.data.epilogue.sC,
		                    work_info.blk_m,
		                    work_info.blk_n,
		                    epi_st_n_idx,
		                    widx,
		                    lane_idx);
		              } else {
		                apply_local_delta_smem_tile_vec8<CtaM, CtaN, EpiN>(
		                    params,
		                    local_delta_stage_tile,
		                    storage.data.epilogue.sC,
		                    work_info.blk_m,
		                    work_info.blk_n,
		                    epi_st_n_idx,
		                    widx,
		                    lane_idx);
		              }
#if HANDWRITTEN_TMA_PHASE_TRACE
		              if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
		                const uint64_t phase_stage_merge_elapsed =
		                    phase_trace_clock() - phase_epilogue_stage_start;
		                phase_trace_add(
		                    params,
		                    PhaseTraceEpilogueLocalDeltaMerge,
		                    phase_stage_merge_elapsed);
		                if (stage_id == 1) {
		                  phase_trace_add(
		                      params,
		                      PhaseTraceEpilogueStage0Merge,
		                      phase_stage_merge_elapsed);
		                } else if (stage_id == 2) {
		                  phase_trace_add(
		                      params,
		                      PhaseTraceEpilogueStage1Merge,
		                      phase_stage_merge_elapsed);
		                }
		              }
#endif
			              ptx::bar_sync(2, epi_bar_threads);
			              const bool skip_stage_consumed_epoch =
			                  HANDWRITTEN_TMA_SKIP_STAGE_CONSUMED_EPOCH != 0 &&
			                  (params.direct_probe_mixed_cta == 34 ||
			                   params.direct_probe_mixed_cta == 37) &&
			                  db_local_delta_mode &&
			                  LocalDeltaStageBuffered;
			              if (!skip_stage_consumed_epoch) {
				              if (widx == 0 && lane_idx == 0) {
#if !HANDWRITTEN_TMA_SKIP_READY_CONSUMER_FENCE
				                __threadfence_block();
#endif
				                atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.epoch),
				                           static_cast<unsigned int>(stage_id));
				              }
				              ptx::bar_sync(2, epi_bar_threads);
			              }
				              if (params.direct_probe_mixed_cta == 31) {
				                if (widx == 0 && lane_idx == 0) {
				                  volatile uint32_t* ready =
			                      reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.done_count);
			                  while (*ready < static_cast<uint32_t>(
			                                     LocalDeltaConsumerStages + stage_id)) {
				                  }
				                  __threadfence_block();
				                }
				                ptx::bar_sync(2, epi_bar_threads);
				              }
			              }
				            } else if (reg_local_delta && local_delta_has_rows) {
		              const int stage_id = epi_st_n_idx / EpiN + 1;
		              if (widx == 0 && lane_idx == 0) {
		                __threadfence_block();
		                atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.epoch),
		                           static_cast<unsigned int>(stage_id));
		              }
		              ptx::bar_sync(2, epi_bar_threads);
		            } else if (smem_subacc32_delta) {
              constexpr int SubN = 32;
              static_assert(EpiN % SubN == 0, "subacc32 expects EpiN divisible by 32");
              #pragma unroll
              for (int sub_n = 0; sub_n < EpiN; sub_n += SubN) {
                const int stage_id = (epi_st_n_idx + sub_n) / SubN + 1;
                if (widx == 0 && lane_idx == 0) {
                  volatile uint32_t* ready =
                      reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.done_count);
                  while (*ready < static_cast<uint32_t>(stage_id)) {
                  }
                  __threadfence_block();
                }
                ptx::bar_sync(2, epi_bar_threads);
                apply_sparse_subacc32_smem_merge_tile<
                    CtaM,
                    CtaN,
                    EpiN,
                    SubN,
                    HANDWRITTEN_TMA_SPARSE_ACC_ROWS>(
                    params,
                    storage.sparse_acc,
                    storage.data.epilogue.sC,
                    work_info.blk_m,
                    work_info.blk_n,
                    epi_st_n_idx,
                    sub_n,
                    widx,
                    lane_idx);
                ptx::bar_sync(2, epi_bar_threads);
                if (widx == 0 && lane_idx == 0) {
                  __threadfence_block();
                  atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.epoch),
                             static_cast<unsigned int>(stage_id));
                }
                ptx::bar_sync(2, epi_bar_threads);
              }
            }
            if (!Mode49Specialized &&
                params.direct_smem_add != 0 && params.direct_smem_add != 14) {
              if (params.direct_smem_add == 11) {
                apply_precomputed_delta_smem_tile<CtaM, CtaN, EpiN>(
                    params,
                    storage.data.epilogue.sC,
                    work_info.blk_m,
                    work_info.blk_n,
                    epi_st_n_idx,
                    widx,
                    lane_idx);
                ptx::fence_shared_async();
              } else if (params.direct_smem_add == 12) {
                apply_precomputed_delta_smem_tile_vec8<CtaM, CtaN, EpiN>(
                    params,
                    storage.data.epilogue.sC,
                    work_info.blk_m,
                    work_info.blk_n,
                    epi_st_n_idx,
                    widx,
                    lane_idx);
                ptx::fence_shared_async();
	              } else if (params.direct_smem_add == 1 || params.direct_smem_add == 3 ||
	                  params.direct_smem_add == 4 || params.direct_smem_add == 5 ||
	                  params.direct_smem_add == 6 || params.direct_smem_add == 7 ||
	                  params.direct_smem_add == 8 || params.direct_smem_add == 9 ||
	                  params.direct_smem_add == 10 || params.direct_smem_add == 13) {
                apply_direct_smem_sparse_tile<CtaM, CtaN, EpiN>(
                    params,
                    storage.data.epilogue.sC,
                    work_info.blk_m,
                    work_info.blk_n,
                    epi_st_n_idx,
                    widx,
                    lane_idx);
                if (params.direct_smem_add == 1) {
                  ptx::fence_shared_async();
                }
              }
              ptx::bar_sync(2, epi_bar_threads);
            }
		            if (widx == 0 && lane_idx == 0) {
#if HANDWRITTEN_TMA_PHASE_TRACE
		              uint64_t phase_merge_done_wait_start = 0;
		              if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
		                phase_merge_done_wait_start = phase_trace_clock();
		              }
#endif
		              if (side_merge_local_delta_mode && local_delta_has_rows) {
		                volatile uint32_t* merge_done =
		                    reinterpret_cast<volatile uint32_t*>(&storage.sparse_probe.merge_done);
		                const uint32_t stage_bit = 1u << (epi_st_n_idx / EpiN);
		                while ((*merge_done & stage_bit) == 0u) {
		                }
		                __threadfence_block();
		              }
#if HANDWRITTEN_TMA_PHASE_TRACE
		              if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
		                phase_trace_add(
		                    params,
		                    PhaseTraceDenseMergeDoneWait,
		                    phase_trace_clock() - phase_merge_done_wait_start);
		              }
#endif
#if HANDWRITTEN_TMA_PHASE_TRACE
		              uint64_t phase_tma_store_start = 0;
		              if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	                phase_tma_store_start = phase_trace_clock();
	              }
#endif
	              ptx::tma_store_2d(
	                &params.tensormap_D,
	                storage.data.epilogue.sC,
                work_info.blk_m * CtaM + epi_st_m_idx,
                work_info.blk_n * CtaN + epi_st_n_idx
	              );
	              ptx::tma_store_commit();
	              ptx::tma_wait();
#if HANDWRITTEN_TMA_PHASE_TRACE
	              if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	                phase_trace_add(
	                    params,
	                    PhaseTraceEpilogueTmaStore,
	                    phase_trace_clock() - phase_tma_store_start);
	              }
#endif
	            }
            __syncwarp();
            ptx::bar_sync(2, epi_bar_threads);
	          }
	        }
#if HANDWRITTEN_TMA_PHASE_TRACE
	        if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
	          phase_trace_add(
	              params,
	              PhaseTraceEpilogueTotal,
	              phase_trace_clock() - phase_epilogue_total_start);
	        }
#endif
	      }

	      if (!Mode49Specialized &&
	          params.direct_row_offsets != nullptr && params.direct_probe_sink == nullptr &&
	          params.direct_smem_add == 0) {
	        const int direct_worker_warps = direct_tailassist_4wg ? 12 : WorkerRepM * WorkerRepN;
	        if (direct_tailassist_4wg && widx == 0 && lane_idx == 0) {
	          __threadfence_block();
	          atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.done_count), 1u);
	        }
	        apply_direct_add_sparse_tile<CtaM, CtaN>(
	            params, work_info.blk_m, work_info.blk_n, widx, lane_idx, direct_worker_warps);
		        ptx::bar_sync(2, epi_bar_threads);
	      }

	      if (widx == 0 && lane_idx == 0) {
        if constexpr (!Mode49Specialized) {
	          const int32_t tile_id =
	              static_cast<int32_t>(work_info.blk_m * real_grid_dim.x + work_info.blk_n);
          if (params.ready_m_queue != nullptr) {
            __threadfence_system();
            const int32_t done = atomicAdd(params.ready_m_counts + work_info.blk_m, 1) + 1;
            if (done == real_grid_dim.x) {
              const int32_t slices = params.ready_m_slices < 1 ? 1 : params.ready_m_slices;
              for (int32_t slice = 0; slice < slices; ++slice) {
                const int32_t slot = atomicAdd(params.ready_m_tail, 1);
                params.ready_m_queue[slot] =
                    static_cast<int32_t>(work_info.blk_m) * slices + slice;
                __threadfence_system();
                params.ready_m_slot_status[slot] = 1;
              }
            }
          } else if (params.ready_queue != nullptr) {
            __threadfence_system();
            const int32_t slot = atomicAdd(params.ready_tail, 1);
            params.ready_queue[slot] = tile_id;
            __threadfence_system();
            params.ready_slot_status[slot] = 1;
          } else if (params.ready_flags != nullptr) {
            __threadfence();
            params.ready_flags[tile_id] = 1;
          }
	        }
	      }

	      if (!Mode49Specialized &&
	          (params.direct_probe_mixed_cta == 9 || params.direct_probe_mixed_cta == 10 ||
	             params.direct_probe_mixed_cta == 11 || params.direct_probe_mixed_cta == 12 ||
	             params.direct_probe_mixed_cta == 13 || params.direct_probe_mixed_cta == 14 ||
	             params.direct_probe_mixed_cta == 15 || params.direct_probe_mixed_cta == 16 ||
	             params.direct_probe_mixed_cta == 17 ||
	             params.direct_probe_mixed_cta == 26) &&
		          widx == 0 && lane_idx == 0) {
	        __threadfence_block();
	        const unsigned done_value =
	            params.direct_probe_mixed_cta == 26
	                ? static_cast<unsigned>(CtaN / EpiN + 1)
	                : 1u;
		        atomicExch(reinterpret_cast<unsigned int*>(&storage.sparse_probe.done_count),
		                   done_value);
		      }

#if HANDWRITTEN_TMA_PHASE_TRACE
		      if (phase_trace_dense_lane(params, wg_id, widx, lane_idx)) {
		        phase_trace_write(params, PhaseTraceKernelExit, phase_trace_clock());
		      }
#endif

		      if (single_tile_cta) {
		        work_info.valid = false;
	      } else {
        // clc
        {
          int4 *clc_ret_ptr = &clcloop.data().clc_ret;
          clcloop.pipe.consumer_wait(clcloop.state);
          auto [ret, valid] = ptx::clc_query(clc_ret_ptr);
          update_work_info(ret, valid, work_info);
        }
        clcloop.state.advance();
      }
    }
  }
}

__global__ void bf16_to_fp32(
  uint16_t *in, float *out,
  size_t size
) {
  uint64_t thr_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thr_id < size) *(out + thr_id) = __bfloat162float(*((__nv_bfloat16*)in+thr_id));
}

__global__ void fp16_to_fp32(
  uint16_t *in, float *out,
  size_t size
) {
  uint64_t thr_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (thr_id < size) *(out + thr_id) = __half2float(*((half*)in+thr_id));
}

__global__ __launch_bounds__(256, 2)
void merge_entry_delta_active_rows_vec8_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ delta_entries,
    const int32_t* __restrict__ active_rows,
    const int32_t* __restrict__ merge_row_offsets,
    const int32_t* __restrict__ merge_entry_indices,
    int64_t active_row_count,
    int64_t n);

__global__ __launch_bounds__(256, 2)
void merge_entry_delta_active_rows_vec8_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ delta_entries,
    const int32_t* __restrict__ active_rows,
    const int32_t* __restrict__ merge_row_offsets,
    const int32_t* __restrict__ merge_entry_indices,
    int64_t active_row_count,
    int64_t n) {
  constexpr int VecN = 8;
  const int64_t groups_per_row = (n + VecN - 1) / VecN;
  const int64_t total_groups = active_row_count * groups_per_row;
  for (int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < total_groups;
       group += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t active_idx = group / groups_per_row;
    const int64_t col_group = group - active_idx * groups_per_row;
    const int64_t row = static_cast<int64_t>(active_rows[active_idx]);
    const int64_t col0 = col_group * VecN;
    const int32_t start = merge_row_offsets[active_idx];
    const int32_t end = merge_row_offsets[active_idx + 1];
    float acc[VecN] = {};

    if (col0 + VecN <= n) {
      if (end == start + 1) {
        const int64_t entry = static_cast<int64_t>(merge_entry_indices[start]);
        const uint4 out = *reinterpret_cast<const uint4*>(output + row * n + col0);
        const uint4 delta =
            *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0);
        *reinterpret_cast<uint4*>(output + row * n + col0) =
            direct_bf16_add_packed_u4(out, delta);
        continue;
      }
      const uint4 out = *reinterpret_cast<const uint4*>(output + row * n + col0);
      acc[0] = direct_bf16_bits_to_float(out.x);
      acc[1] = direct_bf16_bits_hi_to_float(out.x);
      acc[2] = direct_bf16_bits_to_float(out.y);
      acc[3] = direct_bf16_bits_hi_to_float(out.y);
      acc[4] = direct_bf16_bits_to_float(out.z);
      acc[5] = direct_bf16_bits_hi_to_float(out.z);
      acc[6] = direct_bf16_bits_to_float(out.w);
      acc[7] = direct_bf16_bits_hi_to_float(out.w);
      for (int32_t pos = start; pos < end; ++pos) {
        const int64_t entry = static_cast<int64_t>(merge_entry_indices[pos]);
        const uint4 delta = *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0);
        acc[0] += direct_bf16_bits_to_float(delta.x);
        acc[1] += direct_bf16_bits_hi_to_float(delta.x);
        acc[2] += direct_bf16_bits_to_float(delta.y);
        acc[3] += direct_bf16_bits_hi_to_float(delta.y);
        acc[4] += direct_bf16_bits_to_float(delta.z);
        acc[5] += direct_bf16_bits_hi_to_float(delta.z);
        acc[6] += direct_bf16_bits_to_float(delta.w);
        acc[7] += direct_bf16_bits_hi_to_float(delta.w);
      }
      const uint4 packed =
          make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                     direct_pack_bf16x2(acc[2], acc[3]),
                     direct_pack_bf16x2(acc[4], acc[5]),
                     direct_pack_bf16x2(acc[6], acc[7]));
      *reinterpret_cast<uint4*>(output + row * n + col0) = packed;
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < n) {
          acc[cc] = direct_bf16_to_float(output[row * n + col]);
        }
      }
      for (int32_t pos = start; pos < end; ++pos) {
        const int64_t entry = static_cast<int64_t>(merge_entry_indices[pos]);
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t col = col0 + cc;
          if (col < n) {
            acc[cc] += direct_bf16_to_float(delta_entries[entry * n + col]);
          }
        }
      }
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < n) {
          output[row * n + col] = direct_float_to_bf16(acc[cc]);
        }
      }
    }
  }
}

__global__ __launch_bounds__(256, 2)
void merge_entry_delta_active_rows_vec16_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ delta_entries,
    const int32_t* __restrict__ active_rows,
    const int32_t* __restrict__ merge_row_offsets,
    const int32_t* __restrict__ merge_entry_indices,
    int64_t active_row_count,
    int64_t n) {
  constexpr int VecN = 16;
  const int64_t groups_per_row = (n + VecN - 1) / VecN;
  const int64_t total_groups = active_row_count * groups_per_row;
  for (int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < total_groups;
       group += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t active_idx = group / groups_per_row;
    const int64_t col_group = group - active_idx * groups_per_row;
    const int64_t row = static_cast<int64_t>(active_rows[active_idx]);
    const int64_t col0 = col_group * VecN;
    const int32_t start = merge_row_offsets[active_idx];
    const int32_t end = merge_row_offsets[active_idx + 1];

    if (col0 + VecN <= n) {
      if (end == start + 1) {
        const int64_t entry = static_cast<int64_t>(merge_entry_indices[start]);
        const uint4 out0 = *reinterpret_cast<const uint4*>(output + row * n + col0);
        const uint4 out1 = *reinterpret_cast<const uint4*>(output + row * n + col0 + 8);
        const uint4 delta0 =
            *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0);
        const uint4 delta1 =
            *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0 + 8);
        *reinterpret_cast<uint4*>(output + row * n + col0) =
            direct_bf16_add_packed_u4(out0, delta0);
        *reinterpret_cast<uint4*>(output + row * n + col0 + 8) =
            direct_bf16_add_packed_u4(out1, delta1);
        continue;
      }

      float acc[VecN] = {};
      const uint4 out0 = *reinterpret_cast<const uint4*>(output + row * n + col0);
      const uint4 out1 = *reinterpret_cast<const uint4*>(output + row * n + col0 + 8);
      acc[0] = direct_bf16_bits_to_float(out0.x);
      acc[1] = direct_bf16_bits_hi_to_float(out0.x);
      acc[2] = direct_bf16_bits_to_float(out0.y);
      acc[3] = direct_bf16_bits_hi_to_float(out0.y);
      acc[4] = direct_bf16_bits_to_float(out0.z);
      acc[5] = direct_bf16_bits_hi_to_float(out0.z);
      acc[6] = direct_bf16_bits_to_float(out0.w);
      acc[7] = direct_bf16_bits_hi_to_float(out0.w);
      acc[8] = direct_bf16_bits_to_float(out1.x);
      acc[9] = direct_bf16_bits_hi_to_float(out1.x);
      acc[10] = direct_bf16_bits_to_float(out1.y);
      acc[11] = direct_bf16_bits_hi_to_float(out1.y);
      acc[12] = direct_bf16_bits_to_float(out1.z);
      acc[13] = direct_bf16_bits_hi_to_float(out1.z);
      acc[14] = direct_bf16_bits_to_float(out1.w);
      acc[15] = direct_bf16_bits_hi_to_float(out1.w);

      for (int32_t pos = start; pos < end; ++pos) {
        const int64_t entry = static_cast<int64_t>(merge_entry_indices[pos]);
        const uint4 delta0 =
            *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0);
        const uint4 delta1 =
            *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0 + 8);
        acc[0] += direct_bf16_bits_to_float(delta0.x);
        acc[1] += direct_bf16_bits_hi_to_float(delta0.x);
        acc[2] += direct_bf16_bits_to_float(delta0.y);
        acc[3] += direct_bf16_bits_hi_to_float(delta0.y);
        acc[4] += direct_bf16_bits_to_float(delta0.z);
        acc[5] += direct_bf16_bits_hi_to_float(delta0.z);
        acc[6] += direct_bf16_bits_to_float(delta0.w);
        acc[7] += direct_bf16_bits_hi_to_float(delta0.w);
        acc[8] += direct_bf16_bits_to_float(delta1.x);
        acc[9] += direct_bf16_bits_hi_to_float(delta1.x);
        acc[10] += direct_bf16_bits_to_float(delta1.y);
        acc[11] += direct_bf16_bits_hi_to_float(delta1.y);
        acc[12] += direct_bf16_bits_to_float(delta1.z);
        acc[13] += direct_bf16_bits_hi_to_float(delta1.z);
        acc[14] += direct_bf16_bits_to_float(delta1.w);
        acc[15] += direct_bf16_bits_hi_to_float(delta1.w);
      }

      const uint4 packed0 =
          make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                     direct_pack_bf16x2(acc[2], acc[3]),
                     direct_pack_bf16x2(acc[4], acc[5]),
                     direct_pack_bf16x2(acc[6], acc[7]));
      const uint4 packed1 =
          make_uint4(direct_pack_bf16x2(acc[8], acc[9]),
                     direct_pack_bf16x2(acc[10], acc[11]),
                     direct_pack_bf16x2(acc[12], acc[13]),
                     direct_pack_bf16x2(acc[14], acc[15]));
      *reinterpret_cast<uint4*>(output + row * n + col0) = packed0;
      *reinterpret_cast<uint4*>(output + row * n + col0 + 8) = packed1;
    } else {
      float acc[VecN] = {};
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < n) {
          acc[cc] = direct_bf16_to_float(output[row * n + col]);
        }
      }
      for (int32_t pos = start; pos < end; ++pos) {
        const int64_t entry = static_cast<int64_t>(merge_entry_indices[pos]);
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t col = col0 + cc;
          if (col < n) {
            acc[cc] += direct_bf16_to_float(delta_entries[entry * n + col]);
          }
        }
      }
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < n) {
          output[row * n + col] = direct_float_to_bf16(acc[cc]);
        }
      }
    }
  }
}

__global__ __launch_bounds__(256, 2)
void merge_entry_delta_active_rows_chunk_prefix_vec8_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ delta_entries,
    const int32_t* __restrict__ active_rows,
    const int32_t* __restrict__ merge_row_offsets,
    const int32_t* __restrict__ merge_entry_indices,
    int64_t active_row_count,
    int64_t n,
    int64_t chunk_cols,
    int64_t chunks_per_row) {
  constexpr int VecN = 8;
  if (active_row_count <= 0 || n <= 0 || chunk_cols <= 0 || chunks_per_row <= 0) {
    return;
  }

  const int64_t total_chunks_per_row = (n + chunk_cols - 1) / chunk_cols;
  const int64_t active_chunks_per_row =
      chunks_per_row < total_chunks_per_row ? chunks_per_row : total_chunks_per_row;
  const int64_t groups_per_chunk = (chunk_cols + VecN - 1) / VecN;
  const int64_t total_groups = active_row_count * active_chunks_per_row * groups_per_chunk;
  for (int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < total_groups;
       group += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t col_group_in_chunk = group % groups_per_chunk;
    const int64_t row_chunk_linear = group / groups_per_chunk;
    const int64_t chunk_idx = row_chunk_linear % active_chunks_per_row;
    const int64_t active_idx = row_chunk_linear / active_chunks_per_row;
    const int64_t row = static_cast<int64_t>(active_rows[active_idx]);
    const int64_t chunk_start_col = chunk_idx * chunk_cols;
    const int64_t chunk_end_col =
        (chunk_start_col + chunk_cols) < n ? (chunk_start_col + chunk_cols) : n;
    const int64_t col0 = chunk_start_col + col_group_in_chunk * VecN;
    if (col0 >= chunk_end_col) {
      continue;
    }

    const int32_t start = merge_row_offsets[active_idx];
    const int32_t end = merge_row_offsets[active_idx + 1];
    const int32_t count = end - start;
    if (count <= 0) {
      continue;
    }

    if (col0 + VecN <= chunk_end_col) {
      if (count == 1) {
        const int64_t entry = static_cast<int64_t>(merge_entry_indices[start]);
        const uint4 out = *reinterpret_cast<const uint4*>(output + row * n + col0);
        const uint4 delta = *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0);
        *reinterpret_cast<uint4*>(output + row * n + col0) =
            direct_bf16_add_packed_u4(out, delta);
        continue;
      }
      if (count == 2) {
        const int64_t entry0 = static_cast<int64_t>(merge_entry_indices[start]);
        const int64_t entry1 = static_cast<int64_t>(merge_entry_indices[start + 1]);
        const uint4 out = *reinterpret_cast<const uint4*>(output + row * n + col0);
        const uint4 delta0 = *reinterpret_cast<const uint4*>(delta_entries + entry0 * n + col0);
        const uint4 delta1 = *reinterpret_cast<const uint4*>(delta_entries + entry1 * n + col0);
        *reinterpret_cast<uint4*>(output + row * n + col0) =
            direct_bf16_add3_packed_u4(out, delta0, delta1);
        continue;
      }

      float acc[VecN] = {};
      const uint4 out = *reinterpret_cast<const uint4*>(output + row * n + col0);
      acc[0] = direct_bf16_bits_to_float(out.x);
      acc[1] = direct_bf16_bits_hi_to_float(out.x);
      acc[2] = direct_bf16_bits_to_float(out.y);
      acc[3] = direct_bf16_bits_hi_to_float(out.y);
      acc[4] = direct_bf16_bits_to_float(out.z);
      acc[5] = direct_bf16_bits_hi_to_float(out.z);
      acc[6] = direct_bf16_bits_to_float(out.w);
      acc[7] = direct_bf16_bits_hi_to_float(out.w);
      for (int32_t pos = start; pos < end; ++pos) {
        const int64_t entry = static_cast<int64_t>(merge_entry_indices[pos]);
        const uint4 delta = *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0);
        acc[0] += direct_bf16_bits_to_float(delta.x);
        acc[1] += direct_bf16_bits_hi_to_float(delta.x);
        acc[2] += direct_bf16_bits_to_float(delta.y);
        acc[3] += direct_bf16_bits_hi_to_float(delta.y);
        acc[4] += direct_bf16_bits_to_float(delta.z);
        acc[5] += direct_bf16_bits_hi_to_float(delta.z);
        acc[6] += direct_bf16_bits_to_float(delta.w);
        acc[7] += direct_bf16_bits_hi_to_float(delta.w);
      }
      const uint4 packed =
          make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                     direct_pack_bf16x2(acc[2], acc[3]),
                     direct_pack_bf16x2(acc[4], acc[5]),
                     direct_pack_bf16x2(acc[6], acc[7]));
      *reinterpret_cast<uint4*>(output + row * n + col0) = packed;
    } else {
      float acc[VecN] = {};
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < chunk_end_col) {
          acc[cc] = direct_bf16_to_float(output[row * n + col]);
        }
      }
      for (int32_t pos = start; pos < end; ++pos) {
        const int64_t entry = static_cast<int64_t>(merge_entry_indices[pos]);
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t col = col0 + cc;
          if (col < chunk_end_col) {
            acc[cc] += direct_bf16_to_float(delta_entries[entry * n + col]);
          }
        }
      }
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < chunk_end_col) {
          output[row * n + col] = direct_float_to_bf16(acc[cc]);
        }
      }
    }
  }
}

__global__ __launch_bounds__(256, 3)
void merge_single_entry_delta_active_rows_vec16_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ delta_entries,
    const int32_t* __restrict__ active_rows,
    const int32_t* __restrict__ entry_indices,
    int64_t active_row_count,
    int64_t n) {
  constexpr int VecN = 16;
  const int64_t groups_per_row = (n + VecN - 1) / VecN;
  const int64_t total_groups = active_row_count * groups_per_row;
  for (int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < total_groups;
       group += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t active_idx = group / groups_per_row;
    const int64_t col_group = group - active_idx * groups_per_row;
    const int64_t row = static_cast<int64_t>(active_rows[active_idx]);
    const int64_t entry = static_cast<int64_t>(entry_indices[active_idx]);
    const int64_t col0 = col_group * VecN;

    if (col0 + VecN <= n) {
      const uint4 out0 = *reinterpret_cast<const uint4*>(output + row * n + col0);
      const uint4 out1 = *reinterpret_cast<const uint4*>(output + row * n + col0 + 8);
      const uint4 delta0 =
          *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0);
      const uint4 delta1 =
          *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0 + 8);
      *reinterpret_cast<uint4*>(output + row * n + col0) =
          direct_bf16_add_packed_u4(out0, delta0);
      *reinterpret_cast<uint4*>(output + row * n + col0 + 8) =
          direct_bf16_add_packed_u4(out1, delta1);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < n) {
          const float out = direct_bf16_to_float(output[row * n + col]);
          const float delta = direct_bf16_to_float(delta_entries[entry * n + col]);
          output[row * n + col] = direct_float_to_bf16(out + delta);
        }
      }
    }
  }
}

__global__ __launch_bounds__(256, 3)
void merge_single_entry_delta_active_rows_vec32_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ delta_entries,
    const int32_t* __restrict__ active_rows,
    const int32_t* __restrict__ entry_indices,
    int64_t active_row_count,
    int64_t n) {
  constexpr int VecN = 32;
  const int64_t groups_per_row = (n + VecN - 1) / VecN;
  const int64_t total_groups = active_row_count * groups_per_row;
  for (int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < total_groups;
       group += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t active_idx = group / groups_per_row;
    const int64_t col_group = group - active_idx * groups_per_row;
    const int64_t row = static_cast<int64_t>(active_rows[active_idx]);
    const int64_t entry = static_cast<int64_t>(entry_indices[active_idx]);
    const int64_t col0 = col_group * VecN;

    if (col0 + VecN <= n) {
      const uint4 out0 = *reinterpret_cast<const uint4*>(output + row * n + col0);
      const uint4 out1 = *reinterpret_cast<const uint4*>(output + row * n + col0 + 8);
      const uint4 out2 = *reinterpret_cast<const uint4*>(output + row * n + col0 + 16);
      const uint4 out3 = *reinterpret_cast<const uint4*>(output + row * n + col0 + 24);
      const uint4 delta0 =
          *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0);
      const uint4 delta1 =
          *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0 + 8);
      const uint4 delta2 =
          *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0 + 16);
      const uint4 delta3 =
          *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0 + 24);
      *reinterpret_cast<uint4*>(output + row * n + col0) =
          direct_bf16_add_packed_u4(out0, delta0);
      *reinterpret_cast<uint4*>(output + row * n + col0 + 8) =
          direct_bf16_add_packed_u4(out1, delta1);
      *reinterpret_cast<uint4*>(output + row * n + col0 + 16) =
          direct_bf16_add_packed_u4(out2, delta2);
      *reinterpret_cast<uint4*>(output + row * n + col0 + 24) =
          direct_bf16_add_packed_u4(out3, delta3);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < n) {
          const float out = direct_bf16_to_float(output[row * n + col]);
          const float delta = direct_bf16_to_float(delta_entries[entry * n + col]);
          output[row * n + col] = direct_float_to_bf16(out + delta);
        }
      }
    }
  }
}

__global__ __launch_bounds__(256, 2)
void merge_double_entry_delta_active_rows_vec16_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ delta_entries,
    const int32_t* __restrict__ active_rows,
    const int32_t* __restrict__ entry0_indices,
    const int32_t* __restrict__ entry1_indices,
    int64_t active_row_count,
    int64_t n) {
  constexpr int VecN = 16;
  const int64_t groups_per_row = (n + VecN - 1) / VecN;
  const int64_t total_groups = active_row_count * groups_per_row;
  for (int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group < total_groups;
       group += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t active_idx = group / groups_per_row;
    const int64_t col_group = group - active_idx * groups_per_row;
    const int64_t row = static_cast<int64_t>(active_rows[active_idx]);
    const int64_t entry0 = static_cast<int64_t>(entry0_indices[active_idx]);
    const int64_t entry1 = static_cast<int64_t>(entry1_indices[active_idx]);
    const int64_t col0 = col_group * VecN;

    if (col0 + VecN <= n) {
      const uint4 out0 = *reinterpret_cast<const uint4*>(output + row * n + col0);
      const uint4 out1 = *reinterpret_cast<const uint4*>(output + row * n + col0 + 8);
      const uint4 delta00 =
          *reinterpret_cast<const uint4*>(delta_entries + entry0 * n + col0);
      const uint4 delta01 =
          *reinterpret_cast<const uint4*>(delta_entries + entry0 * n + col0 + 8);
      const uint4 delta10 =
          *reinterpret_cast<const uint4*>(delta_entries + entry1 * n + col0);
      const uint4 delta11 =
          *reinterpret_cast<const uint4*>(delta_entries + entry1 * n + col0 + 8);
      *reinterpret_cast<uint4*>(output + row * n + col0) =
          direct_bf16_add3_packed_u4(out0, delta00, delta10);
      *reinterpret_cast<uint4*>(output + row * n + col0 + 8) =
          direct_bf16_add3_packed_u4(out1, delta01, delta11);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < n) {
          const float out = direct_bf16_to_float(output[row * n + col]);
          const float delta0 = direct_bf16_to_float(delta_entries[entry0 * n + col]);
          const float delta1 = direct_bf16_to_float(delta_entries[entry1 * n + col]);
          output[row * n + col] = direct_float_to_bf16(out + delta0 + delta1);
        }
      }
    }
  }
}

__global__ __launch_bounds__(256, 3)
void merge_single_entry_delta_active_rows_rowblock_vec16_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ delta_entries,
    const int32_t* __restrict__ active_rows,
    const int32_t* __restrict__ entry_indices,
    int64_t active_row_count,
    int64_t n) {
  constexpr int VecN = 16;
  const int64_t active_idx = static_cast<int64_t>(blockIdx.x);
  if (active_idx >= active_row_count) {
    return;
  }

  __shared__ int32_t smem[2];
  if (threadIdx.x == 0) {
    smem[0] = active_rows[active_idx];
    smem[1] = entry_indices[active_idx];
  }
  __syncthreads();

  const int64_t row = static_cast<int64_t>(smem[0]);
  const int64_t entry = static_cast<int64_t>(smem[1]);
  const int64_t groups_per_row = (n + VecN - 1) / VecN;
  for (int64_t col_group = threadIdx.x; col_group < groups_per_row; col_group += blockDim.x) {
    const int64_t col0 = col_group * VecN;
    if (col0 + VecN <= n) {
      const uint4 out0 = *reinterpret_cast<const uint4*>(output + row * n + col0);
      const uint4 out1 = *reinterpret_cast<const uint4*>(output + row * n + col0 + 8);
      const uint4 delta0 =
          *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0);
      const uint4 delta1 =
          *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0 + 8);
      *reinterpret_cast<uint4*>(output + row * n + col0) =
          direct_bf16_add_packed_u4(out0, delta0);
      *reinterpret_cast<uint4*>(output + row * n + col0 + 8) =
          direct_bf16_add_packed_u4(out1, delta1);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < n) {
          const float out = direct_bf16_to_float(output[row * n + col]);
          const float delta = direct_bf16_to_float(delta_entries[entry * n + col]);
          output[row * n + col] = direct_float_to_bf16(out + delta);
        }
      }
    }
  }
}

__global__ __launch_bounds__(256, 3)
void merge_double_entry_delta_active_rows_rowblock_vec16_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ delta_entries,
    const int32_t* __restrict__ active_rows,
    const int32_t* __restrict__ entry0_indices,
    const int32_t* __restrict__ entry1_indices,
    int64_t active_row_count,
    int64_t n) {
  constexpr int VecN = 16;
  const int64_t active_idx = static_cast<int64_t>(blockIdx.x);
  if (active_idx >= active_row_count) {
    return;
  }

  __shared__ int32_t smem[3];
  if (threadIdx.x == 0) {
    smem[0] = active_rows[active_idx];
    smem[1] = entry0_indices[active_idx];
    smem[2] = entry1_indices[active_idx];
  }
  __syncthreads();

  const int64_t row = static_cast<int64_t>(smem[0]);
  const int64_t entry0 = static_cast<int64_t>(smem[1]);
  const int64_t entry1 = static_cast<int64_t>(smem[2]);
  const int64_t groups_per_row = (n + VecN - 1) / VecN;
  for (int64_t col_group = threadIdx.x; col_group < groups_per_row; col_group += blockDim.x) {
    const int64_t col0 = col_group * VecN;
    if (col0 + VecN <= n) {
      const uint4 out0 = *reinterpret_cast<const uint4*>(output + row * n + col0);
      const uint4 out1 = *reinterpret_cast<const uint4*>(output + row * n + col0 + 8);
      const uint4 delta00 =
          *reinterpret_cast<const uint4*>(delta_entries + entry0 * n + col0);
      const uint4 delta01 =
          *reinterpret_cast<const uint4*>(delta_entries + entry0 * n + col0 + 8);
      const uint4 delta10 =
          *reinterpret_cast<const uint4*>(delta_entries + entry1 * n + col0);
      const uint4 delta11 =
          *reinterpret_cast<const uint4*>(delta_entries + entry1 * n + col0 + 8);
      *reinterpret_cast<uint4*>(output + row * n + col0) =
          direct_bf16_add3_packed_u4(out0, delta00, delta10);
      *reinterpret_cast<uint4*>(output + row * n + col0 + 8) =
          direct_bf16_add3_packed_u4(out1, delta01, delta11);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < n) {
          const float out = direct_bf16_to_float(output[row * n + col]);
          const float delta0 = direct_bf16_to_float(delta_entries[entry0 * n + col]);
          const float delta1 = direct_bf16_to_float(delta_entries[entry1 * n + col]);
          output[row * n + col] = direct_float_to_bf16(out + delta0 + delta1);
        }
      }
    }
  }
}

__global__ __launch_bounds__(256, 2)
void merge_entry_delta_active_rows_rowblock_vec16_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ delta_entries,
    const int32_t* __restrict__ active_rows,
    const int32_t* __restrict__ merge_row_offsets,
    const int32_t* __restrict__ merge_entry_indices,
    int64_t active_row_count,
    int64_t n) {
  constexpr int VecN = 16;
  const int64_t active_idx = static_cast<int64_t>(blockIdx.x);
  if (active_idx >= active_row_count) {
    return;
  }

  __shared__ int32_t smem[3];
  if (threadIdx.x == 0) {
    smem[0] = active_rows[active_idx];
    smem[1] = merge_row_offsets[active_idx];
    smem[2] = merge_row_offsets[active_idx + 1];
  }
  __syncthreads();

  const int64_t row = static_cast<int64_t>(smem[0]);
  const int32_t start = smem[1];
  const int32_t end = smem[2];
  const int64_t groups_per_row = (n + VecN - 1) / VecN;
	  for (int64_t col_group = threadIdx.x; col_group < groups_per_row; col_group += blockDim.x) {
	    const int64_t col0 = col_group * VecN;
	    if (col0 + VecN <= n) {
	      float acc[VecN] = {};
	      const uint4 out0 = *reinterpret_cast<const uint4*>(output + row * n + col0);
      const uint4 out1 = *reinterpret_cast<const uint4*>(output + row * n + col0 + 8);
      acc[0] = direct_bf16_bits_to_float(out0.x);
      acc[1] = direct_bf16_bits_hi_to_float(out0.x);
      acc[2] = direct_bf16_bits_to_float(out0.y);
      acc[3] = direct_bf16_bits_hi_to_float(out0.y);
      acc[4] = direct_bf16_bits_to_float(out0.z);
      acc[5] = direct_bf16_bits_hi_to_float(out0.z);
      acc[6] = direct_bf16_bits_to_float(out0.w);
      acc[7] = direct_bf16_bits_hi_to_float(out0.w);
      acc[8] = direct_bf16_bits_to_float(out1.x);
      acc[9] = direct_bf16_bits_hi_to_float(out1.x);
      acc[10] = direct_bf16_bits_to_float(out1.y);
      acc[11] = direct_bf16_bits_hi_to_float(out1.y);
      acc[12] = direct_bf16_bits_to_float(out1.z);
      acc[13] = direct_bf16_bits_hi_to_float(out1.z);
      acc[14] = direct_bf16_bits_to_float(out1.w);
      acc[15] = direct_bf16_bits_hi_to_float(out1.w);

      for (int32_t pos = start; pos < end; ++pos) {
        const int64_t entry = static_cast<int64_t>(merge_entry_indices[pos]);
        const uint4 delta0 =
            *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0);
        const uint4 delta1 =
            *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0 + 8);
        acc[0] += direct_bf16_bits_to_float(delta0.x);
        acc[1] += direct_bf16_bits_hi_to_float(delta0.x);
        acc[2] += direct_bf16_bits_to_float(delta0.y);
        acc[3] += direct_bf16_bits_hi_to_float(delta0.y);
        acc[4] += direct_bf16_bits_to_float(delta0.z);
        acc[5] += direct_bf16_bits_hi_to_float(delta0.z);
        acc[6] += direct_bf16_bits_to_float(delta0.w);
        acc[7] += direct_bf16_bits_hi_to_float(delta0.w);
        acc[8] += direct_bf16_bits_to_float(delta1.x);
        acc[9] += direct_bf16_bits_hi_to_float(delta1.x);
        acc[10] += direct_bf16_bits_to_float(delta1.y);
        acc[11] += direct_bf16_bits_hi_to_float(delta1.y);
        acc[12] += direct_bf16_bits_to_float(delta1.z);
        acc[13] += direct_bf16_bits_hi_to_float(delta1.z);
        acc[14] += direct_bf16_bits_to_float(delta1.w);
        acc[15] += direct_bf16_bits_hi_to_float(delta1.w);
      }

      const uint4 packed0 =
          make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                     direct_pack_bf16x2(acc[2], acc[3]),
                     direct_pack_bf16x2(acc[4], acc[5]),
                     direct_pack_bf16x2(acc[6], acc[7]));
      const uint4 packed1 =
          make_uint4(direct_pack_bf16x2(acc[8], acc[9]),
                     direct_pack_bf16x2(acc[10], acc[11]),
                     direct_pack_bf16x2(acc[12], acc[13]),
                     direct_pack_bf16x2(acc[14], acc[15]));
      *reinterpret_cast<uint4*>(output + row * n + col0) = packed0;
      *reinterpret_cast<uint4*>(output + row * n + col0 + 8) = packed1;
    } else {
      float acc[VecN] = {};
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < n) {
          acc[cc] = direct_bf16_to_float(output[row * n + col]);
        }
      }
      for (int32_t pos = start; pos < end; ++pos) {
        const int64_t entry = static_cast<int64_t>(merge_entry_indices[pos]);
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t col = col0 + cc;
          if (col < n) {
            acc[cc] += direct_bf16_to_float(delta_entries[entry * n + col]);
          }
        }
      }
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < n) {
          output[row * n + col] = direct_float_to_bf16(acc[cc]);
        }
      }
    }
  }
}

__global__ __launch_bounds__(256, 2)
void merge_entry_delta_active_rows_rowblock_vec16_fastpath_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ delta_entries,
    const int32_t* __restrict__ active_rows,
    const int32_t* __restrict__ merge_row_offsets,
    const int32_t* __restrict__ merge_entry_indices,
    int64_t active_row_count,
    int64_t n) {
  constexpr int VecN = 16;
  const int64_t active_idx = static_cast<int64_t>(blockIdx.x);
  if (active_idx >= active_row_count) {
    return;
  }

  __shared__ int32_t smem[3];
  if (threadIdx.x == 0) {
    smem[0] = active_rows[active_idx];
    smem[1] = merge_row_offsets[active_idx];
    smem[2] = merge_row_offsets[active_idx + 1];
  }
  __syncthreads();

  const int64_t row = static_cast<int64_t>(smem[0]);
  const int32_t start = smem[1];
  const int32_t end = smem[2];
  const int32_t count = end - start;
  if (count <= 0) {
    return;
  }

  const int64_t groups_per_row = (n + VecN - 1) / VecN;
  if (count == 1) {
    const int64_t entry = static_cast<int64_t>(merge_entry_indices[start]);
    for (int64_t col_group = threadIdx.x; col_group < groups_per_row; col_group += blockDim.x) {
      const int64_t col0 = col_group * VecN;
      if (col0 + VecN <= n) {
        const uint4 out0 = *reinterpret_cast<const uint4*>(output + row * n + col0);
        const uint4 out1 = *reinterpret_cast<const uint4*>(output + row * n + col0 + 8);
        const uint4 delta0 =
            *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0);
        const uint4 delta1 =
            *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0 + 8);
        *reinterpret_cast<uint4*>(output + row * n + col0) =
            direct_bf16_add_packed_u4(out0, delta0);
        *reinterpret_cast<uint4*>(output + row * n + col0 + 8) =
            direct_bf16_add_packed_u4(out1, delta1);
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t col = col0 + cc;
          if (col < n) {
            const float out = direct_bf16_to_float(output[row * n + col]);
            const float delta = direct_bf16_to_float(delta_entries[entry * n + col]);
            output[row * n + col] = direct_float_to_bf16(out + delta);
          }
        }
      }
    }
    return;
  }

  if (count == 2) {
    const int64_t entry0 = static_cast<int64_t>(merge_entry_indices[start]);
    const int64_t entry1 = static_cast<int64_t>(merge_entry_indices[start + 1]);
    for (int64_t col_group = threadIdx.x; col_group < groups_per_row; col_group += blockDim.x) {
      const int64_t col0 = col_group * VecN;
      if (col0 + VecN <= n) {
        const uint4 out0 = *reinterpret_cast<const uint4*>(output + row * n + col0);
        const uint4 out1 = *reinterpret_cast<const uint4*>(output + row * n + col0 + 8);
        const uint4 delta00 =
            *reinterpret_cast<const uint4*>(delta_entries + entry0 * n + col0);
        const uint4 delta01 =
            *reinterpret_cast<const uint4*>(delta_entries + entry0 * n + col0 + 8);
        const uint4 delta10 =
            *reinterpret_cast<const uint4*>(delta_entries + entry1 * n + col0);
        const uint4 delta11 =
            *reinterpret_cast<const uint4*>(delta_entries + entry1 * n + col0 + 8);
        *reinterpret_cast<uint4*>(output + row * n + col0) =
            direct_bf16_add3_packed_u4(out0, delta00, delta10);
        *reinterpret_cast<uint4*>(output + row * n + col0 + 8) =
            direct_bf16_add3_packed_u4(out1, delta01, delta11);
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t col = col0 + cc;
          if (col < n) {
            const float out = direct_bf16_to_float(output[row * n + col]);
            const float delta0 = direct_bf16_to_float(delta_entries[entry0 * n + col]);
            const float delta1 = direct_bf16_to_float(delta_entries[entry1 * n + col]);
            output[row * n + col] = direct_float_to_bf16(out + delta0 + delta1);
          }
        }
      }
    }
    return;
  }

  for (int64_t col_group = threadIdx.x; col_group < groups_per_row; col_group += blockDim.x) {
    const int64_t col0 = col_group * VecN;
    if (col0 + VecN <= n) {
      float acc[VecN] = {};
      const uint4 out0 = *reinterpret_cast<const uint4*>(output + row * n + col0);
      const uint4 out1 = *reinterpret_cast<const uint4*>(output + row * n + col0 + 8);
      acc[0] = direct_bf16_bits_to_float(out0.x);
      acc[1] = direct_bf16_bits_hi_to_float(out0.x);
      acc[2] = direct_bf16_bits_to_float(out0.y);
      acc[3] = direct_bf16_bits_hi_to_float(out0.y);
      acc[4] = direct_bf16_bits_to_float(out0.z);
      acc[5] = direct_bf16_bits_hi_to_float(out0.z);
      acc[6] = direct_bf16_bits_to_float(out0.w);
      acc[7] = direct_bf16_bits_hi_to_float(out0.w);
      acc[8] = direct_bf16_bits_to_float(out1.x);
      acc[9] = direct_bf16_bits_hi_to_float(out1.x);
      acc[10] = direct_bf16_bits_to_float(out1.y);
      acc[11] = direct_bf16_bits_hi_to_float(out1.y);
      acc[12] = direct_bf16_bits_to_float(out1.z);
      acc[13] = direct_bf16_bits_hi_to_float(out1.z);
      acc[14] = direct_bf16_bits_to_float(out1.w);
      acc[15] = direct_bf16_bits_hi_to_float(out1.w);

      for (int32_t pos = start; pos < end; ++pos) {
        const int64_t entry = static_cast<int64_t>(merge_entry_indices[pos]);
        const uint4 delta0 =
            *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0);
        const uint4 delta1 =
            *reinterpret_cast<const uint4*>(delta_entries + entry * n + col0 + 8);
        acc[0] += direct_bf16_bits_to_float(delta0.x);
        acc[1] += direct_bf16_bits_hi_to_float(delta0.x);
        acc[2] += direct_bf16_bits_to_float(delta0.y);
        acc[3] += direct_bf16_bits_hi_to_float(delta0.y);
        acc[4] += direct_bf16_bits_to_float(delta0.z);
        acc[5] += direct_bf16_bits_hi_to_float(delta0.z);
        acc[6] += direct_bf16_bits_to_float(delta0.w);
        acc[7] += direct_bf16_bits_hi_to_float(delta0.w);
        acc[8] += direct_bf16_bits_to_float(delta1.x);
        acc[9] += direct_bf16_bits_hi_to_float(delta1.x);
        acc[10] += direct_bf16_bits_to_float(delta1.y);
        acc[11] += direct_bf16_bits_hi_to_float(delta1.y);
        acc[12] += direct_bf16_bits_to_float(delta1.z);
        acc[13] += direct_bf16_bits_hi_to_float(delta1.z);
        acc[14] += direct_bf16_bits_to_float(delta1.w);
        acc[15] += direct_bf16_bits_hi_to_float(delta1.w);
      }

      const uint4 packed0 =
          make_uint4(direct_pack_bf16x2(acc[0], acc[1]),
                     direct_pack_bf16x2(acc[2], acc[3]),
                     direct_pack_bf16x2(acc[4], acc[5]),
                     direct_pack_bf16x2(acc[6], acc[7]));
      const uint4 packed1 =
          make_uint4(direct_pack_bf16x2(acc[8], acc[9]),
                     direct_pack_bf16x2(acc[10], acc[11]),
                     direct_pack_bf16x2(acc[12], acc[13]),
                     direct_pack_bf16x2(acc[14], acc[15]));
      *reinterpret_cast<uint4*>(output + row * n + col0) = packed0;
      *reinterpret_cast<uint4*>(output + row * n + col0 + 8) = packed1;
    } else {
      float acc[VecN] = {};
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < n) {
          acc[cc] = direct_bf16_to_float(output[row * n + col]);
        }
      }
      for (int32_t pos = start; pos < end; ++pos) {
        const int64_t entry = static_cast<int64_t>(merge_entry_indices[pos]);
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int64_t col = col0 + cc;
          if (col < n) {
            acc[cc] += direct_bf16_to_float(delta_entries[entry * n + col]);
          }
        }
      }
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int64_t col = col0 + cc;
        if (col < n) {
          output[row * n + col] = direct_float_to_bf16(acc[cc]);
        }
      }
    }
  }
}

template<
    int ProbeWarpgroups,
    int CtaM,
    int CtaN,
    int CtaK,
    int EpiM,
    int EpiN,
    bool EnableLocalDelta = false,
    bool Mode49Specialized = false,
    bool Mode49StaticN4096 = false,
    bool CompactConsumerSpecialized = false>
void nvfp4_gemm_launch_typed(Params params,
                             dim3 grid_dim,
                             dim3 real_grid_dim,
                             cudaStream_t stream) {
  constexpr int AtomRepK = CtaK / AtomK;
  constexpr int LaunchThreads = NumThreadsPerWarpGroup * (3 + ProbeWarpgroups);
  using SmemStorage = SmemStorageT<CtaM, CtaN, CtaK, EpiM, EpiN, AtomRepK, EnableLocalDelta>;

  cudaFuncSetAttribute(
    nvfp4_gemm<
        CtaM,
        CtaN,
        CtaK,
        EpiM,
        EpiN,
        LaunchThreads,
        EnableLocalDelta,
        Mode49Specialized,
        Mode49StaticN4096,
        CompactConsumerSpecialized>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    sizeof(SmemStorage)
  );

  nvfp4_gemm<
      CtaM,
      CtaN,
      CtaK,
      EpiM,
      EpiN,
      LaunchThreads,
      EnableLocalDelta,
      Mode49Specialized,
      Mode49StaticN4096,
      CompactConsumerSpecialized>
    <<<grid_dim, dim3(LaunchThreads), sizeof(SmemStorage), stream>>>(params, real_grid_dim);
}

template<int CtaM, int CtaN, int CtaK, int EpiM, int EpiN>
void nvfp4_gemm_launch_mode49_production(Params params,
                                         dim3 grid_dim,
                                         dim3 real_grid_dim,
                                         cudaStream_t stream,
                                         int32_t sparse_warpgroups) {
  const bool static_n4096 = params.n == 4096;
  if (sparse_warpgroups == 1) {
    if (static_n4096) {
      nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN, false, true, true>(
          params, grid_dim, real_grid_dim, stream);
    } else {
      nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN, false, true, false>(
          params, grid_dim, real_grid_dim, stream);
    }
  } else if (sparse_warpgroups == 2) {
    if (static_n4096) {
      nvfp4_gemm_launch_typed<2, CtaM, CtaN, CtaK, EpiM, EpiN, false, true, true>(
          params, grid_dim, real_grid_dim, stream);
    } else {
      nvfp4_gemm_launch_typed<2, CtaM, CtaN, CtaK, EpiM, EpiN, false, true, false>(
          params, grid_dim, real_grid_dim, stream);
    }
  } else {
    TORCH_CHECK(false, "mode49 production path supports one or two sparse warpgroups");
  }
}

template<int CtaM, int CtaN, int CtaK, int EpiM, int EpiN>
void nvfp4_gemm_launch_compact_consumer_posttail(Params params,
                                                 dim3 grid_dim,
                                                 dim3 real_grid_dim,
                                                 cudaStream_t stream) {
  TORCH_CHECK(params.n == HANDWRITTEN_TMA_COMPACT_CONSUMER_STATIC_N,
              "compact consumer post-tail path N does not match its compiled specialization");
  TORCH_CHECK(params.direct_row_offsets != nullptr &&
                  params.direct_row_ks != nullptr &&
                  params.direct_row_values != nullptr &&
                  params.direct_b_comp != nullptr,
              "compact consumer post-tail path requires row payload and B compensation");
  nvfp4_gemm_launch_typed<
      1, CtaM, CtaN, CtaK, EpiM, EpiN, false, true, true, true>(
      params, grid_dim, real_grid_dim, stream);
}

void nvfp4_gemm_launch(
  uint8_t* dev_qA, uint8_t* dev_qB, 
  uint8_t* dev_SFA, uint8_t* dev_SFB,
  const float* dev_amax_A, const float* dev_amax_B,
  uint16_t* dev_C_bf16,
  int32_t* ready_flags,
	  const int m, const int n, const int k,
	  cudaStream_t stream,
	  bool scale_tile_major,
	  int32_t* ready_queue = nullptr,
	  int32_t* ready_slot_status = nullptr,
	  int32_t* ready_tail = nullptr,
	  const int32_t* direct_row_offsets = nullptr,
	  const int32_t* direct_row_ks = nullptr,
	  const c10::BFloat16* direct_row_values = nullptr,
	  const int32_t* direct_active_row_offsets = nullptr,
	  const int32_t* direct_active_rows = nullptr,
	  const c10::BFloat16* direct_b_comp = nullptr,
	  const int32_t* direct_probe_active_mblocks = nullptr,
	  int32_t direct_probe_active_mblock_count = 0,
	  float* direct_probe_sink = nullptr,
	  int32_t* direct_probe_counter = nullptr,
	  int32_t direct_probe_warpgroups = 0,
	  int32_t direct_probe_mixed_cta = 0,
	  const int32_t* direct_kmajor_group_offsets = nullptr,
	  const int32_t* direct_kmajor_group_ks = nullptr,
	  const int32_t* direct_kmajor_entry_offsets = nullptr,
	  const int32_t* direct_kmajor_entry_rows = nullptr,
	  const c10::BFloat16* direct_kmajor_entry_values = nullptr,
	  int32_t direct_probe_kmajor = 0,
	  c10::BFloat16* direct_delta_output = nullptr,
	  int32_t direct_delta_write_mode = 0,
	  int32_t direct_probe_side_warps = 0,
	  int32_t direct_delta_chunk_limit = 0,
	  bool force_dense_4wg = false,
	  int32_t direct_smem_add = 0,
		  int32_t* ready_m_queue = nullptr,
		  int32_t* ready_m_slot_status = nullptr,
		  int32_t* ready_m_tail = nullptr,
		  int32_t* ready_m_counts = nullptr,
		  int32_t ready_m_slices = 1,
		  uint64_t* phase_trace = nullptr,
		  int32_t phase_trace_stride = 0,
		  int32_t phase_trace_max_ctas = 0,
		  int32_t phase_trace_mode = 0,
		  const int32_t* direct_packed_tile_offsets = nullptr,
		  const int64_t* direct_packed_row_records = nullptr,
		  const int32_t* direct_packed_entry_records = nullptr,
		  int32_t direct_packed_payload_mode = 0,
		  const int32_t* direct_kmajor_tile_group_starts = nullptr,
		  const int32_t* direct_kmajor_tile_group_counts = nullptr,
		  const int64_t* direct_kmajor_tile_group_meta = nullptr,
		  int32_t direct_active_row_count = 0
		) {
  constexpr int CtaM = 128;
  constexpr int CtaN = 128;
  constexpr int CtaK = 128;
  constexpr int EpiM = 128;
  constexpr int EpiN = HANDWRITTEN_TMA_EPIN;

  Params params;
  params.m = m;
  params.n = n;
  params.k = k;
  params.inner_scale_A = (uint8_t*)dev_SFA;
  params.inner_scale_B = (uint8_t*)dev_SFB;
  params.amax_A = dev_amax_A;
  params.amax_B = dev_amax_B;
  params.ready_flags = ready_flags;
  params.ready_queue = ready_queue;
	  params.ready_slot_status = ready_slot_status;
	  params.ready_tail = ready_tail;
	  params.ready_m_queue = ready_m_queue;
	  params.ready_m_slot_status = ready_m_slot_status;
	  params.ready_m_tail = ready_m_tail;
	  params.ready_m_counts = ready_m_counts;
	  params.ready_m_slices = ready_m_slices;
	  params.scale_tile_major = scale_tile_major;
	  params.direct_row_offsets = direct_row_offsets;
	  params.direct_row_ks = direct_row_ks;
	  params.direct_row_values = direct_row_values;
	  params.direct_active_row_offsets = direct_active_row_offsets;
	  params.direct_active_rows = direct_active_rows;
	  params.direct_active_row_count =
	      direct_active_row_count > 0 ? direct_active_row_count : direct_delta_chunk_limit;
	  params.direct_packed_tile_offsets = direct_packed_tile_offsets;
	  params.direct_packed_row_records = direct_packed_row_records;
	  params.direct_packed_entry_records = direct_packed_entry_records;
	  params.direct_b_comp = direct_b_comp;
	  params.direct_probe_active_mblocks = direct_probe_active_mblocks;
	  params.direct_kmajor_group_offsets = direct_kmajor_group_offsets;
	  params.direct_kmajor_group_ks = direct_kmajor_group_ks;
	  params.direct_kmajor_entry_offsets = direct_kmajor_entry_offsets;
	  params.direct_kmajor_entry_rows = direct_kmajor_entry_rows;
	  params.direct_kmajor_entry_values = direct_kmajor_entry_values;
	  params.direct_kmajor_tile_group_starts = direct_kmajor_tile_group_starts;
	  params.direct_kmajor_tile_group_counts = direct_kmajor_tile_group_counts;
	  params.direct_kmajor_tile_group_meta = direct_kmajor_tile_group_meta;
	  params.direct_delta_output = direct_delta_output;
	  params.direct_delta_write_mode =
	      direct_delta_output == nullptr ? 0 : direct_delta_write_mode;
		  params.direct_smem_add = direct_smem_add;
		  params.phase_trace = phase_trace;
		  params.phase_trace_stride = phase_trace_stride;
		  params.phase_trace_max_ctas = phase_trace_max_ctas;
		  params.phase_trace_mode = phase_trace_mode;
		  params.direct_packed_payload_mode = direct_packed_payload_mode;
		  params.direct_probe_active_mblock_count = direct_probe_active_mblock_count;
	  params.direct_delta_chunk_limit = direct_delta_chunk_limit;
	  params.direct_probe_sink = direct_probe_sink;
	  params.direct_probe_counter = direct_probe_counter;
	  const int32_t probe_warpgroups_abs =
	      direct_probe_warpgroups < 0 ? -direct_probe_warpgroups : direct_probe_warpgroups;
	  const int32_t probe_side_warps =
	      direct_probe_side_warps <= 0
	          ? 1
	          : (direct_probe_side_warps > 4 ? 4 : direct_probe_side_warps);
	  const int32_t probe_side_warps_with_extra =
	      (direct_probe_mixed_cta == 9 || direct_probe_mixed_cta == 10 ||
	       direct_probe_mixed_cta == 11 || direct_probe_mixed_cta == 12 ||
	       direct_probe_mixed_cta == 13 || direct_probe_mixed_cta == 14 ||
	       direct_probe_mixed_cta == 15 || direct_probe_mixed_cta == 16 ||
	       direct_probe_mixed_cta == 17 || direct_probe_mixed_cta == 18 ||
	       direct_probe_mixed_cta == 19 || direct_probe_mixed_cta == 20 ||
		       direct_probe_mixed_cta == 21 || direct_probe_mixed_cta == 22 ||
				       direct_probe_mixed_cta == 23 || direct_probe_mixed_cta == 24 ||
				       direct_probe_mixed_cta == 25 || direct_probe_mixed_cta == 26 ||
				       direct_probe_mixed_cta == 27 || direct_probe_mixed_cta == 28 ||
				       direct_probe_mixed_cta == 29 || direct_probe_mixed_cta == 30 ||
				       direct_probe_mixed_cta == 31 || direct_probe_mixed_cta == 32 ||
					       direct_probe_mixed_cta == 33 || direct_probe_mixed_cta == 34 ||
					       direct_probe_mixed_cta == 35 || direct_probe_mixed_cta == 36 ||
					       direct_probe_mixed_cta == 37)
	          ? probe_side_warps
	          : probe_side_warps;
	  params.direct_probe_kmajor = direct_probe_kmajor;
	  params.direct_probe_warps =
	      direct_probe_mixed_cta == 1 ? (direct_probe_kmajor == 2 ? 24 : 12)
		                                  : direct_probe_mixed_cta == 2
		                                        ? (direct_probe_kmajor == 2 ? 2 : 1)
		                                  : direct_probe_mixed_cta == 3
	                                        ? (direct_probe_kmajor == 2 ? 8 : 4)
	                                  : direct_probe_mixed_cta == 5
		                                        ? 2
		                                  : direct_probe_mixed_cta == 6
		                                        ? 1
		                                  : direct_probe_mixed_cta == 7
		                                        ? probe_side_warps
		                                  : direct_probe_mixed_cta == 8
		                                        ? 2
		                                  : direct_probe_mixed_cta == 9
		                                        ? probe_side_warps_with_extra
		                                  : direct_probe_mixed_cta == 10
		                                        ? probe_side_warps_with_extra
		                                  : direct_probe_mixed_cta == 11
		                                        ? probe_side_warps_with_extra
		                                  : direct_probe_mixed_cta == 12
		                                        ? probe_side_warps_with_extra
		                                  : direct_probe_mixed_cta == 13
		                                        ? probe_side_warps_with_extra
		                                  : direct_probe_mixed_cta == 14
		                                        ? probe_side_warps_with_extra
		                                  : direct_probe_mixed_cta == 15
		                                        ? probe_side_warps_with_extra
		                                  : direct_probe_mixed_cta == 16
		                                        ? probe_side_warps_with_extra
		                                  : direct_probe_mixed_cta == 17
		                                        ? probe_side_warps_with_extra
		                                  : direct_probe_mixed_cta == 18
		                                        ? probe_side_warps_with_extra
		                                  : direct_probe_mixed_cta == 19
		                                        ? probe_side_warps_with_extra
		                                  : direct_probe_mixed_cta == 20
		                                        ? probe_side_warps_with_extra
	                                  : direct_probe_mixed_cta == 21 ||
	                                    direct_probe_mixed_cta == 22 ||
		                                    direct_probe_mixed_cta == 23 ||
		                                    direct_probe_mixed_cta == 24 ||
			                                    direct_probe_mixed_cta == 25 ||
				                                    direct_probe_mixed_cta == 26 ||
				                                    direct_probe_mixed_cta == 27 ||
				                                    direct_probe_mixed_cta == 28 ||
				                                    direct_probe_mixed_cta == 29 ||
				                                    direct_probe_mixed_cta == 30 ||
				                                    direct_probe_mixed_cta == 31 ||
				                                    direct_probe_mixed_cta == 32 ||
					                                    direct_probe_mixed_cta == 33 ||
					                                    direct_probe_mixed_cta == 34 ||
					                                    direct_probe_mixed_cta == 35 ||
					                                    direct_probe_mixed_cta == 36 ||
					                                    direct_probe_mixed_cta == 37
			                                        ? probe_side_warps_with_extra
			                                  : probe_warpgroups_abs * 4;
	  params.direct_probe_do_math = direct_probe_warpgroups > 0 ? 1 : 0;
	  params.direct_probe_mixed_cta = direct_probe_mixed_cta;
	  params.direct_probe_group_budget = probe_warpgroups_abs;
	  params.direct_probe_persistent_cta_count = 0;
	  params.force_dense_4wg = force_dense_4wg ? 1 : 0;
	  params.tensormap_A = make_tensormap_fp4(dev_qA, m, k, CtaM, CtaK);
  params.tensormap_B = make_tensormap_fp4(dev_qB, n, k, CtaN, CtaK);
  params.tensormap_D = make_tensormap_bf16(dev_C_bf16, m, n, EpiM, EpiN);
  params.D = dev_C_bf16;

  constexpr int GM = HANDWRITTEN_TMA_GM;
  dim3 real_grid_dim(cdiv(n, CtaN), cdiv(m, CtaM));
  const int32_t dense_grid_y = cdiv(real_grid_dim.y, GM) * GM;
  dim3 grid_dim(real_grid_dim.x, dense_grid_y);
	  int current_device = 0;
	  C10_CUDA_CHECK(cudaGetDevice(&current_device));
	  static thread_local int cached_device = -1;
	  static thread_local int cached_sm_count = 0;
	  if (cached_device != current_device || cached_sm_count <= 0) {
	    C10_CUDA_CHECK(cudaDeviceGetAttribute(
	        &cached_sm_count, cudaDevAttrMultiProcessorCount, current_device));
	    cached_device = current_device;
	  }
	  const int64_t active_cta_count =
	      static_cast<int64_t>(real_grid_dim.x) * real_grid_dim.y;
	  params.direct_probe_persistent_cta_count = static_cast<int32_t>(
	      std::max<int64_t>(1, std::min<int64_t>(cached_sm_count, active_cta_count)));
  params.direct_probe_dense_grid_y = dense_grid_y;
  params.direct_probe_total_tiles =
      direct_probe_active_mblock_count > 0
          ? static_cast<int32_t>(real_grid_dim.x * direct_probe_active_mblock_count)
          : static_cast<int32_t>(real_grid_dim.x * real_grid_dim.y);

	  const bool compact_consumer_posttail = direct_probe_mixed_cta == 50;
	  if ((!compact_consumer_posttail &&
	       (direct_probe_sink == nullptr || direct_probe_counter == nullptr)) ||
	      probe_warpgroups_abs <= 0) {
    params.direct_probe_sink = nullptr;
    params.direct_probe_counter = nullptr;
    params.direct_probe_warps = 0;
    params.direct_probe_do_math = 0;
    params.direct_probe_mixed_cta = 0;
    params.direct_probe_total_tiles = 0;
    params.direct_probe_dense_grid_y = 0;
    params.direct_probe_active_mblocks = nullptr;
    params.direct_probe_active_mblock_count = 0;
    params.direct_probe_persistent_cta_count = 0;
    params.direct_active_row_count = 0;
    params.direct_kmajor_group_offsets = nullptr;
    params.direct_kmajor_group_ks = nullptr;
    params.direct_kmajor_entry_offsets = nullptr;
    params.direct_kmajor_entry_rows = nullptr;
    params.direct_kmajor_entry_values = nullptr;
    params.direct_kmajor_tile_group_starts = nullptr;
    params.direct_kmajor_tile_group_counts = nullptr;
    params.direct_kmajor_tile_group_meta = nullptr;
    params.direct_packed_tile_offsets = nullptr;
    params.direct_packed_row_records = nullptr;
    params.direct_packed_entry_records = nullptr;
    params.direct_packed_payload_mode = 0;
    if (direct_smem_add != 11 && direct_smem_add != 12) {
      params.direct_delta_output = nullptr;
      params.direct_delta_write_mode = 0;
    }
    params.direct_probe_kmajor = 0;
    params.direct_probe_group_budget = 0;
	    if (force_dense_4wg) {
	      nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
	          params, grid_dim, real_grid_dim, stream);
	    } else if (direct_smem_add == 14) {
	      nvfp4_gemm_launch_typed<0, CtaM, CtaN, CtaK, EpiM, EpiN, true>(
	          params, grid_dim, real_grid_dim, stream);
	    } else {
	      nvfp4_gemm_launch_typed<0, CtaM, CtaN, CtaK, EpiM, EpiN>(
	          params, grid_dim, real_grid_dim, stream);
    }
  } else if (direct_probe_mixed_cta == 1) {
    grid_dim.y = dense_grid_y + probe_warpgroups_abs;
    nvfp4_gemm_launch_typed<0, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (direct_probe_mixed_cta == 2 || direct_probe_mixed_cta == 3 ||
             direct_probe_mixed_cta == 5 || direct_probe_mixed_cta == 6 ||
             direct_probe_mixed_cta == 7 || direct_probe_mixed_cta == 8) {
    nvfp4_gemm_launch_typed<0, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (direct_probe_mixed_cta == 9) {
    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (direct_probe_mixed_cta == 10) {
    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (direct_probe_mixed_cta == 11) {
    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (direct_probe_mixed_cta == 12) {
    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (direct_probe_mixed_cta == 13) {
    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (direct_probe_mixed_cta == 14) {
    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (direct_probe_mixed_cta == 15) {
    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (direct_probe_mixed_cta == 16) {
    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (direct_probe_mixed_cta == 17) {
    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (direct_probe_mixed_cta == 18) {
    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (direct_probe_mixed_cta == 19) {
    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
	  } else if (direct_probe_mixed_cta == 20) {
	    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
	        params, grid_dim, real_grid_dim, stream);
		  } else if (direct_probe_mixed_cta == 21 ||
		             direct_probe_mixed_cta == 22 ||
				             direct_probe_mixed_cta == 23 ||
				             direct_probe_mixed_cta == 24 ||
					             direct_probe_mixed_cta == 25 ||
					             direct_probe_mixed_cta == 26 ||
					             direct_probe_mixed_cta == 27 ||
					             direct_probe_mixed_cta == 28 ||
					             direct_probe_mixed_cta == 29 ||
					             direct_probe_mixed_cta == 30 ||
					             direct_probe_mixed_cta == 31 ||
						             direct_probe_mixed_cta == 32 ||
						             direct_probe_mixed_cta == 33 ||
						             direct_probe_mixed_cta == 34 ||
						             direct_probe_mixed_cta == 35 ||
						             direct_probe_mixed_cta == 36 ||
						             direct_probe_mixed_cta == 37) {
		    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN, true>(
		        params, grid_dim, real_grid_dim, stream);
		  } else if (direct_probe_mixed_cta == 49) {
		    nvfp4_gemm_launch_mode49_production<CtaM, CtaN, CtaK, EpiM, EpiN>(
		        params, grid_dim, real_grid_dim, stream, probe_warpgroups_abs);
		  } else if (direct_probe_mixed_cta == 50) {
		    nvfp4_gemm_launch_compact_consumer_posttail<
		        CtaM, CtaN, CtaK, EpiM, EpiN>(
		        params, grid_dim, real_grid_dim, stream);
		  } else if (probe_warpgroups_abs == 1) {
    nvfp4_gemm_launch_typed<1, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (probe_warpgroups_abs == 2) {
    nvfp4_gemm_launch_typed<2, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (probe_warpgroups_abs == 4) {
    nvfp4_gemm_launch_typed<4, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else if (probe_warpgroups_abs == 5) {
    nvfp4_gemm_launch_typed<5, CtaM, CtaN, CtaK, EpiM, EpiN>(
        params, grid_dim, real_grid_dim, stream);
  } else {
    TORCH_CHECK(false, "direct_probe_warpgroups must be one of 1, 2, 4, 5");
  }
}
}

namespace {

__device__ __forceinline__ float bf16_bits_to_float_v12(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ uint16_t packed_bf16_slot_v12(uint64_t packed, int slot) {
  return static_cast<uint16_t>((packed >> (16 * slot)) & 0xffffu);
}

__device__ __forceinline__ uint16_t float_to_bf16_bits_v13(float value) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ uint64_t pack_bf16x4_v13(
    float v0,
    float v1,
    float v2,
    float v3) {
  return static_cast<uint64_t>(float_to_bf16_bits_v13(v0)) |
         (static_cast<uint64_t>(float_to_bf16_bits_v13(v1)) << 16) |
         (static_cast<uint64_t>(float_to_bf16_bits_v13(v2)) << 32) |
         (static_cast<uint64_t>(float_to_bf16_bits_v13(v3)) << 48);
}

__device__ __forceinline__ void store_bf16x8_u4_v13(
    c10::BFloat16* __restrict__ dst,
    const float* __restrict__ acc) {
  const uint64_t lo = pack_bf16x4_v13(acc[0], acc[1], acc[2], acc[3]);
  const uint64_t hi = pack_bf16x4_v13(acc[4], acc[5], acc[6], acc[7]);
  *reinterpret_cast<uint4*>(dst) = make_uint4(
      static_cast<uint32_t>(lo),
      static_cast<uint32_t>(lo >> 32),
      static_cast<uint32_t>(hi),
      static_cast<uint32_t>(hi >> 32));
}

__device__ __forceinline__ uint32_t add_bf16x2_u32_v208(uint32_t lhs,
                                                         uint32_t rhs) {
  __nv_bfloat162_raw lhs_raw;
  __nv_bfloat162_raw rhs_raw;
  lhs_raw.x = static_cast<unsigned short>(lhs & 0xffffu);
  lhs_raw.y = static_cast<unsigned short>(lhs >> 16);
  rhs_raw.x = static_cast<unsigned short>(rhs & 0xffffu);
  rhs_raw.y = static_cast<unsigned short>(rhs >> 16);
  const __nv_bfloat162 sum = __nv_bfloat162(lhs_raw) + __nv_bfloat162(rhs_raw);
  const __nv_bfloat162_raw sum_raw = static_cast<__nv_bfloat162_raw>(sum);
  return static_cast<uint32_t>(sum_raw.x) |
         (static_cast<uint32_t>(sum_raw.y) << 16);
}

__device__ __forceinline__ uint4 add_bf16x8_u4_v208(uint4 lhs, uint4 rhs) {
  return make_uint4(add_bf16x2_u32_v208(lhs.x, rhs.x),
                    add_bf16x2_u32_v208(lhs.y, rhs.y),
                    add_bf16x2_u32_v208(lhs.z, rhs.z),
                    add_bf16x2_u32_v208(lhs.w, rhs.w));
}

__device__ __forceinline__ uint64_t ld_global_u64_l1_evict_last_v13(const void* ptr) {
  uint64_t out;
  const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
  asm volatile("ld.global.L1::evict_last.b64 %0, [%1];" : "=l"(out) : "l"(addr));
  return out;
}

__global__ void swizzle_te_scale_to_tma_tile_major_kernel(uint8_t* __restrict__ output,
                                                          const uint8_t* __restrict__ input,
                                                          int64_t rows,
                                                          int64_t k_blocks) {
  constexpr int CtaRows = 128;
  constexpr int CtaKBlocks = 8;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = rows * k_blocks;
  if (idx >= total) {
    return;
  }

  const int64_t tile_k_groups = k_blocks / CtaKBlocks;
  const int64_t tile_linear = idx / (CtaRows * CtaKBlocks);
  const int64_t tile_rem = idx - tile_linear * (CtaRows * CtaKBlocks);
  const int64_t row_in_tile = tile_rem / CtaKBlocks;
  const int64_t k_in_group = tile_rem - row_in_tile * CtaKBlocks;
  const int64_t m_tile = tile_linear / tile_k_groups;
  const int64_t k_group = tile_linear - m_tile * tile_k_groups;
  const int64_t src_row = m_tile * CtaRows + row_in_tile;
  const int64_t src_k = k_group * CtaKBlocks + k_in_group;
  output[idx] = input[src_row * k_blocks + src_k];
}

at::Tensor swizzle_te_scale_to_tma_tile_major_cuda(const at::Tensor& scale,
                                                   int64_t rows,
                                                   int64_t k) {
  const int64_t k_blocks = k / 16;
  auto output = at::empty_like(scale);
  if (rows == 0 || k_blocks == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t total = rows * k_blocks;
  const int64_t blocks = (total + threads - 1) / threads;
  swizzle_te_scale_to_tma_tile_major_kernel<<<static_cast<int>(blocks),
                                             threads,
                                             0,
                                             at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<uint8_t>(),
      scale.data_ptr<uint8_t>(),
      rows,
      k_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

__global__ __launch_bounds__(256, 4)
void sparse_group_ready_value_payload_vec8_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ ready_flags,
    int64_t rows,
    int64_t cols,
    int64_t k,
    int64_t tiles_n,
    int64_t total_group_blocks) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t total_groups = rows * groups_per_row;
  const int64_t lane = static_cast<int64_t>(threadIdx.x);
  const int64_t lane_count = static_cast<int64_t>(blockDim.x);

  for (int64_t group_block = static_cast<int64_t>(blockIdx.x); group_block < total_group_blocks;
       group_block += static_cast<int64_t>(gridDim.x)) {
    const int64_t group_idx = group_block * lane_count + lane;
    if (group_idx >= total_groups) {
      continue;
    }

    const int64_t row = group_idx / groups_per_row;
    const int32_t start = row_offsets[row];
    const int32_t end = row_offsets[row + 1];
    if (start == end) {
      continue;
    }

    const int64_t col_group = group_idx - row * groups_per_row;
    const int64_t base_col = col_group * VecCols;
    const int64_t tile_id = (row / 128) * tiles_n + (base_col / 128);
    while (atomicAdd(const_cast<int32_t*>(ready_flags + tile_id), 0) == 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
      __nanosleep(64);
#endif
    }
    __threadfence();

    const int64_t row_k_base = row * k;
    const int64_t out_base = row * cols + base_col;
    float acc[VecCols];

    const auto* output_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_output = output_u64[pack];
      acc[pack * 4 + 0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 0));
      acc[pack * 4 + 1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 1));
      acc[pack * 4 + 2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 2));
      acc[pack * 4 + 3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 3));
    }

    for (int32_t pos = start; pos < end; ++pos) {
      const int32_t flat = flat_indices[pos];
      const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
      const int64_t weight_base = kk * cols + base_col;
      const float value = static_cast<float>(outlier_values[pos]);
      const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
      for (int pack = 0; pack < Packs; ++pack) {
        const uint64_t packed_weight = weight_u64[pack];
        acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
        acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
        acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
        acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
      }
    }

auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      store_u64[pack] = pack_bf16x4_v13(
          acc[pack * 4 + 0],
          acc[pack * 4 + 1],
          acc[pack * 4 + 2],
          acc[pack * 4 + 3]);
    }
  }
}

__device__ __forceinline__ void apply_sparse_row_group_vec8_vstore(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    int64_t row,
    int64_t base_col,
    int64_t cols,
    int64_t k) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t row_k_base = row * k;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];

  const auto* output_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    const uint64_t packed_output = output_u64[pack];
    acc[pack * 4 + 0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 0));
    acc[pack * 4 + 1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 1));
    acc[pack * 4 + 2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 2));
    acc[pack * 4 + 3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 3));
  }

  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  for (int32_t pos = start; pos < end; ++pos) {
    const int32_t flat = flat_indices[pos];
    const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[pos]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  }

  store_bf16x8_u4_v13(output + out_base, acc);
}

__global__ __launch_bounds__(256, 4)
void sparse_active_row_ready_value_payload_vec8_inplace_vstore_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    const int32_t* __restrict__ ready_flags,
    int64_t active_row_count,
    int64_t cols,
    int64_t k,
    int64_t tiles_n,
    int64_t total_group_blocks,
    int32_t sleep_ns) {
  constexpr int VecCols = 8;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t total_groups = active_row_count * groups_per_row;
  const int64_t lane = static_cast<int64_t>(threadIdx.x);
  const int64_t lane_count = static_cast<int64_t>(blockDim.x);

  for (int64_t group_block = static_cast<int64_t>(blockIdx.x); group_block < total_group_blocks;
       group_block += static_cast<int64_t>(gridDim.x)) {
    const int64_t group_idx = group_block * lane_count + lane;
    if (group_idx >= total_groups) {
      continue;
    }

    const int64_t active_row_idx = group_idx / groups_per_row;
    const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
    const int64_t col_group = group_idx - active_row_idx * groups_per_row;
    const int64_t base_col = col_group * VecCols;
    const int64_t tile_id = (row / 128) * tiles_n + (base_col / 128);
    while (atomicAdd(const_cast<int32_t*>(ready_flags + tile_id), 0) == 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
      if (sleep_ns > 0) {
        __nanosleep(static_cast<unsigned int>(sleep_ns));
      }
#endif
    }
    __threadfence();

    apply_sparse_row_group_vec8_vstore(output,
                                       outlier_values,
                                       weight_t_bf16,
                                       flat_indices,
                                       row_offsets,
                                       row,
                                       base_col,
                                       cols,
                                       k);
  }
}

__global__ __launch_bounds__(256, 2)
void sparse_active_tile_ready_queue_value_payload_vec8_inplace_vstore_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_row_offsets,
    const int32_t* __restrict__ active_rows_local,
    const int32_t* __restrict__ ready_queue,
    const int32_t* __restrict__ ready_slot_status,
    int32_t* __restrict__ ready_head,
    const int32_t* __restrict__ ready_tail,
    int64_t rows,
    int64_t cols,
    int64_t k,
    int64_t tiles_n,
    int64_t total_tiles,
    int32_t sleep_ns) {
  constexpr int VecCols = 8;
  constexpr int TileM = 128;
  constexpr int TileN = 128;
  __shared__ int32_t shared_slot;
  __shared__ int32_t shared_tile_id;
  while (true) {
    if (threadIdx.x == 0) {
      int32_t slot = -1;
      while (slot < 0) {
        const int32_t head = atomicAdd(ready_head, 0);
        const int32_t tail = atomicAdd(const_cast<int32_t*>(ready_tail), 0);
        if (head >= total_tiles && tail >= total_tiles) {
          slot = -2;
          break;
        }
        if (head < tail && atomicCAS(ready_head, head, head + 1) == head) {
          slot = head;
          break;
        }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        if (sleep_ns > 0) {
          __nanosleep(static_cast<unsigned int>(sleep_ns));
        }
#endif
      }
      shared_slot = slot;
      if (slot >= 0) {
        while (atomicAdd(const_cast<int32_t*>(ready_slot_status + slot), 0) == 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
          if (sleep_ns > 0) {
            __nanosleep(static_cast<unsigned int>(sleep_ns));
          }
#endif
        }
        shared_tile_id = ready_queue[slot];
      }
    }
    __syncthreads();

    if (shared_slot == -2) {
      return;
    }

    const int64_t tile_id = static_cast<int64_t>(shared_tile_id);
    const int64_t tile_m = tile_id / tiles_n;
    const int64_t tile_n = tile_id - tile_m * tiles_n;
    const int64_t col_base = tile_n * TileN;
    const int64_t active_start = active_row_offsets[tile_m];
    const int64_t active_end = active_row_offsets[tile_m + 1];
    const int64_t active_count = active_end - active_start;
    const int64_t cols_in_tile = min(static_cast<int64_t>(TileN), cols - col_base);
    const int64_t groups_in_tile_n = cols_in_tile / VecCols;
    const int64_t total_groups = active_count * groups_in_tile_n;

    for (int64_t local_group = static_cast<int64_t>(threadIdx.x);
         local_group < total_groups;
         local_group += static_cast<int64_t>(blockDim.x)) {
      const int64_t active_idx = local_group / groups_in_tile_n;
      const int64_t col_group = local_group - active_idx * groups_in_tile_n;
      const int64_t local_row = active_rows_local[active_start + active_idx];
      const int64_t row = tile_m * TileM + local_row;
      if (row >= rows) {
        continue;
      }
      apply_sparse_row_group_vec8_vstore(output,
                                         outlier_values,
                                         weight_t_bf16,
                                         flat_indices,
                                         row_offsets,
                                         row,
                                         col_base + col_group * VecCols,
                                         cols,
                                         k);
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(256, 2)
void sparse_active_mtile_ready_queue_value_payload_vec8_inplace_vstore_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_row_offsets,
    const int32_t* __restrict__ active_rows_local,
    const int32_t* __restrict__ ready_queue,
    const int32_t* __restrict__ ready_slot_status,
    int32_t* __restrict__ ready_head,
    const int32_t* __restrict__ ready_tail,
    int64_t rows,
    int64_t cols,
    int64_t k,
    int64_t total_items,
    int64_t column_slices,
    int32_t sleep_ns) {
  constexpr int VecCols = 8;
  constexpr int TileM = 128;
  __shared__ int32_t shared_slot;
  __shared__ int32_t shared_mtile;
  const int64_t groups_per_row = cols / VecCols;

  while (true) {
    if (threadIdx.x == 0) {
      int32_t slot = -1;
      while (slot < 0) {
        const int32_t head = atomicAdd(ready_head, 0);
        const int32_t tail = atomicAdd(const_cast<int32_t*>(ready_tail), 0);
        if (head >= total_items && tail >= total_items) {
          slot = -2;
          break;
        }
        if (head < tail && atomicCAS(ready_head, head, head + 1) == head) {
          slot = head;
          break;
        }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        if (sleep_ns > 0) {
          __nanosleep(static_cast<unsigned int>(sleep_ns));
        }
#endif
      }
      shared_slot = slot;
      if (slot >= 0) {
        while (atomicAdd(const_cast<int32_t*>(ready_slot_status + slot), 0) == 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
          if (sleep_ns > 0) {
            __nanosleep(static_cast<unsigned int>(sleep_ns));
          }
#endif
        }
        shared_mtile = ready_queue[slot];
      }
    }
    __syncthreads();

    if (shared_slot == -2) {
      return;
    }

    const int64_t item = static_cast<int64_t>(shared_mtile);
    const int64_t slices = column_slices < 1 ? 1 : column_slices;
    const int64_t tile_m = item / slices;
    const int64_t col_slice = item - tile_m * slices;
    const int64_t active_start = active_row_offsets[tile_m];
    const int64_t active_end = active_row_offsets[tile_m + 1];
    const int64_t active_count = active_end - active_start;
    const int64_t groups_per_slice = (groups_per_row + slices - 1) / slices;
    const int64_t group_start = col_slice * groups_per_slice;
    const int64_t group_end = min(groups_per_row, group_start + groups_per_slice);
    const int64_t slice_groups = group_end > group_start ? group_end - group_start : 0;
    const int64_t total_groups = active_count * slice_groups;

    for (int64_t local_group = static_cast<int64_t>(threadIdx.x);
         local_group < total_groups;
         local_group += static_cast<int64_t>(blockDim.x)) {
      const int64_t active_idx = local_group / slice_groups;
      const int64_t col_group = group_start + local_group - active_idx * slice_groups;
      const int64_t local_row = active_rows_local[active_start + active_idx];
      const int64_t row = tile_m * TileM + local_row;
      if (row >= rows) {
        continue;
      }
      apply_sparse_row_group_vec8_vstore(output,
                                         outlier_values,
                                         weight_t_bf16,
                                         flat_indices,
                                         row_offsets,
                                         row,
                                         col_group * VecCols,
                                         cols,
                                         k);
    }
    __syncthreads();
  }
}

__global__
void sparse_value_payload_vec8_inplace_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    int64_t rows,
    int64_t cols,
    int64_t k) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_groups = rows * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  const int64_t row = group_idx / groups_per_row;
  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  if (start == end) {
    return;
  }

  const int64_t col_group = group_idx - row * groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t row_k_base = row * k;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];

#pragma unroll
  for (int cc = 0; cc < VecCols; ++cc) {
    acc[cc] = static_cast<float>(output[out_base + cc]);
  }

  for (int32_t pos = start; pos < end; ++pos) {
    const int32_t flat = flat_indices[pos];
    const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[pos]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  }

auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    store_u64[pack] = pack_bf16x4_v13(
        acc[pack * 4 + 0],
        acc[pack * 4 + 1],
        acc[pack * 4 + 2],
        acc[pack * 4 + 3]);
  }
}

__global__
void sparse_active_row_value_payload_vec8_inplace_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols,
    int64_t k) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_groups = active_row_count * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  const int64_t active_row_idx = group_idx / groups_per_row;
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  if (row < 0) {
    return;
  }
  const int64_t col_group = group_idx - active_row_idx * groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t row_k_base = row * k;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];

#pragma unroll
  for (int cc = 0; cc < VecCols; ++cc) {
    acc[cc] = static_cast<float>(output[out_base + cc]);
  }

  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  for (int32_t pos = start; pos < end; ++pos) {
    const int32_t flat = flat_indices[pos];
    const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[pos]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  }

auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    store_u64[pack] = pack_bf16x4_v13(
        acc[pack * 4 + 0],
        acc[pack * 4 + 1],
        acc[pack * 4 + 2],
        acc[pack * 4 + 3]);
	  }
	}

__global__
void sparse_active_row_value_payload_vec8_store_vstore_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols,
    int64_t k) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_groups = active_row_count * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  const int64_t active_row_idx = group_idx / groups_per_row;
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  if (row < 0) {
    return;
  }
  const int64_t col_group = group_idx - active_row_idx * groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t row_k_base = row * k;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols] = {};

  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  for (int32_t pos = start; pos < end; ++pos) {
    const int32_t flat = flat_indices[pos];
    const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[pos]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  }

  store_bf16x8_u4_v13(output + out_base, acc);
}

__global__ __launch_bounds__(256, 2)
void merge_full_delta_active_rows_vec8_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ delta_output,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols) {
  constexpr int VecCols = 8;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t total_groups = active_row_count * groups_per_row;
  for (int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group_idx < total_groups;
       group_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t active_row_idx = group_idx / groups_per_row;
    const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
    const int64_t col_group = group_idx - active_row_idx * groups_per_row;
    const int64_t base_col = col_group * VecCols;
    const int64_t out_base = row * cols + base_col;
    const auto* out_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
    const auto* delta_u64 = reinterpret_cast<const uint64_t*>(delta_output + out_base);
    const uint64_t out_lo = out_u64[0];
    const uint64_t out_hi = out_u64[1];
    const uint64_t delta_lo = delta_u64[0];
    const uint64_t delta_hi = delta_u64[1];
    float acc[VecCols];
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      acc[slot] = bf16_bits_to_float_v12(packed_bf16_slot_v12(out_lo, slot)) +
                  bf16_bits_to_float_v12(packed_bf16_slot_v12(delta_lo, slot));
      acc[slot + 4] = bf16_bits_to_float_v12(packed_bf16_slot_v12(out_hi, slot)) +
                      bf16_bits_to_float_v12(packed_bf16_slot_v12(delta_hi, slot));
    }
    store_bf16x8_u4_v13(output + out_base, acc);
  }
}

__global__ __launch_bounds__(256, 2)
void merge_compact_delta_active_rows_vec8_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ compact_delta,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols) {
  constexpr int VecCols = 8;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t total_groups = active_row_count * groups_per_row;
  for (int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group_idx < total_groups;
       group_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t active_row_idx = group_idx / groups_per_row;
    const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
    if (row < 0) {
      continue;
    }
    const int64_t col_group = group_idx - active_row_idx * groups_per_row;
    const int64_t base_col = col_group * VecCols;
    const int64_t out_base = row * cols + base_col;
    const int64_t delta_base = active_row_idx * cols + base_col;
    const auto* out_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
    const auto* delta_u64 = reinterpret_cast<const uint64_t*>(compact_delta + delta_base);
    const uint64_t out_lo = out_u64[0];
    const uint64_t out_hi = out_u64[1];
    const uint64_t delta_lo = delta_u64[0];
    const uint64_t delta_hi = delta_u64[1];
    float acc[VecCols];
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      acc[slot] = bf16_bits_to_float_v12(packed_bf16_slot_v12(out_lo, slot)) +
                  bf16_bits_to_float_v12(packed_bf16_slot_v12(delta_lo, slot));
      acc[slot + 4] = bf16_bits_to_float_v12(packed_bf16_slot_v12(out_hi, slot)) +
                      bf16_bits_to_float_v12(packed_bf16_slot_v12(delta_hi, slot));
    }
    store_bf16x8_u4_v13(output + out_base, acc);
  }
}

__global__ __launch_bounds__(256, 2)
void merge_compact_delta_active_rows_n4096_vec16_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ compact_delta,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count) {
  const int64_t active_row_idx = static_cast<int64_t>(blockIdx.x);
  if (active_row_idx >= active_row_count) {
    return;
  }
  constexpr int64_t Cols = 4096;
  constexpr int64_t VecCols = 16;
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  if (row < 0) {
    return;
  }
  const int64_t base_col = static_cast<int64_t>(threadIdx.x) * VecCols;
  auto* output_vec = reinterpret_cast<uint4*>(output + row * Cols + base_col);
  const auto* delta_vec = reinterpret_cast<const uint4*>(
      compact_delta + active_row_idx * Cols + base_col);
  output_vec[0] = add_bf16x8_u4_v208(output_vec[0], delta_vec[0]);
  output_vec[1] = add_bf16x8_u4_v208(output_vec[1], delta_vec[1]);
}

__global__ __launch_bounds__(256, 2)
void merge_two_compact_delta_active_rows_n4096_vec16_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ first_delta,
    const int32_t* __restrict__ first_rows,
    int64_t first_row_count,
    const c10::BFloat16* __restrict__ second_delta,
    const int32_t* __restrict__ second_rows,
    int64_t second_row_count) {
  const int64_t combined_row_idx = static_cast<int64_t>(blockIdx.x);
  if (combined_row_idx >= first_row_count + second_row_count) {
    return;
  }
  const bool use_second = combined_row_idx >= first_row_count;
  const int64_t compact_row_idx =
      use_second ? combined_row_idx - first_row_count : combined_row_idx;
  const int64_t row = static_cast<int64_t>(
      use_second ? second_rows[compact_row_idx] : first_rows[compact_row_idx]);
  const c10::BFloat16* compact_delta =
      use_second ? second_delta : first_delta;
  constexpr int64_t Cols = 4096;
  constexpr int64_t VecCols = 16;
  const int64_t base_col = static_cast<int64_t>(threadIdx.x) * VecCols;
  auto* output_vec = reinterpret_cast<uint4*>(output + row * Cols + base_col);
  const auto* delta_vec = reinterpret_cast<const uint4*>(
      compact_delta + compact_row_idx * Cols + base_col);
  output_vec[0] = add_bf16x8_u4_v208(output_vec[0], delta_vec[0]);
  output_vec[1] = add_bf16x8_u4_v208(output_vec[1], delta_vec[1]);
}

__global__ __launch_bounds__(256, 2)
void build_padded_light_heavy_rows_kernel(
    const int32_t* __restrict__ row_offsets,
    int32_t* __restrict__ light_rows,
    int32_t* __restrict__ heavy_rows,
    int32_t* __restrict__ light_row_count,
    int32_t* __restrict__ heavy_row_count,
    int64_t rows,
    int32_t heavy_threshold,
    int32_t heavy_capacity) {
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = 8;
  __shared__ int32_t light_warp_counts[kWarpsPerBlock];
  __shared__ int32_t heavy_warp_counts[kWarpsPerBlock];
  __shared__ int32_t light_warp_offsets[kWarpsPerBlock];
  __shared__ int32_t heavy_warp_offsets[kWarpsPerBlock];
  __shared__ int32_t light_block_base;
  __shared__ int32_t heavy_block_base;

  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) & (kWarpSize - 1);
  const int32_t warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
  const int32_t row_count = row < rows ? row_offsets[row + 1] - row_offsets[row] : 0;
  const bool is_light = row_count > 0 && row_count < heavy_threshold;
  const bool is_heavy = row_count >= heavy_threshold;
  const uint32_t light_mask = __ballot_sync(0xffffffffu, is_light);
  const uint32_t heavy_mask = __ballot_sync(0xffffffffu, is_heavy);
  if (lane == 0) {
    light_warp_counts[warp] = __popc(light_mask);
    heavy_warp_counts[warp] = __popc(heavy_mask);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t light_block_count = 0;
    int32_t heavy_block_count = 0;
#pragma unroll
    for (int32_t index = 0; index < kWarpsPerBlock; ++index) {
      light_warp_offsets[index] = light_block_count;
      heavy_warp_offsets[index] = heavy_block_count;
      light_block_count += light_warp_counts[index];
      heavy_block_count += heavy_warp_counts[index];
    }
    light_block_base = atomicAdd(light_row_count, light_block_count);
    heavy_block_base = atomicAdd(heavy_row_count, heavy_block_count);
  }
  __syncthreads();

  const uint32_t preceding_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
  if (is_light) {
    const int32_t light_pos =
        light_block_base + light_warp_offsets[warp] + __popc(light_mask & preceding_lanes);
    light_rows[light_pos] = static_cast<int32_t>(row);
  }
  if (is_heavy) {
    const int32_t heavy_pos =
        heavy_block_base + heavy_warp_offsets[warp] + __popc(heavy_mask & preceding_lanes);
    if (heavy_pos < heavy_capacity) {
      heavy_rows[heavy_pos] = static_cast<int32_t>(row);
    } else {
      const int32_t light_pos = atomicAdd(light_row_count, 1);
      light_rows[light_pos] = static_cast<int32_t>(row);
    }
  }
}

__global__ __launch_bounds__(256, 2)
void zero_compact_dense_residual_vec8_kernel(
    c10::BFloat16* __restrict__ residual,
    int64_t active_row_count,
    int64_t k) {
  constexpr int VecCols = 8;
  const int64_t groups_per_row = k / VecCols;
  const int64_t total_groups = active_row_count * groups_per_row;
  for (int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       group_idx < total_groups;
       group_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t base = group_idx * VecCols;
    *reinterpret_cast<uint4*>(residual + base) = make_uint4(0u, 0u, 0u, 0u);
  }
}

__global__ __launch_bounds__(256, 2)
void scatter_compact_dense_residual_rows_kernel(
    c10::BFloat16* __restrict__ residual,
    const c10::BFloat16* __restrict__ row_values,
    const int32_t* __restrict__ row_ks,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t k) {
  const int64_t active_row_idx = static_cast<int64_t>(blockIdx.x);
  if (active_row_idx >= active_row_count) {
    return;
  }
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  if (row < 0) {
    return;
  }
  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  c10::BFloat16* row_residual = residual + active_row_idx * k;
  for (int32_t pos = start + static_cast<int32_t>(threadIdx.x);
       pos < end;
       pos += static_cast<int32_t>(blockDim.x)) {
    const int32_t kk = row_ks[pos];
    if (kk >= 0 && kk < k) {
      row_residual[kk] = row_values[pos];
    }
  }
}

__global__
void split_hot_dense_padded_cold_rows_kernel(
    c10::BFloat16* __restrict__ hot_dense,
    c10::BFloat16* __restrict__ cold_values,
    int16_t* __restrict__ cold_cols,
    int32_t* __restrict__ cold_counts,
    int32_t* __restrict__ overflow,
    const c10::BFloat16* __restrict__ row_values,
    const int16_t* __restrict__ row_cols,
    const int32_t* __restrict__ row_offsets,
    const int16_t* __restrict__ hot_lut,
    int64_t rows,
    int64_t k,
    int64_t hot_cols,
    int64_t cold_capacity) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  int32_t cold_count = 0;
  for (int32_t pos = start; pos < end; ++pos) {
    const int32_t kk = static_cast<int32_t>(row_cols[pos]);
    if (kk < 0 || kk >= k) {
      continue;
    }
    const int32_t hot_slot = static_cast<int32_t>(hot_lut[kk]);
    if (hot_slot >= 0 && hot_slot < hot_cols) {
      hot_dense[row * hot_cols + hot_slot] = row_values[pos];
    } else if (cold_count < cold_capacity) {
      const int64_t cold_pos = row * cold_capacity + cold_count;
      cold_values[cold_pos] = row_values[pos];
      cold_cols[cold_pos] = static_cast<int16_t>(kk);
      ++cold_count;
    } else {
      atomicExch(overflow, 1);
      ++cold_count;
    }
  }
  cold_counts[row] = cold_count;
}

__global__ __launch_bounds__(256, 2)
void sparse_padded_cold_col_vec16_inplace_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ cold_values,
    const int16_t* __restrict__ cold_cols,
    const int32_t* __restrict__ cold_counts,
    const c10::BFloat16* __restrict__ row_values,
    const int16_t* __restrict__ row_cols,
    const int32_t* __restrict__ row_offsets,
    const int16_t* __restrict__ hot_lut,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    int64_t rows,
    int64_t k,
    int64_t cols,
    int64_t cold_capacity) {
  constexpr int VecCols = 16;
  constexpr int Packs = VecCols / 4;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t col_group =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  const int64_t base_col = col_group * VecCols;
  if (row >= rows || base_col >= cols) {
    return;
  }
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];
  const auto* output_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    const uint64_t packed_output = output_u64[pack];
    acc[pack * 4 + 0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 0));
    acc[pack * 4 + 1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 1));
    acc[pack * 4 + 2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 2));
    acc[pack * 4 + 3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 3));
  }

  const int32_t cold_count = cold_counts[row];
  const bool use_padded = cold_count <= cold_capacity;
  const int32_t start = use_padded ? 0 : row_offsets[row];
  const int32_t end = use_padded ? cold_count : row_offsets[row + 1];
  const int64_t cold_base = row * cold_capacity;
  for (int32_t idx = start; idx < end; ++idx) {
    const int64_t cold_pos = cold_base + idx;
    const int64_t kk = use_padded
        ? static_cast<int64_t>(cold_cols[cold_pos])
        : static_cast<int64_t>(row_cols[idx]);
    if (kk < 0 || kk >= k || (!use_padded && hot_lut[kk] >= 0)) {
      continue;
    }
    const float value = static_cast<float>(
        use_padded ? cold_values[cold_pos] : row_values[idx]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(
        weight_t_bf16 + kk * cols + base_col);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  }

  auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    store_u64[pack] = pack_bf16x4_v13(
        acc[pack * 4 + 0],
        acc[pack * 4 + 1],
        acc[pack * 4 + 2],
        acc[pack * 4 + 3]);
  }
}

__global__
void sparse_active_row_value_payload_vec8_inplace_vstore_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols,
    int64_t k,
    int32_t skip_per_row) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_groups = active_row_count * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  const int64_t active_row_idx = group_idx / groups_per_row;
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  if (row < 0) {
    return;
  }
  const int64_t col_group = group_idx - active_row_idx * groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t row_k_base = row * k;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];

#pragma unroll
  for (int cc = 0; cc < VecCols; ++cc) {
    acc[cc] = static_cast<float>(output[out_base + cc]);
  }

  const int32_t end = row_offsets[row + 1];
  const int32_t row_start = row_offsets[row];
  const int32_t start = min(row_start + skip_per_row, end);
  for (int32_t pos = start; pos < end; ++pos) {
    const int32_t flat = flat_indices[pos];
    const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[pos]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  }

  store_bf16x8_u4_v13(output + out_base, acc);
}

__global__ __launch_bounds__(256, 2)
void sparse_packed_suffix12_vec8_inplace_vstore_kernel(
    c10::BFloat16* __restrict__ output,
    const int32_t* __restrict__ packed_suffix_records,
    const int32_t* __restrict__ active_rows,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    int64_t active_row_count,
    int64_t cols) {
  constexpr int VecCols = 8;
  constexpr int PackedRecords = 12;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t group_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_groups = active_row_count * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  // All supported Transformer shapes have N/8 divisible by a warp.  A warp
  // therefore owns 32 adjacent vec8 column groups from one output row.  Only
  // lane zero reads the compact (K, BF16 value) records; the weight vectors
  // remain full-warp, contiguous requests.
  const int lane = threadIdx.x & 31;
  const int64_t active_row_idx = group_idx / groups_per_row;
  int32_t row32 = 0;
  if (lane == 0) {
    row32 = active_rows[active_row_idx];
  }
  row32 = __shfl_sync(0xffffffffu, row32, 0);
  const int64_t row = static_cast<int64_t>(row32);
  const int64_t col_group = group_idx - active_row_idx * groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];
  const auto* output_u64 = reinterpret_cast<const uint64_t*>(
      output + out_base);
  const uint64_t output_lo = output_u64[0];
  const uint64_t output_hi = output_u64[1];
  acc[0] = bf16_bits_to_float_v12(
      packed_bf16_slot_v12(output_lo, 0));
  acc[1] = bf16_bits_to_float_v12(
      packed_bf16_slot_v12(output_lo, 1));
  acc[2] = bf16_bits_to_float_v12(
      packed_bf16_slot_v12(output_lo, 2));
  acc[3] = bf16_bits_to_float_v12(
      packed_bf16_slot_v12(output_lo, 3));
  acc[4] = bf16_bits_to_float_v12(
      packed_bf16_slot_v12(output_hi, 0));
  acc[5] = bf16_bits_to_float_v12(
      packed_bf16_slot_v12(output_hi, 1));
  acc[6] = bf16_bits_to_float_v12(
      packed_bf16_slot_v12(output_hi, 2));
  acc[7] = bf16_bits_to_float_v12(
      packed_bf16_slot_v12(output_hi, 3));

  auto accumulate_weight = [&](
      uint32_t record, uint64_t weight_lo, uint64_t weight_hi) {
    if ((record & 0xffffu) == 0) {
      return;
    }
    const float value = bf16_bits_to_float_v12(
        static_cast<uint16_t>(record));
    acc[0] = fmaf(
        value,
        bf16_bits_to_float_v12(packed_bf16_slot_v12(weight_lo, 0)),
        acc[0]);
    acc[1] = fmaf(
        value,
        bf16_bits_to_float_v12(packed_bf16_slot_v12(weight_lo, 1)),
        acc[1]);
    acc[2] = fmaf(
        value,
        bf16_bits_to_float_v12(packed_bf16_slot_v12(weight_lo, 2)),
        acc[2]);
    acc[3] = fmaf(
        value,
        bf16_bits_to_float_v12(packed_bf16_slot_v12(weight_lo, 3)),
        acc[3]);
    acc[4] = fmaf(
        value,
        bf16_bits_to_float_v12(packed_bf16_slot_v12(weight_hi, 0)),
        acc[4]);
    acc[5] = fmaf(
        value,
        bf16_bits_to_float_v12(packed_bf16_slot_v12(weight_hi, 1)),
        acc[5]);
    acc[6] = fmaf(
        value,
        bf16_bits_to_float_v12(packed_bf16_slot_v12(weight_hi, 2)),
        acc[6]);
    acc[7] = fmaf(
        value,
        bf16_bits_to_float_v12(packed_bf16_slot_v12(weight_hi, 3)),
        acc[7]);
  };

  int packed_wave_count = PackedRecords / 4;
#pragma unroll
  for (int record_base = 0;
       record_base < PackedRecords;
       record_base += 4) {
    // New-format record zero stores a marker in bit 31 and the number of
    // populated four-record waves in bits 29:30.  K is limited to 13 bits for
    // this fast path.  Legacy records have no marker and conservatively run
    // all three waves.
    if (record_base != 0 &&
        record_base / 4 >= packed_wave_count) {
      break;
    }
    uint4 packed = make_uint4(0, 0, 0, 0);
    if (lane == 0) {
      packed = *reinterpret_cast<const uint4*>(
          packed_suffix_records +
          active_row_idx * PackedRecords +
          record_base);
    }
    const uint32_t record0 =
        __shfl_sync(0xffffffffu, packed.x, 0);
    const uint32_t record1 =
        __shfl_sync(0xffffffffu, packed.y, 0);
    const uint32_t record2 =
        __shfl_sync(0xffffffffu, packed.z, 0);
    const uint32_t record3 =
        __shfl_sync(0xffffffffu, packed.w, 0);
    if (record_base == 0) {
      const int encoded_wave_count =
          static_cast<int>((record0 >> 29) & 0x3u);
      const bool encoded = (record0 & 0x80000000u) != 0;
      packed_wave_count =
          encoded ? encoded_wave_count : PackedRecords / 4;
    }

    uint64_t weight0_lo = 0;
    uint64_t weight0_hi = 0;
    uint64_t weight1_lo = 0;
    uint64_t weight1_hi = 0;
    uint64_t weight2_lo = 0;
    uint64_t weight2_hi = 0;
    uint64_t weight3_lo = 0;
    uint64_t weight3_hi = 0;
    if ((record0 & 0xffffu) != 0) {
      const auto* weight = reinterpret_cast<const uint64_t*>(
          weight_t_bf16 +
          static_cast<int64_t>((record0 >> 16) & 0x1fffu) * cols +
          base_col);
      weight0_lo = weight[0];
      weight0_hi = weight[1];
    }
    if ((record1 & 0xffffu) != 0) {
      const auto* weight = reinterpret_cast<const uint64_t*>(
          weight_t_bf16 +
          static_cast<int64_t>((record1 >> 16) & 0x1fffu) * cols +
          base_col);
      weight1_lo = weight[0];
      weight1_hi = weight[1];
    }
    if ((record2 & 0xffffu) != 0) {
      const auto* weight = reinterpret_cast<const uint64_t*>(
          weight_t_bf16 +
          static_cast<int64_t>((record2 >> 16) & 0x1fffu) * cols +
          base_col);
      weight2_lo = weight[0];
      weight2_hi = weight[1];
    }
    if ((record3 & 0xffffu) != 0) {
      const auto* weight = reinterpret_cast<const uint64_t*>(
          weight_t_bf16 +
          static_cast<int64_t>((record3 >> 16) & 0x1fffu) * cols +
          base_col);
      weight3_lo = weight[0];
      weight3_hi = weight[1];
    }

    // Keep four records' eight packed weight requests independent and live
    // before the first dependent FFMA.
    accumulate_weight(record0, weight0_lo, weight0_hi);
    accumulate_weight(record1, weight1_lo, weight1_hi);
    accumulate_weight(record2, weight2_lo, weight2_hi);
    accumulate_weight(record3, weight3_lo, weight3_hi);
  }

  store_bf16x8_u4_v13(output + out_base, acc);
}

__global__
void sparse_active_row_value_payload_vec8_inplace_strict_vstore_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols,
    int64_t k) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_groups = active_row_count * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  const int64_t active_row_idx = group_idx / groups_per_row;
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  if (row < 0) {
    return;
  }
  const int64_t col_group = group_idx - active_row_idx * groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t row_k_base = row * k;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];

#pragma unroll
  for (int cc = 0; cc < VecCols; ++cc) {
    acc[cc] = static_cast<float>(output[out_base + cc]);
  }

  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  for (int32_t pos = start; pos < end; ++pos) {
    const int32_t flat = flat_indices[pos];
    const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[pos]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] = __fadd_rn(
          acc[pack * 4 + 0],
          __fmul_rn(value, bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0))));
      acc[pack * 4 + 1] = __fadd_rn(
          acc[pack * 4 + 1],
          __fmul_rn(value, bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1))));
      acc[pack * 4 + 2] = __fadd_rn(
          acc[pack * 4 + 2],
          __fmul_rn(value, bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2))));
      acc[pack * 4 + 3] = __fadd_rn(
          acc[pack * 4 + 3],
          __fmul_rn(value, bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3))));
    }
  }

  store_bf16x8_u4_v13(output + out_base, acc);
}

__global__
void sparse_active_row_value_payload_vec8_inplace_sum_then_add_vstore_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols,
    int64_t k) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_groups = active_row_count * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  const int64_t active_row_idx = group_idx / groups_per_row;
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  if (row < 0) {
    return;
  }
  const int64_t col_group = group_idx - active_row_idx * groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t row_k_base = row * k;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];

#pragma unroll
  for (int cc = 0; cc < VecCols; ++cc) {
    acc[cc] = 0.0f;
  }

  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  for (int32_t pos = start; pos < end; ++pos) {
    const int32_t flat = flat_indices[pos];
    const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[pos]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  }

#pragma unroll
  for (int cc = 0; cc < VecCols; ++cc) {
    acc[cc] += static_cast<float>(output[out_base + cc]);
  }

  store_bf16x8_u4_v13(output + out_base, acc);
}

__global__
void sparse_active_row_value_payload_vec8_inplace_b_evict_last_vstore_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols,
    int64_t k) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_groups = active_row_count * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  const int64_t active_row_idx = group_idx / groups_per_row;
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  const int64_t col_group = group_idx - active_row_idx * groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t row_k_base = row * k;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];

#pragma unroll
  for (int cc = 0; cc < VecCols; ++cc) {
    acc[cc] = static_cast<float>(output[out_base + cc]);
  }

  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  for (int32_t pos = start; pos < end; ++pos) {
    const int32_t flat = flat_indices[pos];
    const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[pos]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = ld_global_u64_l1_evict_last_v13(weight_u64 + pack);
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  }

  store_bf16x8_u4_v13(output + out_base, acc);
}

template<int EpiN>
__global__ __launch_bounds__(256, 2)
void sparse_kmajor_epin_delta_store_kernel(
    c10::BFloat16* __restrict__ output,
    const int32_t* __restrict__ active_mblocks,
    const int32_t* __restrict__ active_row_offsets,
    const int32_t* __restrict__ active_rows,
    const int32_t* __restrict__ group_offsets,
    const int32_t* __restrict__ group_ks,
    const int32_t* __restrict__ entry_offsets,
    const int32_t* __restrict__ entry_rows,
    const c10::BFloat16* __restrict__ entry_values,
    const c10::BFloat16* __restrict__ b_comp,
    int64_t m,
    int64_t k,
    int64_t n) {
  constexpr int CtaM = 128;
  constexpr int VecN = 8;
  constexpr int ColGroups = EpiN / VecN;
  constexpr int Threads = 256;
  constexpr int WarpSize = 32;
  constexpr int Warps = Threads / WarpSize;
  static_assert(EpiN % VecN == 0, "EpiN must be divisible by VecN");
  static_assert(ColGroups <= 16, "half-warp owner expects at most 16 col groups");

  __shared__ int32_t row_to_slot[CtaM];
  __shared__ int32_t slot_rows[CtaM];
  __shared__ float acc[CtaM * EpiN];

  const int tid = threadIdx.x;
  const int warp_id = tid / WarpSize;
  const int lane = tid & 31;
  const int half = lane >> 4;
  const int half_lane = lane & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;

  const int active_m_idx = static_cast<int>(blockIdx.y);
  const int blk_m = active_mblocks[active_m_idx];
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int n0 = static_cast<int>(blockIdx.x) * EpiN;
  if (n0 >= n || block_m0 >= m) {
    return;
  }

  const int active_start = active_row_offsets[blk_m];
  const int active_end = active_row_offsets[blk_m + 1];
  const int active_count = active_end - active_start;
  if (active_count <= 0) {
    return;
  }

  for (int idx = tid; idx < CtaM; idx += Threads) {
    row_to_slot[idx] = -1;
  }
  for (int idx = tid; idx < active_count * EpiN; idx += Threads) {
    acc[idx] = 0.0f;
  }
  for (int slot = tid; slot < active_count; slot += Threads) {
    const int local_row = active_rows[active_start + slot];
    slot_rows[slot] = local_row;
    if (local_row >= 0 && local_row < CtaM) {
      row_to_slot[local_row] = slot;
    }
  }
  __syncthreads();

  const int group_start = group_offsets[active_m_idx];
  const int group_end = group_offsets[active_m_idx + 1];
  const int global_col0 = n0 + half_lane * VecN;
  const bool lane_valid = half_lane < ColGroups && global_col0 < n;
  const bool full_packed_vec = global_col0 + VecN <= n;

  for (int group_idx = group_start + warp_id * 2 + half; group_idx < group_end;
       group_idx += Warps * 2) {
    int gk = 0;
    int entry_start = 0;
    int entry_end = 0;
    if (half_lane == 0) {
      gk = group_ks[group_idx];
      entry_start = entry_offsets[group_idx];
      entry_end = entry_offsets[group_idx + 1];
    }
    gk = __shfl_sync(half_mask, gk, half_base_lane);
    entry_start = __shfl_sync(half_mask, entry_start, half_base_lane);
    entry_end = __shfl_sync(half_mask, entry_end, half_base_lane);
    if (entry_start == entry_end || gk < 0 || gk >= k) {
      continue;
    }

    float bvals[VecN] = {};
    if (lane_valid && full_packed_vec) {
      const auto* weight_u64 = reinterpret_cast<const uint64_t*>(
          b_comp + static_cast<int64_t>(gk) * n + global_col0);
      const uint64_t lo = weight_u64[0];
      const uint64_t hi = weight_u64[1];
      bvals[0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(lo, 0));
      bvals[1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(lo, 1));
      bvals[2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(lo, 2));
      bvals[3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(lo, 3));
      bvals[4] = bf16_bits_to_float_v12(packed_bf16_slot_v12(hi, 0));
      bvals[5] = bf16_bits_to_float_v12(packed_bf16_slot_v12(hi, 1));
      bvals[6] = bf16_bits_to_float_v12(packed_bf16_slot_v12(hi, 2));
      bvals[7] = bf16_bits_to_float_v12(packed_bf16_slot_v12(hi, 3));
    } else if (lane_valid) {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int global_col = global_col0 + cc;
        if (global_col < n) {
          bvals[cc] = static_cast<float>(
              b_comp[static_cast<int64_t>(gk) * n + global_col]);
        }
      }
    }

    for (int entry_idx = entry_start; entry_idx < entry_end; ++entry_idx) {
      int local_row = 0;
      if (half_lane == 0) {
        local_row = entry_rows[entry_idx];
      }
      local_row = __shfl_sync(half_mask, local_row, half_base_lane);
      const int slot =
          (local_row >= 0 && local_row < CtaM) ? row_to_slot[local_row] : -1;
      const float av = static_cast<float>(entry_values[entry_idx]);
      if (lane_valid && slot >= 0) {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int local_col = half_lane * VecN + cc;
          if (local_col < EpiN && n0 + local_col < n) {
            atomicAdd(&acc[slot * EpiN + local_col], av * bvals[cc]);
          }
        }
      }
    }
  }
  __syncthreads();

  for (int slot = warp_id * 2 + half; slot < active_count; slot += Warps * 2) {
    const int local_row = slot_rows[slot];
    const int64_t global_row = block_m0 + local_row;
    if (local_row < 0 || local_row >= CtaM || global_row >= m || !lane_valid) {
      continue;
    }
    const int local_col0 = half_lane * VecN;
    const int64_t out_base = global_row * n + n0 + local_col0;
    if (full_packed_vec) {
      float out_vals[VecN];
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        out_vals[cc] = acc[slot * EpiN + local_col0 + cc];
      }
      store_bf16x8_u4_v13(output + out_base, out_vals);
    } else {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int global_col = n0 + local_col0 + cc;
        if (global_col < n) {
          uint16_t* out_ptr = reinterpret_cast<uint16_t*>(output + out_base + cc);
          *out_ptr = float_to_bf16_bits_v13(acc[slot * EpiN + local_col0 + cc]);
        }
      }
    }
  }
}

template<int CtaM>
__global__ __launch_bounds__(64)
void sparse_kmajor_serial_group_inplace_kernel(
    c10::BFloat16* __restrict__ output,
    const int32_t* __restrict__ active_mblocks,
    const int32_t* __restrict__ group_offsets,
    const int32_t* __restrict__ group_ks,
    const int32_t* __restrict__ entry_offsets,
    const int32_t* __restrict__ entry_rows,
    const c10::BFloat16* __restrict__ entry_values,
    const c10::BFloat16* __restrict__ b_comp,
    int64_t m,
    int64_t k,
    int64_t n) {
  constexpr int EpiN = 64;
  __shared__ float acc[CtaM * EpiN];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int active_m_idx = static_cast<int>(blockIdx.y);
  const int blk_m = active_mblocks[active_m_idx];
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int64_t n0 = static_cast<int64_t>(blockIdx.x) * EpiN;
  if (block_m0 >= m || n0 + tid >= n) {
    return;
  }

#pragma unroll
  for (int local_row = 0; local_row < CtaM; ++local_row) {
    const int64_t global_row = block_m0 + local_row;
    acc[local_row * EpiN + tid] =
        global_row < m ? static_cast<float>(output[global_row * n + n0 + tid]) : 0.0f;
  }

  const int group_start = group_offsets[active_m_idx];
  const int group_end = group_offsets[active_m_idx + 1];
  for (int group_idx = group_start; group_idx < group_end; ++group_idx) {
    int gk = 0;
    int entry_start = 0;
    int entry_end = 0;
    if (lane == 0) {
      gk = group_ks[group_idx];
      entry_start = entry_offsets[group_idx];
      entry_end = entry_offsets[group_idx + 1];
    }
    gk = __shfl_sync(0xffffffffu, gk, 0);
    entry_start = __shfl_sync(0xffffffffu, entry_start, 0);
    entry_end = __shfl_sync(0xffffffffu, entry_end, 0);
    if (gk < 0 || gk >= k || entry_start >= entry_end) {
      continue;
    }

    const float b_value =
        static_cast<float>(b_comp[static_cast<int64_t>(gk) * n + n0 + tid]);
    for (int entry_idx = entry_start; entry_idx < entry_end; ++entry_idx) {
      int local_row = 0;
      float a_value = 0.0f;
      if (lane == 0) {
        local_row = entry_rows[entry_idx];
        a_value = static_cast<float>(entry_values[entry_idx]);
      }
      local_row = __shfl_sync(0xffffffffu, local_row, 0);
      a_value = __shfl_sync(0xffffffffu, a_value, 0);
      if (local_row >= 0 && local_row < CtaM && block_m0 + local_row < m) {
        acc[local_row * EpiN + tid] += a_value * b_value;
      }
    }
  }

#pragma unroll
  for (int local_row = 0; local_row < CtaM; ++local_row) {
    const int64_t global_row = block_m0 + local_row;
    if (global_row < m) {
      output[global_row * n + n0 + tid] = acc[local_row * EpiN + tid];
    }
  }
}

__global__ __launch_bounds__(256, 2)
void sparse_kmajor_epin64_direct_store_kernel(
    c10::BFloat16* __restrict__ output,
    const int32_t* __restrict__ active_mblocks,
    const int32_t* __restrict__ group_offsets,
    const int32_t* __restrict__ group_ks,
    const int32_t* __restrict__ entry_offsets,
    const int32_t* __restrict__ entry_rows,
    const c10::BFloat16* __restrict__ entry_values,
    const c10::BFloat16* __restrict__ b_comp,
    int64_t m,
    int64_t k,
    int64_t n) {
  constexpr int CtaM = 128;
  constexpr int EpiN = 64;
  constexpr int VecN = 8;
  constexpr int ColGroups = EpiN / VecN;
  constexpr int Threads = 256;
  constexpr int WarpSize = 32;
  constexpr int Warps = Threads / WarpSize;

  const int tid = threadIdx.x;
  const int warp_id = tid / WarpSize;
  const int lane = tid & 31;
  const int half = lane >> 4;
  const int half_lane = lane & 15;
  const int half_base_lane = half << 4;
  const unsigned half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;

  const int active_m_idx = static_cast<int>(blockIdx.y);
  const int blk_m = active_mblocks[active_m_idx];
  const int64_t block_m0 = static_cast<int64_t>(blk_m) * CtaM;
  const int n0 = static_cast<int>(blockIdx.x) * EpiN;
  if (n0 >= n || block_m0 >= m) {
    return;
  }

  const int group_start = group_offsets[active_m_idx];
  const int group_end = group_offsets[active_m_idx + 1];
  const int global_col0 = n0 + half_lane * VecN;
  const bool lane_valid = half_lane < ColGroups && global_col0 < n;
  const bool full_packed_vec = global_col0 + VecN <= n;

  for (int group_idx = group_start + warp_id * 2 + half; group_idx < group_end;
       group_idx += Warps * 2) {
    int gk = 0;
    int entry_start = 0;
    int entry_end = 0;
    if (half_lane == 0) {
      gk = group_ks[group_idx];
      entry_start = entry_offsets[group_idx];
      entry_end = entry_offsets[group_idx + 1];
    }
    gk = __shfl_sync(half_mask, gk, half_base_lane);
    entry_start = __shfl_sync(half_mask, entry_start, half_base_lane);
    entry_end = __shfl_sync(half_mask, entry_end, half_base_lane);
    if (entry_start == entry_end || gk < 0 || gk >= k) {
      continue;
    }

    float bvals[VecN] = {};
    if (lane_valid && full_packed_vec) {
      const auto* weight_u64 = reinterpret_cast<const uint64_t*>(
          b_comp + static_cast<int64_t>(gk) * n + global_col0);
      const uint64_t lo = weight_u64[0];
      const uint64_t hi = weight_u64[1];
      bvals[0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(lo, 0));
      bvals[1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(lo, 1));
      bvals[2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(lo, 2));
      bvals[3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(lo, 3));
      bvals[4] = bf16_bits_to_float_v12(packed_bf16_slot_v12(hi, 0));
      bvals[5] = bf16_bits_to_float_v12(packed_bf16_slot_v12(hi, 1));
      bvals[6] = bf16_bits_to_float_v12(packed_bf16_slot_v12(hi, 2));
      bvals[7] = bf16_bits_to_float_v12(packed_bf16_slot_v12(hi, 3));
    } else if (lane_valid) {
      #pragma unroll
      for (int cc = 0; cc < VecN; ++cc) {
        const int global_col = global_col0 + cc;
        if (global_col < n) {
          bvals[cc] = static_cast<float>(
              b_comp[static_cast<int64_t>(gk) * n + global_col]);
        }
      }
    }

    for (int entry_idx = entry_start; entry_idx < entry_end; ++entry_idx) {
      int local_row = 0;
      if (half_lane == 0) {
        local_row = entry_rows[entry_idx];
      }
      local_row = __shfl_sync(half_mask, local_row, half_base_lane);
      const int64_t global_row = block_m0 + local_row;
      if (local_row < 0 || local_row >= CtaM || global_row >= m || !lane_valid) {
        continue;
      }
      const float av = static_cast<float>(entry_values[entry_idx]);
      const int64_t out_base = global_row * n + global_col0;
      if (full_packed_vec) {
        float out_vals[VecN];
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          out_vals[cc] = av * bvals[cc];
        }
        store_bf16x8_u4_v13(output + out_base, out_vals);
      } else {
        #pragma unroll
        for (int cc = 0; cc < VecN; ++cc) {
          const int global_col = global_col0 + cc;
          if (global_col < n) {
            uint16_t* out_ptr = reinterpret_cast<uint16_t*>(output + out_base + cc);
            *out_ptr = float_to_bf16_bits_v13(av * bvals[cc]);
          }
        }
      }
    }
  }
}

__global__
void sparse_active_row_value_payload_vec8_inplace_fastpath_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols,
    int64_t k) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_groups = active_row_count * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  const int64_t active_row_idx = group_idx / groups_per_row;
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  const int64_t col_group = group_idx - active_row_idx * groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t row_k_base = row * k;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];

#pragma unroll
  for (int cc = 0; cc < VecCols; ++cc) {
    acc[cc] = static_cast<float>(output[out_base + cc]);
  }

  const int32_t start = row_offsets[row];
  const int32_t count = row_offsets[row + 1] - start;
  if (count == 1) {
    const int32_t flat = flat_indices[start];
    const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[start]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  } else if (count == 2) {
#pragma unroll
    for (int item = 0; item < 2; ++item) {
      const int32_t pos = start + item;
      const int32_t flat = flat_indices[pos];
      const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
      const int64_t weight_base = kk * cols + base_col;
      const float value = static_cast<float>(outlier_values[pos]);
      const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
      for (int pack = 0; pack < Packs; ++pack) {
        const uint64_t packed_weight = weight_u64[pack];
        acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
        acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
        acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
        acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
      }
    }
  } else {
    const int32_t end = start + count;
    for (int32_t pos = start; pos < end; ++pos) {
      const int32_t flat = flat_indices[pos];
      const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
      const int64_t weight_base = kk * cols + base_col;
      const float value = static_cast<float>(outlier_values[pos]);
      const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
      for (int pack = 0; pack < Packs; ++pack) {
        const uint64_t packed_weight = weight_u64[pack];
        acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
        acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
        acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
        acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
      }
    }
  }

  store_bf16x8_u4_v13(output + out_base, acc);
}

	__global__
	void sparse_active_row_value_payload_vec8_inplace_rowblock_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols,
    int64_t k,
    int64_t blocks_per_row) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t active_row_idx = static_cast<int64_t>(blockIdx.x) / blocks_per_row;
  if (active_row_idx >= active_row_count) {
    return;
  }
  const int64_t col_block = static_cast<int64_t>(blockIdx.x) - active_row_idx * blocks_per_row;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t col_group = col_block * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (col_group >= groups_per_row) {
    return;
  }

  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  const int64_t base_col = col_group * VecCols;
  const int64_t row_k_base = row * k;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];

#pragma unroll
  for (int cc = 0; cc < VecCols; ++cc) {
    acc[cc] = static_cast<float>(output[out_base + cc]);
  }

  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  for (int32_t pos = start; pos < end; ++pos) {
    const int32_t flat = flat_indices[pos];
    const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[pos]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  }

  auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    store_u64[pack] = pack_bf16x4_v13(
        acc[pack * 4 + 0],
        acc[pack * 4 + 1],
        acc[pack * 4 + 2],
        acc[pack * 4 + 3]);
  }
}

	__global__
	void sparse_active_row_value_payload_vec8_store_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols,
    int64_t k) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_groups = active_row_count * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  const int64_t active_row_idx = group_idx / groups_per_row;
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  const int64_t col_group = group_idx - active_row_idx * groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t row_k_base = row * k;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols] = {};

  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  for (int32_t pos = start; pos < end; ++pos) {
    const int32_t flat = flat_indices[pos];
    const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[pos]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  }

  auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    store_u64[pack] = pack_bf16x4_v13(
        acc[pack * 4 + 0],
        acc[pack * 4 + 1],
        acc[pack * 4 + 2],
        acc[pack * 4 + 3]);
  }
}

__global__
void sparse_active_row_col_value_payload_vec8_inplace_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const int16_t* __restrict__ outlier_cols,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols,
    int64_t k) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_groups = active_row_count * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  const int64_t active_row_idx = group_idx / groups_per_row;
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  const int64_t col_group = group_idx - active_row_idx * groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];

  const auto* output_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    const uint64_t packed_output = output_u64[pack];
    acc[pack * 4 + 0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 0));
    acc[pack * 4 + 1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 1));
    acc[pack * 4 + 2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 2));
    acc[pack * 4 + 3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 3));
  }

  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  for (int32_t pos = start; pos < end; ++pos) {
    const int64_t kk = static_cast<int64_t>(outlier_cols[pos]);
    if (kk < 0 || kk >= k) {
      continue;
    }
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[pos]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  }

auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    store_u64[pack] = pack_bf16x4_v13(
        acc[pack * 4 + 0],
        acc[pack * 4 + 1],
        acc[pack * 4 + 2],
        acc[pack * 4 + 3]);
  }
}

__global__
void sparse_active_row_col_value_payload_vec8_inplace_vstore_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const int16_t* __restrict__ outlier_cols,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_groups = active_row_count * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  const int64_t active_row_idx = group_idx / groups_per_row;
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  const int64_t col_group = group_idx - active_row_idx * groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];

  const auto* output_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    const uint64_t packed_output = output_u64[pack];
    acc[pack * 4 + 0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 0));
    acc[pack * 4 + 1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 1));
    acc[pack * 4 + 2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 2));
    acc[pack * 4 + 3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 3));
  }

  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  for (int32_t pos = start; pos < end; ++pos) {
    const int64_t kk = static_cast<int64_t>(outlier_cols[pos]);
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[pos]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  }

  store_bf16x8_u4_v13(output + out_base, acc);
}

__global__ __launch_bounds__(256, 2)
void sparse_active_row_col_value_payload_vec16_inplace_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const int16_t* __restrict__ outlier_cols,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols,
    int64_t k) {
  constexpr int VecCols = 16;
  constexpr int Packs = VecCols / 4;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total_groups = active_row_count * groups_per_row;
  if (group_idx >= total_groups) {
    return;
  }

  const int64_t active_row_idx = group_idx / groups_per_row;
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  const int64_t col_group = group_idx - active_row_idx * groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];

  const auto* output_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    const uint64_t packed_output = output_u64[pack];
    acc[pack * 4 + 0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 0));
    acc[pack * 4 + 1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 1));
    acc[pack * 4 + 2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 2));
    acc[pack * 4 + 3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 3));
  }

  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  for (int32_t pos = start; pos < end; ++pos) {
    const int64_t kk = static_cast<int64_t>(outlier_cols[pos]);
    if (kk < 0 || kk >= k) {
      continue;
    }
    const int64_t weight_base = kk * cols + base_col;
    const float value = static_cast<float>(outlier_values[pos]);
    const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_weight = weight_u64[pack];
      acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
      acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
      acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
      acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
    }
  }

auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
  for (int pack = 0; pack < Packs; ++pack) {
    store_u64[pack] = pack_bf16x4_v13(
        acc[pack * 4 + 0],
        acc[pack * 4 + 1],
        acc[pack * 4 + 2],
        acc[pack * 4 + 3]);
  }
}

template <bool SumThenAdd>
__global__ __launch_bounds__(256, 2)
void sparse_active_row_col_value_payload_vec8_shmem_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const int16_t* __restrict__ outlier_cols,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ active_rows,
    int64_t active_row_count,
    int64_t cols,
    int64_t blocks_per_row) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  constexpr int Chunk = 256;
  __shared__ float s_values[Chunk];
  __shared__ int16_t s_cols[Chunk];

  const int64_t active_row_idx = static_cast<int64_t>(blockIdx.x) / blocks_per_row;
  if (active_row_idx >= active_row_count) {
    return;
  }
  const int64_t col_block = static_cast<int64_t>(blockIdx.x) - active_row_idx * blocks_per_row;
  const int64_t groups_per_row = cols / VecCols;
  const int64_t col_group = col_block * blockDim.x + threadIdx.x;
  const int64_t row = static_cast<int64_t>(active_rows[active_row_idx]);
  if (row < 0) {
    return;
  }
  const bool valid_col = col_group < groups_per_row;
  const int64_t base_col = col_group * VecCols;
  const int64_t out_base = row * cols + base_col;
  float acc[VecCols];

  if (valid_col) {
    if constexpr (SumThenAdd) {
#pragma unroll
      for (int cc = 0; cc < VecCols; ++cc) {
        acc[cc] = 0.0f;
      }
    } else {
      const auto* output_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
#pragma unroll
      for (int pack = 0; pack < Packs; ++pack) {
        const uint64_t packed_output = output_u64[pack];
        acc[pack * 4 + 0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 0));
        acc[pack * 4 + 1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 1));
        acc[pack * 4 + 2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 2));
        acc[pack * 4 + 3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 3));
      }
    }
  }

  const int32_t start = row_offsets[row];
  const int32_t end = row_offsets[row + 1];
  for (int32_t chunk_start = start; chunk_start < end; chunk_start += Chunk) {
    const int32_t chunk_count = min(static_cast<int32_t>(Chunk), end - chunk_start);
    if (threadIdx.x < chunk_count) {
      const int32_t pos = chunk_start + threadIdx.x;
      s_values[threadIdx.x] = static_cast<float>(outlier_values[pos]);
      s_cols[threadIdx.x] = outlier_cols[pos];
    }
    __syncthreads();

    if (valid_col) {
      for (int32_t j = 0; j < chunk_count; ++j) {
        const int64_t kk = static_cast<int64_t>(s_cols[j]);
        const int64_t weight_base = kk * cols + base_col;
        const float value = s_values[j];
        const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
        for (int pack = 0; pack < Packs; ++pack) {
          const uint64_t packed_weight = weight_u64[pack];
          acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
          acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
          acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
          acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
        }
      }
    }
    __syncthreads();
  }

  if (valid_col) {
    if constexpr (SumThenAdd) {
      const auto* output_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
#pragma unroll
      for (int pack = 0; pack < Packs; ++pack) {
        const uint64_t packed_output = output_u64[pack];
        acc[pack * 4 + 0] += bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 0));
        acc[pack * 4 + 1] += bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 1));
        acc[pack * 4 + 2] += bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 2));
        acc[pack * 4 + 3] += bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 3));
      }
    }
    auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      store_u64[pack] = pack_bf16x4_v13(
          acc[pack * 4 + 0],
          acc[pack * 4 + 1],
          acc[pack * 4 + 2],
          acc[pack * 4 + 3]);
    }
  }
}

__device__ __forceinline__ void apply_sparse_tile_col_value_payload_vec8(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const int16_t* __restrict__ outlier_cols,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ row_offsets,
    int64_t rows,
    int64_t cols,
    int64_t tiles_n,
    int64_t tile_id) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  constexpr int TileM = 128;
  constexpr int TileN = 128;
  constexpr int GroupsPerTileN = TileN / VecCols;

  const int64_t tile_m = tile_id / tiles_n;
  const int64_t tile_n = tile_id - tile_m * tiles_n;
  const int64_t row_base = tile_m * TileM;
  const int64_t col_base = tile_n * TileN;
  const int64_t rows_in_tile = (row_base + TileM <= rows) ? TileM : (rows - row_base);
  const int64_t groups_in_tile = rows_in_tile * GroupsPerTileN;

  for (int64_t local_group = threadIdx.x; local_group < groups_in_tile;
       local_group += blockDim.x) {
    const int64_t local_row = local_group / GroupsPerTileN;
    const int64_t col_group = local_group - local_row * GroupsPerTileN;
    const int64_t row = row_base + local_row;
    const int32_t start = row_offsets[row];
    const int32_t end = row_offsets[row + 1];
    if (start == end) {
      continue;
    }

    const int64_t base_col = col_base + col_group * VecCols;
    const int64_t out_base = row * cols + base_col;
    float acc[VecCols];

    const auto* output_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_output = output_u64[pack];
      acc[pack * 4 + 0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 0));
      acc[pack * 4 + 1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 1));
      acc[pack * 4 + 2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 2));
      acc[pack * 4 + 3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 3));
    }

    for (int32_t pos = start; pos < end; ++pos) {
      const int64_t kk = static_cast<int64_t>(outlier_cols[pos]);
      const int64_t weight_base = kk * cols + base_col;
      const float value = static_cast<float>(outlier_values[pos]);
      const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
      for (int pack = 0; pack < Packs; ++pack) {
        const uint64_t packed_weight = weight_u64[pack];
        acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
        acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
        acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
        acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
      }
    }

auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      store_u64[pack] = pack_bf16x4_v13(
          acc[pack * 4 + 0],
          acc[pack * 4 + 1],
          acc[pack * 4 + 2],
          acc[pack * 4 + 3]);
    }
  }
}

__global__ __launch_bounds__(256, 2)
void sparse_tile_col_value_payload_vec8_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const int16_t* __restrict__ outlier_cols,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ row_offsets,
    int64_t rows,
    int64_t cols,
    int64_t tiles_n,
    int64_t total_tiles) {
  for (int64_t tile_id = static_cast<int64_t>(blockIdx.x); tile_id < total_tiles;
       tile_id += static_cast<int64_t>(gridDim.x)) {
    apply_sparse_tile_col_value_payload_vec8(output,
                                             outlier_values,
                                             outlier_cols,
                                             weight_t_bf16,
                                             row_offsets,
                                             rows,
                                             cols,
                                             tiles_n,
                                             tile_id);
  }
}

__global__ __launch_bounds__(256, 2)
void sparse_tile_col_value_payload_vec16_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const int16_t* __restrict__ outlier_cols,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ row_offsets,
    int64_t rows,
    int64_t cols,
    int64_t tiles_n,
    int64_t total_tiles) {
  constexpr int VecCols = 16;
  constexpr int Packs = VecCols / 4;
  constexpr int TileM = 128;
  constexpr int TileN = 128;
  constexpr int GroupsPerTileN = TileN / VecCols;

  for (int64_t tile_id = static_cast<int64_t>(blockIdx.x); tile_id < total_tiles;
       tile_id += static_cast<int64_t>(gridDim.x)) {
    const int64_t tile_m = tile_id / tiles_n;
    const int64_t tile_n = tile_id - tile_m * tiles_n;
    const int64_t row_base = tile_m * TileM;
    const int64_t col_base = tile_n * TileN;
    const int64_t rows_in_tile = (row_base + TileM <= rows) ? TileM : (rows - row_base);
    const int64_t groups_in_tile = rows_in_tile * GroupsPerTileN;

    for (int64_t local_group = threadIdx.x; local_group < groups_in_tile;
         local_group += blockDim.x) {
      const int64_t local_row = local_group / GroupsPerTileN;
      const int64_t col_group = local_group - local_row * GroupsPerTileN;
      const int64_t row = row_base + local_row;
      const int32_t start = row_offsets[row];
      const int32_t end = row_offsets[row + 1];
      if (start == end) {
        continue;
      }

      const int64_t base_col = col_base + col_group * VecCols;
      const int64_t out_base = row * cols + base_col;
      float acc[VecCols];

      const auto* output_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
#pragma unroll
      for (int pack = 0; pack < Packs; ++pack) {
        const uint64_t packed_output = output_u64[pack];
        acc[pack * 4 + 0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 0));
        acc[pack * 4 + 1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 1));
        acc[pack * 4 + 2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 2));
        acc[pack * 4 + 3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 3));
      }

      for (int32_t pos = start; pos < end; ++pos) {
        const int64_t kk = static_cast<int64_t>(outlier_cols[pos]);
        const int64_t weight_base = kk * cols + base_col;
        const float value = static_cast<float>(outlier_values[pos]);
        const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
        for (int pack = 0; pack < Packs; ++pack) {
          const uint64_t packed_weight = weight_u64[pack];
          acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
          acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
          acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
          acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
        }
      }

      auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
      for (int pack = 0; pack < Packs; ++pack) {
        store_u64[pack] = pack_bf16x4_v13(
            acc[pack * 4 + 0],
            acc[pack * 4 + 1],
            acc[pack * 4 + 2],
            acc[pack * 4 + 3]);
      }
    }
  }
}

__global__ __launch_bounds__(256, 2)
void sparse_tile_col_value_payload_vec32_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const int16_t* __restrict__ outlier_cols,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ row_offsets,
    int64_t rows,
    int64_t cols,
    int64_t tiles_n,
    int64_t total_tiles) {
  constexpr int VecCols = 32;
  constexpr int Packs = VecCols / 4;
  constexpr int TileM = 128;
  constexpr int TileN = 128;
  constexpr int GroupsPerTileN = TileN / VecCols;

  for (int64_t tile_id = static_cast<int64_t>(blockIdx.x); tile_id < total_tiles;
       tile_id += static_cast<int64_t>(gridDim.x)) {
    const int64_t tile_m = tile_id / tiles_n;
    const int64_t tile_n = tile_id - tile_m * tiles_n;
    const int64_t row_base = tile_m * TileM;
    const int64_t col_base = tile_n * TileN;
    const int64_t rows_in_tile = (row_base + TileM <= rows) ? TileM : (rows - row_base);
    const int64_t groups_in_tile = rows_in_tile * GroupsPerTileN;

    for (int64_t local_group = threadIdx.x; local_group < groups_in_tile;
         local_group += blockDim.x) {
      const int64_t local_row = local_group / GroupsPerTileN;
      const int64_t col_group = local_group - local_row * GroupsPerTileN;
      const int64_t row = row_base + local_row;
      const int32_t start = row_offsets[row];
      const int32_t end = row_offsets[row + 1];
      if (start == end) {
        continue;
      }

      const int64_t base_col = col_base + col_group * VecCols;
      const int64_t out_base = row * cols + base_col;
      float acc[VecCols];

      const auto* output_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
#pragma unroll
      for (int pack = 0; pack < Packs; ++pack) {
        const uint64_t packed_output = output_u64[pack];
        acc[pack * 4 + 0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 0));
        acc[pack * 4 + 1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 1));
        acc[pack * 4 + 2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 2));
        acc[pack * 4 + 3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 3));
      }

      for (int32_t pos = start; pos < end; ++pos) {
        const int64_t kk = static_cast<int64_t>(outlier_cols[pos]);
        const int64_t weight_base = kk * cols + base_col;
        const float value = static_cast<float>(outlier_values[pos]);
        const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
        for (int pack = 0; pack < Packs; ++pack) {
          const uint64_t packed_weight = weight_u64[pack];
          acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
          acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
          acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
          acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
        }
      }

      auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
      for (int pack = 0; pack < Packs; ++pack) {
        store_u64[pack] = pack_bf16x4_v13(
            acc[pack * 4 + 0],
            acc[pack * 4 + 1],
            acc[pack * 4 + 2],
            acc[pack * 4 + 3]);
      }
    }
  }
}

__device__ __forceinline__ int64_t v14_swizzled_tile_from_physical(
    int64_t physical_id,
    int64_t tiles_m,
    int64_t tiles_n) {
  constexpr int GM = HANDWRITTEN_TMA_GM;
  const int64_t physical_y = physical_id / tiles_n;
  const int64_t physical_x = physical_id - physical_y * tiles_n;
  const int64_t blk_idx = physical_y * tiles_n + physical_x;
  const int64_t blk_per_group_row = static_cast<int64_t>(GM) * tiles_n;
  const int64_t group_m = blk_idx / blk_per_group_row;
  const int64_t in_group = blk_idx - group_m * blk_per_group_row;
  const int64_t tile_n = in_group / GM;
  const int64_t tile_m = group_m * GM + (in_group - tile_n * GM);
  if (tile_m >= tiles_m || tile_n >= tiles_n) {
    return -1;
  }
  return tile_m * tiles_n + tile_n;
}

__device__ __forceinline__ int64_t v14_tile_from_scheduler_task(
    int64_t task_id,
    int64_t tiles_m,
    int64_t tiles_n,
    int32_t scheduler_mode) {
  const int64_t total_tiles = tiles_m * tiles_n;
  if (scheduler_mode == 1) {
    return task_id < total_tiles ? task_id : -1;
  }
  if (scheduler_mode == 2) {
    const int64_t tile_n = task_id / tiles_m;
    const int64_t tile_m = task_id - tile_n * tiles_m;
    return tile_n < tiles_n ? tile_m * tiles_n + tile_n : -1;
  }
  return v14_swizzled_tile_from_physical(task_id, tiles_m, tiles_n);
}

__device__ __forceinline__ void apply_sparse_tile_col_value_payload_vec16(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const int16_t* __restrict__ outlier_cols,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ row_offsets,
    int64_t rows,
    int64_t cols,
    int64_t tiles_n,
    int64_t tile_id) {
  constexpr int VecCols = 16;
  constexpr int Packs = VecCols / 4;
  constexpr int TileM = 128;
  constexpr int TileN = 128;
  constexpr int GroupsPerTileN = TileN / VecCols;

  const int64_t tile_m = tile_id / tiles_n;
  const int64_t tile_n = tile_id - tile_m * tiles_n;
  const int64_t row_base = tile_m * TileM;
  const int64_t col_base = tile_n * TileN;
  const int64_t rows_in_tile = (row_base + TileM <= rows) ? TileM : (rows - row_base);
  const int64_t groups_in_tile = rows_in_tile * GroupsPerTileN;

  for (int64_t local_group = threadIdx.x; local_group < groups_in_tile;
       local_group += blockDim.x) {
    const int64_t local_row = local_group / GroupsPerTileN;
    const int64_t col_group = local_group - local_row * GroupsPerTileN;
    const int64_t row = row_base + local_row;
    const int32_t start = row_offsets[row];
    const int32_t end = row_offsets[row + 1];
    if (start == end) {
      continue;
    }

    const int64_t base_col = col_base + col_group * VecCols;
    const int64_t out_base = row * cols + base_col;
    float acc[VecCols];

    const auto* output_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_output = output_u64[pack];
      acc[pack * 4 + 0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 0));
      acc[pack * 4 + 1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 1));
      acc[pack * 4 + 2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 2));
      acc[pack * 4 + 3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 3));
    }

    for (int32_t pos = start; pos < end; ++pos) {
      const int64_t kk = static_cast<int64_t>(outlier_cols[pos]);
      const int64_t weight_base = kk * cols + base_col;
      const float value = static_cast<float>(outlier_values[pos]);
      const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
      for (int pack = 0; pack < Packs; ++pack) {
        const uint64_t packed_weight = weight_u64[pack];
        acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
        acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
        acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
        acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
      }
    }

    auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      store_u64[pack] = pack_bf16x4_v13(
          acc[pack * 4 + 0],
          acc[pack * 4 + 1],
          acc[pack * 4 + 2],
          acc[pack * 4 + 3]);
    }
  }
}

__global__ __launch_bounds__(256, 2)
void sparse_tile_persistent_col_value_payload_vec16_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const int16_t* __restrict__ outlier_cols,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ tile_status,
    int32_t* __restrict__ next_task,
    int64_t rows,
    int64_t cols,
    int64_t tiles_m,
    int64_t tiles_n,
    int64_t task_count,
    int32_t scheduler_mode,
    int32_t sleep_ns) {
  __shared__ int64_t shared_task_id;
  __shared__ int64_t shared_tile_id;
  while (true) {
    if (threadIdx.x == 0) {
      const int64_t task_id = static_cast<int64_t>(atomicAdd(next_task, 1));
      shared_task_id = task_id;
      shared_tile_id =
          v14_tile_from_scheduler_task(task_id, tiles_m, tiles_n, scheduler_mode);
    }
    __syncthreads();

    if (shared_task_id >= task_count) {
      return;
    }
    if (shared_tile_id < 0) {
      __syncthreads();
      continue;
    }

    if (threadIdx.x == 0) {
      while (atomicAdd(const_cast<int32_t*>(tile_status + shared_tile_id), 0) == 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        if (sleep_ns > 0) {
          __nanosleep(static_cast<unsigned int>(sleep_ns));
        }
#endif
      }
    }
    __syncthreads();

    apply_sparse_tile_col_value_payload_vec16(output,
                                              outlier_values,
                                              outlier_cols,
                                              weight_t_bf16,
                                              row_offsets,
                                              rows,
                                              cols,
                                              tiles_n,
                                              shared_tile_id);
    __syncthreads();
  }
}

__global__ __launch_bounds__(256, 2)
void sparse_tile_ready_queue_col_value_payload_vec16_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const int16_t* __restrict__ outlier_cols,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ ready_queue,
    const int32_t* __restrict__ ready_slot_status,
    int32_t* __restrict__ ready_head,
    const int32_t* __restrict__ ready_tail,
    int64_t rows,
    int64_t cols,
    int64_t tiles_n,
    int64_t total_tiles,
    int32_t sleep_ns) {
  __shared__ int32_t shared_slot;
  __shared__ int32_t shared_tile_id;
  while (true) {
    if (threadIdx.x == 0) {
      int32_t slot = -1;
      while (slot < 0) {
        const int32_t head = atomicAdd(ready_head, 0);
        const int32_t tail = atomicAdd(const_cast<int32_t*>(ready_tail), 0);
        if (head >= total_tiles && tail >= total_tiles) {
          slot = -2;
          break;
        }
        if (head < tail && atomicCAS(ready_head, head, head + 1) == head) {
          slot = head;
          break;
        }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        if (sleep_ns > 0) {
          __nanosleep(static_cast<unsigned int>(sleep_ns));
        }
#endif
      }
      shared_slot = slot;
      if (slot >= 0) {
        while (atomicAdd(const_cast<int32_t*>(ready_slot_status + slot), 0) == 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
          if (sleep_ns > 0) {
            __nanosleep(static_cast<unsigned int>(sleep_ns));
          }
#endif
        }
        shared_tile_id = ready_queue[slot];
      }
    }
    __syncthreads();

    if (shared_slot == -2) {
      return;
    }

    apply_sparse_tile_col_value_payload_vec16(output,
                                              outlier_values,
                                              outlier_cols,
                                              weight_t_bf16,
                                              row_offsets,
                                              rows,
                                              cols,
                                              tiles_n,
                                              static_cast<int64_t>(shared_tile_id));
    __syncthreads();
  }
}

__device__ __forceinline__ void apply_sparse_tile_value_payload_vec8(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    int64_t rows,
    int64_t cols,
    int64_t k,
    int64_t tiles_n,
    int64_t tile_id) {
  constexpr int VecCols = 8;
  constexpr int Packs = VecCols / 4;
  constexpr int TileM = 128;
  constexpr int TileN = 128;
  constexpr int GroupsPerTileN = TileN / VecCols;

  const int64_t tile_m = tile_id / tiles_n;
  const int64_t tile_n = tile_id - tile_m * tiles_n;
  const int64_t row_base = tile_m * TileM;
  const int64_t col_base = tile_n * TileN;
  const int64_t rows_in_tile = (row_base + TileM <= rows) ? TileM : (rows - row_base);
  const int64_t groups_in_tile = rows_in_tile * GroupsPerTileN;

  for (int64_t local_group = threadIdx.x; local_group < groups_in_tile;
       local_group += blockDim.x) {
    const int64_t local_row = local_group / GroupsPerTileN;
    const int64_t col_group = local_group - local_row * GroupsPerTileN;
    const int64_t row = row_base + local_row;
    const int32_t start = row_offsets[row];
    const int32_t end = row_offsets[row + 1];
    if (start == end) {
      continue;
    }

    const int64_t base_col = col_base + col_group * VecCols;
    const int64_t row_k_base = row * k;
    const int64_t out_base = row * cols + base_col;
    float acc[VecCols];

    const auto* output_u64 = reinterpret_cast<const uint64_t*>(output + out_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      const uint64_t packed_output = output_u64[pack];
      acc[pack * 4 + 0] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 0));
      acc[pack * 4 + 1] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 1));
      acc[pack * 4 + 2] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 2));
      acc[pack * 4 + 3] = bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_output, 3));
    }

    for (int32_t pos = start; pos < end; ++pos) {
      const int32_t flat = flat_indices[pos];
      const int64_t kk = static_cast<int64_t>(flat) - row_k_base;
      const int64_t weight_base = kk * cols + base_col;
      const float value = static_cast<float>(outlier_values[pos]);
      const auto* weight_u64 = reinterpret_cast<const uint64_t*>(weight_t_bf16 + weight_base);
#pragma unroll
      for (int pack = 0; pack < Packs; ++pack) {
        const uint64_t packed_weight = weight_u64[pack];
        acc[pack * 4 + 0] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 0));
        acc[pack * 4 + 1] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 1));
        acc[pack * 4 + 2] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 2));
        acc[pack * 4 + 3] += value * bf16_bits_to_float_v12(packed_bf16_slot_v12(packed_weight, 3));
      }
    }

auto* store_u64 = reinterpret_cast<uint64_t*>(output + out_base);
#pragma unroll
    for (int pack = 0; pack < Packs; ++pack) {
      store_u64[pack] = pack_bf16x4_v13(
          acc[pack * 4 + 0],
          acc[pack * 4 + 1],
          acc[pack * 4 + 2],
          acc[pack * 4 + 3]);
    }
  }
}

__global__ __launch_bounds__(256, 2)
void sparse_tile_sidecar_value_payload_vec8_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    int32_t* __restrict__ tile_status,
    const int32_t* __restrict__ dense_done,
    int64_t rows,
    int64_t cols,
    int64_t k,
    int64_t tiles_n,
    int64_t total_tiles) {
  __shared__ int32_t block_claimed;
  while (atomicAdd(const_cast<int32_t*>(dense_done), 0) == 0) {
    bool claimed_any = false;
    for (int64_t tile_id = static_cast<int64_t>(blockIdx.x); tile_id < total_tiles;
         tile_id += static_cast<int64_t>(gridDim.x)) {
      if (threadIdx.x == 0) {
        block_claimed = (atomicCAS(tile_status + tile_id, 1, 2) == 1) ? 1 : 0;
      }
      __syncthreads();
      if (block_claimed == 0) {
        continue;
      }
      apply_sparse_tile_value_payload_vec8(output,
                                           outlier_values,
                                           weight_t_bf16,
                                           flat_indices,
                                           row_offsets,
                                           rows,
                                           cols,
                                           k,
                                           tiles_n,
                                           tile_id);
      __syncthreads();
      if (threadIdx.x == 0) {
        __threadfence();
        tile_status[tile_id] = 3;
      }
      __syncthreads();
      claimed_any = true;
    }
    if (!claimed_any) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
      __nanosleep(256);
#endif
    }
  }
}

__global__ __launch_bounds__(256, 2)
void sparse_tile_tail_value_payload_vec8_kernel(
    c10::BFloat16* __restrict__ output,
    const c10::BFloat16* __restrict__ outlier_values,
    const c10::BFloat16* __restrict__ weight_t_bf16,
    const int32_t* __restrict__ flat_indices,
    const int32_t* __restrict__ row_offsets,
    int32_t* __restrict__ tile_status,
    int64_t rows,
    int64_t cols,
    int64_t k,
    int64_t tiles_n,
    int64_t total_tiles) {
  __shared__ int32_t block_claimed;
  for (int64_t tile_id = static_cast<int64_t>(blockIdx.x); tile_id < total_tiles;
       tile_id += static_cast<int64_t>(gridDim.x)) {
    if (threadIdx.x == 0) {
      int32_t old = atomicCAS(tile_status + tile_id, 1, 2);
      if (old != 1) {
        old = atomicCAS(tile_status + tile_id, 0, 2);
      }
      block_claimed = (old == 1 || old == 0) ? 1 : 0;
    }
    __syncthreads();
    if (block_claimed == 0) {
      continue;
    }
    apply_sparse_tile_value_payload_vec8(output,
                                         outlier_values,
                                         weight_t_bf16,
                                         flat_indices,
                                         row_offsets,
                                         rows,
                                         cols,
                                         k,
                                         tiles_n,
                                         tile_id);
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      tile_status[tile_id] = 3;
    }
    __syncthreads();
  }
}

__global__ void mark_dense_done_kernel(int32_t* __restrict__ dense_done) {
  *dense_done = 1;
}

__global__ void v14_sparse_start_delay_kernel(int32_t delay_us) {
  for (int32_t i = 0; i < delay_us; ++i) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    __nanosleep(1000);
#endif
  }
}

cudaStream_t get_v12_sparse_stream() {
  static cudaStream_t stream = [] {
    cudaStream_t created = nullptr;
    int least_priority = 0;
    int greatest_priority = 0;
    C10_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    C10_CUDA_CHECK(cudaStreamCreateWithPriority(&created, cudaStreamNonBlocking, greatest_priority));
    return created;
  }();
  return stream;
}

cudaStream_t create_v12_sparse_stream() {
  cudaStream_t created = nullptr;
  int least_priority = 0;
  int greatest_priority = 0;
  C10_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  C10_CUDA_CHECK(cudaStreamCreateWithPriority(&created, cudaStreamNonBlocking, least_priority));
  return created;
}

int default_sparse_side_worker_blocks(int64_t max_blocks) {
  int device = 0;
  int sm_count = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  int64_t blocks = std::max<int64_t>(1, static_cast<int64_t>(sm_count) / 4);
  blocks = std::min<int64_t>(blocks, std::max<int64_t>(1, max_blocks));
  return static_cast<int>(blocks);
}

void v12_sparse_stream_wait_current(cudaStream_t stream) {
  cudaEvent_t event = nullptr;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  C10_CUDA_CHECK(cudaEventRecord(event, at::cuda::getCurrentCUDAStream().stream()));
  C10_CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
  C10_CUDA_CHECK(cudaEventDestroy(event));
}

void v12_current_stream_wait_sparse(cudaStream_t stream) {
  cudaEvent_t event = nullptr;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  C10_CUDA_CHECK(cudaEventRecord(event, stream));
  C10_CUDA_CHECK(cudaStreamWaitEvent(at::cuda::getCurrentCUDAStream().stream(), event, 0));
  C10_CUDA_CHECK(cudaEventDestroy(event));
}

}  // namespace

at::Tensor swizzle_te_scale_to_tma_tile_major(const at::Tensor& scale,
                                              int64_t rows,
                                              int64_t k) {
  const c10::cuda::CUDAGuard device_guard(scale.device());
  return swizzle_te_scale_to_tma_tile_major_cuda(scale, rows, k);
}

at::Tensor nvfp4_gemm_tma_warpspecialized_cuda(const at::Tensor& a_data,
                                               const at::Tensor& a_scale_inv,
                                               const at::Tensor& b_data,
                                               const at::Tensor& b_scale_inv,
                                               const at::Tensor& a_amax,
                                               const at::Tensor& b_amax,
                                               int64_t m,
                                               int64_t k,
                                               int64_t n) {
  const c10::cuda::CUDAGuard device_guard(a_data.device());
  auto output = at::empty({m, n}, a_data.options().dtype(at::kBFloat16));
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_inv.data_ptr<uint8_t>(),
                         b_scale_inv.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor nvfp4_gemm_tma_tile_scales_cuda(const at::Tensor& a_data,
                                           const at::Tensor& a_scale_tile,
                                           const at::Tensor& b_data,
                                           const at::Tensor& b_scale_tile,
                                           const at::Tensor& a_amax,
                                           const at::Tensor& b_amax,
                                           int64_t m,
                                           int64_t k,
                                           int64_t n) {
  const c10::cuda::CUDAGuard device_guard(a_data.device());
  auto output = at::empty({m, n}, a_data.options().dtype(at::kBFloat16));
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_cuda(const at::Tensor& output,
                                                        const at::Tensor& a_data,
                                                        const at::Tensor& a_scale_tile,
                                                        const at::Tensor& b_data,
                                                        const at::Tensor& b_scale_tile,
                                                        const at::Tensor& a_amax,
                                                        const at::Tensor& b_amax,
                                                        int64_t m,
                                                        int64_t k,
                                                        int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_4wg_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    int64_t m,
    int64_t k,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
	                         at::cuda::getCurrentCUDAStream(),
	                         true,
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
	                         0,
	                         nullptr,
	                         nullptr,
	                         0,
	                         0,
	                         nullptr,
	                         nullptr,
	                         nullptr,
	                         nullptr,
	                         nullptr,
	                         0,
	                         nullptr,
	                         0,
	                         0,
	                         0,
	                         true,
	                         0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_compact_consumer_posttail_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(
      a_data.data_ptr<uint8_t>(),
      b_data.data_ptr<uint8_t>(),
      a_scale_tile.data_ptr<uint8_t>(),
      b_scale_tile.data_ptr<uint8_t>(),
      a_amax.data_ptr<float>(),
      b_amax.data_ptr<float>(),
      reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
      nullptr,
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      at::cuda::getCurrentCUDAStream(),
      true,
      nullptr,
      nullptr,
      nullptr,
      row_offsets.data_ptr<int32_t>(),
      row_ks.data_ptr<int32_t>(),
      row_values.data_ptr<c10::BFloat16>(),
      nullptr,
      nullptr,
      b_comp.data_ptr<c10::BFloat16>(),
      nullptr,
      0,
      nullptr,
      nullptr,
      1,
      50);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

int64_t compact_consumer_max_nnz_cuda() {
  return HANDWRITTEN_TMA_COMPACT_CONSUMER_MAX_NNZ;
}

int64_t compact_consumer_static_n_cuda() {
  return HANDWRITTEN_TMA_COMPACT_CONSUMER_STATIC_N;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_direct_add_active_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
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
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_direct_add_active_4wg_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
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
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         nullptr,
                         0,
                         nullptr,
                         nullptr,
                         0,
                         0,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         nullptr,
                         0,
                         0,
                         0,
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_direct_smem_active_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
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
    int64_t direct_smem_mode) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         nullptr,
                         0,
                         nullptr,
                         nullptr,
                         0,
                         0,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         nullptr,
                         0,
                         0,
                         0,
                         false,
                         static_cast<int32_t>(direct_smem_mode));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_direct_smem_delta_active_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& delta_output,
    const at::Tensor& active_row_offsets,
    const at::Tensor& active_rows,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t direct_smem_mode) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         nullptr,
                         nullptr,
                         0,
                         nullptr,
                         nullptr,
                         0,
                         0,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         const_cast<c10::BFloat16*>(
                             delta_output.data_ptr<c10::BFloat16>()),
                         0,
                         0,
                         0,
                         false,
                         static_cast<int32_t>(direct_smem_mode));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_active_cuda(
    const at::Tensor& output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& probe_active_mblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
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
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         probe_active_mblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(probe_active_mblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_active_sidewarp_cuda(
    const at::Tensor& output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& probe_active_mblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
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
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         probe_active_mblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(probe_active_mblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& probe_active_mblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
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
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         probe_active_mblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(probe_active_mblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         3,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         delta_output.data_ptr<c10::BFloat16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_sidewarp_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& probe_active_mblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
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
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         probe_active_mblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(probe_active_mblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         4,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         delta_output.data_ptr<c10::BFloat16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_rowblock_sidewarp_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         nullptr,
                         nullptr,
                         b_comp.data_ptr<c10::BFloat16>(),
                         nullptr,
                         0,
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         38,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         delta_output.data_ptr<c10::BFloat16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_sidewarp_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& active_rowblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         nullptr,
                         nullptr,
                         b_comp.data_ptr<c10::BFloat16>(),
                         active_rowblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(active_rowblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         39,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         delta_output.data_ptr<c10::BFloat16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_static_persistent_sidewarp_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& active_rowblocks,
    const at::Tensor& active_rows,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         nullptr,
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         active_rowblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(active_rowblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         43,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         delta_output.data_ptr<c10::BFloat16>(),
                         0,
                         0,
                         static_cast<int32_t>(active_rows.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_active_rowblock_static_persistent_sidewarp_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& active_rowblocks,
    const at::Tensor& active_rows,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  (void)delta_output;
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         nullptr,
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         active_rowblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(active_rowblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         44,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         nullptr,
                         0,
                         0,
                         static_cast<int32_t>(active_rows.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

namespace {

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_warp256_active_rowblock_static_persistent_sidewarp_cuda_impl(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& active_rowblocks,
    const at::Tensor& active_rows,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups,
    int32_t mode) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         nullptr,
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         active_rowblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(active_rowblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         mode,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         (mode == 45 || mode == 47 || mode == 49)
                             ? delta_output.data_ptr<c10::BFloat16>()
                             : nullptr,
                         0,
                         0,
                         static_cast<int32_t>(active_rows.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_warp256_active_rowblock_static_persistent_sidewarp_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& active_rowblocks,
    const at::Tensor& active_rows,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_warp256_active_rowblock_static_persistent_sidewarp_cuda_impl(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      active_rowblocks,
      active_rows,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups,
      45);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_warp256_active_rowblock_static_persistent_sidewarp_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& active_rowblocks,
    const at::Tensor& active_rows,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_warp256_active_rowblock_static_persistent_sidewarp_cuda_impl(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      active_rowblocks,
      active_rows,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups,
      46);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_prefetch_active_rowblock_static_persistent_sidewarp_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& active_rowblocks,
    const at::Tensor& active_rows,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_warp256_active_rowblock_static_persistent_sidewarp_cuda_impl(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      active_rowblocks,
      active_rows,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups,
      47);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_prefetch_active_rowblock_static_persistent_sidewarp_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& active_rowblocks,
    const at::Tensor& active_rows,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_warp256_active_rowblock_static_persistent_sidewarp_cuda_impl(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      active_rowblocks,
      active_rows,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups,
      48);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_prefetch_compact_active_rowblock_static_persistent_sidewarp_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& active_rowblocks,
    const at::Tensor& active_rows,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_warp256_active_rowblock_static_persistent_sidewarp_cuda_impl(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      active_rowblocks,
      active_rows,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups,
      49);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_persistent_sidewarp_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& active_rowblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         nullptr,
                         nullptr,
                         b_comp.data_ptr<c10::BFloat16>(),
                         active_rowblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(active_rowblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         42,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         delta_output.data_ptr<c10::BFloat16>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_wg3_ready_active_direct_add_cuda(
    const at::Tensor& output,
    const at::Tensor& ready_flags,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
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
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         ready_flags.data_ptr<int32_t>(),
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         nullptr,
                         0,
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         41,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_packed_rowblock_sidewarp_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& active_rowblocks,
    const at::Tensor& packed_tile_offsets,
    const at::Tensor& packed_row_records,
    const at::Tensor& packed_entry_records,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         b_comp.data_ptr<c10::BFloat16>(),
                         active_rowblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(active_rowblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         40,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         delta_output.data_ptr<c10::BFloat16>(),
                         0,
                         0,
                         0,
                         false,
                         0,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         1,
                         nullptr,
                         0,
                         0,
                         0,
                         packed_tile_offsets.data_ptr<int32_t>(),
                         packed_row_records.data_ptr<int64_t>(),
                         packed_entry_records.data_ptr<int32_t>(),
                         4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_kmajor_cuda(
    const at::Tensor& output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& probe_active_mblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& kmajor_group_offsets,
    const at::Tensor& kmajor_group_ks,
    const at::Tensor& kmajor_entry_offsets,
    const at::Tensor& kmajor_entry_rows,
    const at::Tensor& kmajor_entry_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         b_comp.data_ptr<c10::BFloat16>(),
                         probe_active_mblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(probe_active_mblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         1,
                         kmajor_group_offsets.data_ptr<int32_t>(),
                         kmajor_group_ks.data_ptr<int32_t>(),
                         kmajor_entry_offsets.data_ptr<int32_t>(),
                         kmajor_entry_rows.data_ptr<int32_t>(),
                         kmajor_entry_values.data_ptr<c10::BFloat16>(),
                         1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_hybrid_cuda(
    const at::Tensor& output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& probe_active_mblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& active_row_offsets,
    const at::Tensor& active_rows,
    const at::Tensor& kmajor_group_offsets,
    const at::Tensor& kmajor_group_ks,
    const at::Tensor& kmajor_entry_offsets,
    const at::Tensor& kmajor_entry_rows,
    const at::Tensor& kmajor_entry_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         probe_active_mblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(probe_active_mblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         1,
                         kmajor_group_offsets.data_ptr<int32_t>(),
                         kmajor_group_ks.data_ptr<int32_t>(),
                         kmajor_entry_offsets.data_ptr<int32_t>(),
                         kmajor_entry_rows.data_ptr<int32_t>(),
                         kmajor_entry_values.data_ptr<c10::BFloat16>(),
                         2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_incta_hybrid_cuda(
    const at::Tensor& output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& probe_active_mblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& active_row_offsets,
    const at::Tensor& active_rows,
    const at::Tensor& kmajor_group_offsets,
    const at::Tensor& kmajor_group_ks,
    const at::Tensor& kmajor_entry_offsets,
    const at::Tensor& kmajor_entry_rows,
    const at::Tensor& kmajor_entry_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         probe_active_mblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(probe_active_mblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         3,
                         kmajor_group_offsets.data_ptr<int32_t>(),
                         kmajor_group_ks.data_ptr<int32_t>(),
                         kmajor_entry_offsets.data_ptr<int32_t>(),
                         kmajor_entry_rows.data_ptr<int32_t>(),
                         kmajor_entry_values.data_ptr<c10::BFloat16>(),
                         2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_idlechunk_hybrid_cuda(
    const at::Tensor& output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& probe_active_mblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& active_row_offsets,
    const at::Tensor& active_rows,
    const at::Tensor& kmajor_group_offsets,
    const at::Tensor& kmajor_group_ks,
    const at::Tensor& kmajor_entry_offsets,
    const at::Tensor& kmajor_entry_rows,
    const at::Tensor& kmajor_entry_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t group_budget) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         probe_active_mblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(probe_active_mblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(group_budget),
                         5,
                         kmajor_group_offsets.data_ptr<int32_t>(),
                         kmajor_group_ks.data_ptr<int32_t>(),
                         kmajor_entry_offsets.data_ptr<int32_t>(),
                         kmajor_entry_rows.data_ptr<int32_t>(),
                         kmajor_entry_values.data_ptr<c10::BFloat16>(),
                         2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_scheduler_hybrid_cuda(
    const at::Tensor& output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& probe_active_mblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& active_row_offsets,
    const at::Tensor& active_rows,
    const at::Tensor& kmajor_group_offsets,
    const at::Tensor& kmajor_group_ks,
    const at::Tensor& kmajor_entry_offsets,
    const at::Tensor& kmajor_entry_rows,
    const at::Tensor& kmajor_entry_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t group_budget,
    int64_t side_warps,
    int64_t side_mode,
    const at::Tensor* phase_trace,
    int64_t phase_trace_max_ctas,
    int64_t phase_trace_stride) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  uint64_t* phase_trace_ptr = nullptr;
  if (phase_trace != nullptr && phase_trace->defined() && phase_trace->numel() > 0) {
    phase_trace_ptr = reinterpret_cast<uint64_t*>(phase_trace->data_ptr<int64_t>());
  }
			  const int32_t mixed_cta_base =
			      side_mode == 19 ? 26
			                     : (side_mode == 18 ? 25
		                     : (side_mode == 17 ? 24
	                     : (side_mode == 16 ? 23
                     : (side_mode == 15 ? 22
                     : (side_mode == 14 ? 21
                     : (side_mode == 13 ? 20
                     : (side_mode == 12 ? 19
                     : (side_mode == 11 ? 18
                     : (side_mode == 10 ? 17
                     : (side_mode == 9 ? 16
                     : (side_mode == 8 ? 15
                     : (side_mode == 7 ? 14
                     : (side_mode == 6 ? 13
                     : (side_mode == 5 ? 12
                     : (side_mode == 4 ? 11
                     : (side_mode == 3 ? 10
		                                     : (side_mode == 2 ? 9
			                                                       : (side_mode == 1 ? 8 : (side_warps > 1 ? 7 : 6)))))))))))))))))));
  const int32_t mixed_cta =
      side_mode == 30 ? 37
                      : (side_mode == 29 ? 36
                      : (side_mode == 28 ? 35
                      : (side_mode == 27 ? 34
                      : (side_mode == 26 ? 33
                      : (side_mode == 25 ? 32
                      : (side_mode == 24 ? 31
                                         : (side_mode == 23 ? 30
                                         : (side_mode == 22 ? 29
                                                            : (side_mode == 21 ? 28
                                                                               : (side_mode == 20 ? 27 : mixed_cta_base))))))))));
  const int32_t direct_smem_add = mixed_cta == 27 ? 13 : 0;
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         probe_active_mblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(probe_active_mblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(group_budget),
                         mixed_cta,
                         kmajor_group_offsets.data_ptr<int32_t>(),
                         kmajor_group_ks.data_ptr<int32_t>(),
                         kmajor_entry_offsets.data_ptr<int32_t>(),
                         kmajor_entry_rows.data_ptr<int32_t>(),
                         kmajor_entry_values.data_ptr<c10::BFloat16>(),
	                         2,
	                         nullptr,
	                         0,
	                         static_cast<int32_t>(side_warps),
	                         0,
	                         false,
	                         direct_smem_add,
	                         nullptr,
	                         nullptr,
	                         nullptr,
	                         nullptr,
	                         1,
	                         phase_trace_ptr,
	                         static_cast<int32_t>(phase_trace_stride),
	                         static_cast<int32_t>(phase_trace_max_ctas),
	                         1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& probe_active_mblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& active_row_offsets,
    const at::Tensor& active_rows,
    const at::Tensor& kmajor_group_offsets,
    const at::Tensor& kmajor_group_ks,
    const at::Tensor& kmajor_entry_offsets,
    const at::Tensor& kmajor_entry_rows,
    const at::Tensor& kmajor_entry_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
	    int64_t group_budget,
	    int64_t direct_delta_write_mode,
	    int64_t side_warps,
	    int64_t side_mode,
	    int64_t direct_delta_chunk_limit,
	    const at::Tensor* phase_trace,
	    int64_t phase_trace_max_ctas,
	    int64_t phase_trace_stride,
	    const at::Tensor* packed_tile_offsets = nullptr,
	    const at::Tensor* packed_row_records = nullptr,
	    const at::Tensor* packed_entry_records = nullptr,
	    int64_t packed_payload_mode = 0,
	    const at::Tensor* kmajor_tile_group_starts = nullptr,
	    const at::Tensor* kmajor_tile_group_counts = nullptr,
	    const at::Tensor* kmajor_tile_group_meta = nullptr) {
	  const c10::cuda::CUDAGuard device_guard(output.device());
	  uint64_t* phase_trace_ptr = nullptr;
	  if (phase_trace != nullptr && phase_trace->defined() && phase_trace->numel() > 0) {
	    phase_trace_ptr = reinterpret_cast<uint64_t*>(phase_trace->data_ptr<int64_t>());
	  }
	  const int32_t* packed_tile_offsets_ptr =
	      packed_tile_offsets != nullptr && packed_tile_offsets->defined() &&
	              packed_tile_offsets->numel() > 0
	          ? packed_tile_offsets->data_ptr<int32_t>()
	          : nullptr;
	  const int64_t* packed_row_records_ptr =
	      packed_row_records != nullptr && packed_row_records->defined() &&
	              packed_row_records->numel() > 0
	          ? packed_row_records->data_ptr<int64_t>()
	          : nullptr;
	  const int32_t* packed_entry_records_ptr =
	      packed_entry_records != nullptr && packed_entry_records->defined() &&
	              packed_entry_records->numel() > 0
	          ? packed_entry_records->data_ptr<int32_t>()
	          : nullptr;
	  const int32_t* kmajor_tile_group_starts_ptr =
	      kmajor_tile_group_starts != nullptr && kmajor_tile_group_starts->defined() &&
	              kmajor_tile_group_starts->numel() > 0
	          ? kmajor_tile_group_starts->data_ptr<int32_t>()
	          : nullptr;
	  const int32_t* kmajor_tile_group_counts_ptr =
	      kmajor_tile_group_counts != nullptr && kmajor_tile_group_counts->defined() &&
	              kmajor_tile_group_counts->numel() > 0
	          ? kmajor_tile_group_counts->data_ptr<int32_t>()
	          : nullptr;
	  const int64_t* kmajor_tile_group_meta_ptr =
	      kmajor_tile_group_meta != nullptr && kmajor_tile_group_meta->defined() &&
	              kmajor_tile_group_meta->numel() > 0
	          ? kmajor_tile_group_meta->data_ptr<int64_t>()
	          : nullptr;
				  const int32_t mixed_cta_base =
			      side_mode == 19 ? 26
			                     : (side_mode == 18 ? 25
		                     : (side_mode == 17 ? 24
	                     : (side_mode == 16 ? 23
                     : (side_mode == 15 ? 22
                     : (side_mode == 14 ? 21
                     : (side_mode == 13 ? 20
                     : (side_mode == 12 ? 19
                     : (side_mode == 11 ? 18
                     : (side_mode == 10 ? 17
                     : (side_mode == 9 ? 16
                     : (side_mode == 8 ? 15
                     : (side_mode == 7 ? 14
                     : (side_mode == 6 ? 13
                     : (side_mode == 5 ? 12
                     : (side_mode == 4 ? 11
                     : (side_mode == 3 ? 10
		                                     : (side_mode == 2 ? 9
			                                                       : (side_mode == 1 ? 8 : (side_warps > 1 ? 7 : 6)))))))))))))))))));
  const int32_t mixed_cta =
      side_mode == 30 ? 37
                      : (side_mode == 29 ? 36
                      : (side_mode == 28 ? 35
                      : (side_mode == 27 ? 34
                      : (side_mode == 26 ? 33
                      : (side_mode == 25 ? 32
                      : (side_mode == 24 ? 31
                                         : (side_mode == 23 ? 30
                                         : (side_mode == 22 ? 29
                                                            : (side_mode == 21 ? 28
                                                                               : (side_mode == 20 ? 27 : mixed_cta_base))))))))));
  const int32_t direct_smem_add = mixed_cta == 27 ? 13 : 0;
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         probe_active_mblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(probe_active_mblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(group_budget),
                         mixed_cta,
                         kmajor_group_offsets.data_ptr<int32_t>(),
                         kmajor_group_ks.data_ptr<int32_t>(),
                         kmajor_entry_offsets.data_ptr<int32_t>(),
                         kmajor_entry_rows.data_ptr<int32_t>(),
                         kmajor_entry_values.data_ptr<c10::BFloat16>(),
                         2,
	                         delta_output.data_ptr<c10::BFloat16>(),
	                         static_cast<int32_t>(direct_delta_write_mode),
		                         static_cast<int32_t>(side_warps),
		                         static_cast<int32_t>(direct_delta_chunk_limit),
		                         false,
		                         direct_smem_add,
		                         nullptr,
		                         nullptr,
		                         nullptr,
		                         nullptr,
		                         1,
		                         phase_trace_ptr,
		                         static_cast<int32_t>(phase_trace_stride),
		                         static_cast<int32_t>(phase_trace_max_ctas),
		                         1,
		                         packed_tile_offsets_ptr,
		                         packed_row_records_ptr,
		                         packed_entry_records_ptr,
		                         static_cast<int32_t>(packed_payload_mode),
		                         kmajor_tile_group_starts_ptr,
		                         kmajor_tile_group_counts_ptr,
		                         kmajor_tile_group_meta_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_atomic_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& probe_active_mblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& active_row_offsets,
    const at::Tensor& active_rows,
    const at::Tensor& kmajor_group_offsets,
    const at::Tensor& kmajor_group_ks,
    const at::Tensor& kmajor_entry_offsets,
    const at::Tensor& kmajor_entry_rows,
    const at::Tensor& kmajor_entry_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups,
    int64_t direct_delta_write_mode) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         probe_active_mblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(probe_active_mblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         3,
                         kmajor_group_offsets.data_ptr<int32_t>(),
                         kmajor_group_ks.data_ptr<int32_t>(),
                         kmajor_entry_offsets.data_ptr<int32_t>(),
                         kmajor_entry_rows.data_ptr<int32_t>(),
                         kmajor_entry_values.data_ptr<c10::BFloat16>(),
                         2,
                         delta_output.data_ptr<c10::BFloat16>(),
                         static_cast<int32_t>(direct_delta_write_mode));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_extrawg_kmajor_sharedacc_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& probe_sink,
    const at::Tensor& probe_counter,
    const at::Tensor& probe_active_mblocks,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& row_offsets,
    const at::Tensor& row_ks,
    const at::Tensor& row_values,
    const at::Tensor& active_row_offsets,
    const at::Tensor& active_rows,
    const at::Tensor& kmajor_group_offsets,
    const at::Tensor& kmajor_group_ks,
    const at::Tensor& kmajor_entry_offsets,
    const at::Tensor& kmajor_entry_rows,
    const at::Tensor& kmajor_entry_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_warpgroups) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         nullptr,
                         nullptr,
                         nullptr,
                         row_offsets.data_ptr<int32_t>(),
                         row_ks.data_ptr<int32_t>(),
                         row_values.data_ptr<c10::BFloat16>(),
                         active_row_offsets.data_ptr<int32_t>(),
                         active_rows.data_ptr<int32_t>(),
                         b_comp.data_ptr<c10::BFloat16>(),
                         probe_active_mblocks.data_ptr<int32_t>(),
                         static_cast<int32_t>(probe_active_mblocks.numel()),
                         probe_sink.data_ptr<float>(),
                         probe_counter.data_ptr<int32_t>(),
                         static_cast<int32_t>(sparse_warpgroups),
                         0,
                         kmajor_group_offsets.data_ptr<int32_t>(),
                         kmajor_group_ks.data_ptr<int32_t>(),
                         kmajor_entry_offsets.data_ptr<int32_t>(),
                         kmajor_entry_rows.data_ptr<int32_t>(),
                         kmajor_entry_values.data_ptr<c10::BFloat16>(),
                         2,
                         delta_output.data_ptr<c10::BFloat16>(),
                         3);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor merge_entry_delta_active_rows_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& merge_row_offsets,
    const at::Tensor& merge_entry_indices,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  const int64_t active_row_count = active_rows.numel();
  if (active_row_count == 0) {
    return output;
  }
  constexpr int threads = 256;
  int64_t blocks = active_row_count;
  blocks = blocks > 65535 ? 65535 : blocks;
  fp4::merge_entry_delta_active_rows_rowblock_vec16_kernel<<<static_cast<int>(blocks),
                                                              threads,
                                                              0,
                                                              at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      delta_entries.data_ptr<c10::BFloat16>(),
      active_rows.data_ptr<int32_t>(),
      merge_row_offsets.data_ptr<int32_t>(),
      merge_entry_indices.data_ptr<int32_t>(),
      active_row_count,
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor merge_entry_delta_active_rows_fastpath_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& merge_row_offsets,
    const at::Tensor& merge_entry_indices,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  const int64_t active_row_count = active_rows.numel();
  if (active_row_count == 0) {
    return output;
  }
  constexpr int threads = 256;
  int64_t blocks = active_row_count;
  blocks = blocks > 65535 ? 65535 : blocks;
  fp4::merge_entry_delta_active_rows_rowblock_vec16_fastpath_kernel<<<static_cast<int>(blocks),
                                                                       threads,
                                                                       0,
                                                                       at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      delta_entries.data_ptr<c10::BFloat16>(),
      active_rows.data_ptr<int32_t>(),
      merge_row_offsets.data_ptr<int32_t>(),
      merge_entry_indices.data_ptr<int32_t>(),
      active_row_count,
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor merge_entry_delta_active_rows_vec8_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& merge_row_offsets,
    const at::Tensor& merge_entry_indices,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  const int64_t active_row_count = active_rows.numel();
  if (active_row_count == 0) {
    return output;
  }
  constexpr int threads = 256;
  constexpr int VecN = 8;
  const int64_t groups_per_row = (n + VecN - 1) / VecN;
  int64_t blocks = (active_row_count * groups_per_row + threads - 1) / threads;
  blocks = blocks > 65535 ? 65535 : blocks;
  fp4::merge_entry_delta_active_rows_vec8_kernel<<<static_cast<int>(blocks),
                                                    threads,
                                                    0,
                                                    at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      delta_entries.data_ptr<c10::BFloat16>(),
      active_rows.data_ptr<int32_t>(),
      merge_row_offsets.data_ptr<int32_t>(),
      merge_entry_indices.data_ptr<int32_t>(),
      active_row_count,
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor merge_entry_delta_active_rows_chunk_prefix_vec8_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& merge_row_offsets,
    const at::Tensor& merge_entry_indices,
    int64_t n,
    int64_t chunk_cols,
    int64_t chunks_per_row) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  const int64_t active_row_count = active_rows.numel();
  if (active_row_count == 0 || chunks_per_row <= 0) {
    return output;
  }
  TORCH_CHECK(chunk_cols > 0, "chunk_cols must be positive");
  TORCH_CHECK(chunk_cols % 8 == 0, "chunk_cols must be a multiple of 8");
  constexpr int threads = 256;
  constexpr int VecN = 8;
  const int64_t total_chunks_per_row = (n + chunk_cols - 1) / chunk_cols;
  const int64_t active_chunks_per_row =
      chunks_per_row < total_chunks_per_row ? chunks_per_row : total_chunks_per_row;
  const int64_t groups_per_chunk = (chunk_cols + VecN - 1) / VecN;
  int64_t blocks =
      (active_row_count * active_chunks_per_row * groups_per_chunk + threads - 1) / threads;
  blocks = blocks > 65535 ? 65535 : blocks;
  fp4::merge_entry_delta_active_rows_chunk_prefix_vec8_kernel<<<static_cast<int>(blocks),
                                                                 threads,
                                                                 0,
                                                                 at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      delta_entries.data_ptr<c10::BFloat16>(),
      active_rows.data_ptr<int32_t>(),
      merge_row_offsets.data_ptr<int32_t>(),
      merge_entry_indices.data_ptr<int32_t>(),
      active_row_count,
      n,
      chunk_cols,
      chunks_per_row);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor merge_single_entry_delta_active_rows_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& entry_indices,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  const int64_t active_row_count = active_rows.numel();
  if (active_row_count == 0) {
    return output;
  }
  int64_t blocks = active_row_count;
  blocks = blocks > 65535 ? 65535 : blocks;
  constexpr int threads = 256;
  fp4::merge_single_entry_delta_active_rows_rowblock_vec16_kernel<<<static_cast<int>(blocks),
                                                                    threads,
                                                                    0,
                                                                    at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      delta_entries.data_ptr<c10::BFloat16>(),
      active_rows.data_ptr<int32_t>(),
      entry_indices.data_ptr<int32_t>(),
      active_row_count,
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor merge_double_entry_delta_active_rows_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& entry0_indices,
    const at::Tensor& entry1_indices,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  const int64_t active_row_count = active_rows.numel();
  if (active_row_count == 0) {
    return output;
  }
  int64_t blocks = active_row_count;
  blocks = blocks > 65535 ? 65535 : blocks;
  constexpr int threads = 256;
  fp4::merge_double_entry_delta_active_rows_rowblock_vec16_kernel<<<static_cast<int>(blocks),
                                                                    threads,
                                                                    0,
                                                                    at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      delta_entries.data_ptr<c10::BFloat16>(),
      active_rows.data_ptr<int32_t>(),
      entry0_indices.data_ptr<int32_t>(),
      entry1_indices.data_ptr<int32_t>(),
      active_row_count,
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_tile_scales_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    int64_t m,
    int64_t k,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (flat_indices.numel() == 0) {
    return output;
  }

  constexpr int threads = 256;
  const int64_t total_groups = m * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_value_payload_vec8_inplace_kernel<<<static_cast<int>(blocks),
                                             threads,
                                             0,
                                             at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      m,
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_tile_scales_alloc_cuda(
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    int64_t m,
    int64_t k,
    int64_t n) {
  auto output = at::empty({m, n}, a_data.options().dtype(at::kBFloat16));
  return nvfp4_dense_sparse_tma_value_payload_tile_scales_cuda(output,
                                                              a_data,
                                                              a_scale_tile,
                                                              b_data,
                                                              b_scale_tile,
                                                              a_amax,
                                                              b_amax,
                                                              outlier_values,
                                                              weight_t_bf16,
                                                              flat_indices,
                                                              row_offsets,
                                                              m,
                                                              k,
                                                              n);
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t m,
    int64_t k,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (flat_indices.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }

  constexpr int threads = 256;
  const int64_t total_groups = active_rows.numel() * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_value_payload_vec8_inplace_kernel<<<static_cast<int>(blocks),
                                                        threads,
                                                        0,
                                                        at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_active_row_value_payload_vec8_store_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t total_groups = active_rows.numel() * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_value_payload_vec8_store_kernel<<<static_cast<int>(blocks),
                                                      threads,
                                                      0,
                                                      at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_active_row_value_payload_vec8_store_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t total_groups = active_rows.numel() * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_value_payload_vec8_store_vstore_kernel<<<static_cast<int>(blocks),
                                                             threads,
                                                             0,
                                                             at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor merge_full_delta_active_rows_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& active_rows) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  const int64_t active_row_count = active_rows.numel();
  if (active_row_count == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t total_groups = active_row_count * (n / 8);
  int64_t blocks = (total_groups + threads - 1) / threads;
  blocks = blocks > 65535 ? 65535 : blocks;
  merge_full_delta_active_rows_vec8_kernel<<<static_cast<int>(blocks),
                                             threads,
                                             0,
                                             at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      delta_output.data_ptr<c10::BFloat16>(),
      active_rows.data_ptr<int32_t>(),
      active_row_count,
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor merge_compact_delta_active_rows_cuda(
    const at::Tensor& output,
    const at::Tensor& compact_delta,
    const at::Tensor& active_rows) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  const int64_t active_row_count = active_rows.numel();
  if (active_row_count == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  if (n == 4096) {
    merge_compact_delta_active_rows_n4096_vec16_kernel<<<
        static_cast<int>(active_row_count),
        threads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        output.data_ptr<c10::BFloat16>(),
        compact_delta.data_ptr<c10::BFloat16>(),
        active_rows.data_ptr<int32_t>(),
        active_row_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }
  const int64_t total_groups = active_row_count * (n / 8);
  int64_t blocks = (total_groups + threads - 1) / threads;
  blocks = blocks > 65535 ? 65535 : blocks;
  merge_compact_delta_active_rows_vec8_kernel<<<static_cast<int>(blocks),
                                                threads,
                                                0,
                                                at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      compact_delta.data_ptr<c10::BFloat16>(),
      active_rows.data_ptr<int32_t>(),
      active_row_count,
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor merge_two_compact_delta_active_rows_cuda(
    const at::Tensor& output,
    const at::Tensor& first_delta,
    const at::Tensor& first_rows,
    const at::Tensor& second_delta,
    const at::Tensor& second_rows) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  const int64_t first_row_count = first_rows.numel();
  const int64_t second_row_count = second_rows.numel();
  if (first_row_count == 0) {
    return merge_compact_delta_active_rows_cuda(output, second_delta, second_rows);
  }
  if (second_row_count == 0) {
    return merge_compact_delta_active_rows_cuda(output, first_delta, first_rows);
  }
  if (output.size(1) != 4096) {
    merge_compact_delta_active_rows_cuda(output, first_delta, first_rows);
    return merge_compact_delta_active_rows_cuda(output, second_delta, second_rows);
  }
  constexpr int threads = 256;
  const int64_t combined_row_count = first_row_count + second_row_count;
  merge_two_compact_delta_active_rows_n4096_vec16_kernel<<<
      static_cast<int>(combined_row_count),
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      first_delta.data_ptr<c10::BFloat16>(),
      first_rows.data_ptr<int32_t>(),
      first_row_count,
      second_delta.data_ptr<c10::BFloat16>(),
      second_rows.data_ptr<int32_t>(),
      second_row_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor build_compact_dense_residual_active_rows_cuda(
    const at::Tensor& residual,
    const at::Tensor& row_values,
    const at::Tensor& row_ks,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  const c10::cuda::CUDAGuard device_guard(residual.device());
  const int64_t active_row_count = active_rows.numel();
  if (active_row_count == 0) {
    return residual;
  }
  constexpr int threads = 256;
  const int64_t total_groups = active_row_count * (k / 8);
  int64_t zero_blocks = (total_groups + threads - 1) / threads;
  zero_blocks = zero_blocks > 65535 ? 65535 : zero_blocks;
  zero_compact_dense_residual_vec8_kernel<<<static_cast<int>(zero_blocks),
                                            threads,
                                            0,
                                            at::cuda::getCurrentCUDAStream()>>>(
      residual.data_ptr<c10::BFloat16>(),
      active_row_count,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  scatter_compact_dense_residual_rows_kernel<<<static_cast<int>(active_row_count),
                                               threads,
                                               0,
                                               at::cuda::getCurrentCUDAStream()>>>(
      residual.data_ptr<c10::BFloat16>(),
      row_values.data_ptr<c10::BFloat16>(),
      row_ks.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_row_count,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return residual;
}

std::vector<at::Tensor> build_padded_light_heavy_rows_cuda(
    const at::Tensor& row_offsets,
    int64_t heavy_threshold,
    int64_t heavy_capacity) {
  const c10::cuda::CUDAGuard device_guard(row_offsets.device());
  const int64_t rows = row_offsets.numel() - 1;
  auto light_rows = at::empty({rows}, row_offsets.options());
  auto heavy_rows = at::empty({heavy_capacity}, row_offsets.options());
  auto light_row_count = at::empty({1}, row_offsets.options());
  auto heavy_row_count = at::empty({1}, row_offsets.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (rows > 0) {
    C10_CUDA_CHECK(cudaMemsetAsync(light_rows.data_ptr<int32_t>(),
                                   0xff,
                                   static_cast<size_t>(rows) * sizeof(int32_t),
                                   stream.stream()));
  }
  C10_CUDA_CHECK(cudaMemsetAsync(heavy_rows.data_ptr<int32_t>(),
                                 0xff,
                                 static_cast<size_t>(heavy_capacity) * sizeof(int32_t),
                                 stream.stream()));
  C10_CUDA_CHECK(cudaMemsetAsync(
      light_row_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream.stream()));
  C10_CUDA_CHECK(cudaMemsetAsync(
      heavy_row_count.data_ptr<int32_t>(), 0, sizeof(int32_t), stream.stream()));
  if (rows > 0) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    build_padded_light_heavy_rows_kernel<<<blocks, threads, 0, stream>>>(
        row_offsets.data_ptr<int32_t>(),
        light_rows.data_ptr<int32_t>(),
        heavy_rows.data_ptr<int32_t>(),
        light_row_count.data_ptr<int32_t>(),
        heavy_row_count.data_ptr<int32_t>(),
        rows,
        static_cast<int32_t>(heavy_threshold),
        static_cast<int32_t>(heavy_capacity));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {light_rows, heavy_rows, light_row_count, heavy_row_count};
}

at::Tensor sparse_kmajor_epin_delta_store_cuda(
    const at::Tensor& output,
    const at::Tensor& active_mblocks,
    const at::Tensor& active_row_offsets,
    const at::Tensor& active_rows,
    const at::Tensor& group_offsets,
    const at::Tensor& group_ks,
    const at::Tensor& entry_offsets,
    const at::Tensor& entry_rows,
    const at::Tensor& entry_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t epin) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (active_mblocks.numel() == 0 || entry_values.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const dim3 grid(static_cast<unsigned int>((n + epin - 1) / epin),
                  static_cast<unsigned int>(active_mblocks.numel()),
                  1);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (epin == 32) {
    sparse_kmajor_epin_delta_store_kernel<32><<<grid, threads, 0, stream>>>(
        output.data_ptr<c10::BFloat16>(),
        active_mblocks.data_ptr<int32_t>(),
        active_row_offsets.data_ptr<int32_t>(),
        active_rows.data_ptr<int32_t>(),
        group_offsets.data_ptr<int32_t>(),
        group_ks.data_ptr<int32_t>(),
        entry_offsets.data_ptr<int32_t>(),
        entry_rows.data_ptr<int32_t>(),
        entry_values.data_ptr<c10::BFloat16>(),
        b_comp.data_ptr<c10::BFloat16>(),
        m,
        k,
        n);
  } else if (epin == 64) {
    sparse_kmajor_epin_delta_store_kernel<64><<<grid, threads, 0, stream>>>(
        output.data_ptr<c10::BFloat16>(),
        active_mblocks.data_ptr<int32_t>(),
        active_row_offsets.data_ptr<int32_t>(),
        active_rows.data_ptr<int32_t>(),
        group_offsets.data_ptr<int32_t>(),
        group_ks.data_ptr<int32_t>(),
        entry_offsets.data_ptr<int32_t>(),
        entry_rows.data_ptr<int32_t>(),
        entry_values.data_ptr<c10::BFloat16>(),
        b_comp.data_ptr<c10::BFloat16>(),
        m,
        k,
        n);
  } else {
    TORCH_CHECK(false, "epin must be one of 32, 64");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_kmajor_epin64_delta_store_cuda(
    const at::Tensor& output,
    const at::Tensor& active_mblocks,
    const at::Tensor& active_row_offsets,
    const at::Tensor& active_rows,
    const at::Tensor& group_offsets,
    const at::Tensor& group_ks,
    const at::Tensor& entry_offsets,
    const at::Tensor& entry_rows,
    const at::Tensor& entry_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n) {
  return sparse_kmajor_epin_delta_store_cuda(output,
                                            active_mblocks,
                                            active_row_offsets,
                                            active_rows,
                                            group_offsets,
                                            group_ks,
                                            entry_offsets,
                                            entry_rows,
                                            entry_values,
                                            b_comp,
                                            m,
                                            k,
                                            n,
                                            64);
}

at::Tensor sparse_kmajor_serial_group_inplace_cuda(
    const at::Tensor& output,
    const at::Tensor& active_mblocks,
    const at::Tensor& group_offsets,
    const at::Tensor& group_ks,
    const at::Tensor& entry_offsets,
    const at::Tensor& entry_rows,
    const at::Tensor& entry_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t bm) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (active_mblocks.numel() == 0 || entry_values.numel() == 0) {
    return output;
  }
  constexpr int threads = 64;
  constexpr int epin = 64;
  const dim3 grid(static_cast<unsigned int>((n + epin - 1) / epin),
                  static_cast<unsigned int>(active_mblocks.numel()),
                  1);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (bm == 32) {
    sparse_kmajor_serial_group_inplace_kernel<32><<<grid, threads, 0, stream>>>(
        output.data_ptr<c10::BFloat16>(),
        active_mblocks.data_ptr<int32_t>(),
        group_offsets.data_ptr<int32_t>(),
        group_ks.data_ptr<int32_t>(),
        entry_offsets.data_ptr<int32_t>(),
        entry_rows.data_ptr<int32_t>(),
        entry_values.data_ptr<c10::BFloat16>(),
        b_comp.data_ptr<c10::BFloat16>(),
        m,
        k,
        n);
  } else if (bm == 64) {
    sparse_kmajor_serial_group_inplace_kernel<64><<<grid, threads, 0, stream>>>(
        output.data_ptr<c10::BFloat16>(),
        active_mblocks.data_ptr<int32_t>(),
        group_offsets.data_ptr<int32_t>(),
        group_ks.data_ptr<int32_t>(),
        entry_offsets.data_ptr<int32_t>(),
        entry_rows.data_ptr<int32_t>(),
        entry_values.data_ptr<c10::BFloat16>(),
        b_comp.data_ptr<c10::BFloat16>(),
        m,
        k,
        n);
  } else if (bm == 128) {
    sparse_kmajor_serial_group_inplace_kernel<128><<<grid, threads, 0, stream>>>(
        output.data_ptr<c10::BFloat16>(),
        active_mblocks.data_ptr<int32_t>(),
        group_offsets.data_ptr<int32_t>(),
        group_ks.data_ptr<int32_t>(),
        entry_offsets.data_ptr<int32_t>(),
        entry_rows.data_ptr<int32_t>(),
        entry_values.data_ptr<c10::BFloat16>(),
        b_comp.data_ptr<c10::BFloat16>(),
        m,
        k,
        n);
  } else {
    TORCH_CHECK(false, "bm must be one of 32, 64, 128");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_kmajor_epin64_direct_store_cuda(
    const at::Tensor& output,
    const at::Tensor& active_mblocks,
    const at::Tensor& group_offsets,
    const at::Tensor& group_ks,
    const at::Tensor& entry_offsets,
    const at::Tensor& entry_rows,
    const at::Tensor& entry_values,
    const at::Tensor& b_comp,
    int64_t m,
    int64_t k,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (active_mblocks.numel() == 0 || entry_values.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  constexpr int epin = 64;
  const dim3 grid(static_cast<unsigned int>((n + epin - 1) / epin),
                  static_cast<unsigned int>(active_mblocks.numel()),
                  1);
  sparse_kmajor_epin64_direct_store_kernel<<<grid,
                                              threads,
                                              0,
                                              at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      active_mblocks.data_ptr<int32_t>(),
      group_offsets.data_ptr<int32_t>(),
      group_ks.data_ptr<int32_t>(),
      entry_offsets.data_ptr<int32_t>(),
      entry_rows.data_ptr<int32_t>(),
      entry_values.data_ptr<c10::BFloat16>(),
      b_comp.data_ptr<c10::BFloat16>(),
      m,
      k,
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t total_groups = active_rows.numel() * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_value_payload_vec8_inplace_kernel<<<static_cast<int>(blocks),
                                                        threads,
                                                        0,
                                                        at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t total_groups = active_rows.numel() * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_value_payload_vec8_inplace_vstore_kernel<<<static_cast<int>(blocks),
                                                               threads,
                                                               0,
                                                               at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k,
      0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_skip_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k,
    int64_t skip_per_row) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t total_groups = active_rows.numel() * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_value_payload_vec8_inplace_vstore_kernel<<<static_cast<int>(blocks),
                                                               threads,
                                                               0,
                                                               at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k,
      static_cast<int32_t>(skip_per_row));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_packed_suffix12_vec8_inplace_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& packed_suffix_records,
    const at::Tensor& active_rows,
    const at::Tensor& weight_t_bf16) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  const int64_t active_row_count = active_rows.numel();
  if (active_row_count == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t total_groups = active_row_count * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_packed_suffix12_vec8_inplace_vstore_kernel<<<
      static_cast<int>(blocks),
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      packed_suffix_records.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      active_row_count,
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_strict_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t total_groups = active_rows.numel() * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_value_payload_vec8_inplace_strict_vstore_kernel<<<static_cast<int>(blocks),
                                                                      threads,
                                                                      0,
                                                                      at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_sum_then_add_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t total_groups = active_rows.numel() * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_value_payload_vec8_inplace_sum_then_add_vstore_kernel<<<static_cast<int>(blocks),
                                                                            threads,
                                                                            0,
                                                                            at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_b_evict_last_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t total_groups = active_rows.numel() * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_value_payload_vec8_inplace_b_evict_last_vstore_kernel<<<static_cast<int>(blocks),
                                                                            threads,
                                                                            0,
                                                                            at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_fastpath_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t total_groups = active_rows.numel() * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_value_payload_vec8_inplace_fastpath_kernel<<<static_cast<int>(blocks),
                                                                 threads,
                                                                 0,
                                                                 at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_rowblock_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t groups_per_row = n / 8;
  const int64_t blocks_per_row = (groups_per_row + threads - 1) / threads;
  const int64_t blocks = active_rows.numel() * blocks_per_row;
  sparse_active_row_value_payload_vec8_inplace_rowblock_kernel<<<static_cast<int>(blocks),
                                                                 threads,
                                                                 0,
                                                                 at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k,
      blocks_per_row);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_active_row_col_value_payload_vec16_inplace_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (outlier_values.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t total_groups = active_rows.numel() * (n / 16);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_col_value_payload_vec16_inplace_kernel<<<static_cast<int>(blocks),
                                                             threads,
                                                             0,
                                                             at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      outlier_cols.data_ptr<int16_t>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<at::Tensor> split_hot_dense_padded_cold_rows_cuda(
    const at::Tensor& hot_dense,
    const at::Tensor& cold_values,
    const at::Tensor& cold_cols,
    const at::Tensor& cold_counts,
    const at::Tensor& overflow,
    const at::Tensor& row_values,
    const at::Tensor& row_cols,
    const at::Tensor& row_offsets,
    const at::Tensor& hot_lut,
    int64_t rows,
    int64_t k,
    int64_t hot_cols,
    int64_t cold_capacity) {
  const c10::cuda::CUDAGuard device_guard(hot_dense.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      hot_dense.data_ptr<c10::BFloat16>(),
      0,
      hot_dense.numel() * sizeof(c10::BFloat16),
      stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      overflow.data_ptr<int32_t>(), 0, sizeof(int32_t), stream));
  constexpr int threads = 256;
  const int64_t blocks = (rows + threads - 1) / threads;
  split_hot_dense_padded_cold_rows_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      hot_dense.data_ptr<c10::BFloat16>(),
      cold_values.data_ptr<c10::BFloat16>(),
      cold_cols.data_ptr<int16_t>(),
      cold_counts.data_ptr<int32_t>(),
      overflow.data_ptr<int32_t>(),
      row_values.data_ptr<c10::BFloat16>(),
      row_cols.data_ptr<int16_t>(),
      row_offsets.data_ptr<int32_t>(),
      hot_lut.data_ptr<int16_t>(),
      rows,
      k,
      hot_cols,
      cold_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {hot_dense, cold_values, cold_cols, cold_counts, overflow};
}

at::Tensor sparse_padded_cold_col_vec16_inplace_cuda(
    const at::Tensor& output,
    const at::Tensor& cold_values,
    const at::Tensor& cold_cols,
    const at::Tensor& cold_counts,
    const at::Tensor& row_values,
    const at::Tensor& row_cols,
    const at::Tensor& row_offsets,
    const at::Tensor& hot_lut,
    const at::Tensor& weight_t_bf16,
    int64_t rows,
    int64_t k,
    int64_t cols,
    int64_t cold_capacity) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (rows == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t col_tiles = (cols + threads * 16 - 1) / (threads * 16);
  sparse_padded_cold_col_vec16_inplace_kernel<<<dim3(static_cast<uint32_t>(rows),
                                                      static_cast<uint32_t>(col_tiles)),
                                                threads,
                                                0,
                                                at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      cold_values.data_ptr<c10::BFloat16>(),
      cold_cols.data_ptr<int16_t>(),
      cold_counts.data_ptr<int32_t>(),
      row_values.data_ptr<c10::BFloat16>(),
      row_cols.data_ptr<int16_t>(),
      row_offsets.data_ptr<int32_t>(),
      hot_lut.data_ptr<int16_t>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      rows,
      k,
      cols,
      cold_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_active_row_col_value_payload_vec8_inplace_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  (void)k;
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (outlier_values.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t total_groups = active_rows.numel() * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_col_value_payload_vec8_inplace_vstore_kernel<<<static_cast<int>(blocks),
                                                                   threads,
                                                                   0,
                                                                   at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      outlier_cols.data_ptr<int16_t>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor sparse_active_row_col_value_payload_vec8_shmem_sum_then_add_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  (void)k;
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (outlier_values.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const int64_t n = output.size(1);
  const int64_t groups_per_row = n / 8;
  const int64_t blocks_per_row = (groups_per_row + threads - 1) / threads;
  const int64_t blocks = active_rows.numel() * blocks_per_row;
  sparse_active_row_col_value_payload_vec8_shmem_kernel<true><<<
      static_cast<int>(blocks),
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      outlier_cols.data_ptr<int16_t>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      blocks_per_row);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t m,
    int64_t k,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (outlier_values.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }

  constexpr int threads = 256;
  const int64_t total_groups = active_rows.numel() * (n / 8);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_col_value_payload_vec8_inplace_kernel<<<static_cast<int>(blocks),
                                                            threads,
                                                            0,
                                                            at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      outlier_cols.data_ptr<int16_t>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols_vec16_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t m,
    int64_t k,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (outlier_values.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }

  constexpr int threads = 256;
  const int64_t total_groups = active_rows.numel() * (n / 16);
  const int64_t blocks = (total_groups + threads - 1) / threads;
  sparse_active_row_col_value_payload_vec16_inplace_kernel<<<static_cast<int>(blocks),
                                                             threads,
                                                             0,
                                                             at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      outlier_cols.data_ptr<int16_t>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols_shmem_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t m,
    int64_t k,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (outlier_values.numel() == 0 || active_rows.numel() == 0) {
    return output;
  }

  constexpr int threads = 256;
  const int64_t groups_per_row = n / 8;
  const int64_t blocks_per_row = (groups_per_row + threads - 1) / threads;
  const int64_t blocks = active_rows.numel() * blocks_per_row;
  sparse_active_row_col_value_payload_vec8_shmem_kernel<false><<<
      static_cast<int>(blocks),
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      outlier_cols.data_ptr<int16_t>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      blocks_per_row);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    int64_t m,
    int64_t k,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (outlier_values.numel() == 0) {
    return output;
  }

  constexpr int threads = 256;
  const int64_t tiles_m = (m + 127) / 128;
  const int64_t tiles_n = (n + 127) / 128;
  const int64_t total_tiles = tiles_m * tiles_n;
  sparse_tile_col_value_payload_vec8_kernel<<<static_cast<int>(total_tiles),
                                              threads,
                                              0,
                                              at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      outlier_cols.data_ptr<int16_t>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      row_offsets.data_ptr<int32_t>(),
      m,
      n,
      tiles_n,
      total_tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec16_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    int64_t m,
    int64_t k,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (outlier_values.numel() == 0) {
    return output;
  }

  constexpr int threads = 256;
  const int64_t tiles_m = (m + 127) / 128;
  const int64_t tiles_n = (n + 127) / 128;
  const int64_t total_tiles = tiles_m * tiles_n;
  sparse_tile_col_value_payload_vec16_kernel<<<static_cast<int>(total_tiles),
                                               threads,
                                               0,
                                               at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      outlier_cols.data_ptr<int16_t>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      row_offsets.data_ptr<int32_t>(),
      m,
      n,
      tiles_n,
      total_tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_tile_scales_persistent_cols_vec16_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t worker_blocks,
    int64_t worker_threads,
    int64_t scheduler_mode,
    int64_t sleep_ns,
    int64_t start_delay_us) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (outlier_values.numel() == 0) {
    fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                           b_data.data_ptr<uint8_t>(),
                           a_scale_tile.data_ptr<uint8_t>(),
                           b_scale_tile.data_ptr<uint8_t>(),
                           a_amax.data_ptr<float>(),
                           b_amax.data_ptr<float>(),
                           reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                           nullptr,
                           static_cast<int>(m),
                           static_cast<int>(n),
                           static_cast<int>(k),
                           at::cuda::getCurrentCUDAStream(),
                           true);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  constexpr int TileM = 128;
  constexpr int TileN = 128;
  constexpr int GM = HANDWRITTEN_TMA_GM;
  const int64_t tiles_m = (m + TileM - 1) / TileM;
  const int64_t tiles_n = (n + TileN - 1) / TileN;
  const int64_t total_tiles = tiles_m * tiles_n;
  const int64_t rounded_tiles_m = ((tiles_m + GM - 1) / GM) * GM;
  const int32_t mode = static_cast<int32_t>(scheduler_mode);
  const int64_t task_count = (mode == 0) ? rounded_tiles_m * tiles_n : total_tiles;
  at::Tensor tile_status;
  at::Tensor next_task;
  at::Tensor ready_queue;
  at::Tensor ready_slot_status;
  at::Tensor ready_head;
  at::Tensor ready_tail;
  if (mode == 3) {
    ready_queue = at::empty({total_tiles}, a_data.options().dtype(at::kInt));
    ready_slot_status = at::empty({total_tiles}, a_data.options().dtype(at::kInt));
    ready_head = at::empty({1}, a_data.options().dtype(at::kInt));
    ready_tail = at::empty({1}, a_data.options().dtype(at::kInt));
    ready_slot_status.zero_();
    ready_head.zero_();
    ready_tail.zero_();
  } else {
    tile_status = at::empty({total_tiles}, a_data.options().dtype(at::kInt));
    next_task = at::empty({1}, a_data.options().dtype(at::kInt));
    tile_status.zero_();
    next_task.zero_();
  }

  cudaStream_t sparse_stream = create_v12_sparse_stream();
  v12_sparse_stream_wait_current(sparse_stream);

  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         mode == 3 ? nullptr : tile_status.data_ptr<int32_t>(),
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         mode == 3 ? ready_queue.data_ptr<int32_t>() : nullptr,
                         mode == 3 ? ready_slot_status.data_ptr<int32_t>() : nullptr,
                         mode == 3 ? ready_tail.data_ptr<int32_t>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int64_t blocks = worker_blocks > 0 ? worker_blocks : 128;
  blocks = blocks > task_count ? task_count : blocks;
  blocks = blocks < 1 ? 1 : blocks;
  int threads = worker_threads <= 128 ? 128 : 256;
  const int32_t sleep = static_cast<int32_t>(sleep_ns < 0 ? 0 : sleep_ns);
  const int32_t delay_us = static_cast<int32_t>(start_delay_us < 0 ? 0 : start_delay_us);
  if (delay_us > 0) {
    v14_sparse_start_delay_kernel<<<1, 1, 0, sparse_stream>>>(delay_us);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (mode == 3) {
    sparse_tile_ready_queue_col_value_payload_vec16_kernel<<<static_cast<int>(blocks),
                                                             threads,
                                                             0,
                                                             sparse_stream>>>(
        output.data_ptr<c10::BFloat16>(),
        outlier_values.data_ptr<c10::BFloat16>(),
        outlier_cols.data_ptr<int16_t>(),
        weight_t_bf16.data_ptr<c10::BFloat16>(),
        row_offsets.data_ptr<int32_t>(),
        ready_queue.data_ptr<int32_t>(),
        ready_slot_status.data_ptr<int32_t>(),
        ready_head.data_ptr<int32_t>(),
        ready_tail.data_ptr<int32_t>(),
        m,
        n,
        tiles_n,
        total_tiles,
        sleep);
  } else {
    sparse_tile_persistent_col_value_payload_vec16_kernel<<<static_cast<int>(blocks),
                                                            threads,
                                                            0,
                                                            sparse_stream>>>(
        output.data_ptr<c10::BFloat16>(),
        outlier_values.data_ptr<c10::BFloat16>(),
        outlier_cols.data_ptr<int16_t>(),
        weight_t_bf16.data_ptr<c10::BFloat16>(),
        row_offsets.data_ptr<int32_t>(),
        tile_status.data_ptr<int32_t>(),
        next_task.data_ptr<int32_t>(),
        m,
        n,
        tiles_m,
        tiles_n,
        task_count,
        mode,
        sleep);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  v12_current_stream_wait_sparse(sparse_stream);
  return output;
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec16_threads_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t correction_threads) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (outlier_values.numel() == 0) {
    return output;
  }

  const int threads = correction_threads <= 128 ? 128 : 256;
  const int64_t tiles_m = (m + 127) / 128;
  const int64_t tiles_n = (n + 127) / 128;
  const int64_t total_tiles = tiles_m * tiles_n;
  sparse_tile_col_value_payload_vec16_kernel<<<static_cast<int>(total_tiles),
                                               threads,
                                               0,
                                               at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      outlier_cols.data_ptr<int16_t>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      row_offsets.data_ptr<int32_t>(),
      m,
      n,
      tiles_n,
      total_tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_active_row_ready_flags_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t worker_blocks,
    int64_t sleep_ns) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0 || active_rows.numel() == 0) {
    fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                           b_data.data_ptr<uint8_t>(),
                           a_scale_tile.data_ptr<uint8_t>(),
                           b_scale_tile.data_ptr<uint8_t>(),
                           a_amax.data_ptr<float>(),
                           b_amax.data_ptr<float>(),
                           reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                           nullptr,
                           static_cast<int>(m),
                           static_cast<int>(n),
                           static_cast<int>(k),
                           at::cuda::getCurrentCUDAStream(),
                           true);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  const int64_t tiles_m = (m + 127) / 128;
  const int64_t tiles_n = (n + 127) / 128;
  const int64_t total_tiles = tiles_m * tiles_n;
  auto ready_flags = at::empty({total_tiles}, a_data.options().dtype(at::kInt));
  ready_flags.zero_();
  C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream()));

  cudaStream_t sparse_stream = create_v12_sparse_stream();

  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         ready_flags.data_ptr<int32_t>(),
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int threads = 256;
  const int64_t total_groups = active_rows.numel() * (n / 8);
  const int64_t total_group_blocks = (total_groups + threads - 1) / threads;
  int64_t blocks = worker_blocks > 0 ? worker_blocks : default_sparse_side_worker_blocks(total_group_blocks);
  blocks = blocks < 1 ? 1 : blocks;
  blocks = blocks > total_group_blocks ? total_group_blocks : blocks;
  const int32_t sleep = static_cast<int32_t>(sleep_ns < 0 ? 0 : sleep_ns);
  sparse_active_row_ready_value_payload_vec8_inplace_vstore_kernel<<<static_cast<int>(blocks),
                                                                     threads,
                                                                     0,
                                                                     sparse_stream>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_rows.data_ptr<int32_t>(),
      ready_flags.data_ptr<int32_t>(),
      active_rows.numel(),
      n,
      k,
      tiles_n,
      total_group_blocks,
      sleep);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  v12_current_stream_wait_sparse(sparse_stream);
  C10_CUDA_CHECK(cudaStreamSynchronize(sparse_stream));
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_active_row_ready_queue_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_row_offsets,
    const at::Tensor& active_rows_local,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t worker_blocks,
    int64_t worker_threads,
    int64_t sleep_ns) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0 || active_rows_local.numel() == 0) {
    fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                           b_data.data_ptr<uint8_t>(),
                           a_scale_tile.data_ptr<uint8_t>(),
                           b_scale_tile.data_ptr<uint8_t>(),
                           a_amax.data_ptr<float>(),
                           b_amax.data_ptr<float>(),
                           reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                           nullptr,
                           static_cast<int>(m),
                           static_cast<int>(n),
                           static_cast<int>(k),
                           at::cuda::getCurrentCUDAStream(),
                           true);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  const int64_t tiles_m = (m + 127) / 128;
  const int64_t tiles_n = (n + 127) / 128;
  const int64_t total_tiles = tiles_m * tiles_n;
  auto ready_queue = at::empty({total_tiles}, a_data.options().dtype(at::kInt));
  auto ready_slot_status = at::empty({total_tiles}, a_data.options().dtype(at::kInt));
  auto ready_head = at::empty({1}, a_data.options().dtype(at::kInt));
  auto ready_tail = at::empty({1}, a_data.options().dtype(at::kInt));
  ready_slot_status.zero_();
  ready_head.zero_();
  ready_tail.zero_();
  C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream()));

  cudaStream_t sparse_stream = create_v12_sparse_stream();

  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
                         ready_queue.data_ptr<int32_t>(),
                         ready_slot_status.data_ptr<int32_t>(),
                         ready_tail.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int64_t blocks = worker_blocks > 0 ? worker_blocks : default_sparse_side_worker_blocks(total_tiles);
  blocks = blocks < 1 ? 1 : blocks;
  blocks = blocks > total_tiles ? total_tiles : blocks;
  const int threads = worker_threads <= 128 ? 128 : 256;
  const int32_t sleep = static_cast<int32_t>(sleep_ns < 0 ? 0 : sleep_ns);
  sparse_active_tile_ready_queue_value_payload_vec8_inplace_vstore_kernel<<<static_cast<int>(blocks),
                                                                            threads,
                                                                            0,
                                                                            sparse_stream>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_row_offsets.data_ptr<int32_t>(),
      active_rows_local.data_ptr<int32_t>(),
      ready_queue.data_ptr<int32_t>(),
      ready_slot_status.data_ptr<int32_t>(),
      ready_head.data_ptr<int32_t>(),
      ready_tail.data_ptr<int32_t>(),
      m,
      n,
      k,
      tiles_n,
      total_tiles,
      sleep);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  v12_current_stream_wait_sparse(sparse_stream);
  C10_CUDA_CHECK(cudaStreamSynchronize(sparse_stream));
  return output;
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_active_mtile_ready_queue_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_row_offsets,
    const at::Tensor& active_rows_local,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t worker_blocks,
    int64_t worker_threads,
    int64_t sleep_ns,
    int64_t mtile_slices) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0 || active_rows_local.numel() == 0) {
    fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                           b_data.data_ptr<uint8_t>(),
                           a_scale_tile.data_ptr<uint8_t>(),
                           b_scale_tile.data_ptr<uint8_t>(),
                           a_amax.data_ptr<float>(),
                           b_amax.data_ptr<float>(),
                           reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                           nullptr,
                           static_cast<int>(m),
                           static_cast<int>(n),
                           static_cast<int>(k),
                           at::cuda::getCurrentCUDAStream(),
                           true);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  const int64_t tiles_m = (m + 127) / 128;
  const int64_t slices = mtile_slices < 1 ? 1 : mtile_slices;
  const int64_t total_items = tiles_m * slices;
  auto ready_queue = at::empty({total_items}, a_data.options().dtype(at::kInt));
  auto ready_slot_status = at::empty({total_items}, a_data.options().dtype(at::kInt));
  auto ready_head = at::empty({1}, a_data.options().dtype(at::kInt));
  auto ready_tail = at::empty({1}, a_data.options().dtype(at::kInt));
  auto ready_counts = at::empty({tiles_m}, a_data.options().dtype(at::kInt));
  ready_slot_status.zero_();
  ready_head.zero_();
  ready_tail.zero_();
  ready_counts.zero_();
  C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream()));

  cudaStream_t sparse_stream = create_v12_sparse_stream();

  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true,
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
                         0,
                         nullptr,
                         nullptr,
                         0,
                         0,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         0,
                         nullptr,
                         0,
                         0,
                         0,
                         false,
                         0,
                         ready_queue.data_ptr<int32_t>(),
                         ready_slot_status.data_ptr<int32_t>(),
                         ready_tail.data_ptr<int32_t>(),
                         ready_counts.data_ptr<int32_t>(),
                         static_cast<int32_t>(slices));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int64_t blocks = worker_blocks > 0 ? worker_blocks : default_sparse_side_worker_blocks(total_items);
  blocks = blocks < 1 ? 1 : blocks;
  blocks = blocks > total_items ? total_items : blocks;
  const int threads = worker_threads <= 128 ? 128 : 256;
  const int32_t sleep = static_cast<int32_t>(sleep_ns < 0 ? 0 : sleep_ns);
  sparse_active_mtile_ready_queue_value_payload_vec8_inplace_vstore_kernel<<<static_cast<int>(blocks),
                                                                             threads,
                                                                             0,
                                                                             sparse_stream>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      active_row_offsets.data_ptr<int32_t>(),
      active_rows_local.data_ptr<int32_t>(),
      ready_queue.data_ptr<int32_t>(),
      ready_slot_status.data_ptr<int32_t>(),
      ready_head.data_ptr<int32_t>(),
      ready_tail.data_ptr<int32_t>(),
      m,
      n,
      k,
      total_items,
      slices,
      sleep);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  v12_current_stream_wait_sparse(sparse_stream);
  C10_CUDA_CHECK(cudaStreamSynchronize(sparse_stream));
  return output;
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec32_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    int64_t m,
    int64_t k,
    int64_t n) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         nullptr,
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (outlier_values.numel() == 0) {
    return output;
  }

  constexpr int threads = 256;
  const int64_t tiles_m = (m + 127) / 128;
  const int64_t tiles_n = (n + 127) / 128;
  const int64_t total_tiles = tiles_m * tiles_n;
  sparse_tile_col_value_payload_vec32_kernel<<<static_cast<int>(total_tiles),
                                               threads,
                                               0,
                                               at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      outlier_cols.data_ptr<int16_t>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      row_offsets.data_ptr<int32_t>(),
      m,
      n,
      tiles_n,
      total_tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_tile_scales_sidecar_cuda(
    const at::Tensor& output,
    const at::Tensor& a_data,
    const at::Tensor& a_scale_tile,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_tile,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sidecar_worker_blocks) {
  const c10::cuda::CUDAGuard device_guard(output.device());
  if (flat_indices.numel() == 0) {
    fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                           b_data.data_ptr<uint8_t>(),
                           a_scale_tile.data_ptr<uint8_t>(),
                           b_scale_tile.data_ptr<uint8_t>(),
                           a_amax.data_ptr<float>(),
                           b_amax.data_ptr<float>(),
                           reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                           nullptr,
                           static_cast<int>(m),
                           static_cast<int>(n),
                           static_cast<int>(k),
                           at::cuda::getCurrentCUDAStream(),
                           true);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }

  constexpr int TileM = 128;
  constexpr int TileN = 128;
  constexpr int threads = 256;
  const int64_t tiles_m = (m + TileM - 1) / TileM;
  const int64_t tiles_n = (n + TileN - 1) / TileN;
  const int64_t total_tiles = tiles_m * tiles_n;
  auto tile_status = at::empty({total_tiles}, a_data.options().dtype(at::kInt));
  auto dense_done = at::empty({1}, a_data.options().dtype(at::kInt));
  tile_status.zero_();
  dense_done.zero_();

  cudaStream_t sparse_stream = get_v12_sparse_stream();
  int64_t sidecar_blocks = sidecar_worker_blocks < 0 ? 0 : sidecar_worker_blocks;
  sidecar_blocks = sidecar_blocks > total_tiles ? total_tiles : sidecar_blocks;
  if (sidecar_blocks > 0) {
    v12_sparse_stream_wait_current(sparse_stream);
  }

  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_tile.data_ptr<uint8_t>(),
                         b_scale_tile.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         tile_status.data_ptr<int32_t>(),
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (sidecar_blocks > 0) {
    sparse_tile_sidecar_value_payload_vec8_kernel<<<static_cast<int>(sidecar_blocks),
                                                    threads,
                                                    0,
                                                    sparse_stream>>>(
        output.data_ptr<c10::BFloat16>(),
        outlier_values.data_ptr<c10::BFloat16>(),
        weight_t_bf16.data_ptr<c10::BFloat16>(),
        flat_indices.data_ptr<int32_t>(),
        row_offsets.data_ptr<int32_t>(),
        tile_status.data_ptr<int32_t>(),
        dense_done.data_ptr<int32_t>(),
        m,
        n,
        k,
        tiles_n,
        total_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  mark_dense_done_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      dense_done.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int64_t tail_blocks = total_tiles;
  sparse_tile_tail_value_payload_vec8_kernel<<<static_cast<int>(tail_blocks),
                                               threads,
                                               0,
                                               at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      tile_status.data_ptr<int32_t>(),
      m,
      n,
      k,
      tiles_n,
      total_tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (sidecar_blocks > 0) {
    v12_current_stream_wait_sparse(sparse_stream);
  }
  return output;
}

at::Tensor nvfp4_gemm_tma_swizzled_scale_cuda(const at::Tensor& a_data,
                                              const at::Tensor& a_scale_inv,
                                              const at::Tensor& b_data,
                                              const at::Tensor& b_scale_inv,
                                              const at::Tensor& a_amax,
                                              const at::Tensor& b_amax,
                                              int64_t m,
                                              int64_t k,
                                              int64_t n) {
  const c10::cuda::CUDAGuard device_guard(a_data.device());
  auto a_scale_tile = swizzle_te_scale_to_tma_tile_major_cuda(a_scale_inv, m, k);
  auto b_scale_tile = swizzle_te_scale_to_tma_tile_major_cuda(b_scale_inv, n, k);
  return nvfp4_gemm_tma_tile_scales_cuda(a_data,
                                         a_scale_tile,
                                         b_data,
                                         b_scale_tile,
                                         a_amax,
                                         b_amax,
                                         m,
                                         k,
                                         n);
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_overlap_cuda(
    const at::Tensor& a_data,
    const at::Tensor& a_scale_inv,
    const at::Tensor& b_data,
    const at::Tensor& b_scale_inv,
    const at::Tensor& a_amax,
    const at::Tensor& b_amax,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t sparse_worker_blocks) {
  const c10::cuda::CUDAGuard device_guard(a_data.device());
  if (flat_indices.numel() == 0) {
    return nvfp4_gemm_tma_warpspecialized_cuda(
        a_data, a_scale_inv, b_data, b_scale_inv, a_amax, b_amax, m, k, n);
  }

  auto output = at::empty({m, n}, a_data.options().dtype(at::kBFloat16));
  const int64_t tiles_m = (m + 127) / 128;
  const int64_t tiles_n = (n + 127) / 128;
  const int64_t tiles = tiles_m * tiles_n;
  auto ready_flags = at::empty({tiles}, a_data.options().dtype(at::kInt));
  ready_flags.zero_();

  cudaStream_t sparse_stream = get_v12_sparse_stream();
  v12_sparse_stream_wait_current(sparse_stream);

  fp4::nvfp4_gemm_launch(a_data.data_ptr<uint8_t>(),
                         b_data.data_ptr<uint8_t>(),
                         a_scale_inv.data_ptr<uint8_t>(),
                         b_scale_inv.data_ptr<uint8_t>(),
                         a_amax.data_ptr<float>(),
                         b_amax.data_ptr<float>(),
                         reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>()),
                         ready_flags.data_ptr<int32_t>(),
                         static_cast<int>(m),
                         static_cast<int>(n),
                         static_cast<int>(k),
                         at::cuda::getCurrentCUDAStream(),
                         false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int threads = 256;
  const int64_t total_groups = m * (n / 8);
  const int64_t total_group_blocks = (total_groups + threads - 1) / threads;
  int64_t sparse_blocks =
      sparse_worker_blocks > 0 ? sparse_worker_blocks : total_group_blocks;
  sparse_blocks = sparse_blocks < 1 ? 1 : sparse_blocks;
  sparse_blocks = sparse_blocks > total_group_blocks ? total_group_blocks : sparse_blocks;
  sparse_group_ready_value_payload_vec8_kernel<<<static_cast<int>(sparse_blocks),
                                                 threads,
                                                 0,
                                                 sparse_stream>>>(
      output.data_ptr<c10::BFloat16>(),
      outlier_values.data_ptr<c10::BFloat16>(),
      weight_t_bf16.data_ptr<c10::BFloat16>(),
      flat_indices.data_ptr<int32_t>(),
      row_offsets.data_ptr<int32_t>(),
      ready_flags.data_ptr<int32_t>(),
      m,
      n,
      k,
      tiles_n,
      total_group_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  v12_current_stream_wait_sparse(sparse_stream);
  return output;
}
