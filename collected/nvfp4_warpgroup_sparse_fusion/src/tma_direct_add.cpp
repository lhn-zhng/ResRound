#include <torch/extension.h>

#include <cstdint>

at::Tensor nvfp4_gemm_tma_warpspecialized_cuda(const at::Tensor& a_data,
                                               const at::Tensor& a_scale_inv,
                                               const at::Tensor& b_data,
                                               const at::Tensor& b_scale_inv,
                                               const at::Tensor& a_amax,
                                               const at::Tensor& b_amax,
                                               int64_t m,
                                               int64_t k,
                                               int64_t n);

at::Tensor nvfp4_gemm_tma_swizzled_scale_cuda(const at::Tensor& a_data,
                                              const at::Tensor& a_scale_inv,
                                              const at::Tensor& b_data,
                                              const at::Tensor& b_scale_inv,
                                              const at::Tensor& a_amax,
                                              const at::Tensor& b_amax,
                                              int64_t m,
                                              int64_t k,
                                              int64_t n);

at::Tensor nvfp4_gemm_tma_tile_scales_cuda(const at::Tensor& a_data,
                                           const at::Tensor& a_scale_tile,
                                           const at::Tensor& b_data,
                                           const at::Tensor& b_scale_tile,
                                           const at::Tensor& a_amax,
                                           const at::Tensor& b_amax,
                                           int64_t m,
                                           int64_t k,
                                           int64_t n);

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_cuda(const at::Tensor& output,
                                                        const at::Tensor& a_data,
                                                        const at::Tensor& a_scale_tile,
                                                        const at::Tensor& b_data,
                                                        const at::Tensor& b_scale_tile,
                                                        const at::Tensor& a_amax,
                                                        const at::Tensor& b_amax,
                                                        int64_t m,
                                                        int64_t k,
                                                        int64_t n);

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
    int64_t n);

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
    int64_t n);

int64_t compact_consumer_max_nnz_cuda();
int64_t compact_consumer_static_n_cuda();

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
    int64_t sleep_ns);

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
    int64_t sleep_ns);

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
    int64_t mtile_slices);

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
    int64_t n);

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
    int64_t n);

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
    int64_t direct_smem_mode);

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
    int64_t direct_smem_mode);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t sparse_warpgroups);

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
    int64_t group_budget);

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
    const at::Tensor* phase_trace = nullptr,
    int64_t phase_trace_max_ctas = 0,
    int64_t phase_trace_stride = 0);

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
	    const at::Tensor* phase_trace = nullptr,
	    int64_t phase_trace_max_ctas = 0,
	    int64_t phase_trace_stride = 0,
	    const at::Tensor* packed_tile_offsets = nullptr,
	    const at::Tensor* packed_row_records = nullptr,
	    const at::Tensor* packed_entry_records = nullptr,
	    int64_t packed_payload_mode = 0,
	    const at::Tensor* kmajor_tile_group_starts = nullptr,
	    const at::Tensor* kmajor_tile_group_counts = nullptr,
	    const at::Tensor* kmajor_tile_group_meta = nullptr);

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
    int64_t direct_delta_write_mode);

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
    int64_t sparse_warpgroups);

at::Tensor merge_entry_delta_active_rows_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& merge_row_offsets,
    const at::Tensor& merge_entry_indices,
    int64_t n);

at::Tensor merge_entry_delta_active_rows_fastpath_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& merge_row_offsets,
    const at::Tensor& merge_entry_indices,
    int64_t n);

at::Tensor merge_entry_delta_active_rows_vec8_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& merge_row_offsets,
    const at::Tensor& merge_entry_indices,
    int64_t n);

at::Tensor merge_entry_delta_active_rows_chunk_prefix_vec8_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& merge_row_offsets,
    const at::Tensor& merge_entry_indices,
    int64_t n,
    int64_t chunk_cols,
    int64_t chunks_per_row);

at::Tensor merge_single_entry_delta_active_rows_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& entry_indices,
    int64_t n);

at::Tensor merge_double_entry_delta_active_rows_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& entry0_indices,
    const at::Tensor& entry1_indices,
    int64_t n);

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
    int64_t n);

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
    int64_t n);

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
    int64_t sidecar_worker_blocks);

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
    int64_t n);

at::Tensor sparse_active_row_value_payload_vec8_store_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k);

at::Tensor sparse_active_row_value_payload_vec8_store_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k);

at::Tensor merge_full_delta_active_rows_cuda(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& active_rows);

at::Tensor merge_compact_delta_active_rows_cuda(
    const at::Tensor& output,
    const at::Tensor& compact_delta,
    const at::Tensor& active_rows);

at::Tensor merge_two_compact_delta_active_rows_cuda(
    const at::Tensor& output,
    const at::Tensor& first_delta,
    const at::Tensor& first_rows,
    const at::Tensor& second_delta,
    const at::Tensor& second_rows);

at::Tensor build_compact_dense_residual_active_rows_cuda(
    const at::Tensor& residual,
    const at::Tensor& row_values,
    const at::Tensor& row_ks,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k);

std::vector<at::Tensor> build_padded_light_heavy_rows_cuda(
    const at::Tensor& row_offsets,
    int64_t heavy_threshold,
    int64_t heavy_capacity);

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
    int64_t n);

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
    int64_t epin);

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
    int64_t n);

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
    int64_t bm);

at::Tensor sparse_active_row_value_payload_vec8_inplace_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k);

at::Tensor sparse_active_row_value_payload_vec8_inplace_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k);

at::Tensor sparse_active_row_value_payload_vec8_inplace_skip_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k,
    int64_t skip_per_row);

at::Tensor sparse_packed_suffix12_vec8_inplace_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& packed_suffix_records,
    const at::Tensor& active_rows,
    const at::Tensor& weight_t_bf16);

at::Tensor sparse_active_row_value_payload_vec8_inplace_strict_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k);

at::Tensor sparse_active_row_value_payload_vec8_inplace_sum_then_add_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k);

at::Tensor sparse_active_row_value_payload_vec8_inplace_b_evict_last_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k);

at::Tensor sparse_active_row_value_payload_vec8_inplace_fastpath_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k);

at::Tensor sparse_active_row_value_payload_vec8_inplace_rowblock_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k);

at::Tensor sparse_active_row_col_value_payload_vec16_inplace_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k);

at::Tensor sparse_active_row_col_value_payload_vec8_inplace_vstore_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k);

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
    int64_t cold_capacity);

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
    int64_t cold_capacity);

at::Tensor sparse_active_row_col_value_payload_vec8_shmem_sum_then_add_cuda(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k);

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
    int64_t n);

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
    int64_t n);

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
    int64_t n);

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
    int64_t n);

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
    int64_t n);

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
    int64_t start_delay_us);

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
    int64_t n);

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
    int64_t correction_threads);

at::Tensor swizzle_te_scale_to_tma_tile_major(const at::Tensor& scale,
                                              int64_t rows,
                                              int64_t k);

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
    int64_t sparse_worker_blocks);

namespace {

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void check_inputs(const at::Tensor& a_data,
                  const at::Tensor& a_scale_inv,
                  const at::Tensor& b_data,
                  const at::Tensor& b_scale_inv,
                  const at::Tensor& a_amax,
                  const at::Tensor& b_amax,
                  int64_t m,
                  int64_t k,
                  int64_t n) {
  CHECK_CUDA(a_data);
  CHECK_CUDA(a_scale_inv);
  CHECK_CUDA(b_data);
  CHECK_CUDA(b_scale_inv);
  CHECK_CUDA(a_amax);
  CHECK_CUDA(b_amax);
  CHECK_CONTIGUOUS(a_data);
  CHECK_CONTIGUOUS(a_scale_inv);
  CHECK_CONTIGUOUS(b_data);
  CHECK_CONTIGUOUS(b_scale_inv);
  CHECK_CONTIGUOUS(a_amax);
  CHECK_CONTIGUOUS(b_amax);
  TORCH_CHECK(a_data.scalar_type() == at::kByte && b_data.scalar_type() == at::kByte,
              "packed data must be uint8");
  TORCH_CHECK(a_scale_inv.scalar_type() == at::kByte && b_scale_inv.scalar_type() == at::kByte,
              "scale must be uint8 UE4M3");
  TORCH_CHECK(a_amax.scalar_type() == at::kFloat && b_amax.scalar_type() == at::kFloat,
              "amax must be fp32");
  TORCH_CHECK(m % 128 == 0 && n % 128 == 0 && k % 128 == 0,
              "v12 TMA path currently requires M/N/K divisible by 128");
  TORCH_CHECK(a_data.size(0) == m && a_data.size(1) == k / 2, "A packed shape mismatch");
  TORCH_CHECK(b_data.size(0) == n && b_data.size(1) == k / 2, "B packed shape mismatch");
  TORCH_CHECK(a_scale_inv.size(0) == m && a_scale_inv.size(1) == k / 16,
              "A scale shape mismatch");
  TORCH_CHECK(b_scale_inv.size(0) == n && b_scale_inv.size(1) == k / 16,
              "B scale shape mismatch");
}

void check_value_payload_inputs(const at::Tensor& outlier_values,
                                const at::Tensor& weight_t_bf16,
                                const at::Tensor& flat_indices,
                                const at::Tensor& row_offsets,
                                int64_t m,
                                int64_t k,
                                int64_t n) {
  CHECK_CUDA(outlier_values);
  CHECK_CUDA(weight_t_bf16);
  CHECK_CUDA(flat_indices);
  CHECK_CUDA(row_offsets);
  CHECK_CONTIGUOUS(outlier_values);
  CHECK_CONTIGUOUS(weight_t_bf16);
  CHECK_CONTIGUOUS(flat_indices);
  CHECK_CONTIGUOUS(row_offsets);
  TORCH_CHECK(outlier_values.scalar_type() == at::kBFloat16,
              "outlier_values must be bf16");
  TORCH_CHECK(weight_t_bf16.scalar_type() == at::kBFloat16,
              "weight_t_bf16 must be bf16");
  TORCH_CHECK(flat_indices.scalar_type() == at::kInt, "flat_indices must be int32");
  TORCH_CHECK(row_offsets.scalar_type() == at::kInt, "row_offsets must be int32");
  TORCH_CHECK(weight_t_bf16.size(0) == k && weight_t_bf16.size(1) == n,
              "weight_t_bf16 shape mismatch");
  TORCH_CHECK(row_offsets.numel() == m + 1, "row_offsets shape must be M+1");
  TORCH_CHECK(outlier_values.numel() >= flat_indices.numel(),
              "outlier_values must cover flat_indices");
}

void check_output(const at::Tensor& output, int64_t m, int64_t n) {
  CHECK_CUDA(output);
  CHECK_CONTIGUOUS(output);
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bf16");
  TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape mismatch");
}

void check_compact_delta_output(const at::Tensor& output, int64_t rows, int64_t n) {
  CHECK_CUDA(output);
  CHECK_CONTIGUOUS(output);
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "compact delta output must be bf16");
  TORCH_CHECK(output.dim() == 2, "compact delta output must be 2D");
  TORCH_CHECK(output.size(0) >= rows && output.size(1) == n,
              "compact delta output shape mismatch");
}

void check_merge_entry_delta_inputs(const at::Tensor& output,
                                    const at::Tensor& delta_entries,
                                    const at::Tensor& active_rows,
                                    const at::Tensor& merge_row_offsets,
                                    const at::Tensor& merge_entry_indices) {
  CHECK_CUDA(output);
  CHECK_CUDA(delta_entries);
  CHECK_CUDA(active_rows);
  CHECK_CUDA(merge_row_offsets);
  CHECK_CUDA(merge_entry_indices);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(delta_entries);
  CHECK_CONTIGUOUS(active_rows);
  CHECK_CONTIGUOUS(merge_row_offsets);
  CHECK_CONTIGUOUS(merge_entry_indices);
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bf16");
  TORCH_CHECK(delta_entries.scalar_type() == at::kBFloat16, "delta_entries must be bf16");
  TORCH_CHECK(active_rows.scalar_type() == at::kInt, "active_rows must be int32");
  TORCH_CHECK(merge_row_offsets.scalar_type() == at::kInt,
              "merge_row_offsets must be int32");
  TORCH_CHECK(merge_entry_indices.scalar_type() == at::kInt,
              "merge_entry_indices must be int32");
  TORCH_CHECK(output.dim() == 2 && delta_entries.dim() == 2,
              "output and delta_entries must be 2D");
  TORCH_CHECK(output.size(1) == delta_entries.size(1),
              "output/delta_entries N mismatch");
  TORCH_CHECK(merge_row_offsets.numel() == active_rows.numel() + 1,
              "merge_row_offsets length mismatch");
  TORCH_CHECK(merge_entry_indices.numel() <= delta_entries.size(0),
              "merge_entry_indices is longer than delta_entries");
}

void check_probe_sink(const at::Tensor& probe_sink,
                      const at::Tensor& probe_counter,
                      const at::Tensor& probe_active_mblocks,
                      int64_t m,
                      int64_t n,
                      int64_t sparse_warpgroups) {
  CHECK_CUDA(probe_sink);
  CHECK_CUDA(probe_counter);
  CHECK_CUDA(probe_active_mblocks);
  CHECK_CONTIGUOUS(probe_sink);
  CHECK_CONTIGUOUS(probe_counter);
  CHECK_CONTIGUOUS(probe_active_mblocks);
  TORCH_CHECK(probe_sink.scalar_type() == at::kFloat, "probe_sink must be fp32");
  TORCH_CHECK(probe_counter.scalar_type() == at::kInt, "probe_counter must be int32");
  TORCH_CHECK(probe_active_mblocks.scalar_type() == at::kInt,
              "probe_active_mblocks must be int32");
  TORCH_CHECK(probe_counter.numel() >= 1, "probe_counter must have at least one element");
  TORCH_CHECK(probe_active_mblocks.numel() >= 1,
              "probe_active_mblocks must have at least one element");
  const int64_t sparse_warpgroups_abs =
      sparse_warpgroups < 0 ? -sparse_warpgroups : sparse_warpgroups;
  TORCH_CHECK(sparse_warpgroups_abs >= 1 && sparse_warpgroups_abs <= 64,
              "sparse_warpgroups must be in [1, 64]");
  const int64_t tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= tiles * sparse_warpgroups_abs * 4,
              "probe_sink is too small");
}

void check_probe_sink_slots(const at::Tensor& probe_sink,
                            const at::Tensor& probe_counter,
                            const at::Tensor& probe_active_mblocks,
                            int64_t m,
                            int64_t n,
                            int64_t slots_per_tile) {
  CHECK_CUDA(probe_sink);
  CHECK_CUDA(probe_counter);
  CHECK_CUDA(probe_active_mblocks);
  CHECK_CONTIGUOUS(probe_sink);
  CHECK_CONTIGUOUS(probe_counter);
  CHECK_CONTIGUOUS(probe_active_mblocks);
  TORCH_CHECK(probe_sink.scalar_type() == at::kFloat, "probe_sink must be fp32");
  TORCH_CHECK(probe_counter.scalar_type() == at::kInt, "probe_counter must be int32");
  TORCH_CHECK(probe_active_mblocks.scalar_type() == at::kInt,
              "probe_active_mblocks must be int32");
  TORCH_CHECK(probe_counter.numel() >= 1, "probe_counter must have at least one element");
  TORCH_CHECK(probe_active_mblocks.numel() >= 1,
              "probe_active_mblocks must have at least one element");
  TORCH_CHECK(slots_per_tile >= 1, "slots_per_tile must be positive");
  const int64_t tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= tiles * slots_per_tile,
              "probe_sink is too small");
}

void check_active_rows(const at::Tensor& active_rows, int64_t m) {
  CHECK_CUDA(active_rows);
  CHECK_CONTIGUOUS(active_rows);
  TORCH_CHECK(active_rows.scalar_type() == at::kInt, "active_rows must be int32");
  TORCH_CHECK(active_rows.numel() <= m, "active_rows cannot be longer than M");
}

void check_direct_add_inputs(const at::Tensor& row_offsets,
                             const at::Tensor& row_ks,
                             const at::Tensor& row_values,
                             const at::Tensor& active_row_offsets,
                             const at::Tensor& active_rows,
                             const at::Tensor& b_comp,
                             int64_t m,
                             int64_t k,
                             int64_t n) {
  CHECK_CUDA(row_offsets);
  CHECK_CUDA(row_ks);
  CHECK_CUDA(row_values);
  CHECK_CUDA(active_row_offsets);
  CHECK_CUDA(active_rows);
  CHECK_CUDA(b_comp);
  CHECK_CONTIGUOUS(row_offsets);
  CHECK_CONTIGUOUS(row_ks);
  CHECK_CONTIGUOUS(row_values);
  CHECK_CONTIGUOUS(active_row_offsets);
  CHECK_CONTIGUOUS(active_rows);
  CHECK_CONTIGUOUS(b_comp);
  TORCH_CHECK(row_offsets.scalar_type() == at::kInt, "row_offsets must be int32");
  TORCH_CHECK(row_ks.scalar_type() == at::kInt, "row_ks must be int32");
  TORCH_CHECK(row_values.scalar_type() == at::kBFloat16, "row_values must be bf16");
  TORCH_CHECK(active_row_offsets.scalar_type() == at::kInt,
              "active_row_offsets must be int32");
  TORCH_CHECK(active_rows.scalar_type() == at::kInt, "active_rows must be int32");
  TORCH_CHECK(b_comp.scalar_type() == at::kBFloat16, "b_comp must be bf16");
  TORCH_CHECK(row_offsets.numel() == m + 1, "row_offsets shape must be M+1");
  TORCH_CHECK(row_ks.numel() == row_values.numel(), "row_ks/row_values length mismatch");
  TORCH_CHECK(active_row_offsets.numel() == (m + 127) / 128 + 1,
              "active_row_offsets must be tile_m_count+1 for BM=128");
  TORCH_CHECK(active_rows.numel() <= m * 2,
              "active_rows is unexpectedly large");
  TORCH_CHECK(b_comp.size(0) == k && b_comp.size(1) == n, "b_comp shape mismatch");
}

void check_rowblock_payload_inputs(const at::Tensor& row_offsets,
                                   const at::Tensor& row_ks,
                                   const at::Tensor& row_values,
                                   const at::Tensor& b_comp,
                                   int64_t m,
                                   int64_t k,
                                   int64_t n) {
  CHECK_CUDA(row_offsets);
  CHECK_CUDA(row_ks);
  CHECK_CUDA(row_values);
  CHECK_CUDA(b_comp);
  CHECK_CONTIGUOUS(row_offsets);
  CHECK_CONTIGUOUS(row_ks);
  CHECK_CONTIGUOUS(row_values);
  CHECK_CONTIGUOUS(b_comp);
  TORCH_CHECK(row_offsets.scalar_type() == at::kInt, "row_offsets must be int32");
  TORCH_CHECK(row_ks.scalar_type() == at::kInt, "row_ks must be int32");
  TORCH_CHECK(row_values.scalar_type() == at::kBFloat16, "row_values must be bf16");
  TORCH_CHECK(b_comp.scalar_type() == at::kBFloat16, "b_comp must be bf16");
  TORCH_CHECK(row_offsets.numel() == m + 1, "row_offsets shape must be M+1");
  TORCH_CHECK(row_ks.numel() == row_values.numel(), "row_ks/row_values length mismatch");
  TORCH_CHECK(b_comp.size(0) == k && b_comp.size(1) == n, "b_comp shape mismatch");
}

void check_kmajor_probe_inputs(const at::Tensor& kmajor_group_offsets,
                               const at::Tensor& kmajor_group_ks,
                               const at::Tensor& kmajor_entry_offsets,
                               const at::Tensor& kmajor_entry_rows,
                               const at::Tensor& kmajor_entry_values,
                               const at::Tensor& b_comp,
                               const at::Tensor& probe_active_mblocks,
                               int64_t k,
                               int64_t n) {
  CHECK_CUDA(kmajor_group_offsets);
  CHECK_CUDA(kmajor_group_ks);
  CHECK_CUDA(kmajor_entry_offsets);
  CHECK_CUDA(kmajor_entry_rows);
  CHECK_CUDA(kmajor_entry_values);
  CHECK_CUDA(b_comp);
  CHECK_CUDA(probe_active_mblocks);
  CHECK_CONTIGUOUS(kmajor_group_offsets);
  CHECK_CONTIGUOUS(kmajor_group_ks);
  CHECK_CONTIGUOUS(kmajor_entry_offsets);
  CHECK_CONTIGUOUS(kmajor_entry_rows);
  CHECK_CONTIGUOUS(kmajor_entry_values);
  CHECK_CONTIGUOUS(b_comp);
  CHECK_CONTIGUOUS(probe_active_mblocks);
  TORCH_CHECK(kmajor_group_offsets.scalar_type() == at::kInt,
              "kmajor_group_offsets must be int32");
  TORCH_CHECK(kmajor_group_ks.scalar_type() == at::kInt,
              "kmajor_group_ks must be int32");
  TORCH_CHECK(kmajor_entry_offsets.scalar_type() == at::kInt,
              "kmajor_entry_offsets must be int32");
  TORCH_CHECK(kmajor_entry_rows.scalar_type() == at::kInt,
              "kmajor_entry_rows must be int32");
  TORCH_CHECK(kmajor_entry_values.scalar_type() == at::kBFloat16,
              "kmajor_entry_values must be bf16");
  TORCH_CHECK(b_comp.scalar_type() == at::kBFloat16, "b_comp must be bf16");
  TORCH_CHECK(probe_active_mblocks.scalar_type() == at::kInt,
              "probe_active_mblocks must be int32");
  TORCH_CHECK(kmajor_group_offsets.numel() == probe_active_mblocks.numel() + 1,
              "kmajor_group_offsets must be active_mblocks+1");
  TORCH_CHECK(kmajor_entry_offsets.numel() == kmajor_group_ks.numel() + 1,
              "kmajor_entry_offsets must be group_count+1");
  TORCH_CHECK(kmajor_entry_rows.numel() == kmajor_entry_values.numel(),
              "kmajor entry row/value length mismatch");
  TORCH_CHECK(kmajor_entry_offsets.numel() >= 1,
              "kmajor_entry_offsets must have at least one element");
  TORCH_CHECK(b_comp.size(0) == k && b_comp.size(1) == n, "b_comp shape mismatch");
}

void check_packed_local_delta_payload(const at::Tensor& packed_tile_offsets,
                                      const at::Tensor& packed_row_records,
                                      const at::Tensor& packed_entry_records,
                                      int64_t m) {
  CHECK_CUDA(packed_tile_offsets);
  CHECK_CUDA(packed_row_records);
  CHECK_CUDA(packed_entry_records);
  CHECK_CONTIGUOUS(packed_tile_offsets);
  CHECK_CONTIGUOUS(packed_row_records);
  CHECK_CONTIGUOUS(packed_entry_records);
  TORCH_CHECK(packed_tile_offsets.scalar_type() == at::kInt,
              "packed_tile_offsets must be int32");
  TORCH_CHECK(packed_row_records.scalar_type() == at::kLong,
              "packed_row_records must be int64");
  TORCH_CHECK(packed_entry_records.scalar_type() == at::kInt,
              "packed_entry_records must be int32");
  TORCH_CHECK(packed_tile_offsets.dim() == 1, "packed_tile_offsets must be 1D");
  TORCH_CHECK(packed_row_records.dim() == 1, "packed_row_records must be 1D");
  TORCH_CHECK(packed_entry_records.dim() == 1, "packed_entry_records must be 1D");
  TORCH_CHECK(packed_tile_offsets.numel() == (m + 127) / 128 + 1,
              "packed_tile_offsets must be tile_m_count+1");
}

void check_packed_rowblock_payload(const at::Tensor& active_rowblocks,
                                   const at::Tensor& packed_tile_offsets,
                                   const at::Tensor& packed_row_records,
                                   const at::Tensor& packed_entry_records,
                                   int64_t m) {
  CHECK_CUDA(active_rowblocks);
  CHECK_CUDA(packed_tile_offsets);
  CHECK_CUDA(packed_row_records);
  CHECK_CUDA(packed_entry_records);
  CHECK_CONTIGUOUS(active_rowblocks);
  CHECK_CONTIGUOUS(packed_tile_offsets);
  CHECK_CONTIGUOUS(packed_row_records);
  CHECK_CONTIGUOUS(packed_entry_records);
  TORCH_CHECK(active_rowblocks.scalar_type() == at::kInt,
              "active_rowblocks must be int32");
  TORCH_CHECK(packed_tile_offsets.scalar_type() == at::kInt,
              "packed_tile_offsets must be int32");
  TORCH_CHECK(packed_row_records.scalar_type() == at::kLong,
              "packed_row_records must be int64");
  TORCH_CHECK(packed_entry_records.scalar_type() == at::kInt,
              "packed_entry_records must be int32");
  TORCH_CHECK(active_rowblocks.dim() == 1, "active_rowblocks must be 1D");
  TORCH_CHECK(packed_tile_offsets.dim() == 1, "packed_tile_offsets must be 1D");
  TORCH_CHECK(packed_row_records.dim() == 1, "packed_row_records must be 1D");
  TORCH_CHECK(packed_entry_records.dim() == 1, "packed_entry_records must be 1D");
  TORCH_CHECK(active_rowblocks.numel() >= 1, "active_rowblocks must be non-empty");
  TORCH_CHECK(active_rowblocks.numel() <= (m + 7) / 8,
              "active_rowblocks is unexpectedly large");
  TORCH_CHECK(packed_tile_offsets.numel() == active_rowblocks.numel() + 1,
              "packed_tile_offsets must be active_rowblocks+1");
  TORCH_CHECK(packed_row_records.numel() == active_rowblocks.numel() * 8,
              "packed rowblock path expects exactly 8 row records per rowblock");
}

void check_kmajor_tile_metadata(const at::Tensor& tile_group_starts,
                                const at::Tensor& tile_group_counts,
                                const at::Tensor& tile_group_meta,
                                int64_t m) {
  CHECK_CUDA(tile_group_starts);
  CHECK_CUDA(tile_group_counts);
  CHECK_CUDA(tile_group_meta);
  CHECK_CONTIGUOUS(tile_group_starts);
  CHECK_CONTIGUOUS(tile_group_counts);
  CHECK_CONTIGUOUS(tile_group_meta);
  TORCH_CHECK(tile_group_starts.scalar_type() == at::kInt,
              "tile_group_starts must be int32");
  TORCH_CHECK(tile_group_counts.scalar_type() == at::kInt,
              "tile_group_counts must be int32");
  TORCH_CHECK(tile_group_meta.scalar_type() == at::kLong,
              "tile_group_meta must be int64");
  TORCH_CHECK(tile_group_starts.dim() == 1, "tile_group_starts must be 1D");
  TORCH_CHECK(tile_group_counts.dim() == 1, "tile_group_counts must be 1D");
  TORCH_CHECK(tile_group_meta.dim() == 1, "tile_group_meta must be 1D");
  const int64_t tile_m_count = (m + 127) / 128;
  TORCH_CHECK(tile_group_starts.numel() == tile_m_count,
              "tile_group_starts must be tile_m_count");
  TORCH_CHECK(tile_group_counts.numel() == tile_m_count,
              "tile_group_counts must be tile_m_count");
  TORCH_CHECK(tile_group_meta.numel() == tile_m_count,
              "tile_group_meta must be tile_m_count");
}

void check_outlier_cols(const at::Tensor& outlier_cols,
                        const at::Tensor& outlier_values,
                        int64_t k) {
  CHECK_CUDA(outlier_cols);
  CHECK_CONTIGUOUS(outlier_cols);
  TORCH_CHECK(outlier_cols.scalar_type() == at::kShort, "outlier_cols must be int16");
  TORCH_CHECK(outlier_cols.numel() == outlier_values.numel(),
              "outlier_cols must match outlier_values length");
  TORCH_CHECK(k <= 32767, "int16 outlier_cols requires K <= 32767");
}

void check_col_value_payload_inputs(const at::Tensor& outlier_values,
                                    const at::Tensor& outlier_cols,
                                    const at::Tensor& weight_t_bf16,
                                    const at::Tensor& row_offsets,
                                    int64_t m,
                                    int64_t k,
                                    int64_t n) {
  CHECK_CUDA(outlier_values);
  CHECK_CUDA(weight_t_bf16);
  CHECK_CUDA(row_offsets);
  CHECK_CONTIGUOUS(outlier_values);
  CHECK_CONTIGUOUS(weight_t_bf16);
  CHECK_CONTIGUOUS(row_offsets);
  TORCH_CHECK(outlier_values.scalar_type() == at::kBFloat16,
              "outlier_values must be bf16");
  TORCH_CHECK(weight_t_bf16.scalar_type() == at::kBFloat16,
              "weight_t_bf16 must be bf16");
  TORCH_CHECK(row_offsets.scalar_type() == at::kInt, "row_offsets must be int32");
  TORCH_CHECK(weight_t_bf16.size(0) == k && weight_t_bf16.size(1) == n,
              "weight_t_bf16 shape mismatch");
  TORCH_CHECK(row_offsets.numel() == m + 1, "row_offsets shape must be M+1");
  check_outlier_cols(outlier_cols, outlier_values, k);
}

}  // namespace

at::Tensor swizzle_scale_to_tma_tile_major(const at::Tensor& scale,
                                           int64_t rows,
                                           int64_t k) {
  CHECK_CUDA(scale);
  CHECK_CONTIGUOUS(scale);
  TORCH_CHECK(scale.scalar_type() == at::kByte, "scale must be uint8 UE4M3");
  TORCH_CHECK(rows % 128 == 0 && k % 128 == 0,
              "scale swizzle requires rows and K divisible by 128");
  TORCH_CHECK(scale.size(0) == rows && scale.size(1) == k / 16,
              "scale shape mismatch");
  return swizzle_te_scale_to_tma_tile_major(scale, rows, k);
}

at::Tensor nvfp4_gemm_tma_warpspecialized(const at::Tensor& a_data,
                                          const at::Tensor& a_scale_inv,
                                          const at::Tensor& b_data,
                                          const at::Tensor& b_scale_inv,
                                          const at::Tensor& a_amax,
                                          const at::Tensor& b_amax,
                                          int64_t m,
                                          int64_t k,
                                          int64_t n) {
  check_inputs(a_data, a_scale_inv, b_data, b_scale_inv, a_amax, b_amax, m, k, n);
  return nvfp4_gemm_tma_warpspecialized_cuda(a_data,
                                            a_scale_inv,
                                            b_data,
                                            b_scale_inv,
                                            a_amax,
                                            b_amax,
                                            m,
                                            k,
                                            n);
}

at::Tensor nvfp4_gemm_tma_tile_scales(const at::Tensor& a_data,
                                      const at::Tensor& a_scale_tile,
                                      const at::Tensor& b_data,
                                      const at::Tensor& b_scale_tile,
                                      const at::Tensor& a_amax,
                                      const at::Tensor& b_amax,
                                      int64_t m,
                                      int64_t k,
                                      int64_t n) {
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
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

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  return preallocated_nvfp4_gemm_tma_tile_scales_cuda(output,
                                                     a_data,
                                                     a_scale_tile,
                                                     b_data,
                                                     b_scale_tile,
                                                     a_amax,
                                                     b_amax,
                                                     m,
                                                     k,
                                                     n);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_4wg(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  return preallocated_nvfp4_gemm_tma_tile_scales_4wg_cuda(output,
                                                        a_data,
                                                        a_scale_tile,
                                                        b_data,
                                                        b_scale_tile,
                                                        a_amax,
                                                        b_amax,
                                                        m,
                                                        k,
      n);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_compact_consumer_posttail(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_rowblock_payload_inputs(row_offsets, row_ks, row_values, b_comp, m, k, n);
  TORCH_CHECK(k <= 65535, "compact consumer post-tail packed K requires K <= 65535");
  return preallocated_nvfp4_gemm_tma_tile_scales_compact_consumer_posttail_cuda(
      output,
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
      n);
}

int64_t compact_consumer_max_nnz() {
  return compact_consumer_max_nnz_cuda();
}

int64_t compact_consumer_static_n() {
  return compact_consumer_static_n_cuda();
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_active_row_ready_flags_vstore(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  TORCH_CHECK(n % 8 == 0, "active-row ready flags path requires N divisible by 8");
  return preallocated_nvfp4_gemm_tma_tile_scales_active_row_ready_flags_vstore_cuda(
      output,
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
      active_rows,
      m,
      k,
      n,
      worker_blocks,
      sleep_ns);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_active_row_ready_queue_vstore(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  TORCH_CHECK(active_row_offsets.is_cuda() && active_row_offsets.scalar_type() == at::kInt,
              "active_row_offsets must be CUDA int32");
  TORCH_CHECK(active_rows_local.is_cuda() && active_rows_local.scalar_type() == at::kInt,
              "active_rows_local must be CUDA int32");
  TORCH_CHECK(active_row_offsets.numel() == ((m + 127) / 128) + 1,
              "active_row_offsets must have one entry per M tile plus one");
  TORCH_CHECK(worker_threads == 128 || worker_threads == 256,
              "worker_threads must be 128 or 256");
  TORCH_CHECK(n % 8 == 0, "active-row ready queue path requires N divisible by 8");
  return preallocated_nvfp4_gemm_tma_tile_scales_active_row_ready_queue_vstore_cuda(
      output,
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
      active_row_offsets,
      active_rows_local,
      m,
      k,
      n,
      worker_blocks,
      worker_threads,
      sleep_ns);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_active_mtile_ready_queue_vstore(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  TORCH_CHECK(active_row_offsets.is_cuda() && active_row_offsets.scalar_type() == at::kInt,
              "active_row_offsets must be CUDA int32");
  TORCH_CHECK(active_rows_local.is_cuda() && active_rows_local.scalar_type() == at::kInt,
              "active_rows_local must be CUDA int32");
  TORCH_CHECK(active_row_offsets.numel() == ((m + 127) / 128) + 1,
              "active_row_offsets must have one entry per M tile plus one");
  TORCH_CHECK(worker_threads == 128 || worker_threads == 256,
              "worker_threads must be 128 or 256");
  TORCH_CHECK(mtile_slices >= 1, "mtile_slices must be >= 1");
  TORCH_CHECK(n % 8 == 0, "active-M-tile ready queue path requires N divisible by 8");
  return preallocated_nvfp4_gemm_tma_tile_scales_active_mtile_ready_queue_vstore_cuda(
      output,
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
      active_row_offsets,
      active_rows_local,
      m,
      k,
      n,
      worker_blocks,
      worker_threads,
      sleep_ns,
      mtile_slices);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_direct_add_active(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  return preallocated_nvfp4_gemm_tma_tile_scales_direct_add_active_cuda(
      output,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
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
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_direct_add_active_4wg(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  return preallocated_nvfp4_gemm_tma_tile_scales_direct_add_active_4wg_cuda(
      output,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
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
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_direct_smem_active(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  return preallocated_nvfp4_gemm_tma_tile_scales_direct_smem_active_cuda(
      output,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
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
      n,
      direct_smem_mode);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_direct_smem_delta_active(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  CHECK_CUDA(active_row_offsets);
  CHECK_CUDA(active_rows);
  CHECK_CONTIGUOUS(active_row_offsets);
  CHECK_CONTIGUOUS(active_rows);
  TORCH_CHECK(active_row_offsets.scalar_type() == at::kInt,
              "active_row_offsets must be int32");
  TORCH_CHECK(active_rows.scalar_type() == at::kInt, "active_rows must be int32");
  TORCH_CHECK(active_row_offsets.numel() == (m + 127) / 128 + 1,
              "active_row_offsets must be tile_m_count+1 for BM=128");
  TORCH_CHECK(active_rows.numel() <= m * 2,
              "active_rows is unexpectedly large");
  TORCH_CHECK(direct_smem_mode == 11 || direct_smem_mode == 12,
              "direct_smem_delta_active currently expects direct_smem_mode=11 or 12");
  return preallocated_nvfp4_gemm_tma_tile_scales_direct_smem_delta_active_cuda(
      output,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      delta_output,
      active_row_offsets,
      active_rows,
      m,
      k,
      n,
      direct_smem_mode);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_active(
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
  check_output(output, m, n);
  check_probe_sink(probe_sink, probe_counter, probe_active_mblocks, m, n, sparse_warpgroups);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_active_cuda(
      output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
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
      n,
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_active_sidewarp(
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
  check_output(output, m, n);
  check_probe_sink(probe_sink, probe_counter, probe_active_mblocks, m, n, sparse_warpgroups);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_active_sidewarp_cuda(
      output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
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
      n,
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  check_probe_sink(probe_sink, probe_counter, probe_active_mblocks, m, n, sparse_warpgroups);
  const int64_t tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= tiles * 4,
              "write-active probe_sink needs at least 4 slots per tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
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
      n,
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_sidewarp(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  check_probe_sink(probe_sink, probe_counter, probe_active_mblocks, m, n, sparse_warpgroups);
  const int64_t tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= tiles * 4,
              "write-active sidewarp probe_sink needs at least 4 slots per tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_sidewarp_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
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
      n,
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_rowblock_sidewarp(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  CHECK_CUDA(probe_sink);
  CHECK_CUDA(probe_counter);
  CHECK_CONTIGUOUS(probe_sink);
  CHECK_CONTIGUOUS(probe_counter);
  TORCH_CHECK(probe_sink.scalar_type() == at::kFloat, "probe_sink must be fp32");
  TORCH_CHECK(probe_counter.scalar_type() == at::kInt, "probe_counter must be int32");
  TORCH_CHECK(probe_counter.numel() >= 1, "probe_counter must have at least one element");
  const int64_t sparse_warpgroups_abs =
      sparse_warpgroups < 0 ? -sparse_warpgroups : sparse_warpgroups;
  TORCH_CHECK(sparse_warpgroups_abs == 1,
              "rowblock same-CTA sidewarp path currently supports exactly one sparse warpgroup");
  const int64_t dense_tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= dense_tiles * 4,
              "rowblock sidewarp probe_sink needs at least 4 slots per dense tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_rowblock_payload_inputs(row_offsets, row_ks, row_values, b_comp, m, k, n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_rowblock_sidewarp_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
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
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_sidewarp(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  CHECK_CUDA(probe_sink);
  CHECK_CUDA(probe_counter);
  CHECK_CUDA(active_rowblocks);
  CHECK_CONTIGUOUS(probe_sink);
  CHECK_CONTIGUOUS(probe_counter);
  CHECK_CONTIGUOUS(active_rowblocks);
  TORCH_CHECK(probe_sink.scalar_type() == at::kFloat, "probe_sink must be fp32");
  TORCH_CHECK(probe_counter.scalar_type() == at::kInt, "probe_counter must be int32");
  TORCH_CHECK(active_rowblocks.scalar_type() == at::kInt,
              "active_rowblocks must be int32");
  TORCH_CHECK(probe_counter.numel() >= 1, "probe_counter must have at least one element");
  TORCH_CHECK(active_rowblocks.numel() >= 1, "active_rowblocks must be non-empty");
  TORCH_CHECK(active_rowblocks.numel() <= (m + 7) / 8,
              "active_rowblocks is unexpectedly large");
  const int64_t sparse_warpgroups_abs =
      sparse_warpgroups < 0 ? -sparse_warpgroups : sparse_warpgroups;
  TORCH_CHECK(sparse_warpgroups_abs == 1,
              "active-rowblock same-CTA sidewarp path currently supports exactly one sparse warpgroup");
  const int64_t dense_tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= dense_tiles * 4,
              "active-rowblock sidewarp probe_sink needs at least 4 slots per dense tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_rowblock_payload_inputs(row_offsets, row_ks, row_values, b_comp, m, k, n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_sidewarp_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      active_rowblocks,
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
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_static_persistent_sidewarp(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  CHECK_CUDA(probe_sink);
  CHECK_CUDA(probe_counter);
  CHECK_CUDA(active_rowblocks);
  CHECK_CUDA(active_rows);
  CHECK_CONTIGUOUS(probe_sink);
  CHECK_CONTIGUOUS(probe_counter);
  CHECK_CONTIGUOUS(active_rowblocks);
  CHECK_CONTIGUOUS(active_rows);
  TORCH_CHECK(probe_sink.scalar_type() == at::kFloat, "probe_sink must be fp32");
  TORCH_CHECK(probe_counter.scalar_type() == at::kInt, "probe_counter must be int32");
  TORCH_CHECK(active_rowblocks.scalar_type() == at::kInt,
              "active_rowblocks must be int32");
  TORCH_CHECK(active_rows.scalar_type() == at::kInt,
              "active_rows must be int32");
  TORCH_CHECK(probe_counter.numel() >= 1, "probe_counter must have at least one element");
  TORCH_CHECK(active_rowblocks.numel() >= 1, "active_rowblocks must be non-empty");
  TORCH_CHECK(active_rowblocks.numel() <= (m + 7) / 8,
              "active_rowblocks is unexpectedly large");
  check_active_rows(active_rows, m);
  const int64_t sparse_warpgroups_abs =
      sparse_warpgroups < 0 ? -sparse_warpgroups : sparse_warpgroups;
  TORCH_CHECK(sparse_warpgroups_abs == 1,
              "active-rowblock static persistent sidewarp path currently supports exactly one sparse warpgroup");
  const int64_t dense_tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= dense_tiles * 4,
              "active-rowblock static persistent sidewarp probe_sink needs at least 4 slots per dense tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_rowblock_payload_inputs(row_offsets, row_ks, row_values, b_comp, m, k, n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_static_persistent_sidewarp_cuda(
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
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_active_rowblock_static_persistent_sidewarp(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  CHECK_CUDA(probe_sink);
  CHECK_CUDA(probe_counter);
  CHECK_CUDA(active_rowblocks);
  CHECK_CUDA(active_rows);
  CHECK_CONTIGUOUS(probe_sink);
  CHECK_CONTIGUOUS(probe_counter);
  CHECK_CONTIGUOUS(active_rowblocks);
  CHECK_CONTIGUOUS(active_rows);
  TORCH_CHECK(probe_sink.scalar_type() == at::kFloat, "probe_sink must be fp32");
  TORCH_CHECK(probe_counter.scalar_type() == at::kInt, "probe_counter must be int32");
  TORCH_CHECK(active_rowblocks.scalar_type() == at::kInt,
              "active_rowblocks must be int32");
  TORCH_CHECK(active_rows.scalar_type() == at::kInt,
              "active_rows must be int32");
  TORCH_CHECK(probe_counter.numel() >= 1, "probe_counter must have at least one element");
  TORCH_CHECK(active_rowblocks.numel() >= 1, "active_rowblocks must be non-empty");
  TORCH_CHECK(active_rowblocks.numel() <= (m + 7) / 8,
              "active_rowblocks is unexpectedly large");
  check_active_rows(active_rows, m);
  const int64_t sparse_warpgroups_abs =
      sparse_warpgroups < 0 ? -sparse_warpgroups : sparse_warpgroups;
  TORCH_CHECK(sparse_warpgroups_abs == 1,
              "active-rowblock static persistent sidewarp no-store path currently supports exactly one sparse warpgroup");
  const int64_t dense_tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= dense_tiles * 4,
              "active-rowblock static persistent sidewarp probe_sink needs at least 4 slots per dense tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_rowblock_payload_inputs(row_offsets, row_ks, row_values, b_comp, m, k, n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_active_rowblock_static_persistent_sidewarp_cuda(
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
      sparse_warpgroups);
}

namespace {

void check_warp256_active_rowblock_static_persistent_sidewarp_args(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  CHECK_CUDA(probe_sink);
  CHECK_CUDA(probe_counter);
  CHECK_CUDA(active_rowblocks);
  CHECK_CUDA(active_rows);
  CHECK_CONTIGUOUS(probe_sink);
  CHECK_CONTIGUOUS(probe_counter);
  CHECK_CONTIGUOUS(active_rowblocks);
  CHECK_CONTIGUOUS(active_rows);
  TORCH_CHECK(probe_sink.scalar_type() == at::kFloat, "probe_sink must be fp32");
  TORCH_CHECK(probe_counter.scalar_type() == at::kInt, "probe_counter must be int32");
  TORCH_CHECK(active_rowblocks.scalar_type() == at::kInt,
              "active_rowblocks must be int32");
  TORCH_CHECK(active_rows.scalar_type() == at::kInt,
              "active_rows must be int32");
  TORCH_CHECK(probe_counter.numel() >= 1, "probe_counter must have at least one element");
  TORCH_CHECK(active_rowblocks.numel() >= 1, "active_rowblocks must be non-empty");
  TORCH_CHECK(active_rowblocks.numel() <= (m + 7) / 8,
              "active_rowblocks is unexpectedly large");
  check_active_rows(active_rows, m);
  const int64_t sparse_warpgroups_abs =
      sparse_warpgroups < 0 ? -sparse_warpgroups : sparse_warpgroups;
  TORCH_CHECK(sparse_warpgroups_abs == 1,
              "active-rowblock static persistent warp256 sidewarp path currently supports exactly one sparse warpgroup");
  const int64_t dense_tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= dense_tiles * 4,
              "active-rowblock static persistent warp256 sidewarp probe_sink needs at least 4 slots per dense tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_rowblock_payload_inputs(row_offsets, row_ks, row_values, b_comp, m, k, n);
}

}  // namespace

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_warp256_active_rowblock_static_persistent_sidewarp(
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
  check_warp256_active_rowblock_static_persistent_sidewarp_args(
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
      sparse_warpgroups);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_warp256_active_rowblock_static_persistent_sidewarp_cuda(
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
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_warp256_active_rowblock_static_persistent_sidewarp(
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
  check_warp256_active_rowblock_static_persistent_sidewarp_args(
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
      sparse_warpgroups);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_warp256_active_rowblock_static_persistent_sidewarp_cuda(
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
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_prefetch_active_rowblock_static_persistent_sidewarp(
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
  check_warp256_active_rowblock_static_persistent_sidewarp_args(
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
      sparse_warpgroups);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_prefetch_active_rowblock_static_persistent_sidewarp_cuda(
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
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_prefetch_active_rowblock_static_persistent_sidewarp(
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
  check_warp256_active_rowblock_static_persistent_sidewarp_args(
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
      sparse_warpgroups);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_prefetch_active_rowblock_static_persistent_sidewarp_cuda(
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
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_prefetch_compact_active_rowblock_static_persistent_sidewarp(
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
  check_output(output, m, n);
  check_output(delta_output, active_rows.numel(), n);
  CHECK_CUDA(probe_sink);
  CHECK_CUDA(probe_counter);
  CHECK_CUDA(active_rowblocks);
  CHECK_CUDA(active_rows);
  CHECK_CONTIGUOUS(probe_sink);
  CHECK_CONTIGUOUS(probe_counter);
  CHECK_CONTIGUOUS(active_rowblocks);
  CHECK_CONTIGUOUS(active_rows);
  TORCH_CHECK(probe_sink.scalar_type() == at::kFloat, "probe_sink must be fp32");
  TORCH_CHECK(probe_counter.scalar_type() == at::kInt, "probe_counter must be int32");
  TORCH_CHECK(active_rowblocks.scalar_type() == at::kInt,
              "active_rowblocks must be int32");
  TORCH_CHECK(active_rows.scalar_type() == at::kInt,
              "active_rows must be int32");
  TORCH_CHECK(probe_counter.numel() >= 1, "probe_counter must have at least one element");
  TORCH_CHECK(active_rowblocks.numel() >= 1, "active_rowblocks must be non-empty");
  TORCH_CHECK(active_rowblocks.numel() <= (m + 7) / 8,
              "active_rowblocks is unexpectedly large");
  check_active_rows(active_rows, m);
  const int64_t sparse_warpgroups_abs =
      sparse_warpgroups < 0 ? -sparse_warpgroups : sparse_warpgroups;
  TORCH_CHECK(sparse_warpgroups_abs == 1 || sparse_warpgroups_abs == 2,
              "active-row compact persistent path supports one or two sparse warpgroups");
  TORCH_CHECK(probe_sink.numel() >= 1,
              "active-row compact persistent path needs one dummy sink element");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_rowblock_payload_inputs(row_offsets, row_ks, row_values, b_comp, m, k, n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_prefetch_compact_active_rowblock_static_persistent_sidewarp_cuda(
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
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_persistent_sidewarp(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  CHECK_CUDA(probe_sink);
  CHECK_CUDA(probe_counter);
  CHECK_CUDA(active_rowblocks);
  CHECK_CONTIGUOUS(probe_sink);
  CHECK_CONTIGUOUS(probe_counter);
  CHECK_CONTIGUOUS(active_rowblocks);
  TORCH_CHECK(probe_sink.scalar_type() == at::kFloat, "probe_sink must be fp32");
  TORCH_CHECK(probe_counter.scalar_type() == at::kInt, "probe_counter must be int32");
  TORCH_CHECK(active_rowblocks.scalar_type() == at::kInt,
              "active_rowblocks must be int32");
  TORCH_CHECK(probe_counter.numel() >= 1, "probe_counter must have at least one element");
  TORCH_CHECK(active_rowblocks.numel() >= 1, "active_rowblocks must be non-empty");
  TORCH_CHECK(active_rowblocks.numel() <= ((m + 7) / 8) * ((n + 127) / 128) * 4,
              "active_rowblocks/task_records is unexpectedly large");
  const int64_t sparse_warpgroups_abs =
      sparse_warpgroups < 0 ? -sparse_warpgroups : sparse_warpgroups;
  TORCH_CHECK(sparse_warpgroups_abs == 1,
              "persistent active-rowblock same-CTA sidewarp path currently supports exactly one sparse warpgroup");
  TORCH_CHECK(probe_sink.numel() >= 4,
              "persistent active-rowblock sidewarp probe_sink needs at least 4 slots");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_rowblock_payload_inputs(row_offsets, row_ks, row_values, b_comp, m, k, n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_persistent_sidewarp_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      active_rowblocks,
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
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_wg3_ready_active_direct_add(
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
  check_output(output, m, n);
  CHECK_CUDA(ready_flags);
  CHECK_CUDA(probe_sink);
  CHECK_CUDA(probe_counter);
  CHECK_CONTIGUOUS(ready_flags);
  CHECK_CONTIGUOUS(probe_sink);
  CHECK_CONTIGUOUS(probe_counter);
  TORCH_CHECK(ready_flags.scalar_type() == at::kInt, "ready_flags must be int32");
  TORCH_CHECK(probe_sink.scalar_type() == at::kFloat, "probe_sink must be fp32");
  TORCH_CHECK(probe_counter.scalar_type() == at::kInt, "probe_counter must be int32");
  TORCH_CHECK(probe_counter.numel() >= 1, "probe_counter must have at least one element");
  const int64_t sparse_warpgroups_abs =
      sparse_warpgroups < 0 ? -sparse_warpgroups : sparse_warpgroups;
  TORCH_CHECK(sparse_warpgroups_abs == 1,
              "WG3 ready direct-add path currently supports exactly one sparse warpgroup");
  const int64_t dense_tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(ready_flags.numel() >= dense_tiles,
              "ready_flags needs one int32 per dense tile");
  TORCH_CHECK(probe_sink.numel() >= dense_tiles * 4,
              "probe_sink needs at least 4 slots per dense tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(
      row_offsets, row_ks, row_values, active_row_offsets, active_rows, b_comp, m, k, n);
  TORCH_CHECK(n % 8 == 0, "WG3 ready direct-add path requires N divisible by 8");
  return preallocated_nvfp4_gemm_tma_tile_scales_wg3_ready_active_direct_add_cuda(
      output,
      ready_flags,
      probe_sink,
      probe_counter,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
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
      n,
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_packed_rowblock_sidewarp(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  CHECK_CUDA(probe_sink);
  CHECK_CUDA(probe_counter);
  CHECK_CUDA(b_comp);
  CHECK_CONTIGUOUS(probe_sink);
  CHECK_CONTIGUOUS(probe_counter);
  CHECK_CONTIGUOUS(b_comp);
  TORCH_CHECK(probe_sink.scalar_type() == at::kFloat, "probe_sink must be fp32");
  TORCH_CHECK(probe_counter.scalar_type() == at::kInt, "probe_counter must be int32");
  TORCH_CHECK(b_comp.scalar_type() == at::kBFloat16, "b_comp must be bf16");
  TORCH_CHECK(probe_counter.numel() >= 1, "probe_counter must have at least one element");
  TORCH_CHECK(b_comp.size(0) == k && b_comp.size(1) == n, "b_comp shape mismatch");
  const int64_t sparse_warpgroups_abs =
      sparse_warpgroups < 0 ? -sparse_warpgroups : sparse_warpgroups;
  TORCH_CHECK(sparse_warpgroups_abs == 1,
              "packed rowblock same-CTA sidewarp path currently supports exactly one sparse warpgroup");
  const int64_t dense_tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= dense_tiles * 4,
              "packed rowblock sidewarp probe_sink needs at least 4 slots per dense tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_packed_rowblock_payload(
      active_rowblocks, packed_tile_offsets, packed_row_records, packed_entry_records, m);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_packed_rowblock_sidewarp_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      active_rowblocks,
      packed_tile_offsets,
      packed_row_records,
      packed_entry_records,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_kmajor(
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
  check_output(output, m, n);
  check_probe_sink(probe_sink, probe_counter, probe_active_mblocks, m, n, sparse_warpgroups);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_kmajor_cuda(
      output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_hybrid(
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
  check_output(output, m, n);
  check_probe_sink(probe_sink, probe_counter, probe_active_mblocks, m, n, sparse_warpgroups);
  const int64_t tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= tiles * 24,
              "hybrid probe_sink needs at least 24 slots per tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_hybrid_cuda(
      output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_incta_hybrid(
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
  check_output(output, m, n);
  check_probe_sink(probe_sink, probe_counter, probe_active_mblocks, m, n, sparse_warpgroups);
  const int64_t tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= tiles * 8,
              "in-CTA hybrid probe_sink needs at least 8 slots per tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_incta_hybrid_cuda(
      output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_idlechunk_hybrid(
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
  check_output(output, m, n);
  TORCH_CHECK(group_budget > 0, "group_budget must be positive");
  check_probe_sink_slots(probe_sink, probe_counter, probe_active_mblocks, m, n, 2);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_idlechunk_hybrid_cuda(
      output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      group_budget);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_scheduler_hybrid(
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
    int64_t side_mode) {
  check_output(output, m, n);
  TORCH_CHECK(group_budget > 0, "group_budget must be positive");
  TORCH_CHECK(side_warps >= 1 && side_warps <= 4, "side_warps must be in [1, 4]");
  TORCH_CHECK(side_mode == 0 || side_mode == 1 || side_mode == 2 ||
                  side_mode == 3 || side_mode == 4 || side_mode == 5 ||
                  side_mode == 6 || side_mode == 7 || side_mode == 8 ||
                  side_mode == 9 || side_mode == 10 || side_mode == 11 ||
	                  side_mode == 12 || side_mode == 13 || side_mode == 14 ||
		                  side_mode == 15 || side_mode == 16 ||
		                  side_mode == 17 || side_mode == 18 ||
		                  side_mode == 19 || side_mode == 20 ||
		                  side_mode == 21 || side_mode == 22 ||
		                  side_mode == 23 || side_mode == 24 ||
		                  side_mode == 25 || side_mode == 26 ||
		                  side_mode == 27 || side_mode == 28 ||
		                  side_mode == 29 || side_mode == 30,
		              "side_mode must be 0..30");
  if (side_mode == 1) {
    TORCH_CHECK(side_warps == 2, "idle-stage side_mode currently requires side_warps=2");
  }
	  if (side_mode == 3) {
	    TORCH_CHECK(side_warps == 4, "extra-wg-idle side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 18) {
	    TORCH_CHECK(side_warps == 4, "colsplit local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 19) {
	    TORCH_CHECK(side_warps == 4, "heavyrowadd local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 20) {
	    TORCH_CHECK(side_warps == 4, "heavysmem local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 21) {
	    TORCH_CHECK(side_warps == 4, "superhot colsplit local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 22) {
	    TORCH_CHECK(side_warps == 4, "superhot atomic local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 23) {
	    TORCH_CHECK(side_warps == 4, "heavy pipeline local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 24) {
	    TORCH_CHECK(side_warps == 4, "side smem heavy local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 26 || side_mode == 27 || side_mode == 28 ||
	      side_mode == 29 || side_mode == 30) {
	    TORCH_CHECK(side_warps == 4, "stage-split local-delta side_mode currently requires side_warps=4");
	  }
  check_probe_sink_slots(probe_sink, probe_counter, probe_active_mblocks, m, n, side_warps);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_scheduler_hybrid_cuda(
      output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      group_budget,
      side_warps,
      side_mode);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_scheduler_hybrid_phase_trace(
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
    const at::Tensor& phase_trace,
    int64_t phase_trace_max_ctas,
    int64_t phase_trace_stride) {
  check_output(output, m, n);
  TORCH_CHECK(group_budget != 0, "group_budget must be non-zero");
  TORCH_CHECK(side_warps >= 1 && side_warps <= 4, "side_warps must be in [1, 4]");
  TORCH_CHECK(side_mode == 0 || side_mode == 1 || side_mode == 2 ||
                  side_mode == 3 || side_mode == 4 || side_mode == 5 ||
                  side_mode == 6 || side_mode == 7 || side_mode == 8 ||
                  side_mode == 9 || side_mode == 10 || side_mode == 11 ||
                  side_mode == 12 || side_mode == 13 || side_mode == 14 ||
                  side_mode == 15 || side_mode == 16 ||
                  side_mode == 17 || side_mode == 18 ||
                  side_mode == 19 || side_mode == 20 ||
                  side_mode == 21 || side_mode == 22 ||
                  side_mode == 23 || side_mode == 24 ||
                  side_mode == 25 || side_mode == 26 ||
                  side_mode == 27 || side_mode == 28 ||
                  side_mode == 29 || side_mode == 30,
              "side_mode must be 0..30");
  if (side_mode == 1) {
    TORCH_CHECK(side_warps == 2, "idle-stage side_mode currently requires side_warps=2");
  }
  if (side_mode == 3) {
    TORCH_CHECK(side_warps == 4, "extra-wg-idle side_mode currently requires side_warps=4");
  }
  if (side_mode == 18) {
    TORCH_CHECK(side_warps == 4, "colsplit local-delta side_mode currently requires side_warps=4");
  }
  if (side_mode == 19) {
    TORCH_CHECK(side_warps == 4, "heavyrowadd local-delta side_mode currently requires side_warps=4");
  }
  if (side_mode == 20) {
    TORCH_CHECK(side_warps == 4, "heavysmem local-delta side_mode currently requires side_warps=4");
  }
  if (side_mode == 21) {
    TORCH_CHECK(side_warps == 4, "superhot colsplit local-delta side_mode currently requires side_warps=4");
  }
  if (side_mode == 22) {
    TORCH_CHECK(side_warps == 4, "superhot atomic local-delta side_mode currently requires side_warps=4");
  }
  if (side_mode == 23) {
    TORCH_CHECK(side_warps == 4, "heavy pipeline local-delta side_mode currently requires side_warps=4");
  }
  if (side_mode == 24) {
    TORCH_CHECK(side_warps == 4, "side smem heavy local-delta side_mode currently requires side_warps=4");
  }
  if (side_mode == 26 || side_mode == 27 || side_mode == 28 ||
      side_mode == 29 || side_mode == 30) {
    TORCH_CHECK(side_warps == 4, "stage-split local-delta side_mode currently requires side_warps=4");
  }
  TORCH_CHECK(phase_trace.is_cuda(), "phase_trace must be a CUDA tensor");
  TORCH_CHECK(phase_trace.scalar_type() == at::kLong, "phase_trace must be int64");
  TORCH_CHECK(phase_trace.is_contiguous(), "phase_trace must be contiguous");
  TORCH_CHECK(phase_trace.dim() == 2, "phase_trace must be [ctas, slots]");
  TORCH_CHECK(phase_trace_stride >= 52, "phase_trace_stride must cover all trace slots");
  TORCH_CHECK(phase_trace_stride <= phase_trace.size(1), "phase_trace_stride exceeds trace width");
  TORCH_CHECK(phase_trace_max_ctas >= 0 && phase_trace_max_ctas <= phase_trace.size(0),
              "phase_trace_max_ctas out of range");
  check_probe_sink_slots(probe_sink, probe_counter, probe_active_mblocks, m, n, side_warps);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_scheduler_hybrid_cuda(
      output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      group_budget,
      side_warps,
      side_mode,
      &phase_trace,
      phase_trace_max_ctas,
      phase_trace_stride);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid(
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
    int64_t direct_delta_chunk_limit) {
  check_output(output, m, n);
  if (direct_delta_write_mode == 4) {
    check_compact_delta_output(delta_output, kmajor_entry_values.numel(), n);
  } else {
    check_output(delta_output, m, n);
  }
  TORCH_CHECK(group_budget > 0, "group_budget must be positive");
  TORCH_CHECK(direct_delta_chunk_limit >= 0, "direct_delta_chunk_limit must be non-negative");
  TORCH_CHECK(side_warps >= 1 && side_warps <= 4, "side_warps must be in [1, 4]");
  TORCH_CHECK(side_mode == 0 || side_mode == 1 || side_mode == 2 ||
                  side_mode == 3 || side_mode == 4 || side_mode == 5 ||
                  side_mode == 6 || side_mode == 7 || side_mode == 8 ||
                  side_mode == 9 || side_mode == 10 || side_mode == 11 ||
	                  side_mode == 12 || side_mode == 13 || side_mode == 14 ||
		                  side_mode == 15 || side_mode == 16 ||
		                  side_mode == 17 || side_mode == 18 ||
		                  side_mode == 19 || side_mode == 20 ||
		                  side_mode == 21 || side_mode == 22 ||
		                  side_mode == 23 || side_mode == 24 ||
		                  side_mode == 25 || side_mode == 26 ||
		                  side_mode == 27 || side_mode == 28 ||
		                  side_mode == 29 || side_mode == 30,
		              "side_mode must be 0..30");
  if (side_mode == 1) {
    TORCH_CHECK(side_warps == 2, "idle-stage side_mode currently requires side_warps=2");
  }
	  if (side_mode == 3) {
	    TORCH_CHECK(side_warps == 4, "extra-wg-idle side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 18) {
	    TORCH_CHECK(side_warps == 4, "colsplit local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 19) {
	    TORCH_CHECK(side_warps == 4, "heavyrowadd local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 20) {
	    TORCH_CHECK(side_warps == 4, "heavysmem local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 21) {
	    TORCH_CHECK(side_warps == 4, "superhot colsplit local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 22) {
	    TORCH_CHECK(side_warps == 4, "superhot atomic local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 23) {
	    TORCH_CHECK(side_warps == 4, "heavy pipeline local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 24) {
	    TORCH_CHECK(side_warps == 4, "side smem heavy local-delta side_mode currently requires side_warps=4");
	  }
	  if (side_mode == 26 || side_mode == 27 || side_mode == 28 ||
	      side_mode == 29 || side_mode == 30) {
	    TORCH_CHECK(side_warps == 4, "stage-split local-delta side_mode currently requires side_warps=4");
	  }
  check_probe_sink_slots(probe_sink, probe_counter, probe_active_mblocks, m, n, side_warps);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
	  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_cuda(
	      output,
	      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      group_budget,
      direct_delta_write_mode,
      side_warps,
	      side_mode,
	      direct_delta_chunk_limit);
	}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_tile_meta(
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
    const at::Tensor& tile_group_starts,
    const at::Tensor& tile_group_counts,
    const at::Tensor& tile_group_meta) {
  check_output(output, m, n);
  if (direct_delta_write_mode == 4) {
    check_compact_delta_output(delta_output, kmajor_entry_values.numel(), n);
  } else {
    check_output(delta_output, m, n);
  }
  TORCH_CHECK(group_budget > 0, "group_budget must be positive");
  TORCH_CHECK(direct_delta_chunk_limit >= 0, "direct_delta_chunk_limit must be non-negative");
  TORCH_CHECK(side_warps >= 1 && side_warps <= 4, "side_warps must be in [1, 4]");
  TORCH_CHECK(side_mode >= 0 && side_mode <= 30, "side_mode must be 0..30");
  if (side_mode == 1) {
    TORCH_CHECK(side_warps == 2, "idle-stage side_mode currently requires side_warps=2");
  }
  if (side_mode == 3 || side_mode == 18 || side_mode == 19 ||
      side_mode == 20 || side_mode == 21 || side_mode == 22 ||
      side_mode == 23 || side_mode == 24 || side_mode == 26 || side_mode == 27 ||
      side_mode == 28 || side_mode == 29 || side_mode == 30) {
    TORCH_CHECK(side_warps == 4, "this local-delta side_mode currently requires side_warps=4");
  }
  check_probe_sink_slots(probe_sink, probe_counter, probe_active_mblocks, m, n, side_warps);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  check_kmajor_tile_metadata(tile_group_starts, tile_group_counts, tile_group_meta, m);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      group_budget,
      direct_delta_write_mode,
      side_warps,
      side_mode,
      direct_delta_chunk_limit,
      nullptr,
      0,
      0,
      nullptr,
      nullptr,
      nullptr,
      0,
      &tile_group_starts,
      &tile_group_counts,
      &tile_group_meta);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_packed(
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
    const at::Tensor& packed_tile_offsets,
    const at::Tensor& packed_row_records,
    const at::Tensor& packed_entry_records,
    int64_t packed_payload_mode) {
  check_output(output, m, n);
  if (direct_delta_write_mode == 4) {
    check_compact_delta_output(delta_output, kmajor_entry_values.numel(), n);
  } else {
    check_output(delta_output, m, n);
  }
  TORCH_CHECK(group_budget > 0, "group_budget must be positive");
  TORCH_CHECK(direct_delta_chunk_limit >= 0, "direct_delta_chunk_limit must be non-negative");
  TORCH_CHECK(side_warps >= 1 && side_warps <= 4, "side_warps must be in [1, 4]");
  TORCH_CHECK(side_mode >= 0 && side_mode <= 30, "side_mode must be 0..30");
  if (side_mode == 1) {
    TORCH_CHECK(side_warps == 2, "idle-stage side_mode currently requires side_warps=2");
  }
  if (side_mode == 3 || side_mode == 18 || side_mode == 19 ||
      side_mode == 20 || side_mode == 21 || side_mode == 22 ||
      side_mode == 23 || side_mode == 24 || side_mode == 26 || side_mode == 27 ||
      side_mode == 28 || side_mode == 29 || side_mode == 30) {
    TORCH_CHECK(side_warps == 4, "this local-delta side_mode currently requires side_warps=4");
  }
  check_probe_sink_slots(probe_sink, probe_counter, probe_active_mblocks, m, n, side_warps);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  check_packed_local_delta_payload(
      packed_tile_offsets, packed_row_records, packed_entry_records, m);
  TORCH_CHECK(packed_payload_mode == 1 || packed_payload_mode == 2 ||
                  packed_payload_mode == 3,
              "packed_payload_mode must be 1, 2, or 3");
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      group_budget,
      direct_delta_write_mode,
      side_warps,
      side_mode,
      direct_delta_chunk_limit,
      nullptr,
      0,
      0,
      &packed_tile_offsets,
      &packed_row_records,
      &packed_entry_records,
      packed_payload_mode);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_packed_tile_meta(
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
    const at::Tensor& packed_tile_offsets,
    const at::Tensor& packed_row_records,
    const at::Tensor& packed_entry_records,
    int64_t packed_payload_mode,
    const at::Tensor& tile_group_starts,
    const at::Tensor& tile_group_counts,
    const at::Tensor& tile_group_meta) {
  check_output(output, m, n);
  if (direct_delta_write_mode == 4) {
    check_compact_delta_output(delta_output, kmajor_entry_values.numel(), n);
  } else {
    check_output(delta_output, m, n);
  }
  TORCH_CHECK(group_budget > 0, "group_budget must be positive");
  TORCH_CHECK(direct_delta_chunk_limit >= 0, "direct_delta_chunk_limit must be non-negative");
  TORCH_CHECK(side_warps >= 1 && side_warps <= 4, "side_warps must be in [1, 4]");
  TORCH_CHECK(side_mode >= 0 && side_mode <= 30, "side_mode must be 0..30");
  if (side_mode == 1) {
    TORCH_CHECK(side_warps == 2, "idle-stage side_mode currently requires side_warps=2");
  }
  if (side_mode == 3 || side_mode == 18 || side_mode == 19 ||
      side_mode == 20 || side_mode == 21 || side_mode == 22 ||
      side_mode == 23 || side_mode == 24 || side_mode == 26 || side_mode == 27 ||
      side_mode == 28 || side_mode == 29 || side_mode == 30) {
    TORCH_CHECK(side_warps == 4, "this local-delta side_mode currently requires side_warps=4");
  }
  check_probe_sink_slots(probe_sink, probe_counter, probe_active_mblocks, m, n, side_warps);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  check_packed_local_delta_payload(
      packed_tile_offsets, packed_row_records, packed_entry_records, m);
  TORCH_CHECK(packed_payload_mode == 1 || packed_payload_mode == 2 ||
                  packed_payload_mode == 3,
              "packed_payload_mode must be 1, 2, or 3");
  check_kmajor_tile_metadata(tile_group_starts, tile_group_counts, tile_group_meta, m);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      group_budget,
      direct_delta_write_mode,
      side_warps,
      side_mode,
      direct_delta_chunk_limit,
      nullptr,
      0,
      0,
      &packed_tile_offsets,
      &packed_row_records,
      &packed_entry_records,
      packed_payload_mode,
      &tile_group_starts,
      &tile_group_counts,
      &tile_group_meta);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace(
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
    const at::Tensor& phase_trace,
    int64_t phase_trace_max_ctas,
    int64_t phase_trace_stride) {
  TORCH_CHECK(phase_trace.is_cuda(), "phase_trace must be a CUDA tensor");
  TORCH_CHECK(phase_trace.scalar_type() == at::kLong, "phase_trace must be int64");
  TORCH_CHECK(phase_trace.is_contiguous(), "phase_trace must be contiguous");
  TORCH_CHECK(phase_trace.dim() == 2, "phase_trace must be [ctas, slots]");
  TORCH_CHECK(phase_trace_stride >= 52, "phase_trace_stride must be at least 52");
  TORCH_CHECK(phase_trace_max_ctas >= 0, "phase_trace_max_ctas must be non-negative");
  TORCH_CHECK(phase_trace_max_ctas <= phase_trace.size(0),
              "phase_trace_max_ctas exceeds phase_trace rows");
  TORCH_CHECK(phase_trace_stride <= phase_trace.size(1),
              "phase_trace_stride exceeds phase_trace columns");
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      group_budget,
      direct_delta_write_mode,
      side_warps,
      side_mode,
      direct_delta_chunk_limit,
      &phase_trace,
      phase_trace_max_ctas,
      phase_trace_stride);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace_tile_meta(
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
    const at::Tensor& phase_trace,
    int64_t phase_trace_max_ctas,
    int64_t phase_trace_stride,
    const at::Tensor& tile_group_starts,
    const at::Tensor& tile_group_counts,
    const at::Tensor& tile_group_meta) {
  TORCH_CHECK(phase_trace.is_cuda(), "phase_trace must be a CUDA tensor");
  TORCH_CHECK(phase_trace.scalar_type() == at::kLong, "phase_trace must be int64");
  TORCH_CHECK(phase_trace.is_contiguous(), "phase_trace must be contiguous");
  TORCH_CHECK(phase_trace.dim() == 2, "phase_trace must be [ctas, slots]");
  TORCH_CHECK(phase_trace_stride >= 52, "phase_trace_stride must be at least 52");
  TORCH_CHECK(phase_trace_max_ctas >= 0, "phase_trace_max_ctas must be non-negative");
  TORCH_CHECK(phase_trace_max_ctas <= phase_trace.size(0),
              "phase_trace_max_ctas exceeds phase_trace rows");
  TORCH_CHECK(phase_trace_stride <= phase_trace.size(1),
              "phase_trace_stride exceeds phase_trace columns");
  check_kmajor_tile_metadata(tile_group_starts, tile_group_counts, tile_group_meta, m);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      group_budget,
      direct_delta_write_mode,
      side_warps,
      side_mode,
      direct_delta_chunk_limit,
      &phase_trace,
      phase_trace_max_ctas,
      phase_trace_stride,
      nullptr,
      nullptr,
      nullptr,
      0,
      &tile_group_starts,
      &tile_group_counts,
      &tile_group_meta);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace_packed(
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
    const at::Tensor& phase_trace,
    int64_t phase_trace_max_ctas,
    int64_t phase_trace_stride,
    const at::Tensor& packed_tile_offsets,
    const at::Tensor& packed_row_records,
    const at::Tensor& packed_entry_records,
    int64_t packed_payload_mode) {
  TORCH_CHECK(phase_trace.is_cuda(), "phase_trace must be a CUDA tensor");
  TORCH_CHECK(phase_trace.scalar_type() == at::kLong, "phase_trace must be int64");
  TORCH_CHECK(phase_trace.is_contiguous(), "phase_trace must be contiguous");
  TORCH_CHECK(phase_trace.dim() == 2, "phase_trace must be [ctas, slots]");
  TORCH_CHECK(phase_trace_stride >= 52, "phase_trace_stride must be at least 52");
  TORCH_CHECK(phase_trace_max_ctas >= 0, "phase_trace_max_ctas must be non-negative");
  TORCH_CHECK(phase_trace_max_ctas <= phase_trace.size(0),
              "phase_trace_max_ctas exceeds phase_trace rows");
  TORCH_CHECK(phase_trace_stride <= phase_trace.size(1),
              "phase_trace_stride exceeds phase_trace columns");
  check_packed_local_delta_payload(
      packed_tile_offsets, packed_row_records, packed_entry_records, m);
  TORCH_CHECK(packed_payload_mode == 1 || packed_payload_mode == 2 ||
                  packed_payload_mode == 3,
              "packed_payload_mode must be 1, 2, or 3");
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      group_budget,
      direct_delta_write_mode,
      side_warps,
      side_mode,
      direct_delta_chunk_limit,
      &phase_trace,
      phase_trace_max_ctas,
      phase_trace_stride,
      &packed_tile_offsets,
      &packed_row_records,
      &packed_entry_records,
      packed_payload_mode);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace_packed_tile_meta(
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
    const at::Tensor& phase_trace,
    int64_t phase_trace_max_ctas,
    int64_t phase_trace_stride,
    const at::Tensor& packed_tile_offsets,
    const at::Tensor& packed_row_records,
    const at::Tensor& packed_entry_records,
    int64_t packed_payload_mode,
    const at::Tensor& tile_group_starts,
    const at::Tensor& tile_group_counts,
    const at::Tensor& tile_group_meta) {
  TORCH_CHECK(phase_trace.is_cuda(), "phase_trace must be a CUDA tensor");
  TORCH_CHECK(phase_trace.scalar_type() == at::kLong, "phase_trace must be int64");
  TORCH_CHECK(phase_trace.is_contiguous(), "phase_trace must be contiguous");
  TORCH_CHECK(phase_trace.dim() == 2, "phase_trace must be [ctas, slots]");
  TORCH_CHECK(phase_trace_stride >= 52, "phase_trace_stride must be at least 52");
  TORCH_CHECK(phase_trace_max_ctas >= 0, "phase_trace_max_ctas must be non-negative");
  TORCH_CHECK(phase_trace_max_ctas <= phase_trace.size(0),
              "phase_trace_max_ctas exceeds phase_trace rows");
  TORCH_CHECK(phase_trace_stride <= phase_trace.size(1),
              "phase_trace_stride exceeds phase_trace columns");
  check_packed_local_delta_payload(
      packed_tile_offsets, packed_row_records, packed_entry_records, m);
  TORCH_CHECK(packed_payload_mode == 1 || packed_payload_mode == 2 ||
                  packed_payload_mode == 3,
              "packed_payload_mode must be 1, 2, or 3");
  check_kmajor_tile_metadata(tile_group_starts, tile_group_counts, tile_group_meta, m);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      group_budget,
      direct_delta_write_mode,
      side_warps,
      side_mode,
      direct_delta_chunk_limit,
      &phase_trace,
      phase_trace_max_ctas,
      phase_trace_stride,
      &packed_tile_offsets,
      &packed_row_records,
      &packed_entry_records,
      packed_payload_mode,
      &tile_group_starts,
      &tile_group_counts,
      &tile_group_meta);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_atomic(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  check_probe_sink(probe_sink, probe_counter, probe_active_mblocks, m, n, sparse_warpgroups);
  const int64_t tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= tiles * 8,
              "in-CTA kmajor atomic write probe_sink needs at least 8 slots per tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_atomic_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups,
      1);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_direct(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  check_probe_sink(probe_sink, probe_counter, probe_active_mblocks, m, n, sparse_warpgroups);
  const int64_t tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= tiles * 8,
              "in-CTA kmajor direct write probe_sink needs at least 8 slots per tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_atomic_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups,
      2);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_sharedacc(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  check_probe_sink(probe_sink, probe_counter, probe_active_mblocks, m, n, sparse_warpgroups);
  const int64_t tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= tiles * 8,
              "in-CTA kmajor sharedacc write probe_sink needs at least 8 slots per tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_atomic_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups,
      3);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_extrawg_kmajor_sharedacc(
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
  check_output(output, m, n);
  check_output(delta_output, m, n);
  check_probe_sink(probe_sink, probe_counter, probe_active_mblocks, m, n, sparse_warpgroups);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_extrawg_kmajor_sharedacc_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups);
}

at::Tensor preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_entry_direct(
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
  check_output(output, m, n);
  check_compact_delta_output(delta_output, kmajor_entry_values.numel(), n);
  check_probe_sink(probe_sink, probe_counter, probe_active_mblocks, m, n, sparse_warpgroups);
  const int64_t tiles = ((m + 127) / 128) * ((n + 127) / 128);
  TORCH_CHECK(probe_sink.numel() >= tiles * 8,
              "in-CTA kmajor entry-direct write probe_sink needs at least 8 slots per tile");
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_direct_add_inputs(row_offsets,
                          row_ks,
                          row_values,
                          active_row_offsets,
                          active_rows,
                          b_comp,
                          m,
                          k,
                          n);
  check_kmajor_probe_inputs(kmajor_group_offsets,
                            kmajor_group_ks,
                            kmajor_entry_offsets,
                            kmajor_entry_rows,
                            kmajor_entry_values,
                            b_comp,
                            probe_active_mblocks,
                            k,
                            n);
  return preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_atomic_cuda(
      output,
      delta_output,
      probe_sink,
      probe_counter,
      probe_active_mblocks,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      row_offsets,
      row_ks,
      row_values,
      active_row_offsets,
      active_rows,
      kmajor_group_offsets,
      kmajor_group_ks,
      kmajor_entry_offsets,
      kmajor_entry_rows,
      kmajor_entry_values,
      b_comp,
      m,
      k,
      n,
      sparse_warpgroups,
      4);
}

at::Tensor merge_entry_delta_active_rows(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& merge_row_offsets,
    const at::Tensor& merge_entry_indices) {
  check_merge_entry_delta_inputs(output,
                                 delta_entries,
                                 active_rows,
                                 merge_row_offsets,
                                 merge_entry_indices);
  return merge_entry_delta_active_rows_cuda(output,
                                           delta_entries,
                                           active_rows,
                                           merge_row_offsets,
                                           merge_entry_indices,
                                           output.size(1));
}

at::Tensor merge_entry_delta_active_rows_fastpath(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& merge_row_offsets,
    const at::Tensor& merge_entry_indices) {
  check_merge_entry_delta_inputs(output,
                                 delta_entries,
                                 active_rows,
                                 merge_row_offsets,
                                 merge_entry_indices);
  return merge_entry_delta_active_rows_fastpath_cuda(output,
                                                    delta_entries,
                                                    active_rows,
                                                    merge_row_offsets,
                                                    merge_entry_indices,
                                                    output.size(1));
}

at::Tensor merge_entry_delta_active_rows_vec8(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& merge_row_offsets,
    const at::Tensor& merge_entry_indices) {
  check_merge_entry_delta_inputs(output,
                                 delta_entries,
                                 active_rows,
                                 merge_row_offsets,
                                 merge_entry_indices);
  return merge_entry_delta_active_rows_vec8_cuda(output,
                                                delta_entries,
                                                active_rows,
                                                merge_row_offsets,
                                                merge_entry_indices,
                                                output.size(1));
}

at::Tensor merge_entry_delta_active_rows_chunk_prefix_vec8(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& merge_row_offsets,
    const at::Tensor& merge_entry_indices,
    int64_t chunk_cols,
    int64_t chunks_per_row) {
  check_merge_entry_delta_inputs(output,
                                 delta_entries,
                                 active_rows,
                                 merge_row_offsets,
                                 merge_entry_indices);
  TORCH_CHECK(chunk_cols > 0, "chunk_cols must be positive");
  TORCH_CHECK(chunk_cols % 8 == 0, "chunk_cols must be a multiple of 8");
  TORCH_CHECK(chunks_per_row > 0, "chunks_per_row must be positive");
  return merge_entry_delta_active_rows_chunk_prefix_vec8_cuda(output,
                                                             delta_entries,
                                                             active_rows,
                                                             merge_row_offsets,
                                                             merge_entry_indices,
                                                             output.size(1),
                                                             chunk_cols,
                                                             chunks_per_row);
}

at::Tensor merge_single_entry_delta_active_rows(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& entry_indices) {
  CHECK_CUDA(output);
  CHECK_CUDA(delta_entries);
  CHECK_CUDA(active_rows);
  CHECK_CUDA(entry_indices);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(delta_entries);
  CHECK_CONTIGUOUS(active_rows);
  CHECK_CONTIGUOUS(entry_indices);
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bf16");
  TORCH_CHECK(delta_entries.scalar_type() == at::kBFloat16, "delta_entries must be bf16");
  TORCH_CHECK(active_rows.scalar_type() == at::kInt, "active_rows must be int32");
  TORCH_CHECK(entry_indices.scalar_type() == at::kInt, "entry_indices must be int32");
  TORCH_CHECK(output.dim() == 2 && delta_entries.dim() == 2,
              "output and delta_entries must be 2D");
  TORCH_CHECK(output.size(1) == delta_entries.size(1),
              "output and delta_entries must have the same N");
  TORCH_CHECK(active_rows.dim() == 1 && entry_indices.dim() == 1,
              "active_rows and entry_indices must be 1D");
  TORCH_CHECK(active_rows.numel() == entry_indices.numel(),
              "active_rows and entry_indices length mismatch");
  return merge_single_entry_delta_active_rows_cuda(output,
                                                  delta_entries,
                                                  active_rows,
                                                  entry_indices,
                                                  output.size(1));
}

at::Tensor merge_double_entry_delta_active_rows(
    const at::Tensor& output,
    const at::Tensor& delta_entries,
    const at::Tensor& active_rows,
    const at::Tensor& entry0_indices,
    const at::Tensor& entry1_indices) {
  CHECK_CUDA(output);
  CHECK_CUDA(delta_entries);
  CHECK_CUDA(active_rows);
  CHECK_CUDA(entry0_indices);
  CHECK_CUDA(entry1_indices);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(delta_entries);
  CHECK_CONTIGUOUS(active_rows);
  CHECK_CONTIGUOUS(entry0_indices);
  CHECK_CONTIGUOUS(entry1_indices);
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bf16");
  TORCH_CHECK(delta_entries.scalar_type() == at::kBFloat16, "delta_entries must be bf16");
  TORCH_CHECK(active_rows.scalar_type() == at::kInt, "active_rows must be int32");
  TORCH_CHECK(entry0_indices.scalar_type() == at::kInt, "entry0_indices must be int32");
  TORCH_CHECK(entry1_indices.scalar_type() == at::kInt, "entry1_indices must be int32");
  TORCH_CHECK(output.dim() == 2 && delta_entries.dim() == 2,
              "output and delta_entries must be 2D");
  TORCH_CHECK(output.size(1) == delta_entries.size(1),
              "output and delta_entries must have the same N");
  TORCH_CHECK(active_rows.dim() == 1 && entry0_indices.dim() == 1 &&
                  entry1_indices.dim() == 1,
              "active_rows and entry indices must be 1D");
  TORCH_CHECK(active_rows.numel() == entry0_indices.numel() &&
                  active_rows.numel() == entry1_indices.numel(),
              "active_rows and entry indices length mismatch");
  return merge_double_entry_delta_active_rows_cuda(output,
                                                  delta_entries,
                                                  active_rows,
                                                  entry0_indices,
                                                  entry1_indices,
                                                  output.size(1));
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_tile_scales(
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
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  return nvfp4_dense_sparse_tma_value_payload_tile_scales_alloc_cuda(a_data,
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

at::Tensor preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
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

at::Tensor preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_sidecar(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  return nvfp4_dense_sparse_tma_value_payload_tile_scales_sidecar_cuda(
      output,
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
      n,
      sidecar_worker_blocks);
}

at::Tensor preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  return nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cuda(
      output,
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
      active_rows,
      m,
      k,
      n);
}

at::Tensor sparse_active_row_value_payload_vec8_store(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  return sparse_active_row_value_payload_vec8_store_cuda(
      output,
      outlier_values,
      weight_t_bf16,
      flat_indices,
      row_offsets,
      active_rows,
      k);
}

at::Tensor sparse_active_row_value_payload_vec8_store_vstore(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  return sparse_active_row_value_payload_vec8_store_vstore_cuda(
      output,
      outlier_values,
      weight_t_bf16,
      flat_indices,
      row_offsets,
      active_rows,
      k);
}

at::Tensor merge_full_delta_active_rows(
    const at::Tensor& output,
    const at::Tensor& delta_output,
    const at::Tensor& active_rows) {
  TORCH_CHECK(output.dim() == 2 && delta_output.dim() == 2,
              "output and delta_output must be 2D");
  CHECK_CUDA(output);
  CHECK_CUDA(delta_output);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(delta_output);
  TORCH_CHECK(output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  TORCH_CHECK(delta_output.scalar_type() == at::kBFloat16,
              "delta_output must be CUDA BF16");
  TORCH_CHECK(output.device() == delta_output.device(),
              "output and delta_output must be on the same device");
  TORCH_CHECK(output.sizes() == delta_output.sizes(),
              "output and delta_output shape mismatch");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  check_active_rows(active_rows, m);
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  return merge_full_delta_active_rows_cuda(output, delta_output, active_rows);
}

at::Tensor merge_compact_delta_active_rows(
    const at::Tensor& output,
    const at::Tensor& compact_delta,
    const at::Tensor& active_rows) {
  TORCH_CHECK(output.dim() == 2 && compact_delta.dim() == 2,
              "output and compact_delta must be 2D");
  CHECK_CUDA(output);
  CHECK_CUDA(compact_delta);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(compact_delta);
  TORCH_CHECK(output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  TORCH_CHECK(compact_delta.scalar_type() == at::kBFloat16,
              "compact_delta must be CUDA BF16");
  TORCH_CHECK(output.device() == compact_delta.device(),
              "output and compact_delta must be on the same device");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  check_active_rows(active_rows, m);
  TORCH_CHECK(compact_delta.size(0) == active_rows.numel(),
              "compact_delta rows must match active_rows");
  TORCH_CHECK(compact_delta.size(1) == n,
              "compact_delta N must match output N");
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  return merge_compact_delta_active_rows_cuda(output, compact_delta, active_rows);
}

at::Tensor merge_two_compact_delta_active_rows(
    const at::Tensor& output,
    const at::Tensor& first_delta,
    const at::Tensor& first_rows,
    const at::Tensor& second_delta,
    const at::Tensor& second_rows) {
  TORCH_CHECK(output.dim() == 2 && first_delta.dim() == 2 && second_delta.dim() == 2,
              "output and compact deltas must be 2D");
  CHECK_CUDA(output);
  CHECK_CUDA(first_delta);
  CHECK_CUDA(second_delta);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(first_delta);
  CHECK_CONTIGUOUS(second_delta);
  TORCH_CHECK(output.scalar_type() == at::kBFloat16 &&
                  first_delta.scalar_type() == at::kBFloat16 &&
                  second_delta.scalar_type() == at::kBFloat16,
              "output and compact deltas must be CUDA BF16");
  TORCH_CHECK(output.device() == first_delta.device() &&
                  output.device() == second_delta.device(),
              "output and compact deltas must be on the same device");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  check_active_rows(first_rows, m);
  check_active_rows(second_rows, m);
  TORCH_CHECK(first_delta.size(0) == first_rows.numel() &&
                  second_delta.size(0) == second_rows.numel(),
              "compact delta rows must match active rows");
  TORCH_CHECK(first_delta.size(1) == n && second_delta.size(1) == n,
              "compact delta N must match output N");
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  return merge_two_compact_delta_active_rows_cuda(
      output, first_delta, first_rows, second_delta, second_rows);
}

at::Tensor build_compact_dense_residual_active_rows(
    const at::Tensor& residual,
    const at::Tensor& row_values,
    const at::Tensor& row_ks,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  TORCH_CHECK(residual.dim() == 2, "residual must be 2D");
  CHECK_CUDA(residual);
  CHECK_CUDA(row_values);
  CHECK_CUDA(row_ks);
  CHECK_CUDA(row_offsets);
  CHECK_CUDA(active_rows);
  CHECK_CONTIGUOUS(residual);
  CHECK_CONTIGUOUS(row_values);
  CHECK_CONTIGUOUS(row_ks);
  CHECK_CONTIGUOUS(row_offsets);
  CHECK_CONTIGUOUS(active_rows);
  TORCH_CHECK(residual.scalar_type() == at::kBFloat16,
              "residual must be CUDA BF16");
  TORCH_CHECK(row_values.scalar_type() == at::kBFloat16,
              "row_values must be CUDA BF16");
  TORCH_CHECK(row_ks.scalar_type() == at::kInt,
              "row_ks must be int32");
  TORCH_CHECK(row_offsets.scalar_type() == at::kInt,
              "row_offsets must be int32");
  TORCH_CHECK(active_rows.scalar_type() == at::kInt,
              "active_rows must be int32");
  TORCH_CHECK(row_values.device() == residual.device() &&
              row_ks.device() == residual.device() &&
              row_offsets.device() == residual.device() &&
              active_rows.device() == residual.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(residual.size(0) == active_rows.numel(),
              "residual rows must match active_rows");
  TORCH_CHECK(residual.size(1) == k,
              "residual K must match k");
  TORCH_CHECK(k % 8 == 0, "k must be divisible by 8");
  TORCH_CHECK(row_offsets.numel() >= 2, "row_offsets must be non-empty CSR offsets");
  const int64_t m = row_offsets.numel() - 1;
  check_active_rows(active_rows, m);
  return build_compact_dense_residual_active_rows_cuda(
      residual, row_values, row_ks, row_offsets, active_rows, k);
}

std::vector<at::Tensor> build_padded_light_heavy_rows(
    const at::Tensor& row_offsets,
    int64_t heavy_threshold,
    int64_t heavy_capacity) {
  CHECK_CUDA(row_offsets);
  CHECK_CONTIGUOUS(row_offsets);
  TORCH_CHECK(row_offsets.scalar_type() == at::kInt,
              "row_offsets must be int32");
  TORCH_CHECK(row_offsets.numel() >= 2,
              "row_offsets must be non-empty CSR offsets");
  const int64_t rows = row_offsets.numel() - 1;
  TORCH_CHECK(heavy_threshold > 0 && heavy_threshold <= 2147483647LL,
              "heavy_threshold must fit a positive int32");
  TORCH_CHECK(heavy_capacity > 0 && heavy_capacity <= rows,
              "heavy_capacity must be in (0, rows]");
  return build_padded_light_heavy_rows_cuda(
      row_offsets, heavy_threshold, heavy_capacity);
}

at::Tensor sparse_kmajor_epin64_delta_store(
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
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  TORCH_CHECK(b_comp.dim() == 2 && b_comp.size(0) == k && b_comp.size(1) == n,
              "b_comp must be [K, N]");
  TORCH_CHECK(b_comp.is_cuda() && b_comp.scalar_type() == at::kBFloat16,
              "b_comp must be CUDA BF16");
  TORCH_CHECK(active_mblocks.is_cuda() && active_mblocks.scalar_type() == at::kInt,
              "active_mblocks must be CUDA int32");
  TORCH_CHECK(active_row_offsets.is_cuda() && active_row_offsets.scalar_type() == at::kInt,
              "active_row_offsets must be CUDA int32");
  TORCH_CHECK(active_rows.is_cuda() && active_rows.scalar_type() == at::kInt,
              "active_rows must be CUDA int32");
  TORCH_CHECK(group_offsets.is_cuda() && group_offsets.scalar_type() == at::kInt,
              "group_offsets must be CUDA int32");
  TORCH_CHECK(group_ks.is_cuda() && group_ks.scalar_type() == at::kInt,
              "group_ks must be CUDA int32");
  TORCH_CHECK(entry_offsets.is_cuda() && entry_offsets.scalar_type() == at::kInt,
              "entry_offsets must be CUDA int32");
  TORCH_CHECK(entry_rows.is_cuda() && entry_rows.scalar_type() == at::kInt,
              "entry_rows must be CUDA int32");
  TORCH_CHECK(entry_values.is_cuda() && entry_values.scalar_type() == at::kBFloat16,
              "entry_values must be CUDA BF16");
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  TORCH_CHECK(active_row_offsets.numel() >= (m + 127) / 128 + 1,
              "active_row_offsets is too small");
  TORCH_CHECK(group_offsets.numel() >= active_mblocks.numel() + 1,
              "group_offsets is too small");
  return sparse_kmajor_epin64_delta_store_cuda(
      output,
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
	      n);
}

at::Tensor sparse_kmajor_epin_delta_store(
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
    int64_t k,
    int64_t epin) {
  TORCH_CHECK(epin == 32 || epin == 64,
              "epin must be one of 32, 64");
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  TORCH_CHECK(b_comp.dim() == 2 && b_comp.size(0) == k && b_comp.size(1) == n,
              "b_comp must be [K, N]");
  TORCH_CHECK(b_comp.is_cuda() && b_comp.scalar_type() == at::kBFloat16,
              "b_comp must be CUDA BF16");
  TORCH_CHECK(active_mblocks.is_cuda() && active_mblocks.scalar_type() == at::kInt,
              "active_mblocks must be CUDA int32");
  TORCH_CHECK(active_row_offsets.is_cuda() && active_row_offsets.scalar_type() == at::kInt,
              "active_row_offsets must be CUDA int32");
  TORCH_CHECK(active_rows.is_cuda() && active_rows.scalar_type() == at::kInt,
              "active_rows must be CUDA int32");
  TORCH_CHECK(group_offsets.is_cuda() && group_offsets.scalar_type() == at::kInt,
              "group_offsets must be CUDA int32");
  TORCH_CHECK(group_ks.is_cuda() && group_ks.scalar_type() == at::kInt,
              "group_ks must be CUDA int32");
  TORCH_CHECK(entry_offsets.is_cuda() && entry_offsets.scalar_type() == at::kInt,
              "entry_offsets must be CUDA int32");
  TORCH_CHECK(entry_rows.is_cuda() && entry_rows.scalar_type() == at::kInt,
              "entry_rows must be CUDA int32");
  TORCH_CHECK(entry_values.is_cuda() && entry_values.scalar_type() == at::kBFloat16,
              "entry_values must be CUDA BF16");
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  TORCH_CHECK(active_row_offsets.numel() >= (m + 127) / 128 + 1,
              "active_row_offsets is too small");
  TORCH_CHECK(group_offsets.numel() >= active_mblocks.numel() + 1,
              "group_offsets is too small");
  return sparse_kmajor_epin_delta_store_cuda(
      output,
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
      epin);
}

at::Tensor sparse_kmajor_epin64_direct_store(
    const at::Tensor& output,
    const at::Tensor& active_mblocks,
    const at::Tensor& group_offsets,
    const at::Tensor& group_ks,
    const at::Tensor& entry_offsets,
    const at::Tensor& entry_rows,
    const at::Tensor& entry_values,
    const at::Tensor& b_comp,
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  TORCH_CHECK(b_comp.dim() == 2 && b_comp.size(0) == k && b_comp.size(1) == n,
              "b_comp must be [K, N]");
  TORCH_CHECK(b_comp.is_cuda() && b_comp.scalar_type() == at::kBFloat16,
              "b_comp must be CUDA BF16");
  TORCH_CHECK(active_mblocks.is_cuda() && active_mblocks.scalar_type() == at::kInt,
              "active_mblocks must be CUDA int32");
  TORCH_CHECK(group_offsets.is_cuda() && group_offsets.scalar_type() == at::kInt,
              "group_offsets must be CUDA int32");
  TORCH_CHECK(group_ks.is_cuda() && group_ks.scalar_type() == at::kInt,
              "group_ks must be CUDA int32");
  TORCH_CHECK(entry_offsets.is_cuda() && entry_offsets.scalar_type() == at::kInt,
              "entry_offsets must be CUDA int32");
  TORCH_CHECK(entry_rows.is_cuda() && entry_rows.scalar_type() == at::kInt,
              "entry_rows must be CUDA int32");
  TORCH_CHECK(entry_values.is_cuda() && entry_values.scalar_type() == at::kBFloat16,
              "entry_values must be CUDA BF16");
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  TORCH_CHECK(group_offsets.numel() >= active_mblocks.numel() + 1,
              "group_offsets is too small");
  return sparse_kmajor_epin64_direct_store_cuda(
      output,
      active_mblocks,
      group_offsets,
      group_ks,
      entry_offsets,
      entry_rows,
      entry_values,
      b_comp,
      m,
      k,
      n);
}

at::Tensor sparse_kmajor_serial_group_inplace(
    const at::Tensor& output,
    const at::Tensor& active_mblocks,
    const at::Tensor& group_offsets,
    const at::Tensor& group_ks,
    const at::Tensor& entry_offsets,
    const at::Tensor& entry_rows,
    const at::Tensor& entry_values,
    const at::Tensor& b_comp,
    int64_t k,
    int64_t bm) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  TORCH_CHECK(bm == 32 || bm == 64 || bm == 128,
              "bm must be one of 32, 64, 128");
  TORCH_CHECK(n % 64 == 0, "output N must be divisible by 64");
  TORCH_CHECK(b_comp.dim() == 2 && b_comp.size(0) == k && b_comp.size(1) == n,
              "b_comp must be [K, N]");
  TORCH_CHECK(b_comp.is_cuda() && b_comp.scalar_type() == at::kBFloat16,
              "b_comp must be CUDA BF16");
  TORCH_CHECK(active_mblocks.is_cuda() && active_mblocks.scalar_type() == at::kInt,
              "active_mblocks must be CUDA int32");
  TORCH_CHECK(group_offsets.is_cuda() && group_offsets.scalar_type() == at::kInt,
              "group_offsets must be CUDA int32");
  TORCH_CHECK(group_ks.is_cuda() && group_ks.scalar_type() == at::kInt,
              "group_ks must be CUDA int32");
  TORCH_CHECK(entry_offsets.is_cuda() && entry_offsets.scalar_type() == at::kInt,
              "entry_offsets must be CUDA int32");
  TORCH_CHECK(entry_rows.is_cuda() && entry_rows.scalar_type() == at::kInt,
              "entry_rows must be CUDA int32");
  TORCH_CHECK(entry_values.is_cuda() && entry_values.scalar_type() == at::kBFloat16,
              "entry_values must be CUDA BF16");
  TORCH_CHECK(group_offsets.numel() >= active_mblocks.numel() + 1,
              "group_offsets is too small");
  return sparse_kmajor_serial_group_inplace_cuda(
      output,
      active_mblocks,
      group_offsets,
      group_ks,
      entry_offsets,
      entry_rows,
      entry_values,
      b_comp,
      m,
      k,
      n,
      bm);
}

at::Tensor sparse_active_row_value_payload_vec8_inplace(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  return sparse_active_row_value_payload_vec8_inplace_cuda(
      output,
      outlier_values,
      weight_t_bf16,
      flat_indices,
      row_offsets,
      active_rows,
      k);
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_vstore(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  return sparse_active_row_value_payload_vec8_inplace_vstore_cuda(
      output,
      outlier_values,
      weight_t_bf16,
      flat_indices,
      row_offsets,
      active_rows,
      k);
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_skip_vstore(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k,
    int64_t skip_per_row) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  TORCH_CHECK(skip_per_row >= 0 && skip_per_row <= std::numeric_limits<int32_t>::max(),
              "skip_per_row must fit a non-negative int32");
  return sparse_active_row_value_payload_vec8_inplace_skip_vstore_cuda(
      output,
      outlier_values,
      weight_t_bf16,
      flat_indices,
      row_offsets,
      active_rows,
      k,
      skip_per_row);
}

at::Tensor sparse_packed_suffix12_vec8_inplace_vstore(
    const at::Tensor& output,
    const at::Tensor& packed_suffix_records,
    const at::Tensor& active_rows,
    const at::Tensor& weight_t_bf16) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  TORCH_CHECK(
      packed_suffix_records.is_cuda() &&
          packed_suffix_records.scalar_type() == at::kInt &&
          packed_suffix_records.dim() == 2 &&
          packed_suffix_records.size(1) == 12 &&
          packed_suffix_records.is_contiguous(),
      "packed_suffix_records must be contiguous CUDA int32 [R, 12]");
  TORCH_CHECK(
      active_rows.is_cuda() &&
          active_rows.scalar_type() == at::kInt &&
          active_rows.dim() == 1 &&
          active_rows.is_contiguous(),
      "active_rows must be contiguous CUDA int32 [R]");
  TORCH_CHECK(
      packed_suffix_records.size(0) == active_rows.numel(),
      "packed_suffix_records and active_rows must have the same R");
  TORCH_CHECK(
      weight_t_bf16.is_cuda() &&
          weight_t_bf16.scalar_type() == at::kBFloat16 &&
          weight_t_bf16.dim() == 2 &&
          weight_t_bf16.is_contiguous(),
      "weight_t_bf16 must be contiguous CUDA BF16 [K, N]");
  TORCH_CHECK(
      weight_t_bf16.device() == output.device() &&
          active_rows.device() == output.device() &&
          packed_suffix_records.device() == output.device(),
      "all tensors must be on the output device");
  TORCH_CHECK(
      weight_t_bf16.size(1) == output.size(1),
      "weight_t_bf16 N must match output N");
  TORCH_CHECK(
      weight_t_bf16.size(0) > 0 && weight_t_bf16.size(0) <= 8192,
      "packed suffix kernel requires 0 < K <= 8192 for its 13-bit K encoding");
  TORCH_CHECK(
      output.size(1) % 256 == 0,
      "packed suffix kernel requires output N divisible by 256");
  check_active_rows(active_rows, output.size(0));
  return sparse_packed_suffix12_vec8_inplace_vstore_cuda(
      output,
      packed_suffix_records,
      active_rows,
      weight_t_bf16);
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_strict_vstore(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  return sparse_active_row_value_payload_vec8_inplace_strict_vstore_cuda(
      output,
      outlier_values,
      weight_t_bf16,
      flat_indices,
      row_offsets,
      active_rows,
      k);
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_sum_then_add_vstore(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  return sparse_active_row_value_payload_vec8_inplace_sum_then_add_vstore_cuda(
      output,
      outlier_values,
      weight_t_bf16,
      flat_indices,
      row_offsets,
      active_rows,
      k);
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_b_evict_last_vstore(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  return sparse_active_row_value_payload_vec8_inplace_b_evict_last_vstore_cuda(
      output,
      outlier_values,
      weight_t_bf16,
      flat_indices,
      row_offsets,
      active_rows,
      k);
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_fastpath(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  return sparse_active_row_value_payload_vec8_inplace_fastpath_cuda(
      output,
      outlier_values,
      weight_t_bf16,
      flat_indices,
      row_offsets,
      active_rows,
      k);
}

at::Tensor sparse_active_row_value_payload_vec8_inplace_rowblock(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& flat_indices,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  return sparse_active_row_value_payload_vec8_inplace_rowblock_cuda(
      output,
      outlier_values,
      weight_t_bf16,
      flat_indices,
      row_offsets,
      active_rows,
      k);
}

at::Tensor sparse_active_row_col_value_payload_vec16_inplace(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  TORCH_CHECK(n % 16 == 0, "output N must be divisible by 16");
  check_col_value_payload_inputs(outlier_values, outlier_cols, weight_t_bf16, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  return sparse_active_row_col_value_payload_vec16_inplace_cuda(
      output,
      outlier_values,
      outlier_cols,
      weight_t_bf16,
      row_offsets,
      active_rows,
      k);
}

std::vector<at::Tensor> split_hot_dense_padded_cold_rows(
    const at::Tensor& hot_dense,
    const at::Tensor& cold_values,
    const at::Tensor& cold_cols,
    const at::Tensor& cold_counts,
    const at::Tensor& overflow,
    const at::Tensor& row_values,
    const at::Tensor& row_cols,
    const at::Tensor& row_offsets,
    const at::Tensor& hot_lut) {
  TORCH_CHECK(hot_dense.dim() == 2 && hot_dense.is_cuda() &&
                  hot_dense.scalar_type() == at::kBFloat16,
              "hot_dense must be CUDA BF16 [M, H]");
  const int64_t rows = hot_dense.size(0);
  const int64_t hot_cols = hot_dense.size(1);
  TORCH_CHECK(cold_values.dim() == 2 && cold_values.size(0) == rows &&
                  cold_values.is_cuda() && cold_values.scalar_type() == at::kBFloat16,
              "cold_values must be CUDA BF16 [M, capacity]");
  const int64_t cold_capacity = cold_values.size(1);
  TORCH_CHECK(cold_cols.sizes() == cold_values.sizes() && cold_cols.is_cuda() &&
                  cold_cols.scalar_type() == at::kShort,
              "cold_cols must be CUDA int16 with cold_values shape");
  TORCH_CHECK(cold_counts.dim() == 1 && cold_counts.numel() == rows &&
                  cold_counts.is_cuda() && cold_counts.scalar_type() == at::kInt,
              "cold_counts must be CUDA int32 [M]");
  TORCH_CHECK(overflow.is_cuda() && overflow.scalar_type() == at::kInt &&
                  overflow.numel() == 1,
              "overflow must be a CUDA int32 scalar");
  TORCH_CHECK(row_values.dim() == 1 && row_values.is_cuda() &&
                  row_values.scalar_type() == at::kBFloat16,
              "row_values must be CUDA BF16");
  TORCH_CHECK(row_cols.dim() == 1 && row_cols.numel() == row_values.numel() &&
                  row_cols.is_cuda() && row_cols.scalar_type() == at::kShort,
              "row_cols must be CUDA int16 matching row_values");
  TORCH_CHECK(row_offsets.dim() == 1 && row_offsets.numel() == rows + 1 &&
                  row_offsets.is_cuda() && row_offsets.scalar_type() == at::kInt,
              "row_offsets must be CUDA int32 [M+1]");
  TORCH_CHECK(hot_lut.dim() == 1 && hot_lut.is_cuda() &&
                  hot_lut.scalar_type() == at::kShort,
              "hot_lut must be CUDA int16 [K]");
  TORCH_CHECK(hot_cols > 0 && hot_cols <= 32767,
              "hot column count must fit int16");
  TORCH_CHECK(cold_capacity > 0, "cold capacity must be positive");
  return split_hot_dense_padded_cold_rows_cuda(
      hot_dense,
      cold_values,
      cold_cols,
      cold_counts,
      overflow,
      row_values,
      row_cols,
      row_offsets,
      hot_lut,
      rows,
      hot_lut.numel(),
      hot_cols,
      cold_capacity);
}

at::Tensor sparse_padded_cold_col_vec16_inplace(
    const at::Tensor& output,
    const at::Tensor& cold_values,
    const at::Tensor& cold_cols,
    const at::Tensor& cold_counts,
    const at::Tensor& row_values,
    const at::Tensor& row_cols,
    const at::Tensor& row_offsets,
    const at::Tensor& hot_lut,
    const at::Tensor& weight_t_bf16) {
  TORCH_CHECK(output.dim() == 2 && output.is_cuda() &&
                  output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16 [M, N]");
  const int64_t rows = output.size(0);
  const int64_t cols = output.size(1);
  TORCH_CHECK(cols > 0 && cols % 16 == 0,
              "padded cold vec16 kernel requires positive N divisible by 16");
  TORCH_CHECK(cold_values.dim() == 2 && cold_values.size(0) == rows &&
                  cold_values.is_cuda() && cold_values.scalar_type() == at::kBFloat16,
              "cold_values must be CUDA BF16 [M, capacity]");
  TORCH_CHECK(cold_cols.sizes() == cold_values.sizes() && cold_cols.is_cuda() &&
                  cold_cols.scalar_type() == at::kShort,
              "cold_cols must be CUDA int16 with cold_values shape");
  TORCH_CHECK(cold_counts.dim() == 1 && cold_counts.numel() == rows &&
                  cold_counts.is_cuda() && cold_counts.scalar_type() == at::kInt,
              "cold_counts must be CUDA int32 [M]");
  TORCH_CHECK(row_values.dim() == 1 && row_values.is_cuda() &&
                  row_values.scalar_type() == at::kBFloat16,
              "row_values must be CUDA BF16");
  TORCH_CHECK(row_cols.dim() == 1 && row_cols.numel() == row_values.numel() &&
                  row_cols.is_cuda() && row_cols.scalar_type() == at::kShort,
              "row_cols must be CUDA int16 matching row_values");
  TORCH_CHECK(row_offsets.dim() == 1 && row_offsets.numel() == rows + 1 &&
                  row_offsets.is_cuda() && row_offsets.scalar_type() == at::kInt,
              "row_offsets must be CUDA int32 [M+1]");
  TORCH_CHECK(hot_lut.dim() == 1 && hot_lut.is_cuda() &&
                  hot_lut.scalar_type() == at::kShort,
              "hot_lut must be CUDA int16 [K]");
  TORCH_CHECK(weight_t_bf16.dim() == 2 && weight_t_bf16.size(1) == cols &&
                  weight_t_bf16.is_cuda() && weight_t_bf16.scalar_type() == at::kBFloat16,
              "weight_t_bf16 must be CUDA BF16 [K, N]");
  return sparse_padded_cold_col_vec16_inplace_cuda(
      output,
      cold_values,
      cold_cols,
      cold_counts,
      row_values,
      row_cols,
      row_offsets,
      hot_lut,
      weight_t_bf16,
      rows,
      hot_lut.numel(),
      cols,
      cold_values.size(1));
}

at::Tensor sparse_active_row_col_value_payload_vec8_inplace_vstore(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  check_col_value_payload_inputs(outlier_values, outlier_cols, weight_t_bf16, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  return sparse_active_row_col_value_payload_vec8_inplace_vstore_cuda(
      output,
      outlier_values,
      outlier_cols,
      weight_t_bf16,
      row_offsets,
      active_rows,
      k);
}

at::Tensor sparse_active_row_col_value_payload_vec8_shmem_sum_then_add(
    const at::Tensor& output,
    const at::Tensor& outlier_values,
    const at::Tensor& outlier_cols,
    const at::Tensor& weight_t_bf16,
    const at::Tensor& row_offsets,
    const at::Tensor& active_rows,
    int64_t k) {
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16,
              "output must be CUDA BF16");
  const int64_t m = output.size(0);
  const int64_t n = output.size(1);
  TORCH_CHECK(n % 8 == 0, "output N must be divisible by 8");
  check_col_value_payload_inputs(outlier_values, outlier_cols, weight_t_bf16, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  return sparse_active_row_col_value_payload_vec8_shmem_sum_then_add_cuda(
      output,
      outlier_values,
      outlier_cols,
      weight_t_bf16,
      row_offsets,
      active_rows,
      k);
}

at::Tensor preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_col_value_payload_inputs(outlier_values, outlier_cols, weight_t_bf16, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  return nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols_cuda(
      output,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      outlier_values,
      outlier_cols,
      weight_t_bf16,
      row_offsets,
      active_rows,
      m,
      k,
      n);
}

at::Tensor preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols_vec16(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  TORCH_CHECK(n % 16 == 0, "vec16 correction requires N divisible by 16");
  check_col_value_payload_inputs(outlier_values, outlier_cols, weight_t_bf16, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  return nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols_vec16_cuda(
      output,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      outlier_values,
      outlier_cols,
      weight_t_bf16,
      row_offsets,
      active_rows,
      m,
      k,
      n);
}

at::Tensor preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols_shmem(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_col_value_payload_inputs(outlier_values, outlier_cols, weight_t_bf16, row_offsets, m, k, n);
  check_active_rows(active_rows, m);
  return nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols_shmem_cuda(
      output,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      outlier_values,
      outlier_cols,
      weight_t_bf16,
      row_offsets,
      active_rows,
      m,
      k,
      n);
}

at::Tensor preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  check_col_value_payload_inputs(outlier_values, outlier_cols, weight_t_bf16, row_offsets, m, k, n);
  return nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_cuda(
      output,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      outlier_values,
      outlier_cols,
      weight_t_bf16,
      row_offsets,
      m,
      k,
      n);
}

at::Tensor preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec16(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  TORCH_CHECK(n % 16 == 0, "vec16 tile correction requires N divisible by 16");
  check_col_value_payload_inputs(outlier_values, outlier_cols, weight_t_bf16, row_offsets, m, k, n);
  return nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec16_cuda(
      output,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      outlier_values,
      outlier_cols,
      weight_t_bf16,
      row_offsets,
      m,
      k,
      n);
}

at::Tensor preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_persistent_cols_vec16(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  TORCH_CHECK(n % 16 == 0, "persistent vec16 correction requires N divisible by 16");
  TORCH_CHECK(scheduler_mode >= 0 && scheduler_mode <= 3,
              "scheduler_mode must be 0 swizzled, 1 row-major, 2 column-major, or 3 ready-queue");
  check_col_value_payload_inputs(outlier_values, outlier_cols, weight_t_bf16, row_offsets, m, k, n);
  return nvfp4_dense_sparse_tma_value_payload_tile_scales_persistent_cols_vec16_cuda(
      output,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      outlier_values,
      outlier_cols,
      weight_t_bf16,
      row_offsets,
      m,
      k,
      n,
      worker_blocks,
      worker_threads,
      scheduler_mode,
      sleep_ns,
      start_delay_us);
}

at::Tensor preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec32(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  TORCH_CHECK(n % 32 == 0, "vec32 tile correction requires N divisible by 32");
  check_col_value_payload_inputs(outlier_values, outlier_cols, weight_t_bf16, row_offsets, m, k, n);
  return nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec32_cuda(
      output,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      outlier_values,
      outlier_cols,
      weight_t_bf16,
      row_offsets,
      m,
      k,
      n);
}

at::Tensor preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec16_threads(
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
  check_output(output, m, n);
  check_inputs(a_data, a_scale_tile, b_data, b_scale_tile, a_amax, b_amax, m, k, n);
  TORCH_CHECK(n % 16 == 0, "vec16 tile correction requires N divisible by 16");
  TORCH_CHECK(correction_threads == 128 || correction_threads == 256,
              "correction_threads must be 128 or 256");
  check_col_value_payload_inputs(outlier_values, outlier_cols, weight_t_bf16, row_offsets, m, k, n);
  return nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec16_threads_cuda(
      output,
      a_data,
      a_scale_tile,
      b_data,
      b_scale_tile,
      a_amax,
      b_amax,
      outlier_values,
      outlier_cols,
      weight_t_bf16,
      row_offsets,
      m,
      k,
      n,
      correction_threads);
}

at::Tensor nvfp4_gemm_tma_swizzled_scale(const at::Tensor& a_data,
                                         const at::Tensor& a_scale_inv,
                                         const at::Tensor& b_data,
                                         const at::Tensor& b_scale_inv,
                                         const at::Tensor& a_amax,
                                         const at::Tensor& b_amax,
                                         int64_t m,
                                         int64_t k,
                                         int64_t n) {
  check_inputs(a_data, a_scale_inv, b_data, b_scale_inv, a_amax, b_amax, m, k, n);
  return nvfp4_gemm_tma_swizzled_scale_cuda(a_data,
                                           a_scale_inv,
                                           b_data,
                                           b_scale_inv,
                                           a_amax,
                                           b_amax,
                                           m,
                                           k,
                                           n);
}

at::Tensor nvfp4_dense_sparse_tma_value_payload_overlap(
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
  check_inputs(a_data, a_scale_inv, b_data, b_scale_inv, a_amax, b_amax, m, k, n);
  check_value_payload_inputs(outlier_values, weight_t_bf16, flat_indices, row_offsets, m, k, n);
  return nvfp4_dense_sparse_tma_value_payload_overlap_cuda(a_data,
                                                          a_scale_inv,
                                                          b_data,
                                                          b_scale_inv,
                                                          a_amax,
                                                          b_amax,
                                                          outlier_values,
                                                          weight_t_bf16,
                                                          flat_indices,
                                                          row_offsets,
                                                          m,
                                                          k,
                                                          n,
                                                          sparse_worker_blocks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("swizzle_scale_to_tma_tile_major",
        &swizzle_scale_to_tma_tile_major,
        "Swizzle TE row-major NVFP4 scale bytes to v13 TMA tile-major layout");
  m.def("nvfp4_gemm_tma_warpspecialized",
        &nvfp4_gemm_tma_warpspecialized,
        "v12 TMA warp-specialized handwritten NVFP4 GEMM");
  m.def("nvfp4_gemm_tma_tile_scales",
        &nvfp4_gemm_tma_tile_scales,
        "v13 TMA warp-specialized handwritten NVFP4 GEMM using pre-swizzled tile-major scales");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales",
        &preallocated_nvfp4_gemm_tma_tile_scales,
        "Preallocated v13 TMA warp-specialized handwritten NVFP4 GEMM");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_4wg",
        &preallocated_nvfp4_gemm_tma_tile_scales_4wg,
        "Preallocated v13 TMA dense-only NVFP4 GEMM launched with a 4-WG CTA");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_compact_consumer_posttail",
        &preallocated_nvfp4_gemm_tma_tile_scales_compact_consumer_posttail,
        "Specialized 4WG NVFP4 GEMM: WG3 stages compact row inputs and consumers apply correction post-mainloop");
  m.def("compact_consumer_max_nnz",
        &compact_consumer_max_nnz,
        "Return the compile-time per-row cap of the compact-consumer kernel");
  m.def("compact_consumer_static_n",
        &compact_consumer_static_n,
        "Return the compile-time output width of the compact-consumer kernel");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_active_row_ready_flags_vstore",
        &preallocated_nvfp4_gemm_tma_tile_scales_active_row_ready_flags_vstore,
        "Preallocated TMA NVFP4 dense with active-row tile-ready flag sparse consumer");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_active_row_ready_queue_vstore",
        &preallocated_nvfp4_gemm_tma_tile_scales_active_row_ready_queue_vstore,
        "Preallocated TMA NVFP4 dense with active-row ready-queue sparse consumer");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_active_mtile_ready_queue_vstore",
        &preallocated_nvfp4_gemm_tma_tile_scales_active_mtile_ready_queue_vstore,
        "Preallocated TMA NVFP4 dense with active-M-tile ready-queue sparse consumer");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_direct_add_active",
        &preallocated_nvfp4_gemm_tma_tile_scales_direct_add_active,
        "Preallocated v13 TMA NVFP4 GEMM with in-kernel tile-local direct-add sparse correction");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_direct_add_active_4wg",
        &preallocated_nvfp4_gemm_tma_tile_scales_direct_add_active_4wg,
        "Preallocated v13 TMA NVFP4 GEMM with 4-WG post-store direct-add tail assistance");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_direct_smem_active",
        &preallocated_nvfp4_gemm_tma_tile_scales_direct_smem_active,
        "Preallocated v13 TMA NVFP4 GEMM with pre-store shared-memory direct sparse correction");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_direct_smem_delta_active",
        &preallocated_nvfp4_gemm_tma_tile_scales_direct_smem_delta_active,
        "Preallocated v13 TMA NVFP4 GEMM with pre-store shared-memory precomputed sparse delta add");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_active",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_active,
        "Preallocated v13 TMA NVFP4 GEMM with scheduler-warp sparse load+FMA checksum probe");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_active_sidewarp",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_active_sidewarp,
        "Preallocated v13 TMA NVFP4 GEMM with same-CTA extra side-warp sparse load+FMA checksum probe");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active,
        "Preallocated v13 TMA NVFP4 GEMM with in-CTA row-owned sparse load+FMA writeback to a separate delta output");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_sidewarp",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_sidewarp,
        "Preallocated v13 TMA NVFP4 GEMM with same-CTA extra sidewarp row-owned sparse load+FMA writeback to a separate delta output");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_rowblock_sidewarp",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_rowblock_sidewarp,
        "Preallocated v13 TMA NVFP4 GEMM with same-CTA WG3 sparse-friendly rowblock/N-block delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_sidewarp",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_sidewarp,
        "Preallocated v13 TMA NVFP4 GEMM with same-CTA WG3 active-rowblock sparse-friendly delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_static_persistent_sidewarp",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_static_persistent_sidewarp,
        "Preallocated v13 TMA NVFP4 GEMM with persistent dense scheduling plus WG3 static active-rowblock vec8 delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_active_rowblock_static_persistent_sidewarp",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_active_rowblock_static_persistent_sidewarp,
        "Preallocated v13 TMA NVFP4 GEMM with persistent dense scheduling plus WG3 static active-rowblock vec8 load+FMA sink, no delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_warp256_active_rowblock_static_persistent_sidewarp",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_warp256_active_rowblock_static_persistent_sidewarp,
        "Preallocated v13 TMA NVFP4 GEMM with persistent dense scheduling plus WG3 row-owned warp256 delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_warp256_active_rowblock_static_persistent_sidewarp",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_warp256_active_rowblock_static_persistent_sidewarp,
        "Preallocated v13 TMA NVFP4 GEMM with persistent dense scheduling plus WG3 row-owned warp256 load+FMA sink, no delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_prefetch_active_rowblock_static_persistent_sidewarp",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_prefetch_active_rowblock_static_persistent_sidewarp,
        "Preallocated v13 TMA NVFP4 GEMM with persistent dense scheduling plus WG3 vec8 one-step-prefetch delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_prefetch_active_rowblock_static_persistent_sidewarp",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_only_prefetch_active_rowblock_static_persistent_sidewarp,
        "Preallocated v13 TMA NVFP4 GEMM with persistent dense scheduling plus WG3 vec8 one-step-prefetch load+FMA sink, no delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_prefetch_compact_active_rowblock_static_persistent_sidewarp",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_prefetch_compact_active_rowblock_static_persistent_sidewarp,
        "Preallocated v13 TMA NVFP4 GEMM with persistent dense scheduling plus WG3 vec8 one-step-prefetch compact active-row delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_persistent_sidewarp",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_active_rowblock_persistent_sidewarp,
        "Preallocated v13 TMA NVFP4 GEMM with same-CTA WG3 persistent active-rowblock/N-block delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_wg3_ready_active_direct_add",
        &preallocated_nvfp4_gemm_tma_tile_scales_wg3_ready_active_direct_add,
        "Preallocated v13 TMA NVFP4 GEMM with same-CTA WG3 coarse-ready active-row direct-add");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_packed_rowblock_sidewarp",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_packed_rowblock_sidewarp,
        "Preallocated v13 TMA NVFP4 GEMM with same-CTA WG3 packed active-rowblock sparse-friendly delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_kmajor",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_kmajor,
        "Preallocated v13 TMA NVFP4 GEMM with mixed-CTA k-major sparse load+FMA checksum probe");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_hybrid",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_hybrid,
        "Preallocated v13 TMA NVFP4 GEMM with mixed-CTA hot-kmajor/cold-row sparse load+FMA checksum probe");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_incta_hybrid",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_incta_hybrid,
        "Preallocated v13 TMA NVFP4 GEMM with in-CTA scheduler-warp hot-kmajor/cold-row sparse load+FMA checksum probe");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_idlechunk_hybrid",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_idlechunk_hybrid,
        "Preallocated v13 TMA NVFP4 GEMM with bounded producer-idle k-major sparse load+FMA checksum probe");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_scheduler_hybrid",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_scheduler_hybrid,
        "Preallocated v13 TMA NVFP4 GEMM with scheduler-warp bounded k-major sparse load+FMA checksum probe");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_scheduler_hybrid_phase_trace",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_probe_scheduler_hybrid_phase_trace,
        "Preallocated v13 TMA NVFP4 GEMM scheduler-warp sparse load+FMA probe with guarded phase trace");
	  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid",
	        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid,
	        "Preallocated v13 TMA NVFP4 GEMM with scheduler-warp bounded k-major sparse load+FMA BF16 delta writeback");
	  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_tile_meta",
	        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_tile_meta,
	        "Preallocated v13 TMA NVFP4 GEMM sparse load+FMA path with precomputed K-major tile metadata");
	  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_packed",
	        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_packed,
	        "Preallocated v13 TMA NVFP4 GEMM sparse load+FMA path with packed local-delta payload");
	  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_packed_tile_meta",
	        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_packed_tile_meta,
	        "Preallocated v13 TMA NVFP4 GEMM sparse load+FMA path with packed local-delta payload and precomputed K-major tile metadata");
	  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace",
	        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace,
	        "Preallocated v13 TMA NVFP4 GEMM sparse load+FMA path with guarded phase trace");
	  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace_tile_meta",
	        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace_tile_meta,
	        "Preallocated v13 TMA NVFP4 GEMM sparse load+FMA path with guarded phase trace and precomputed K-major tile metadata");
	  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace_packed",
	        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace_packed,
	        "Preallocated v13 TMA NVFP4 GEMM sparse load+FMA path with guarded phase trace and packed local-delta payload");
	  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace_packed_tile_meta",
	        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_scheduler_hybrid_phase_trace_packed_tile_meta,
	        "Preallocated v13 TMA NVFP4 GEMM sparse load+FMA path with guarded phase trace, packed local-delta payload, and precomputed K-major tile metadata");
	  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_atomic",
	        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_atomic,
        "Preallocated v13 TMA NVFP4 GEMM with in-CTA k-major sparse load+FMA atomic BF16 delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_direct",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_direct,
        "Preallocated v13 TMA NVFP4 GEMM with in-CTA k-major sparse load+FMA direct BF16 delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_sharedacc",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_sharedacc,
        "Preallocated v13 TMA NVFP4 GEMM with in-CTA k-major sparse load+FMA shared-accumulator BF16 delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_extrawg_kmajor_sharedacc",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_extrawg_kmajor_sharedacc,
        "Preallocated v13 TMA NVFP4 GEMM with extra sparse warpgroup k-major shared-accumulator BF16 delta writeback");
  m.def("preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_entry_direct",
        &preallocated_nvfp4_gemm_tma_tile_scales_loadfma_write_incta_kmajor_entry_direct,
        "Preallocated v13 TMA NVFP4 GEMM with in-CTA k-major sparse load+FMA compact entry BF16 delta writeback");
  m.def("merge_entry_delta_active_rows",
        &merge_entry_delta_active_rows,
        "Merge compact entry delta vectors into active rows of a BF16 output");
  m.def("merge_entry_delta_active_rows_fastpath",
        &merge_entry_delta_active_rows_fastpath,
        "Merge compact entry delta vectors with single/double-entry fast paths");
  m.def("merge_entry_delta_active_rows_vec8",
        &merge_entry_delta_active_rows_vec8,
        "Merge compact entry delta vectors with flattened Vec8 N chunks");
  m.def("merge_entry_delta_active_rows_chunk_prefix_vec8",
        &merge_entry_delta_active_rows_chunk_prefix_vec8,
        "Merge compact entry delta vectors into a prefix of N chunks with flattened Vec8 chunks");
  m.def("merge_single_entry_delta_active_rows",
        &merge_single_entry_delta_active_rows,
        "Merge one compact delta vector per active row into a BF16 output");
  m.def("merge_double_entry_delta_active_rows",
        &merge_double_entry_delta_active_rows,
        "Merge two compact delta vectors per active row into a BF16 output");
  m.def("nvfp4_dense_sparse_tma_value_payload_tile_scales",
        &nvfp4_dense_sparse_tma_value_payload_tile_scales,
        "v13 TMA tile-scale NVFP4 GEMM followed by exact value-payload sparse correction");
  m.def("preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales",
        &preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales,
        "Preallocated v13 TMA tile-scale NVFP4 GEMM followed by exact value-payload sparse correction");
  m.def("preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_sidecar",
        &preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_sidecar,
        "Preallocated v13 TMA tile-scale NVFP4 GEMM with opportunistic tile sparse sidecar");
  m.def("preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows",
        &preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows,
        "Preallocated v13 TMA tile-scale NVFP4 GEMM followed by active-row value-payload sparse correction");
  m.def("sparse_active_row_value_payload_vec8_store",
        &sparse_active_row_value_payload_vec8_store,
        "Standalone active-row BF16 sparse correction store kernel");
  m.def("sparse_active_row_value_payload_vec8_store_vstore",
        &sparse_active_row_value_payload_vec8_store_vstore,
        "Standalone active-row BF16 sparse correction store kernel with 128-bit vec8 store");
  m.def("merge_full_delta_active_rows",
        &merge_full_delta_active_rows,
        "Merge a full [M, N] BF16 delta buffer into active rows of a BF16 output");
  m.def("merge_compact_delta_active_rows",
        &merge_compact_delta_active_rows,
        "Merge a compact [active_rows, N] BF16 delta buffer into active rows of a BF16 output");
  m.def("merge_two_compact_delta_active_rows",
        &merge_two_compact_delta_active_rows,
        "Merge two compact BF16 delta buffers into disjoint active rows of a BF16 output");
  m.def("build_compact_dense_residual_active_rows",
        &build_compact_dense_residual_active_rows,
        "Build a compact [active_rows, K] BF16 dense residual matrix from CSR row payload");
  m.def("build_padded_light_heavy_rows",
        &build_padded_light_heavy_rows,
        "Partition CSR rows into padded light and fixed-capacity heavy row lists without host sync");
  m.def("sparse_kmajor_epin64_delta_store",
        &sparse_kmajor_epin64_delta_store,
        "Standalone full-topk k-major EpiN=64 BF16 sparse correction delta-store kernel");
  m.def("sparse_kmajor_epin_delta_store",
        &sparse_kmajor_epin_delta_store,
        "Standalone full-topk k-major EpiN={32,64} BF16 sparse correction delta-store kernel");
  m.def("sparse_kmajor_epin64_direct_store",
        &sparse_kmajor_epin64_direct_store,
        "Standalone conflict-free k-major EpiN=64 BF16 sparse correction direct-store kernel");
  m.def("sparse_kmajor_serial_group_inplace",
        &sparse_kmajor_serial_group_inplace,
        "Standalone atomics-free k-major EpiN=64 BF16 sparse correction in-place kernel");
  m.def("sparse_active_row_value_payload_vec8_inplace",
        &sparse_active_row_value_payload_vec8_inplace,
        "Standalone active-row BF16 sparse correction in-place add kernel");
  m.def("sparse_active_row_value_payload_vec8_inplace_vstore",
        &sparse_active_row_value_payload_vec8_inplace_vstore,
        "Standalone active-row BF16 sparse correction in-place add kernel with 128-bit vec8 store");
  m.def("sparse_active_row_value_payload_vec8_inplace_skip_vstore",
        &sparse_active_row_value_payload_vec8_inplace_skip_vstore,
        "Standalone active-row BF16 sparse tail add starting after a fixed per-row prefix");
  m.def("sparse_packed_suffix12_vec8_inplace_vstore",
        &sparse_packed_suffix12_vec8_inplace_vstore,
        "Standalone packed <=12-record active-row BF16 sparse suffix kernel");
  m.def("sparse_active_row_value_payload_vec8_inplace_strict_vstore",
        &sparse_active_row_value_payload_vec8_inplace_strict_vstore,
        "Standalone active-row BF16 sparse correction in-place add kernel with strict FP32 mul/add and 128-bit vec8 store");
  m.def("sparse_active_row_value_payload_vec8_inplace_sum_then_add_vstore",
        &sparse_active_row_value_payload_vec8_inplace_sum_then_add_vstore,
        "Standalone active-row BF16 sparse correction in-place add kernel that sums correction before adding output");
  m.def("sparse_active_row_value_payload_vec8_inplace_b_evict_last_vstore",
        &sparse_active_row_value_payload_vec8_inplace_b_evict_last_vstore,
        "Standalone active-row BF16 sparse correction in-place add kernel with B evict_last loads and 128-bit vec8 store");
  m.def("sparse_active_row_value_payload_vec8_inplace_fastpath",
        &sparse_active_row_value_payload_vec8_inplace_fastpath,
        "Standalone active-row BF16 sparse correction in-place add kernel with single/double-entry fast paths");
  m.def("sparse_active_row_value_payload_vec8_inplace_rowblock",
        &sparse_active_row_value_payload_vec8_inplace_rowblock,
        "Standalone active-row BF16 sparse correction in-place add kernel with row-block grid mapping");
  m.def("sparse_active_row_col_value_payload_vec16_inplace",
        &sparse_active_row_col_value_payload_vec16_inplace,
        "Standalone active-row int16-column vec16 BF16 sparse correction in-place add kernel");
  m.def("split_hot_dense_padded_cold_rows",
        &split_hot_dense_padded_cold_rows,
        "Split row-indexed BF16 payload into dense hot columns and padded cold rows");
  m.def("sparse_padded_cold_col_vec16_inplace",
        &sparse_padded_cold_col_vec16_inplace,
        "Apply padded cold-row BF16 correction with one vec16 CTA per output row");
  m.def("sparse_active_row_col_value_payload_vec8_inplace_vstore",
        &sparse_active_row_col_value_payload_vec8_inplace_vstore,
        "Standalone active-row int16-column vec8 BF16 sparse correction in-place add kernel with 128-bit store");
  m.def("sparse_active_row_col_value_payload_vec8_shmem_sum_then_add",
        &sparse_active_row_col_value_payload_vec8_shmem_sum_then_add,
        "Standalone active-row int16-column shared-memory BF16 sparse correction that sums before adding output");
  m.def("preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols",
        &preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols,
        "Preallocated v13 TMA tile-scale NVFP4 GEMM followed by active-row int16-column value-payload sparse correction");
  m.def("preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols_vec16",
        &preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols_vec16,
        "Preallocated v13 TMA tile-scale NVFP4 GEMM followed by active-row int16-column vec16 value-payload sparse correction");
  m.def("preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols_shmem",
        &preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_active_rows_cols_shmem,
        "Preallocated v13 TMA tile-scale NVFP4 GEMM followed by active-row int16-column shared-memory value-payload sparse correction");
  m.def("preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols",
        &preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols,
        "Preallocated v13 TMA tile-scale NVFP4 GEMM followed by tile int16-column value-payload sparse correction");
  m.def("preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec16",
        &preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec16,
        "Preallocated v13 TMA tile-scale NVFP4 GEMM followed by tile int16-column vec16 value-payload sparse correction");
  m.def("preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_persistent_cols_vec16",
        &preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_persistent_cols_vec16,
        "Preallocated v14 TMA GEMM with persistent tile-ready int16-column vec16 sparse correction");
  m.def("preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec32",
        &preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec32,
        "Preallocated v14 TMA tile-scale NVFP4 GEMM followed by tile int16-column vec32 value-payload sparse correction");
  m.def("preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec16_threads",
        &preallocated_nvfp4_dense_sparse_tma_value_payload_tile_scales_tile_cols_vec16_threads,
        "Preallocated v14 TMA tile-scale NVFP4 GEMM followed by tile int16-column vec16 correction with selectable CTA threads");
  m.def("nvfp4_gemm_tma_swizzled_scale",
        &nvfp4_gemm_tma_swizzled_scale,
        "v13 TMA warp-specialized handwritten NVFP4 GEMM with tile-major scale staging");
  m.def("nvfp4_dense_sparse_tma_value_payload_overlap",
        &nvfp4_dense_sparse_tma_value_payload_overlap,
        "v12 TMA dense GEMM with tile-ready value-payload sparse overlap");
}
