#include <torch/extension.h>

#include <cstdint>

at::Tensor preallocated_nvfp4_dense_cuda(const at::Tensor& output,
                                         const at::Tensor& a_data,
                                         const at::Tensor& a_scale_inv,
                                         const at::Tensor& b_data,
                                         const at::Tensor& b_scale_inv,
                                         const at::Tensor& a_amax,
                                         const at::Tensor& b_amax,
                                         int64_t m,
                                         int64_t k,
                                         int64_t n);

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
    int64_t dense_ntiles);

namespace {

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void check_nvfp4_dense_inputs(const at::Tensor& output,
                              const at::Tensor& a_data,
                              const at::Tensor& a_scale_inv,
                              const at::Tensor& b_data,
                              const at::Tensor& b_scale_inv,
                              const at::Tensor& a_amax,
                              const at::Tensor& b_amax,
                              int64_t m,
                              int64_t k,
                              int64_t n) {
  CHECK_CUDA(output);
  CHECK_CUDA(a_data);
  CHECK_CUDA(a_scale_inv);
  CHECK_CUDA(b_data);
  CHECK_CUDA(b_scale_inv);
  CHECK_CUDA(a_amax);
  CHECK_CUDA(b_amax);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(a_data);
  CHECK_CONTIGUOUS(a_scale_inv);
  CHECK_CONTIGUOUS(b_data);
  CHECK_CONTIGUOUS(b_scale_inv);
  CHECK_CONTIGUOUS(a_amax);
  CHECK_CONTIGUOUS(b_amax);
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
  TORCH_CHECK(a_data.scalar_type() == at::kByte, "a_data must be uint8");
  TORCH_CHECK(a_scale_inv.scalar_type() == at::kByte, "a_scale_inv must be uint8");
  TORCH_CHECK(b_data.scalar_type() == at::kByte, "b_data must be uint8");
  TORCH_CHECK(b_scale_inv.scalar_type() == at::kByte, "b_scale_inv must be uint8");
  TORCH_CHECK(a_amax.scalar_type() == at::kFloat, "a_amax must be FP32");
  TORCH_CHECK(b_amax.scalar_type() == at::kFloat, "b_amax must be FP32");
  TORCH_CHECK(output.dim() == 2 && output.size(0) == m && output.size(1) == n,
              "output shape mismatch");
  TORCH_CHECK(k % 64 == 0, "K must be divisible by 64");
  TORCH_CHECK(a_data.dim() == 2 && a_data.size(0) == m && a_data.size(1) == k / 2,
              "a_data shape must be M x K/2");
  TORCH_CHECK(b_data.dim() == 2 && b_data.size(0) == n && b_data.size(1) == k / 2,
              "b_data shape must be N x K/2");
  TORCH_CHECK(a_scale_inv.dim() == 2 && a_scale_inv.size(0) >= m &&
                  a_scale_inv.size(1) >= k / 16,
              "a_scale_inv shape must cover M x K/16");
  TORCH_CHECK(b_scale_inv.dim() == 2 && b_scale_inv.size(0) >= n &&
                  b_scale_inv.size(1) >= k / 16,
              "b_scale_inv shape must cover N x K/16");
}

void check_row_payload_inputs(const at::Tensor& row_offsets,
                              const at::Tensor& row_ks,
                              const at::Tensor& row_values,
                              int64_t m) {
  CHECK_CUDA(row_offsets);
  CHECK_CUDA(row_ks);
  CHECK_CUDA(row_values);
  CHECK_CONTIGUOUS(row_offsets);
  CHECK_CONTIGUOUS(row_ks);
  CHECK_CONTIGUOUS(row_values);
  TORCH_CHECK(row_offsets.scalar_type() == at::kInt, "row_offsets must be int32");
  TORCH_CHECK(row_ks.scalar_type() == at::kInt, "row_ks must be int32");
  TORCH_CHECK(row_values.scalar_type() == at::kBFloat16, "row_values must be BF16");
  TORCH_CHECK(row_offsets.dim() == 1 && row_offsets.numel() == m + 1,
              "row_offsets length must be M+1");
  TORCH_CHECK(row_ks.dim() == 1 && row_values.dim() == 1,
              "row payload arrays must be 1D");
  TORCH_CHECK(row_ks.numel() == row_values.numel(),
              "row_ks and row_values must have matching length");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "preallocated_nvfp4_dense",
      [](const at::Tensor& output,
         const at::Tensor& a_data,
         const at::Tensor& a_scale_inv,
         const at::Tensor& b_data,
         const at::Tensor& b_scale_inv,
         const at::Tensor& a_amax,
         const at::Tensor& b_amax,
         int64_t m,
         int64_t k,
         int64_t n) {
        check_nvfp4_dense_inputs(
            output, a_data, a_scale_inv, b_data, b_scale_inv, a_amax, b_amax, m, k, n);
        return preallocated_nvfp4_dense_cuda(
            output, a_data, a_scale_inv, b_data, b_scale_inv, a_amax, b_amax, m, k, n);
      },
      "Preallocated NVFP4 dense baseline");

  m.def(
      "preallocated_nvfp4_dense16_sparse_tail_add_active",
      [](const at::Tensor& output,
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
        check_nvfp4_dense_inputs(
            output, a_data, a_scale_inv, b_data, b_scale_inv, a_amax, b_amax, m, k, n);
        check_row_payload_inputs(row_offsets, row_ks, row_values, m);
        CHECK_CUDA(active_row_offsets);
        CHECK_CUDA(active_rows);
        CHECK_CUDA(b_comp);
        CHECK_CONTIGUOUS(active_row_offsets);
        CHECK_CONTIGUOUS(active_rows);
        CHECK_CONTIGUOUS(b_comp);
        TORCH_CHECK(active_row_offsets.scalar_type() == at::kInt,
                    "active_row_offsets must be int32");
        TORCH_CHECK(active_rows.scalar_type() == at::kInt, "active_rows must be int32");
        TORCH_CHECK(b_comp.scalar_type() == at::kBFloat16, "b_comp must be BF16");
        TORCH_CHECK(b_comp.dim() == 2 && b_comp.size(0) == k && b_comp.size(1) == n,
                    "b_comp shape must be K x N");
        const int64_t tiles_m = (m + 255) / 256;
        TORCH_CHECK(active_row_offsets.dim() == 1 &&
                        active_row_offsets.numel() == tiles_m + 1,
                    "active_row_offsets length must be ceil(M/256)+1");
        TORCH_CHECK(active_rows.dim() == 1, "active_rows must be 1D");
        return preallocated_nvfp4_dense16_sparse_tail_add_active_cuda(
            output,
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
            n,
            r,
            kb,
            c,
            dense_ntiles);
      },
      "NVFP4 dense16 + same-CTA sparse correction directly accumulated into output");
}
