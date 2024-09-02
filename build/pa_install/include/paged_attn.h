#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif
void fmha_page_kvcache_fwd(
  void* q_ptr,
  void* kcache_ptr,
  void* vcache_ptr,
  void* k_ptr,
  void* v_ptr,
  void* o_ptr,
  void* block_table_ptr,
  void* cache_seqlens_k_ptr,
  const int32_t max_cache_seq_k,
  const int32_t seqlen_q,
  const int32_t seqlen_k,
  const int32_t batch_size,
  const int32_t num_heads,
  const int32_t num_heads_k,
  const int32_t head_size,
  const int32_t page_block_size,
  hipStream_t stream,
  const float softmax_scale,
  int window_size_left,
  int window_size_right,
  const int32_t num_splits,
  void* cache_batch_idx_ptr,
  void* rotary_cos_ptr,
  void* rotary_sin_ptr,
  bool is_causal,
  bool is_rotary_interleaved, bool is_fp16
);
#ifdef __cplusplus
} // 结束 extern "C"
#endif
