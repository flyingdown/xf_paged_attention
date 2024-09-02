#include "paged_attn.h"

void paged_attention_v1(
    void* out,    // [num_seqs, num_heads, head_size]
    void* query,  // [num_seqs, num_heads, head_size]
    void* key_cache,  // [num_blocks, block_size, num_heads, head_size]
    void* value_cache, // [num_blocks, block_size, num_heads, head_size]
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    void* block_tables,  // [num_seqs, max_num_blocks_per_seq]
    void* seq_lens,      // [num_seqs]
    int64_t block_size, int64_t max_seq_len, 
    const int32_t seqlen_q, 
    const int32_t seqlen_k,
    const int32_t batch_size, 
    const int32_t num_heads,
    const int32_t head_size, 
    hipStream_t stream, bool is_fp16);

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
) {
  paged_attention_v1(o_ptr, q_ptr, kcache_ptr, vcache_ptr, num_heads_k, 
    softmax_scale, block_table_ptr, cache_seqlens_k_ptr, page_block_size, 
    max_cache_seq_k, seqlen_q, seqlen_k, batch_size, num_heads, head_size,
    stream, is_fp16);
}
