#pragma once

#include "sllmrf/internlm2.h"

namespace sllmrf::internlm2_cuda {

void apply_rope_inplace(
    TensorBuffer &q,
    TensorBuffer &k,
    const Internlm2Config &config,
    uint32_t start_position);

void write_kv_cache(
    KvCacheLayer &layer,
    const TensorBuffer &k,
    const TensorBuffer &v,
    uint32_t start_position,
    const Internlm2Config &config);
[[nodiscard]] TensorBuffer causal_attention(
    const TensorBuffer &q,
    const KvCacheLayer &cache,
    uint32_t start_position,
    const Internlm2Config &config);

}  // namespace sllmrf::internlm2_cuda
