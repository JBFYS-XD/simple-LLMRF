#pragma once

#include <vector>

#include "sllmrf/gguf.h"
#include "sllmrf/tensor.h"

namespace sllmrf::ops::cuda {

[[nodiscard]] TensorBuffer rms_norm(
    const TensorBuffer &input,
    const std::vector<float> &weight,
    float epsilon);
[[nodiscard]] TensorBuffer rms_norm(
    const TensorBuffer &input,
    const TensorView &weight,
    float epsilon);
[[nodiscard]] TensorBuffer add(
    const TensorBuffer &lhs,
    const TensorBuffer &rhs);
void add_inplace(
    TensorBuffer &lhs,
    const TensorBuffer &rhs);
[[nodiscard]] TensorBuffer silu(const TensorBuffer &input);
[[nodiscard]] TensorBuffer multiply(
    const TensorBuffer &lhs,
    const TensorBuffer &rhs);
[[nodiscard]] TensorBuffer linear_project(
    const TensorView &weight,
    const TensorBuffer &input);
[[nodiscard]] TensorBuffer embedding_lookup(
    const TensorView &weight,
    const std::vector<uint32_t> &token_ids,
    Device device);
[[nodiscard]] std::vector<float> project_last_token_logits(
    const TensorView &weight,
    const TensorBuffer &hidden_state);

}  // namespace sllmrf::ops::cuda
