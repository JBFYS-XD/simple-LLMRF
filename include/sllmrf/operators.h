#pragma once

#include <vector>

#include "sllmrf/gguf.h"
#include "sllmrf/tensor.h"

namespace sllmrf::ops {

[[nodiscard]] TensorBuffer rms_norm(
    const TensorBuffer &input,
    const std::vector<float> &weight,
    float epsilon);
[[nodiscard]] TensorBuffer add(const TensorBuffer &lhs, const TensorBuffer &rhs);
void add_inplace(TensorBuffer &lhs, const TensorBuffer &rhs);
[[nodiscard]] TensorBuffer silu(const TensorBuffer &input);
[[nodiscard]] TensorBuffer multiply(const TensorBuffer &lhs, const TensorBuffer &rhs);

}  // namespace sllmrf::ops
