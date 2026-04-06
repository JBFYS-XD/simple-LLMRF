#pragma once

#include <vector>

#include "sllmrf/gguf.h"
#include "sllmrf/tensor.h"

namespace sllmrf::ops {

struct OperatorContext {
    Device device {Device::cpu()};

    [[nodiscard]] static constexpr OperatorContext cpu() noexcept {
        return OperatorContext {.device = Device::cpu()};
    }

    [[nodiscard]] static constexpr OperatorContext cuda(uint32_t device_index = 0U) noexcept {
        return OperatorContext {.device = Device::cuda(device_index)};
    }
};

[[nodiscard]] TensorBuffer rms_norm(
    const TensorBuffer &input,
    const std::vector<float> &weight,
    float epsilon,
    OperatorContext context = OperatorContext::cpu());
[[nodiscard]] TensorBuffer rms_norm(
    const TensorBuffer &input,
    const TensorView &weight,
    float epsilon,
    OperatorContext context = OperatorContext::cpu());
[[nodiscard]] TensorBuffer add(
    const TensorBuffer &lhs,
    const TensorBuffer &rhs,
    OperatorContext context = OperatorContext::cpu());
void add_inplace(
    TensorBuffer &lhs,
    const TensorBuffer &rhs,
    OperatorContext context = OperatorContext::cpu());
[[nodiscard]] TensorBuffer silu(
    const TensorBuffer &input,
    OperatorContext context = OperatorContext::cpu());
[[nodiscard]] TensorBuffer multiply(
    const TensorBuffer &lhs,
    const TensorBuffer &rhs,
    OperatorContext context = OperatorContext::cpu());

}  // namespace sllmrf::ops
