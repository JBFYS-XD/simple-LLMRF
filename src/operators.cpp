#include "sllmrf/operators.h"

#include <bit>
#include <cstdint>
#include <cmath>
#include <string>
#include <string_view>

#if defined(SLLMRF_USE_OPENMP) && SLLMRF_USE_OPENMP
#include <omp.h>
#endif

#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
#include "operators_cuda.h"
#endif

namespace sllmrf::ops {

namespace {

void require_tensor_device(
    const TensorBuffer &tensor,
    const Device &device,
    std::string_view op_name,
    std::string_view arg_name) {
    if (tensor.device() != device) {
        throw GgufError(
            std::string(op_name) + " requires " + std::string(arg_name) +
            " on device " + device.to_string() + ", but got " + tensor.placement_string());
    }
}

void require_same_device(
    const TensorBuffer &lhs,
    const TensorBuffer &rhs,
    std::string_view op_name) {
    if (lhs.device() != rhs.device()) {
        throw GgufError(
            std::string(op_name) + " requires tensors on the same device, but got " +
            lhs.placement_string() + " and " + rhs.placement_string());
    }
}

TensorBuffer cpu_rms_norm(
    const TensorBuffer &input,
    const std::vector<float> &weight,
    float epsilon) {
    require_tensor_device(input, Device::cpu(), "rms_norm", "input");
    if (input.cols() != weight.size()) {
        throw GgufError("rms_norm weight size does not match tensor width");
    }

    TensorBuffer output(input.shape(), 0.0F, input.device());
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (std::int64_t row = 0; row < static_cast<std::int64_t>(input.rows()); ++row) {
        float mean_square = 0.0F;
        for (uint64_t col = 0; col < input.cols(); ++col) {
            const auto value = input.at(static_cast<uint64_t>(row), col);
            mean_square += value * value;
        }
        mean_square /= static_cast<float>(input.cols());

        const auto scale = 1.0F / std::sqrt(mean_square + epsilon);
        for (uint64_t col = 0; col < input.cols(); ++col) {
            output.at(static_cast<uint64_t>(row), col) =
                input.at(static_cast<uint64_t>(row), col) * scale * weight[static_cast<std::size_t>(col)];
        }
    }

    return output;
}

float half_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
    uint32_t exponent = static_cast<uint32_t>(bits & 0x7C00U) >> 10U;
    uint32_t mantissa = static_cast<uint32_t>(bits & 0x03FFU);

    uint32_t result = 0U;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            result = sign;
        } else {
            exponent = 127U - 15U + 1U;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                --exponent;
            }
            mantissa &= 0x03FFU;
            result = sign | (exponent << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        result = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        exponent += 127U - 15U;
        result = sign | (exponent << 23U) | (mantissa << 13U);
    }

    return std::bit_cast<float>(result);
}

std::vector<float> tensor_to_f32(const TensorView &tensor) {
    if (tensor.type != GgmlType::F32 && tensor.type != GgmlType::F16) {
        throw GgufError("tensor conversion to f32 is not implemented for tensor type " + to_string(tensor.type));
    }

    const auto element_count = static_cast<std::size_t>(
        tensor.dimensions.empty()
            ? 0U
            : tensor.row_width() * tensor.row_count());
    std::vector<float> values(element_count, 0.0F);
    if (tensor.type == GgmlType::F32) {
        const auto *source = reinterpret_cast<const float *>(tensor.data);
        for (std::size_t index = 0; index < element_count; ++index) {
            values[index] = source[index];
        }
        return values;
    }

    const auto *source = reinterpret_cast<const uint16_t *>(tensor.data);
    for (std::size_t index = 0; index < element_count; ++index) {
        values[index] = half_to_float(source[index]);
    }
    return values;
}

TensorBuffer cpu_rms_norm(
    const TensorBuffer &input,
    const TensorView &weight,
    float epsilon) {
    if (weight.dimensions.size() != 1U) {
        throw GgufError("rms_norm weight tensor must be rank 1: " + weight.name);
    }
    return cpu_rms_norm(input, tensor_to_f32(weight), epsilon);
}

TensorBuffer cpu_add(const TensorBuffer &lhs, const TensorBuffer &rhs) {
    require_same_device(lhs, rhs, "add");
    require_tensor_device(lhs, Device::cpu(), "add", "lhs");
    if (lhs.shape() != rhs.shape()) {
        throw GgufError("add requires tensors with identical shapes");
    }

    TensorBuffer output(lhs.shape(), 0.0F, lhs.device());
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(lhs.values().size()); ++index) {
        output.values()[static_cast<std::size_t>(index)] =
            lhs.values()[static_cast<std::size_t>(index)] + rhs.values()[static_cast<std::size_t>(index)];
    }
    return output;
}

void cpu_add_inplace(TensorBuffer &lhs, const TensorBuffer &rhs) {
    require_same_device(lhs, rhs, "add_inplace");
    require_tensor_device(lhs, Device::cpu(), "add_inplace", "lhs");
    if (lhs.shape() != rhs.shape()) {
        throw GgufError("add_inplace requires tensors with identical shapes");
    }

    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(lhs.values().size()); ++index) {
        lhs.values()[static_cast<std::size_t>(index)] += rhs.values()[static_cast<std::size_t>(index)];
    }
}

TensorBuffer cpu_silu(const TensorBuffer &input) {
    require_tensor_device(input, Device::cpu(), "silu", "input");
    TensorBuffer output(input.shape(), 0.0F, input.device());
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(input.values().size()); ++index) {
        const auto value = input.values()[static_cast<std::size_t>(index)];
        output.values()[static_cast<std::size_t>(index)] = value / (1.0F + std::exp(-value));
    }
    return output;
}

TensorBuffer cpu_multiply(const TensorBuffer &lhs, const TensorBuffer &rhs) {
    require_same_device(lhs, rhs, "multiply");
    require_tensor_device(lhs, Device::cpu(), "multiply", "lhs");
    if (lhs.shape() != rhs.shape()) {
        throw GgufError("multiply requires tensors with identical shapes");
    }

    TensorBuffer output(lhs.shape(), 0.0F, lhs.device());
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(lhs.values().size()); ++index) {
        output.values()[static_cast<std::size_t>(index)] =
            lhs.values()[static_cast<std::size_t>(index)] * rhs.values()[static_cast<std::size_t>(index)];
    }
    return output;
}

template <typename Func>
TensorBuffer emulate_cuda_unary(
    const TensorBuffer &input,
    const Device &device,
    Func &&func) {
    auto cpu_input = input.copy_to(Device::cpu());
    auto cpu_output = func(cpu_input);
    return cpu_output.copy_to(device);
}

template <typename Func>
TensorBuffer emulate_cuda_binary(
    const TensorBuffer &lhs,
    const TensorBuffer &rhs,
    const Device &device,
    Func &&func) {
    auto cpu_lhs = lhs.copy_to(Device::cpu());
    auto cpu_rhs = rhs.copy_to(Device::cpu());
    auto cpu_output = func(cpu_lhs, cpu_rhs);
    return cpu_output.copy_to(device);
}

template <typename Func>
void emulate_cuda_binary_inplace(
    TensorBuffer &lhs,
    const TensorBuffer &rhs,
    Func &&func) {
    auto cpu_lhs = lhs.copy_to(Device::cpu());
    auto cpu_rhs = rhs.copy_to(Device::cpu());
    func(cpu_lhs, cpu_rhs);
    lhs = cpu_lhs.copy_to(lhs.device());
}

void require_context_compatible(
    const Device &context_device,
    const Device &tensor_device,
    std::string_view op_name,
    std::string_view arg_name) {
    if (context_device != tensor_device) {
        throw GgufError(
            std::string(op_name) + " requires " + std::string(arg_name) +
            " on execution device " + context_device.to_string() + ", but got " +
            tensor_device.to_string());
    }
}

}  // namespace

TensorBuffer rms_norm(
    const TensorBuffer &input,
    const std::vector<float> &weight,
    float epsilon,
    OperatorContext context) {
    require_context_compatible(context.device, input.device(), "rms_norm", "input");
    if (context.device.is_cpu()) {
        return cpu_rms_norm(input, weight, epsilon);
    }
    if (context.device.is_cuda()) {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
        if (cuda_backend_enabled()) {
            return cuda::rms_norm(input, weight, epsilon);
        }
        return emulate_cuda_unary(
            input,
            context.device,
            [&](const TensorBuffer &cpu_input) { return cpu_rms_norm(cpu_input, weight, epsilon); });
#else
        return emulate_cuda_unary(
            input,
            context.device,
            [&](const TensorBuffer &cpu_input) { return cpu_rms_norm(cpu_input, weight, epsilon); });
#endif
    }

    throw GgufError("rms_norm native CUDA backend is not implemented yet");
}

TensorBuffer rms_norm(
    const TensorBuffer &input,
    const TensorView &weight,
    float epsilon,
    OperatorContext context) {
    require_context_compatible(context.device, input.device(), "rms_norm", "input");
    if (context.device.is_cpu()) {
        return cpu_rms_norm(input, weight, epsilon);
    }
    if (context.device.is_cuda()) {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
        if (cuda_backend_enabled()) {
            return cuda::rms_norm(input, weight, epsilon);
        }
        return emulate_cuda_unary(
            input,
            context.device,
            [&](const TensorBuffer &cpu_input) { return cpu_rms_norm(cpu_input, weight, epsilon); });
#else
        return emulate_cuda_unary(
            input,
            context.device,
            [&](const TensorBuffer &cpu_input) { return cpu_rms_norm(cpu_input, weight, epsilon); });
#endif
    }

    throw GgufError("rms_norm native CUDA backend is not implemented yet");
}

TensorBuffer add(const TensorBuffer &lhs, const TensorBuffer &rhs, OperatorContext context) {
    require_same_device(lhs, rhs, "add");
    require_context_compatible(context.device, lhs.device(), "add", "lhs");
    if (context.device.is_cpu()) {
        return cpu_add(lhs, rhs);
    }
    if (context.device.is_cuda()) {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
        if (cuda_backend_enabled()) {
            return cuda::add(lhs, rhs);
        }
        return emulate_cuda_binary(
            lhs,
            rhs,
            context.device,
            [](const TensorBuffer &cpu_lhs, const TensorBuffer &cpu_rhs) {
                return cpu_add(cpu_lhs, cpu_rhs);
            });
#else
        return emulate_cuda_binary(
            lhs,
            rhs,
            context.device,
            [](const TensorBuffer &cpu_lhs, const TensorBuffer &cpu_rhs) {
                return cpu_add(cpu_lhs, cpu_rhs);
            });
#endif
    }

    throw GgufError("add native CUDA backend is not implemented yet");
}

void add_inplace(TensorBuffer &lhs, const TensorBuffer &rhs, OperatorContext context) {
    require_same_device(lhs, rhs, "add_inplace");
    require_context_compatible(context.device, lhs.device(), "add_inplace", "lhs");
    if (context.device.is_cpu()) {
        cpu_add_inplace(lhs, rhs);
        return;
    }
    if (context.device.is_cuda()) {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
        if (cuda_backend_enabled()) {
            cuda::add_inplace(lhs, rhs);
            return;
        }
        emulate_cuda_binary_inplace(
            lhs,
            rhs,
            [](TensorBuffer &cpu_lhs, const TensorBuffer &cpu_rhs) {
                cpu_add_inplace(cpu_lhs, cpu_rhs);
            });
        return;
#else
        emulate_cuda_binary_inplace(
            lhs,
            rhs,
            [](TensorBuffer &cpu_lhs, const TensorBuffer &cpu_rhs) {
                cpu_add_inplace(cpu_lhs, cpu_rhs);
            });
        return;
#endif
    }

    throw GgufError("add_inplace native CUDA backend is not implemented yet");
}

TensorBuffer silu(const TensorBuffer &input, OperatorContext context) {
    require_context_compatible(context.device, input.device(), "silu", "input");
    if (context.device.is_cpu()) {
        return cpu_silu(input);
    }
    if (context.device.is_cuda()) {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
        if (cuda_backend_enabled()) {
            return cuda::silu(input);
        }
        return emulate_cuda_unary(
            input,
            context.device,
            [](const TensorBuffer &cpu_input) { return cpu_silu(cpu_input); });
#else
        return emulate_cuda_unary(
            input,
            context.device,
            [](const TensorBuffer &cpu_input) { return cpu_silu(cpu_input); });
#endif
    }

    throw GgufError("silu native CUDA backend is not implemented yet");
}

TensorBuffer multiply(const TensorBuffer &lhs, const TensorBuffer &rhs, OperatorContext context) {
    require_same_device(lhs, rhs, "multiply");
    require_context_compatible(context.device, lhs.device(), "multiply", "lhs");
    if (context.device.is_cpu()) {
        return cpu_multiply(lhs, rhs);
    }
    if (context.device.is_cuda()) {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
        if (cuda_backend_enabled()) {
            return cuda::multiply(lhs, rhs);
        }
        return emulate_cuda_binary(
            lhs,
            rhs,
            context.device,
            [](const TensorBuffer &cpu_lhs, const TensorBuffer &cpu_rhs) {
                return cpu_multiply(cpu_lhs, cpu_rhs);
            });
#else
        return emulate_cuda_binary(
            lhs,
            rhs,
            context.device,
            [](const TensorBuffer &cpu_lhs, const TensorBuffer &cpu_rhs) {
                return cpu_multiply(cpu_lhs, cpu_rhs);
            });
#endif
    }

    throw GgufError("multiply native CUDA backend is not implemented yet");
}

}  // namespace sllmrf::ops
