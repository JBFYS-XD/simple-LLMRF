#include "sllmrf/operators.h"

#include <cstdint>
#include <cmath>

#if defined(SLLMRF_USE_OPENMP) && SLLMRF_USE_OPENMP
#include <omp.h>
#endif

namespace sllmrf::ops {

TensorBuffer rms_norm(
    const TensorBuffer &input,
    const std::vector<float> &weight,
    float epsilon) {
    if (input.cols() != weight.size()) {
        throw GgufError("rms_norm weight size does not match tensor width");
    }

    TensorBuffer output(input.shape());
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

TensorBuffer add(const TensorBuffer &lhs, const TensorBuffer &rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw GgufError("add requires tensors with identical shapes");
    }

    TensorBuffer output(lhs.shape());
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(lhs.values().size()); ++index) {
        output.values()[static_cast<std::size_t>(index)] =
            lhs.values()[static_cast<std::size_t>(index)] + rhs.values()[static_cast<std::size_t>(index)];
    }
    return output;
}

void add_inplace(TensorBuffer &lhs, const TensorBuffer &rhs) {
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

TensorBuffer silu(const TensorBuffer &input) {
    TensorBuffer output(input.shape());
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(input.values().size()); ++index) {
        const auto value = input.values()[static_cast<std::size_t>(index)];
        output.values()[static_cast<std::size_t>(index)] = value / (1.0F + std::exp(-value));
    }
    return output;
}

TensorBuffer multiply(const TensorBuffer &lhs, const TensorBuffer &rhs) {
    if (lhs.shape() != rhs.shape()) {
        throw GgufError("multiply requires tensors with identical shapes");
    }

    TensorBuffer output(lhs.shape());
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(lhs.values().size()); ++index) {
        output.values()[static_cast<std::size_t>(index)] =
            lhs.values()[static_cast<std::size_t>(index)] * rhs.values()[static_cast<std::size_t>(index)];
    }
    return output;
}

}  // namespace sllmrf::ops
