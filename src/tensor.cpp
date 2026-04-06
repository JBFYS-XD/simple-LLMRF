#include "sllmrf/tensor.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace sllmrf {

TensorBuffer::TensorBuffer(std::vector<uint64_t> shape, float initial_value)
    : shape_(std::move(shape)),
      values_(static_cast<std::size_t>(element_count()), initial_value) {}

TensorBuffer::TensorBuffer(std::vector<uint64_t> shape, std::vector<float> values)
    : shape_(std::move(shape)),
      values_(std::move(values)) {
    if (values_.size() != static_cast<std::size_t>(element_count())) {
        throw std::runtime_error("tensor value count does not match tensor shape");
    }
}

const std::vector<uint64_t> &TensorBuffer::shape() const noexcept {
    return shape_;
}

std::size_t TensorBuffer::rank() const noexcept {
    return shape_.size();
}

uint64_t TensorBuffer::element_count() const noexcept {
    if (shape_.empty()) {
        return 0U;
    }

    return std::accumulate(
        shape_.begin(),
        shape_.end(),
        uint64_t {1},
        [](uint64_t lhs, uint64_t rhs) { return lhs * rhs; });
}

uint64_t TensorBuffer::rows() const noexcept {
    if (shape_.empty()) {
        return 0U;
    }
    if (shape_.size() == 1U) {
        return 1U;
    }
    return shape_.front();
}

uint64_t TensorBuffer::cols() const noexcept {
    if (shape_.empty()) {
        return 0U;
    }
    if (shape_.size() == 1U) {
        return shape_.front();
    }

    uint64_t cols = 1U;
    for (std::size_t index = 1; index < shape_.size(); ++index) {
        cols *= shape_[index];
    }
    return cols;
}

std::string TensorBuffer::shape_string() const {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < shape_.size(); ++index) {
        if (index > 0U) {
            stream << " x ";
        }
        stream << shape_[index];
    }
    stream << ']';
    return stream.str();
}

float &TensorBuffer::at(uint64_t row, uint64_t col) {
    return values_.at(static_cast<std::size_t>(row * cols() + col));
}

const float &TensorBuffer::at(uint64_t row, uint64_t col) const {
    return values_.at(static_cast<std::size_t>(row * cols() + col));
}

std::vector<float> TensorBuffer::row(uint64_t row_index) const {
    std::vector<float> values(cols(), 0.0F);
    const auto start = static_cast<std::size_t>(row_index * cols());
    for (uint64_t col = 0; col < cols(); ++col) {
        values[static_cast<std::size_t>(col)] = values_.at(start + static_cast<std::size_t>(col));
    }
    return values;
}

void TensorBuffer::fill(float value) {
    std::fill(values_.begin(), values_.end(), value);
}

const std::vector<float> &TensorBuffer::values() const noexcept {
    return values_;
}

std::vector<float> &TensorBuffer::values() noexcept {
    return values_;
}

float *TensorBuffer::data() noexcept {
    return values_.data();
}

const float *TensorBuffer::data() const noexcept {
    return values_.data();
}

}  // namespace sllmrf
