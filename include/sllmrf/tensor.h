#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sllmrf {

class TensorBuffer {
public:
    TensorBuffer() = default;
    explicit TensorBuffer(std::vector<uint64_t> shape, float initial_value = 0.0F);
    TensorBuffer(std::vector<uint64_t> shape, std::vector<float> values);

    [[nodiscard]] const std::vector<uint64_t> &shape() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] uint64_t element_count() const noexcept;
    [[nodiscard]] uint64_t rows() const noexcept;
    [[nodiscard]] uint64_t cols() const noexcept;
    [[nodiscard]] std::string shape_string() const;
    [[nodiscard]] float &at(uint64_t row, uint64_t col);
    [[nodiscard]] const float &at(uint64_t row, uint64_t col) const;
    [[nodiscard]] std::vector<float> row(uint64_t row_index) const;
    void fill(float value);

    [[nodiscard]] const std::vector<float> &values() const noexcept;
    [[nodiscard]] std::vector<float> &values() noexcept;
    [[nodiscard]] float *data() noexcept;
    [[nodiscard]] const float *data() const noexcept;

private:
    std::vector<uint64_t> shape_;
    std::vector<float> values_;
};

}  // namespace sllmrf
