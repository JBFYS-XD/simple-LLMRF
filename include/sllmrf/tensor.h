#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "sllmrf/device.h"

namespace sllmrf {

class TensorBuffer {
public:
    TensorBuffer() = default;
    explicit TensorBuffer(
        std::vector<uint64_t> shape,
        float initial_value = 0.0F,
        Device device = Device::cpu());
    TensorBuffer(
        std::vector<uint64_t> shape,
        std::vector<float> values,
        Device device = Device::cpu());

    [[nodiscard]] const std::vector<uint64_t> &shape() const noexcept;
    [[nodiscard]] std::size_t rank() const noexcept;
    [[nodiscard]] uint64_t element_count() const noexcept;
    [[nodiscard]] uint64_t rows() const noexcept;
    [[nodiscard]] uint64_t cols() const noexcept;
    [[nodiscard]] std::string shape_string() const;
    [[nodiscard]] const Device &device() const noexcept;
    [[nodiscard]] std::string placement_string() const;
    [[nodiscard]] bool is_on_device(const Device &device) const noexcept;
    [[nodiscard]] bool has_device_allocation() const noexcept;
    [[nodiscard]] bool is_device_allocation_emulated() const noexcept;
    [[nodiscard]] void *device_data() noexcept;
    [[nodiscard]] const void *device_data() const noexcept;
    [[nodiscard]] bool host_dirty() const noexcept;
    [[nodiscard]] bool device_dirty() const noexcept;
    void set_device(Device device);
    void sync_host_to_device();
    void sync_device_to_host();
    void mark_device_dirty() noexcept;
    [[nodiscard]] TensorBuffer copy_to(Device device) const;
    [[nodiscard]] float &at(uint64_t row, uint64_t col);
    [[nodiscard]] const float &at(uint64_t row, uint64_t col) const;
    [[nodiscard]] std::vector<float> row(uint64_t row_index) const;
    void fill(float value);

    [[nodiscard]] const std::vector<float> &values() const noexcept;
    [[nodiscard]] std::vector<float> &values() noexcept;
    [[nodiscard]] float *data() noexcept;
    [[nodiscard]] const float *data() const noexcept;

private:
    [[nodiscard]] std::size_t byte_size() const noexcept;
    [[nodiscard]] std::vector<float> snapshot_host_values() const;
    void mark_host_dirty() noexcept;

    std::vector<uint64_t> shape_;
    std::vector<float> values_;
    Device device_ {Device::cpu()};
    std::optional<DeviceAllocation> device_allocation_;
    bool host_dirty_ {false};
    bool device_dirty_ {false};
};

}  // namespace sllmrf
