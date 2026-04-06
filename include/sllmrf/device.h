#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sllmrf {

enum class DeviceType {
    Cpu,
    Cuda,
};

struct Device {
    DeviceType type {DeviceType::Cpu};
    uint32_t index {0};

    [[nodiscard]] static constexpr Device cpu() noexcept {
        return Device {.type = DeviceType::Cpu, .index = 0U};
    }

    [[nodiscard]] static constexpr Device cuda(uint32_t device_index = 0U) noexcept {
        return Device {.type = DeviceType::Cuda, .index = device_index};
    }

    [[nodiscard]] constexpr bool is_cpu() const noexcept {
        return type == DeviceType::Cpu;
    }

    [[nodiscard]] constexpr bool is_cuda() const noexcept {
        return type == DeviceType::Cuda;
    }

    [[nodiscard]] std::string to_string() const {
        if (is_cpu()) {
            return "cpu:0";
        }
        return "cuda:" + std::to_string(index);
    }

    [[nodiscard]] constexpr bool operator==(const Device &other) const noexcept {
        return type == other.type && index == other.index;
    }

    [[nodiscard]] constexpr bool operator!=(const Device &other) const noexcept {
        return !(*this == other);
    }
};

[[nodiscard]] bool cuda_backend_enabled() noexcept;
[[nodiscard]] std::string describe_device_backend(Device device);

class DeviceAllocation {
public:
    DeviceAllocation() = default;
    ~DeviceAllocation() noexcept;
    DeviceAllocation(Device device, std::size_t size_bytes);
    DeviceAllocation(const DeviceAllocation &other);
    DeviceAllocation &operator=(const DeviceAllocation &other);
    DeviceAllocation(DeviceAllocation &&other) noexcept;
    DeviceAllocation &operator=(DeviceAllocation &&other) noexcept;

    void resize(Device device, std::size_t size_bytes);
    void reset() noexcept;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t size_bytes() const noexcept;
    [[nodiscard]] const Device &device() const noexcept;
    [[nodiscard]] bool is_emulated() const noexcept;
    [[nodiscard]] void *data() noexcept;
    [[nodiscard]] const void *data() const noexcept;

    void copy_from_host(const void *source, std::size_t size_bytes);
    void copy_to_host(void *destination, std::size_t size_bytes) const;

private:
    Device device_ {Device::cpu()};
    std::vector<std::byte> bytes_;
    void *native_ptr_ {nullptr};
    std::size_t size_bytes_ {0U};
    bool emulated_ {false};
};

}  // namespace sllmrf
