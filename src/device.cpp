#include "sllmrf/device.h"

#include <cstring>
#include <stdexcept>
#include <string_view>
#include <utility>

#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace sllmrf {

namespace {

bool is_emulated_device(Device device) noexcept {
    return device.is_cuda() && !cuda_backend_enabled();
}

#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
void throw_on_cuda_error(cudaError_t error, std::string_view operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(error));
    }
}

void set_cuda_device(Device device) {
    if (device.is_cuda()) {
        throw_on_cuda_error(cudaSetDevice(static_cast<int>(device.index)), "cudaSetDevice");
    }
}

bool detect_native_cuda_backend() noexcept {
    int device_count = 0;
    const auto error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return device_count > 0;
}
#endif

}  // namespace

bool cuda_backend_enabled() noexcept {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
    static const bool enabled = detect_native_cuda_backend();
    return enabled;
#else
    return false;
#endif
}

std::string describe_device_backend(Device device) {
    if (device.is_cpu()) {
        return "native cpu backend";
    }
    if (cuda_backend_enabled()) {
        return "native cuda backend";
    }
    return "emulated cuda staging backend";
}

DeviceAllocation::~DeviceAllocation() noexcept {
    reset();
}

DeviceAllocation::DeviceAllocation(Device device, std::size_t size_bytes) {
    resize(device, size_bytes);
}

DeviceAllocation::DeviceAllocation(const DeviceAllocation &other) {
    if (!other.empty()) {
        resize(other.device_, other.size_bytes_);
        if (size_bytes_ > 0U) {
            if (emulated_) {
                bytes_ = other.bytes_;
            } else {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
                set_cuda_device(device_);
                throw_on_cuda_error(
                    cudaMemcpy(native_ptr_, other.native_ptr_, size_bytes_, cudaMemcpyDeviceToDevice),
                    "cudaMemcpyDeviceToDevice");
#else
                (void)other;
                throw std::runtime_error("native CUDA copy requested without CUDA support");
#endif
            }
        }
    }
}

DeviceAllocation &DeviceAllocation::operator=(const DeviceAllocation &other) {
    if (this == &other) {
        return *this;
    }

    reset();
    if (!other.empty()) {
        resize(other.device_, other.size_bytes_);
        if (size_bytes_ > 0U) {
            if (emulated_) {
                bytes_ = other.bytes_;
            } else {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
                set_cuda_device(device_);
                throw_on_cuda_error(
                    cudaMemcpy(native_ptr_, other.native_ptr_, size_bytes_, cudaMemcpyDeviceToDevice),
                    "cudaMemcpyDeviceToDevice");
#else
                (void)other;
                throw std::runtime_error("native CUDA copy requested without CUDA support");
#endif
            }
        }
    }
    return *this;
}

DeviceAllocation::DeviceAllocation(DeviceAllocation &&other) noexcept
    : device_(other.device_),
      bytes_(std::move(other.bytes_)),
      native_ptr_(other.native_ptr_),
      size_bytes_(other.size_bytes_),
      emulated_(other.emulated_) {
    other.device_ = Device::cpu();
    other.native_ptr_ = nullptr;
    other.size_bytes_ = 0U;
    other.emulated_ = false;
}

DeviceAllocation &DeviceAllocation::operator=(DeviceAllocation &&other) noexcept {
    if (this == &other) {
        return *this;
    }

    reset();
    device_ = other.device_;
    bytes_ = std::move(other.bytes_);
    native_ptr_ = other.native_ptr_;
    size_bytes_ = other.size_bytes_;
    emulated_ = other.emulated_;

    other.device_ = Device::cpu();
    other.native_ptr_ = nullptr;
    other.size_bytes_ = 0U;
    other.emulated_ = false;
    return *this;
}

void DeviceAllocation::resize(Device device, std::size_t size_bytes) {
    reset();
    device_ = device;
    size_bytes_ = size_bytes;
    emulated_ = device.is_cpu() || is_emulated_device(device);

    if (size_bytes_ == 0U) {
        return;
    }

    if (emulated_) {
        bytes_.assign(size_bytes_, std::byte {0});
        return;
    }

#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
    set_cuda_device(device_);
    throw_on_cuda_error(cudaMalloc(&native_ptr_, size_bytes_), "cudaMalloc");
#else
    throw std::runtime_error("native CUDA allocation requested without CUDA support");
#endif
}

void DeviceAllocation::reset() noexcept {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
    if (native_ptr_ != nullptr) {
        if (device_.is_cuda()) {
            (void)cudaSetDevice(static_cast<int>(device_.index));
        }
        (void)cudaFree(native_ptr_);
    }
#endif
    device_ = Device::cpu();
    bytes_.clear();
    native_ptr_ = nullptr;
    size_bytes_ = 0U;
    emulated_ = false;
}

bool DeviceAllocation::empty() const noexcept {
    return size_bytes_ == 0U;
}

std::size_t DeviceAllocation::size_bytes() const noexcept {
    return size_bytes_;
}

const Device &DeviceAllocation::device() const noexcept {
    return device_;
}

bool DeviceAllocation::is_emulated() const noexcept {
    return emulated_;
}

void *DeviceAllocation::data() noexcept {
    return emulated_ ? static_cast<void *>(bytes_.data()) : native_ptr_;
}

const void *DeviceAllocation::data() const noexcept {
    return emulated_ ? static_cast<const void *>(bytes_.data()) : native_ptr_;
}

void DeviceAllocation::copy_from_host(const void *source, std::size_t size_bytes) {
    if (size_bytes != size_bytes_) {
        throw std::runtime_error("device allocation copy_from_host size mismatch");
    }
    if (size_bytes == 0U) {
        return;
    }

    if (emulated_) {
        std::memcpy(bytes_.data(), source, size_bytes);
        return;
    }

#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
    set_cuda_device(device_);
    throw_on_cuda_error(
        cudaMemcpy(native_ptr_, source, size_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpyHostToDevice");
#else
    throw std::runtime_error("native CUDA upload requested without CUDA support");
#endif
}

void DeviceAllocation::copy_to_host(void *destination, std::size_t size_bytes) const {
    if (size_bytes != size_bytes_) {
        throw std::runtime_error("device allocation copy_to_host size mismatch");
    }
    if (size_bytes == 0U) {
        return;
    }

    if (emulated_) {
        std::memcpy(destination, bytes_.data(), size_bytes);
        return;
    }

#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
    set_cuda_device(device_);
    throw_on_cuda_error(
        cudaMemcpy(destination, native_ptr_, size_bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpyDeviceToHost");
#else
    throw std::runtime_error("native CUDA download requested without CUDA support");
#endif
}

}  // namespace sllmrf
