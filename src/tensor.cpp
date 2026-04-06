#include "sllmrf/tensor.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace sllmrf {

TensorBuffer::TensorBuffer(std::vector<uint64_t> shape, float initial_value, Device device)
    : shape_(std::move(shape)),
      values_(static_cast<std::size_t>(element_count()), initial_value),
      device_(device) {
    if (!device_.is_cpu()) {
        sync_host_to_device();
    }
}

TensorBuffer::TensorBuffer(std::vector<uint64_t> shape, std::vector<float> values, Device device)
    : shape_(std::move(shape)),
      values_(std::move(values)),
      device_(device) {
    if (values_.size() != static_cast<std::size_t>(element_count())) {
        throw std::runtime_error("tensor value count does not match tensor shape");
    }
    if (!device_.is_cpu()) {
        sync_host_to_device();
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

const Device &TensorBuffer::device() const noexcept {
    return device_;
}

std::string TensorBuffer::placement_string() const {
    return device_.to_string();
}

bool TensorBuffer::is_on_device(const Device &device) const noexcept {
    return device_ == device;
}

bool TensorBuffer::has_device_allocation() const noexcept {
    return device_allocation_.has_value() && !device_allocation_->empty();
}

bool TensorBuffer::is_device_allocation_emulated() const noexcept {
    return device_allocation_.has_value() && device_allocation_->is_emulated();
}

void *TensorBuffer::device_data() noexcept {
    return device_allocation_.has_value() ? device_allocation_->data() : nullptr;
}

const void *TensorBuffer::device_data() const noexcept {
    return device_allocation_.has_value() ? device_allocation_->data() : nullptr;
}

bool TensorBuffer::host_dirty() const noexcept {
    return host_dirty_;
}

bool TensorBuffer::device_dirty() const noexcept {
    return device_dirty_;
}

void TensorBuffer::set_device(Device device) {
    if (device == device_) {
        if (!device.is_cpu() && (host_dirty_ || !has_device_allocation())) {
            sync_host_to_device();
        }
        return;
    }

    if (device_dirty_) {
        sync_device_to_host();
    }

    device_ = device;
    if (device_.is_cpu()) {
        device_allocation_.reset();
        host_dirty_ = false;
        device_dirty_ = false;
        return;
    }

    sync_host_to_device();
}

void TensorBuffer::sync_host_to_device() {
    if (device_.is_cpu()) {
        device_allocation_.reset();
        device_dirty_ = false;
        return;
    }

    if (!device_allocation_.has_value()) {
        device_allocation_.emplace(device_, byte_size());
    } else if (device_allocation_->device() != device_ || device_allocation_->size_bytes() != byte_size()) {
        device_allocation_->resize(device_, byte_size());
    }

    device_allocation_->copy_from_host(values_.data(), byte_size());
    host_dirty_ = false;
    device_dirty_ = false;
}

void TensorBuffer::sync_device_to_host() {
    if (device_.is_cpu()) {
        device_dirty_ = false;
        return;
    }
    if (!has_device_allocation()) {
        throw std::runtime_error("tensor does not have a device allocation to copy from");
    }

    device_allocation_->copy_to_host(values_.data(), byte_size());
    host_dirty_ = false;
    device_dirty_ = false;
}

void TensorBuffer::mark_device_dirty() noexcept {
    device_dirty_ = true;
    host_dirty_ = false;
}

TensorBuffer TensorBuffer::copy_to(Device device) const {
    TensorBuffer result(shape_, snapshot_host_values(), device);
    if (!device.is_cpu()) {
        result.sync_host_to_device();
    }
    return result;
}

float &TensorBuffer::at(uint64_t row, uint64_t col) {
    mark_host_dirty();
    return values_.at(static_cast<std::size_t>(row * cols() + col));
}

const float &TensorBuffer::at(uint64_t row, uint64_t col) const {
    return values_.at(static_cast<std::size_t>(row * cols() + col));
}

std::vector<float> TensorBuffer::row(uint64_t row_index) const {
    const auto host_values = snapshot_host_values();
    std::vector<float> values(cols(), 0.0F);
    const auto start = static_cast<std::size_t>(row_index * cols());
    for (uint64_t col = 0; col < cols(); ++col) {
        values[static_cast<std::size_t>(col)] =
            host_values.at(start + static_cast<std::size_t>(col));
    }
    return values;
}

void TensorBuffer::fill(float value) {
    std::fill(values_.begin(), values_.end(), value);
    mark_host_dirty();
}

const std::vector<float> &TensorBuffer::values() const noexcept {
    return values_;
}

std::vector<float> &TensorBuffer::values() noexcept {
    mark_host_dirty();
    return values_;
}

float *TensorBuffer::data() noexcept {
    mark_host_dirty();
    return values_.data();
}

const float *TensorBuffer::data() const noexcept {
    return values_.data();
}

std::size_t TensorBuffer::byte_size() const noexcept {
    return values_.size() * sizeof(float);
}

std::vector<float> TensorBuffer::snapshot_host_values() const {
    if (!device_dirty_) {
        return values_;
    }

    if (!has_device_allocation()) {
        throw std::runtime_error("tensor device state is dirty but no device allocation exists");
    }

    std::vector<float> snapshot(values_.size(), 0.0F);
    device_allocation_->copy_to_host(snapshot.data(), byte_size());
    return snapshot;
}

void TensorBuffer::mark_host_dirty() noexcept {
    host_dirty_ = true;
    if (!device_.is_cpu()) {
        device_dirty_ = false;
    }
}

}  // namespace sllmrf
