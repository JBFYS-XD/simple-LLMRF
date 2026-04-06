#include "operators_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "sllmrf/gguf.h"

namespace sllmrf::ops::cuda {

namespace {

void throw_on_cuda_error(cudaError_t error, std::string_view operation) {
    if (error != cudaSuccess) {
        throw GgufError(
            std::string(operation) + " failed: " + cudaGetErrorString(error));
    }
}

void set_cuda_device(const Device &device) {
    if (!device.is_cuda()) {
        throw GgufError("native CUDA operator received a non-CUDA tensor");
    }
    throw_on_cuda_error(cudaSetDevice(static_cast<int>(device.index)), "cudaSetDevice");
}

void require_cuda_tensor(
    const TensorBuffer &tensor,
    std::string_view op_name,
    std::string_view arg_name) {
    if (!tensor.device().is_cuda()) {
        throw GgufError(
            std::string(op_name) + " requires " + std::string(arg_name) + " on a CUDA device");
    }
    if (!tensor.has_device_allocation() || tensor.device_data() == nullptr) {
        throw GgufError(
            std::string(op_name) + " requires " + std::string(arg_name) +
            " to have an allocated device buffer");
    }
}

void require_same_shape(
    const TensorBuffer &lhs,
    const TensorBuffer &rhs,
    std::string_view op_name) {
    if (lhs.shape() != rhs.shape()) {
        throw GgufError(std::string(op_name) + " requires tensors with identical shapes");
    }
}

void require_same_device(
    const TensorBuffer &lhs,
    const TensorBuffer &rhs,
    std::string_view op_name) {
    if (lhs.device() != rhs.device()) {
        throw GgufError(
            std::string(op_name) + " requires tensors on the same CUDA device");
    }
}

struct WeightCacheKey {
    Device device;
    const void *host_data {nullptr};
    std::size_t byte_size {0U};
    GgmlType type {GgmlType::F32};

    [[nodiscard]] bool operator==(const WeightCacheKey &other) const noexcept {
        return device == other.device &&
            host_data == other.host_data &&
            byte_size == other.byte_size &&
            type == other.type;
    }
};

struct WeightCacheKeyHasher {
    [[nodiscard]] std::size_t operator()(const WeightCacheKey &key) const noexcept {
        auto hash = std::hash<const void *> {}(key.host_data);
        hash ^= std::hash<std::size_t> {}(key.byte_size) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        hash ^= std::hash<uint32_t> {}(static_cast<uint32_t>(key.type)) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        hash ^= std::hash<uint32_t> {}(key.device.index) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        hash ^= std::hash<uint32_t> {}(static_cast<uint32_t>(key.device.type)) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
        return hash;
    }
};

struct CachedWeight {
    DeviceAllocation allocation;
};

template <typename T>
class ScopedCudaBuffer {
public:
    explicit ScopedCudaBuffer(std::size_t count) : count_(count) {
        if (count_ == 0U) {
            return;
        }
        throw_on_cuda_error(cudaMalloc(&ptr_, count_ * sizeof(T)), "cudaMalloc");
    }

    ~ScopedCudaBuffer() {
        if (ptr_ != nullptr) {
            (void)cudaFree(ptr_);
        }
    }

    ScopedCudaBuffer(const ScopedCudaBuffer &) = delete;
    ScopedCudaBuffer &operator=(const ScopedCudaBuffer &) = delete;

    [[nodiscard]] T *data() noexcept {
        return ptr_;
    }

    [[nodiscard]] const T *data() const noexcept {
        return ptr_;
    }

private:
    T *ptr_ {nullptr};
    std::size_t count_ {0U};
};

int default_block_size(std::size_t element_count) {
    if (element_count <= 64U) {
        return 64;
    }
    if (element_count <= 128U) {
        return 128;
    }
    return 256;
}

void finalize_kernel(TensorBuffer &tensor, std::string_view op_name) {
    throw_on_cuda_error(cudaGetLastError(), op_name);
    throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    tensor.mark_device_dirty();
}

const void *get_cached_weight_device_data(
    const TensorView &weight,
    const Device &device) {
    static std::mutex cache_mutex;
    static std::unordered_map<WeightCacheKey, CachedWeight, WeightCacheKeyHasher> cache;

    const WeightCacheKey key {
        .device = device,
        .host_data = weight.data,
        .byte_size = weight.byte_size,
        .type = weight.type,
    };

    std::scoped_lock lock(cache_mutex);
    auto found = cache.find(key);
    if (found == cache.end()) {
        CachedWeight cached_weight;
        cached_weight.allocation.resize(device, weight.byte_size);
        cached_weight.allocation.copy_from_host(weight.data, weight.byte_size);
        found = cache.emplace(key, std::move(cached_weight)).first;
    }

    return found->second.allocation.data();
}

__global__ void add_kernel(
    const float *lhs,
    const float *rhs,
    float *output,
    std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = lhs[index] + rhs[index];
    }
}

__global__ void add_inplace_kernel(
    float *lhs,
    const float *rhs,
    std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        lhs[index] += rhs[index];
    }
}

__global__ void silu_kernel(
    const float *input,
    float *output,
    std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        const auto value = input[index];
        output[index] = value / (1.0F + __expf(-value));
    }
}

__global__ void multiply_kernel(
    const float *lhs,
    const float *rhs,
    float *output,
    std::size_t element_count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = lhs[index] * rhs[index];
    }
}

__global__ void rms_norm_kernel(
    const float *input,
    const float *weight,
    float *output,
    std::size_t rows,
    std::size_t cols,
    float epsilon) {
    const auto row = static_cast<std::size_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    extern __shared__ float partial_sums[];

    float local_sum = 0.0F;
    const auto row_offset = row * cols;
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const auto value = input[row_offset + col];
        local_sum += value * value;
    }

    partial_sums[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const auto scale = rsqrtf(partial_sums[0] / static_cast<float>(cols) + epsilon);
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        output[row_offset + col] = input[row_offset + col] * scale * weight[col];
    }
}

__global__ void rms_norm_f16_kernel(
    const float *input,
    const __half *weight,
    float *output,
    std::size_t rows,
    std::size_t cols,
    float epsilon) {
    const auto row = static_cast<std::size_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }

    extern __shared__ float partial_sums[];

    float local_sum = 0.0F;
    const auto row_offset = row * cols;
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const auto value = input[row_offset + col];
        local_sum += value * value;
    }

    partial_sums[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const auto scale = rsqrtf(partial_sums[0] / static_cast<float>(cols) + epsilon);
    for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        output[row_offset + col] =
            input[row_offset + col] * scale * __half2float(weight[col]);
    }
}

__global__ void linear_project_f32_kernel(
    const float *input,
    const float *weight,
    float *output,
    std::size_t row_count,
    std::size_t input_width,
    std::size_t output_width) {
    const auto flat_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = row_count * output_width;
    if (flat_index >= total) {
        return;
    }

    const auto row = flat_index / output_width;
    const auto output_col = flat_index % output_width;
    const auto *input_row = input + row * input_width;
    const auto *weight_row = weight + output_col * input_width;

    float sum = 0.0F;
    for (std::size_t index = 0; index < input_width; ++index) {
        sum += input_row[index] * weight_row[index];
    }
    output[flat_index] = sum;
}

__global__ void linear_project_f16_kernel(
    const float *input,
    const __half *weight,
    float *output,
    std::size_t row_count,
    std::size_t input_width,
    std::size_t output_width) {
    const auto flat_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = row_count * output_width;
    if (flat_index >= total) {
        return;
    }

    const auto row = flat_index / output_width;
    const auto output_col = flat_index % output_width;
    const auto *input_row = input + row * input_width;
    const auto *weight_row = weight + output_col * input_width;

    float sum = 0.0F;
    for (std::size_t index = 0; index < input_width; ++index) {
        sum += input_row[index] * __half2float(weight_row[index]);
    }
    output[flat_index] = sum;
}

__global__ void output_projection_f32_kernel(
    const float *last_token,
    const float *weight,
    float *logits,
    std::size_t vocab_size,
    std::size_t hidden_size) {
    const auto token_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token_index >= vocab_size) {
        return;
    }

    const auto *weight_row = weight + token_index * hidden_size;
    float sum = 0.0F;
    for (std::size_t hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
        sum += last_token[hidden_index] * weight_row[hidden_index];
    }
    logits[token_index] = sum;
}

__global__ void output_projection_f16_kernel(
    const float *last_token,
    const __half *weight,
    float *logits,
    std::size_t vocab_size,
    std::size_t hidden_size) {
    const auto token_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token_index >= vocab_size) {
        return;
    }

    const auto *weight_row = weight + token_index * hidden_size;
    float sum = 0.0F;
    for (std::size_t hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
        sum += last_token[hidden_index] * __half2float(weight_row[hidden_index]);
    }
    logits[token_index] = sum;
}

__global__ void embedding_lookup_f32_kernel(
    const uint32_t *token_ids,
    const float *weight,
    float *output,
    std::size_t token_count,
    std::size_t hidden_size) {
    const auto flat_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = token_count * hidden_size;
    if (flat_index >= total) {
        return;
    }

    const auto token_index = flat_index / hidden_size;
    const auto hidden_index = flat_index % hidden_size;
    const auto vocab_row = static_cast<std::size_t>(token_ids[token_index]);
    output[flat_index] = weight[vocab_row * hidden_size + hidden_index];
}

__global__ void embedding_lookup_f16_kernel(
    const uint32_t *token_ids,
    const __half *weight,
    float *output,
    std::size_t token_count,
    std::size_t hidden_size) {
    const auto flat_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = token_count * hidden_size;
    if (flat_index >= total) {
        return;
    }

    const auto token_index = flat_index / hidden_size;
    const auto hidden_index = flat_index % hidden_size;
    const auto vocab_row = static_cast<std::size_t>(token_ids[token_index]);
    output[flat_index] = __half2float(weight[vocab_row * hidden_size + hidden_index]);
}

}  // namespace

TensorBuffer rms_norm(
    const TensorBuffer &input,
    const std::vector<float> &weight,
    float epsilon) {
    require_cuda_tensor(input, "rms_norm", "input");
    if (input.cols() != weight.size()) {
        throw GgufError("rms_norm weight size does not match tensor width");
    }

    TensorBuffer output(input.shape(), 0.0F, input.device());
    if (input.element_count() == 0U) {
        return output;
    }

    set_cuda_device(input.device());
    ScopedCudaBuffer<float> weight_buffer(weight.size());
    throw_on_cuda_error(
        cudaMemcpy(
            weight_buffer.data(),
            weight.data(),
            weight.size() * sizeof(float),
            cudaMemcpyHostToDevice),
        "cudaMemcpyHostToDevice");

    const auto block_size = default_block_size(weight.size());
    rms_norm_kernel<<<static_cast<unsigned int>(input.rows()), block_size, static_cast<std::size_t>(block_size) * sizeof(float)>>>(
        static_cast<const float *>(input.device_data()),
        weight_buffer.data(),
        static_cast<float *>(output.device_data()),
        static_cast<std::size_t>(input.rows()),
        static_cast<std::size_t>(input.cols()),
        epsilon);
    finalize_kernel(output, "rms_norm");
    return output;
}

TensorBuffer rms_norm(
    const TensorBuffer &input,
    const TensorView &weight,
    float epsilon) {
    require_cuda_tensor(input, "rms_norm", "input");
    if (weight.dimensions.size() != 1U) {
        throw GgufError("rms_norm weight tensor must be rank 1: " + weight.name);
    }
    if (input.cols() != weight.dimensions[0]) {
        throw GgufError("rms_norm weight size does not match tensor width");
    }
    if (weight.type != GgmlType::F32 && weight.type != GgmlType::F16) {
        throw GgufError("rms_norm is not implemented for tensor type " + to_string(weight.type));
    }

    TensorBuffer output(input.shape(), 0.0F, input.device());
    if (input.element_count() == 0U) {
        return output;
    }

    set_cuda_device(input.device());
    const auto *cached_weight = get_cached_weight_device_data(weight, input.device());
    const auto block_size = default_block_size(static_cast<std::size_t>(weight.dimensions[0]));
    if (weight.type == GgmlType::F32) {
        rms_norm_kernel<<<static_cast<unsigned int>(input.rows()), block_size, static_cast<std::size_t>(block_size) * sizeof(float)>>>(
            static_cast<const float *>(input.device_data()),
            static_cast<const float *>(cached_weight),
            static_cast<float *>(output.device_data()),
            static_cast<std::size_t>(input.rows()),
            static_cast<std::size_t>(input.cols()),
            epsilon);
    } else {
        rms_norm_f16_kernel<<<static_cast<unsigned int>(input.rows()), block_size, static_cast<std::size_t>(block_size) * sizeof(float)>>>(
            static_cast<const float *>(input.device_data()),
            static_cast<const __half *>(cached_weight),
            static_cast<float *>(output.device_data()),
            static_cast<std::size_t>(input.rows()),
            static_cast<std::size_t>(input.cols()),
            epsilon);
    }
    finalize_kernel(output, "rms_norm");
    return output;
}

TensorBuffer add(const TensorBuffer &lhs, const TensorBuffer &rhs) {
    require_same_shape(lhs, rhs, "add");
    require_same_device(lhs, rhs, "add");
    require_cuda_tensor(lhs, "add", "lhs");
    require_cuda_tensor(rhs, "add", "rhs");

    TensorBuffer output(lhs.shape(), 0.0F, lhs.device());
    const auto element_count = static_cast<std::size_t>(lhs.element_count());
    if (element_count == 0U) {
        return output;
    }

    set_cuda_device(lhs.device());
    const auto block_size = default_block_size(element_count);
    const auto grid_size = static_cast<unsigned int>((element_count + static_cast<std::size_t>(block_size) - 1U) /
                                                     static_cast<std::size_t>(block_size));
    add_kernel<<<grid_size, block_size>>>(
        static_cast<const float *>(lhs.device_data()),
        static_cast<const float *>(rhs.device_data()),
        static_cast<float *>(output.device_data()),
        element_count);
    finalize_kernel(output, "add");
    return output;
}

void add_inplace(TensorBuffer &lhs, const TensorBuffer &rhs) {
    require_same_shape(lhs, rhs, "add_inplace");
    require_same_device(lhs, rhs, "add_inplace");
    require_cuda_tensor(lhs, "add_inplace", "lhs");
    require_cuda_tensor(rhs, "add_inplace", "rhs");

    const auto element_count = static_cast<std::size_t>(lhs.element_count());
    if (element_count == 0U) {
        return;
    }

    set_cuda_device(lhs.device());
    const auto block_size = default_block_size(element_count);
    const auto grid_size = static_cast<unsigned int>((element_count + static_cast<std::size_t>(block_size) - 1U) /
                                                     static_cast<std::size_t>(block_size));
    add_inplace_kernel<<<grid_size, block_size>>>(
        static_cast<float *>(lhs.device_data()),
        static_cast<const float *>(rhs.device_data()),
        element_count);
    finalize_kernel(lhs, "add_inplace");
}

TensorBuffer silu(const TensorBuffer &input) {
    require_cuda_tensor(input, "silu", "input");

    TensorBuffer output(input.shape(), 0.0F, input.device());
    const auto element_count = static_cast<std::size_t>(input.element_count());
    if (element_count == 0U) {
        return output;
    }

    set_cuda_device(input.device());
    const auto block_size = default_block_size(element_count);
    const auto grid_size = static_cast<unsigned int>((element_count + static_cast<std::size_t>(block_size) - 1U) /
                                                     static_cast<std::size_t>(block_size));
    silu_kernel<<<grid_size, block_size>>>(
        static_cast<const float *>(input.device_data()),
        static_cast<float *>(output.device_data()),
        element_count);
    finalize_kernel(output, "silu");
    return output;
}

TensorBuffer multiply(const TensorBuffer &lhs, const TensorBuffer &rhs) {
    require_same_shape(lhs, rhs, "multiply");
    require_same_device(lhs, rhs, "multiply");
    require_cuda_tensor(lhs, "multiply", "lhs");
    require_cuda_tensor(rhs, "multiply", "rhs");

    TensorBuffer output(lhs.shape(), 0.0F, lhs.device());
    const auto element_count = static_cast<std::size_t>(lhs.element_count());
    if (element_count == 0U) {
        return output;
    }

    set_cuda_device(lhs.device());
    const auto block_size = default_block_size(element_count);
    const auto grid_size = static_cast<unsigned int>((element_count + static_cast<std::size_t>(block_size) - 1U) /
                                                     static_cast<std::size_t>(block_size));
    multiply_kernel<<<grid_size, block_size>>>(
        static_cast<const float *>(lhs.device_data()),
        static_cast<const float *>(rhs.device_data()),
        static_cast<float *>(output.device_data()),
        element_count);
    finalize_kernel(output, "multiply");
    return output;
}

TensorBuffer linear_project(
    const TensorView &weight,
    const TensorBuffer &input) {
    require_cuda_tensor(input, "linear_project", "input");
    if (weight.dimensions.size() != 2U) {
        throw GgufError("linear projection requires a 2D tensor: " + weight.name);
    }
    if (input.cols() != weight.row_width()) {
        throw GgufError("linear projection input width does not match tensor width: " + weight.name);
    }
    if (weight.type != GgmlType::F32 && weight.type != GgmlType::F16) {
        throw GgufError("linear projection is not implemented for tensor type " + to_string(weight.type));
    }

    const auto input_width = static_cast<std::size_t>(input.cols());
    const auto output_width = static_cast<std::size_t>(weight.row_count());
    const auto row_count = static_cast<std::size_t>(input.rows());
    TensorBuffer output({input.rows(), static_cast<uint64_t>(output_width)}, 0.0F, input.device());
    if (row_count == 0U || output_width == 0U) {
        return output;
    }

    set_cuda_device(input.device());
    const auto *cached_weight = get_cached_weight_device_data(weight, input.device());

    const auto total = row_count * output_width;
    const auto block_size = default_block_size(total);
    const auto grid_size = static_cast<unsigned int>(
        (total + static_cast<std::size_t>(block_size) - 1U) / static_cast<std::size_t>(block_size));

    if (weight.type == GgmlType::F32) {
        linear_project_f32_kernel<<<grid_size, block_size>>>(
            static_cast<const float *>(input.device_data()),
            static_cast<const float *>(cached_weight),
            static_cast<float *>(output.device_data()),
            row_count,
            input_width,
            output_width);
    } else {
        linear_project_f16_kernel<<<grid_size, block_size>>>(
            static_cast<const float *>(input.device_data()),
            static_cast<const __half *>(cached_weight),
            static_cast<float *>(output.device_data()),
            row_count,
            input_width,
            output_width);
    }

    finalize_kernel(output, "linear_project");
    return output;
}

TensorBuffer embedding_lookup(
    const TensorView &weight,
    const std::vector<uint32_t> &token_ids,
    Device device) {
    if (!device.is_cuda()) {
        throw GgufError("embedding_lookup requires a CUDA execution device");
    }
    if (weight.dimensions.size() != 2U) {
        throw GgufError("embedding lookup requires a 2D tensor: " + weight.name);
    }
    if (weight.type != GgmlType::F32 && weight.type != GgmlType::F16) {
        throw GgufError("embedding lookup is not implemented for tensor type " + to_string(weight.type));
    }

    const auto hidden_size = static_cast<std::size_t>(weight.row_width());
    const auto vocab_size = static_cast<std::size_t>(weight.row_count());
    for (uint32_t token_id : token_ids) {
        if (static_cast<std::size_t>(token_id) >= vocab_size) {
            throw GgufError("token id exceeds token embedding vocabulary size");
        }
    }

    TensorBuffer output(
        {static_cast<uint64_t>(token_ids.size()), static_cast<uint64_t>(hidden_size)},
        0.0F,
        device);
    if (token_ids.empty() || hidden_size == 0U) {
        return output;
    }

    set_cuda_device(device);
    const auto *cached_weight = get_cached_weight_device_data(weight, device);
    ScopedCudaBuffer<uint32_t> token_buffer(token_ids.size());
    throw_on_cuda_error(
        cudaMemcpy(
            token_buffer.data(),
            token_ids.data(),
            token_ids.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice),
        "cudaMemcpyHostToDevice");

    const auto total = token_ids.size() * hidden_size;
    const auto block_size = default_block_size(total);
    const auto grid_size = static_cast<unsigned int>(
        (total + static_cast<std::size_t>(block_size) - 1U) /
        static_cast<std::size_t>(block_size));

    if (weight.type == GgmlType::F32) {
        embedding_lookup_f32_kernel<<<grid_size, block_size>>>(
            token_buffer.data(),
            static_cast<const float *>(cached_weight),
            static_cast<float *>(output.device_data()),
            token_ids.size(),
            hidden_size);
    } else {
        embedding_lookup_f16_kernel<<<grid_size, block_size>>>(
            token_buffer.data(),
            static_cast<const __half *>(cached_weight),
            static_cast<float *>(output.device_data()),
            token_ids.size(),
            hidden_size);
    }

    finalize_kernel(output, "embedding_lookup");
    return output;
}

std::vector<float> project_last_token_logits(
    const TensorView &weight,
    const TensorBuffer &hidden_state) {
    require_cuda_tensor(hidden_state, "project_last_token_logits", "hidden_state");
    if (hidden_state.rank() != 2U) {
        throw GgufError("project_last_token_logits expects a rank-2 hidden state");
    }
    if (weight.dimensions.size() != 2U) {
        throw GgufError("output projection requires a 2D tensor: " + weight.name);
    }
    if (hidden_state.cols() != weight.row_width()) {
        throw GgufError("output projection hidden size does not match output.weight width");
    }
    if (weight.type != GgmlType::F32 && weight.type != GgmlType::F16) {
        throw GgufError("output projection is not implemented for tensor type " + to_string(weight.type));
    }

    const auto vocab_size = static_cast<std::size_t>(weight.row_count());
    const auto hidden_size = static_cast<std::size_t>(weight.row_width());
    TensorBuffer logits_tensor({static_cast<uint64_t>(vocab_size)}, 0.0F, hidden_state.device());
    if (vocab_size == 0U) {
        return {};
    }

    set_cuda_device(hidden_state.device());
    const auto *cached_weight = get_cached_weight_device_data(weight, hidden_state.device());
    const auto *last_token =
        static_cast<const float *>(hidden_state.device_data()) +
        (static_cast<std::size_t>(hidden_state.rows()) - 1U) * hidden_size;
    const auto block_size = default_block_size(vocab_size);
    const auto grid_size = static_cast<unsigned int>(
        (vocab_size + static_cast<std::size_t>(block_size) - 1U) /
        static_cast<std::size_t>(block_size));

    if (weight.type == GgmlType::F32) {
        output_projection_f32_kernel<<<grid_size, block_size>>>(
            last_token,
            static_cast<const float *>(cached_weight),
            static_cast<float *>(logits_tensor.device_data()),
            vocab_size,
            hidden_size);
    } else {
        output_projection_f16_kernel<<<grid_size, block_size>>>(
            last_token,
            static_cast<const __half *>(cached_weight),
            static_cast<float *>(logits_tensor.device_data()),
            vocab_size,
            hidden_size);
    }

    finalize_kernel(logits_tensor, "project_last_token_logits");
    const auto logits_cpu = logits_tensor.copy_to(Device::cpu());
    return logits_cpu.values();
}

}  // namespace sllmrf::ops::cuda
