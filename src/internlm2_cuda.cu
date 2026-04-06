#include "internlm2_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include <cuda_runtime_api.h>

#include "sllmrf/gguf.h"

namespace sllmrf::internlm2_cuda {

namespace {

void throw_on_cuda_error(cudaError_t error, std::string_view operation) {
    if (error != cudaSuccess) {
        throw GgufError(
            std::string(operation) + " failed: " + cudaGetErrorString(error));
    }
}

void set_cuda_device(const Device &device) {
    if (!device.is_cuda()) {
        throw GgufError("internlm2 CUDA helper received a non-CUDA tensor");
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

void finalize_tensor(TensorBuffer &tensor, std::string_view op_name) {
    throw_on_cuda_error(cudaGetLastError(), op_name);
    throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    tensor.mark_device_dirty();
}

int default_block_size(std::size_t element_count) {
    if (element_count <= 64U) {
        return 64;
    }
    if (element_count <= 128U) {
        return 128;
    }
    return 256;
}

__global__ void rope_kernel(
    float *tensor,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dim,
    uint32_t start_position,
    float rope_freq_base) {
    const auto pair_count_per_head = head_dim / 2U;
    const auto flat_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total_pairs = token_count * head_count * pair_count_per_head;
    if (flat_index >= total_pairs) {
        return;
    }

    const auto token = flat_index / (head_count * pair_count_per_head);
    const auto head_slot = flat_index % (head_count * pair_count_per_head);
    const auto head = head_slot / pair_count_per_head;
    const auto pair_slot = head_slot % pair_count_per_head;
    const auto pair = pair_slot * 2U;

    const auto position = static_cast<float>(start_position + static_cast<uint32_t>(token));
    const auto exponent = static_cast<float>(pair) / static_cast<float>(head_dim);
    const auto angle = position / powf(rope_freq_base, exponent);
    const auto cos_value = cosf(angle);
    const auto sin_value = sinf(angle);

    const auto base_offset = token * head_count * head_dim + head * head_dim + pair;
    const auto x0 = tensor[base_offset];
    const auto x1 = tensor[base_offset + 1U];
    tensor[base_offset] = x0 * cos_value - x1 * sin_value;
    tensor[base_offset + 1U] = x0 * sin_value + x1 * cos_value;
}

__global__ void causal_attention_kernel(
    const float *q,
    const float *cache_key,
    const float *cache_value,
    float *output,
    std::size_t token_count,
    std::size_t total_cache_tokens,
    std::size_t attention_head_count,
    std::size_t attention_head_count_kv,
    std::size_t head_dim,
    std::size_t kv_group_size,
    uint32_t start_position,
    float scale) {
    const auto task_index = static_cast<std::size_t>(blockIdx.x);
    const auto total_tasks = token_count * attention_head_count;
    if (task_index >= total_tasks) {
        return;
    }

    extern __shared__ float partials[];

    const auto token = task_index / attention_head_count;
    const auto head = task_index % attention_head_count;
    const auto kv_head = head / kv_group_size;
    const auto causal_limit = static_cast<std::size_t>(start_position) + token;
    const auto q_offset = token * attention_head_count * head_dim + head * head_dim;
    const auto output_offset = token * attention_head_count * head_dim + head * head_dim;

    float local_max = -INFINITY;
    for (std::size_t past = threadIdx.x; past <= causal_limit; past += blockDim.x) {
        float score = 0.0F;
        const auto k_offset = past * attention_head_count_kv * head_dim + kv_head * head_dim;
        for (std::size_t dim = 0; dim < head_dim; ++dim) {
            score += q[q_offset + dim] * cache_key[k_offset + dim];
        }
        score *= scale;
        local_max = fmaxf(local_max, score);
    }

    partials[threadIdx.x] = local_max;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] = fmaxf(partials[threadIdx.x], partials[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const auto max_score = partials[0];

    float local_denom = 0.0F;
    for (std::size_t past = threadIdx.x; past <= causal_limit; past += blockDim.x) {
        float score = 0.0F;
        const auto k_offset = past * attention_head_count_kv * head_dim + kv_head * head_dim;
        for (std::size_t dim = 0; dim < head_dim; ++dim) {
            score += q[q_offset + dim] * cache_key[k_offset + dim];
        }
        local_denom += expf(score * scale - max_score);
    }

    partials[threadIdx.x] = local_denom;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const auto denominator = partials[0];
    for (std::size_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        float value = 0.0F;
        for (std::size_t past = 0; past <= causal_limit; ++past) {
            float score = 0.0F;
            const auto k_offset = past * attention_head_count_kv * head_dim + kv_head * head_dim;
            const auto v_offset = past * attention_head_count_kv * head_dim + kv_head * head_dim;
            for (std::size_t inner_dim = 0; inner_dim < head_dim; ++inner_dim) {
                score += q[q_offset + inner_dim] * cache_key[k_offset + inner_dim];
            }
            const auto probability = expf(score * scale - max_score) / denominator;
            value += probability * cache_value[v_offset + dim];
        }
        output[output_offset + dim] = value;
    }
    (void)total_cache_tokens;
}

}  // namespace

void apply_rope_inplace(
    TensorBuffer &q,
    TensorBuffer &k,
    const Internlm2Config &config,
    uint32_t start_position) {
    require_cuda_tensor(q, "rope", "q");
    require_cuda_tensor(k, "rope", "k");
    if (q.device() != k.device()) {
        throw GgufError("rope requires q and k on the same CUDA device");
    }

    const auto head_dim = static_cast<std::size_t>(config.head_dimension());
    if ((head_dim % 2U) != 0U) {
        throw GgufError("RoPE currently requires an even head dimension");
    }

    set_cuda_device(q.device());

    const auto q_total_pairs =
        static_cast<std::size_t>(q.rows()) * static_cast<std::size_t>(config.attention_head_count) * (head_dim / 2U);
    if (q_total_pairs > 0U) {
        const auto block_size = default_block_size(q_total_pairs);
        const auto grid_size = static_cast<unsigned int>(
            (q_total_pairs + static_cast<std::size_t>(block_size) - 1U) /
            static_cast<std::size_t>(block_size));
        rope_kernel<<<grid_size, block_size>>>(
            static_cast<float *>(q.device_data()),
            static_cast<std::size_t>(q.rows()),
            static_cast<std::size_t>(config.attention_head_count),
            head_dim,
            start_position,
            config.rope_freq_base);
        finalize_tensor(q, "rope.q");
    }

    const auto k_total_pairs =
        static_cast<std::size_t>(k.rows()) * static_cast<std::size_t>(config.attention_head_count_kv) * (head_dim / 2U);
    if (k_total_pairs > 0U) {
        const auto block_size = default_block_size(k_total_pairs);
        const auto grid_size = static_cast<unsigned int>(
            (k_total_pairs + static_cast<std::size_t>(block_size) - 1U) /
            static_cast<std::size_t>(block_size));
        rope_kernel<<<grid_size, block_size>>>(
            static_cast<float *>(k.device_data()),
            static_cast<std::size_t>(k.rows()),
            static_cast<std::size_t>(config.attention_head_count_kv),
            head_dim,
            start_position,
            config.rope_freq_base);
        finalize_tensor(k, "rope.k");
    }
}

void write_kv_cache(
    KvCacheLayer &layer,
    const TensorBuffer &k,
    const TensorBuffer &v,
    uint32_t start_position,
    const Internlm2Config &config) {
    require_cuda_tensor(k, "write_kv_cache", "k");
    require_cuda_tensor(v, "write_kv_cache", "v");
    require_cuda_tensor(layer.key, "write_kv_cache", "layer.key");
    require_cuda_tensor(layer.value, "write_kv_cache", "layer.value");
    if (layer.key.device() != k.device() || layer.value.device() != v.device() || k.device() != v.device()) {
        throw GgufError("write_kv_cache requires all tensors on the same CUDA device");
    }
    if (k.shape() != v.shape()) {
        throw GgufError("write_kv_cache requires matching k/v shapes");
    }
    if (k.rank() != 2U || v.rank() != 2U) {
        throw GgufError("write_kv_cache expects rank-2 k/v tensors");
    }
    if (layer.key.rank() != 3U || layer.value.rank() != 3U) {
        throw GgufError("write_kv_cache expects rank-3 cache tensors");
    }
    if (k.cols() !=
        static_cast<uint64_t>(config.attention_head_count_kv) * static_cast<uint64_t>(config.head_dimension())) {
        throw GgufError("write_kv_cache k/v shape does not match InternLM2 KV layout");
    }
    if (layer.key.shape()[1] != config.attention_head_count_kv ||
        layer.key.shape()[2] != config.head_dimension() ||
        layer.value.shape()[1] != config.attention_head_count_kv ||
        layer.value.shape()[2] != config.head_dimension()) {
        throw GgufError("write_kv_cache cache tensor shape does not match InternLM2 KV layout");
    }
    if (start_position + k.rows() > layer.key.shape()[0] || start_position + v.rows() > layer.value.shape()[0]) {
        throw GgufError("write_kv_cache would exceed allocated cache capacity");
    }

    const auto token_stride_bytes =
        static_cast<std::size_t>(config.attention_head_count_kv) *
        static_cast<std::size_t>(config.head_dimension()) *
        sizeof(float);
    const auto copy_bytes = static_cast<std::size_t>(k.rows()) * token_stride_bytes;
    if (copy_bytes == 0U) {
        return;
    }

    set_cuda_device(k.device());
    auto *key_destination = static_cast<std::byte *>(layer.key.device_data()) +
        static_cast<std::size_t>(start_position) * token_stride_bytes;
    auto *value_destination = static_cast<std::byte *>(layer.value.device_data()) +
        static_cast<std::size_t>(start_position) * token_stride_bytes;

    throw_on_cuda_error(
        cudaMemcpy(
            key_destination,
            k.device_data(),
            copy_bytes,
            cudaMemcpyDeviceToDevice),
        "cudaMemcpyDeviceToDevice");
    throw_on_cuda_error(
        cudaMemcpy(
            value_destination,
            v.device_data(),
            copy_bytes,
            cudaMemcpyDeviceToDevice),
        "cudaMemcpyDeviceToDevice");
    throw_on_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    layer.key.mark_device_dirty();
    layer.value.mark_device_dirty();
}

TensorBuffer causal_attention(
    const TensorBuffer &q,
    const KvCacheLayer &cache,
    uint32_t start_position,
    const Internlm2Config &config) {
    require_cuda_tensor(q, "causal_attention", "q");
    require_cuda_tensor(cache.key, "causal_attention", "cache.key");
    require_cuda_tensor(cache.value, "causal_attention", "cache.value");
    if (q.device() != cache.key.device() || q.device() != cache.value.device()) {
        throw GgufError("causal_attention requires all tensors on the same CUDA device");
    }
    if (q.rank() != 2U) {
        throw GgufError("causal_attention expects q to be rank 2");
    }
    if (cache.key.rank() != 3U || cache.value.rank() != 3U) {
        throw GgufError("causal_attention expects cache tensors to be rank 3");
    }

    const auto head_dim = static_cast<std::size_t>(config.head_dimension());
    const auto attention_head_count = static_cast<std::size_t>(config.attention_head_count);
    const auto attention_head_count_kv = static_cast<std::size_t>(config.attention_head_count_kv);
    if (q.cols() != config.embedding_length) {
        throw GgufError("causal_attention q width does not match embedding length");
    }
    if (cache.key.shape()[1] != config.attention_head_count_kv ||
        cache.key.shape()[2] != config.head_dimension() ||
        cache.value.shape()[1] != config.attention_head_count_kv ||
        cache.value.shape()[2] != config.head_dimension()) {
        throw GgufError("causal_attention cache tensor shape does not match InternLM2 KV layout");
    }
    if (start_position + q.rows() > cache.key.shape()[0] || start_position + q.rows() > cache.value.shape()[0]) {
        throw GgufError("causal_attention exceeds allocated cache length");
    }
    if (attention_head_count_kv == 0U || (attention_head_count % attention_head_count_kv) != 0U) {
        throw GgufError("causal_attention requires attention heads divisible by kv heads");
    }

    TensorBuffer output({q.rows(), q.cols()}, 0.0F, q.device());
    if (q.rows() == 0U || q.cols() == 0U) {
        return output;
    }

    set_cuda_device(q.device());
    const auto task_count = static_cast<std::size_t>(q.rows()) * attention_head_count;
    const auto block_size = default_block_size(std::max(head_dim, static_cast<std::size_t>(32U)));
    causal_attention_kernel<<<
        static_cast<unsigned int>(task_count),
        block_size,
        static_cast<std::size_t>(block_size) * sizeof(float)>>>(
        static_cast<const float *>(q.device_data()),
        static_cast<const float *>(cache.key.device_data()),
        static_cast<const float *>(cache.value.device_data()),
        static_cast<float *>(output.device_data()),
        static_cast<std::size_t>(q.rows()),
        static_cast<std::size_t>(cache.key.shape()[0]),
        attention_head_count,
        attention_head_count_kv,
        head_dim,
        attention_head_count / attention_head_count_kv,
        start_position,
        1.0F / sqrtf(static_cast<float>(head_dim)));
    finalize_tensor(output, "causal_attention");
    return output;
}

}  // namespace sllmrf::internlm2_cuda
