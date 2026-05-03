#include "sllmrf/internlm2.h"

#include "mermaid.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>

#if defined(SLLMRF_USE_OPENMP) && SLLMRF_USE_OPENMP
#include <omp.h>
#endif

#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
#include "internlm2_cuda.h"
#include "operators_cuda.h"
#endif

namespace sllmrf {

namespace {

const MetadataValue &require_metadata(const GgufFile &file, std::string_view key) {
    const auto *entry = file.find_metadata(key);
    if (entry == nullptr) {
        throw GgufError("missing required internlm2 metadata: " + std::string(key));
    }
    return entry->value;
}

const MetadataValue *find_metadata(const GgufFile &file, std::string_view key) {
    const auto *entry = file.find_metadata(key);
    return entry == nullptr ? nullptr : &entry->value;
}

std::string read_string(const MetadataValue &value, std::string_view key) {
    if (!value.is<std::string>()) {
        throw GgufError("metadata is not a string: " + std::string(key));
    }
    return value.as<std::string>();
}

uint32_t read_u32(const MetadataValue &value, std::string_view key) {
    if (value.is<uint32_t>()) {
        return value.as<uint32_t>();
    }
    if (value.is<uint64_t>()) {
        return static_cast<uint32_t>(value.as<uint64_t>());
    }
    throw GgufError("metadata is not an unsigned integer: " + std::string(key));
}

float read_f32(const MetadataValue &value, std::string_view key) {
    if (value.is<float>()) {
        return value.as<float>();
    }
    if (value.is<double>()) {
        return static_cast<float>(value.as<double>());
    }
    throw GgufError("metadata is not a floating-point number: " + std::string(key));
}

void ensure_tensor_exists(const GgufTensorReader &weights, std::string_view name) {
    if (!weights.has_tensor(name)) {
        throw GgufError("required tensor missing from internlm2 GGUF: " + std::string(name));
    }
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

std::string to_string(Internlm2OpType type) {
    switch (type) {
        case Internlm2OpType::Embedding:
            return "embedding";
        case Internlm2OpType::RmsNorm:
            return "rms_norm";
        case Internlm2OpType::Linear:
            return "linear";
        case Internlm2OpType::Rope:
            return "rope";
        case Internlm2OpType::Attention:
            return "attention";
        case Internlm2OpType::ResidualAdd:
            return "residual_add";
        case Internlm2OpType::SwiGLU:
            return "swiglu";
        case Internlm2OpType::OutputProjection:
            return "output_projection";
    }

    return "unknown";
}

uint32_t sample_stochastic_impl(
    const std::vector<float> &logits,
    float temperature,
    std::mt19937_64 &rng) {
    if (logits.empty()) {
        throw GgufError("cannot sample from empty logits");
    }
    if (!(temperature > 0.0F)) {
        throw GgufError("stochastic sampling requires temperature > 0");
    }

    const auto max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> weights(logits.size(), 0.0);
    for (std::size_t index = 0; index < logits.size(); ++index) {
        const auto scaled = static_cast<double>((logits[index] - max_logit) / temperature);
        weights[index] = std::exp(scaled);
    }

    const auto total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (!(total_weight > 0.0) || !std::isfinite(total_weight)) {
        throw GgufError("stochastic sampling produced an invalid probability distribution");
    }

    std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
    return static_cast<uint32_t>(distribution(rng));
}

struct GraphModuleSpec {
    std::string id;
    std::string title;
    std::string file_name;
    std::string description;
    std::string class_name;
    std::string input;
    std::string output;
    std::string flow_detail;
    std::string data_detail;
};

struct GraphModuleConnection {
    std::string from;
    std::string to;
    std::string data_label;
};

struct GraphModuleSet {
    std::vector<GraphModuleSpec> modules;
    std::vector<GraphModuleConnection> connections;
};

GraphModuleSet build_graph_module_specs(
    uint32_t block_count,
    uint32_t embedding_length,
    uint32_t context_length,
    std::size_t vocab_size) {
    GraphModuleSet specs;
    specs.modules = {
        {
            .id = "gguf",
            .title = "01 GGUF Loading Module",
            .file_name = "01-gguf-loading.mmd",
            .description = "GGUF loading module graph",
            .class_name = "file",
            .input = "model path",
            .output = "metadata, tokenizer metadata, config, tensor reader",
            .flow_detail = "load GGUF file and expose model objects",
            .data_detail = "metadata + tensor infos + F16/F32 weights",
        },
        {
            .id = "prompt",
            .title = "02 Prompt and Tokenizer Module",
            .file_name = "02-prompt-tokenizer.mmd",
            .description = "prompt and tokenizer module graph",
            .class_name = "group",
            .input = "prompt text + tokenizer metadata",
            .output = "PromptBatch token_ids + positions, decode support",
            .flow_detail = "encode prompt and prepare positions",
            .data_detail = "prompt text -> token ids -> PromptBatch",
        },
        {
            .id = "runtime",
            .title = "03 Runtime and Tensor Module",
            .file_name = "03-runtime-tensor.mmd",
            .description = "runtime and tensor module graph",
            .class_name = "runtime",
            .input = "config + device options",
            .output = "OperatorContext, KV cache, TensorBuffer placement",
            .flow_detail = "select device and allocate runtime state",
            .data_detail = "runtime state + KV Cache",
        },
        {
            .id = "forward",
            .title = "04 Model Forward Module",
            .file_name = "04-model-forward.mmd",
            .description = "model forward module graph",
            .class_name = "attention",
            .input = "PromptBatch + weights + runtime",
            .output = "hidden state + logits",
            .flow_detail = "embedding -> Decoder Block x" + std::to_string(block_count) + " -> final norm -> LM head",
            .data_detail = "embedding TensorBuffer [seq, " + std::to_string(embedding_length) +
                "] -> Decoder Block x" + std::to_string(block_count) +
                " -> logits [vocab: " + std::to_string(vocab_size) + "]",
        },
        {
            .id = "generation",
            .title = "05 Generation Output Module",
            .file_name = "05-generation-output.mmd",
            .description = "generation output module graph",
            .class_name = "activation",
            .input = "logits + tokenizer decode support",
            .output = "generated ids and text",
            .flow_detail = "sample next token and decode output",
            .data_detail = "logits -> sampled token ids -> generated text",
        },
    };
    specs.connections = {
        {.from = "gguf", .to = "prompt", .data_label = "Tokenizer"},
        {.from = "gguf", .to = "runtime", .data_label = "Internlm2Config"},
        {.from = "gguf", .to = "forward", .data_label = "GgufTensorReader + TensorLayout"},
        {.from = "prompt", .to = "forward", .data_label = "PromptBatch token_ids + positions"},
        {.from = "runtime", .to = "forward", .data_label = "OperatorContext + KV Cache"},
        {.from = "forward", .to = "generation", .data_label = "logits"},
        {.from = "prompt", .to = "generation", .data_label = "decode support"},
    };
    (void)context_length;
    return specs;
}

GraphModuleSet build_graph_module_specs_from_plan(
    const std::vector<Internlm2PlanStep> &steps) {
    std::size_t decoder_start = steps.size();
    std::size_t decoder_end = steps.size();
    for (std::size_t index = 0; index < steps.size(); ++index) {
        if (steps[index].name.rfind("layer_", 0) == 0U) {
            decoder_start = std::min(decoder_start, index);
            decoder_end = index + 1U;
        }
    }

    const auto decoder_step_count = decoder_end > decoder_start ? decoder_end - decoder_start : 0U;
    const auto layer_count = static_cast<uint32_t>(decoder_step_count / 11U);
    return build_graph_module_specs(layer_count, 0U, 0U, 0U);
}

std::string module_node_label(const GraphModuleSpec &module) {
    return mermaid::escape(module.title) + "<br/>input: " +
        mermaid::escape(module.input) + "<br/>output: " +
        mermaid::escape(module.output);
}

std::string module_detail_node_id(
    std::string_view module_id,
    std::string_view suffix) {
    return mermaid::compact_node_id(std::string(module_id) + "_" + std::string(suffix));
}

const GraphModuleSpec *find_module_spec(
    const GraphModuleSet &specs,
    std::string_view id) {
    for (const auto &module : specs.modules) {
        if (module.id == id) {
            return &module;
        }
    }
    return nullptr;
}

std::string module_overview_mermaid(
    std::string_view graph_name,
    const GraphModuleSet &specs) {
    std::ostringstream stream;
    stream << "%% " << mermaid::escape(graph_name) << "\n";
    stream << "flowchart TD\n";
    stream << "    model_path([model path])\n";
    stream << "    prompt_text([prompt text])\n";
    stream << "    device_options([device and generation options])\n";
    stream << "    generated_text([generated text and debug previews])\n";
    for (const auto &module : specs.modules) {
        stream << "    " << module.id << "[\"" << module_node_label(module) << "\"]\n";
    }
    stream << "    model_path --> gguf\n";
    stream << "    prompt_text --> prompt\n";
    stream << "    device_options --> runtime\n";
    for (const auto &connection : specs.connections) {
        stream << "    " << connection.from << " -->|" << mermaid::escape(connection.data_label)
               << "| " << connection.to << "\n";
    }
    stream << "    generation --> generated_text\n";
    mermaid::write_class_defs(stream);
    stream << "    class model_path,prompt_text,device_options,generated_text io\n";
    for (const auto &module : specs.modules) {
        stream << "    class " << module.id << " " << module.class_name << "\n";
    }
    return stream.str();
}

std::string forward_module_mermaid(
    std::string_view graph_name,
    std::string_view module_input,
    std::string_view module_output,
    const Internlm2Config &config) {
    std::ostringstream stream;
    stream << "%% " << mermaid::escape(graph_name) << "\n";
    stream << "flowchart TD\n";
    stream << "    module_input([\"input: " << mermaid::escape(module_input) << "\"])\n";
    stream << "    module_output([\"output: " << mermaid::escape(module_output) << "\"])\n";
    stream << "    prompt_batch([PromptBatch])\n";
    stream << "    weight_access([GgufTensorReader + TensorLayout])\n";
    stream << "    runtime_state([runtime + KV cache])\n";
    stream << "    embedding[\"Token Embedding<br/>token ids -> TensorBuffer [seq, "
           << config.embedding_length << "]\"]\n";
    stream << "    decoder[\"Decoder Block x" << config.block_count
           << "<br/>same block template repeated\"]\n";
    stream << "    final_norm[\"Final RMSNorm\"]\n";
    stream << "    lm_head[\"LM Head<br/>last hidden state -> logits\"]\n";
    stream << "    logits([logits for generation module])\n";
    stream << "    subgraph block_template[Single decoder block template]\n";
    stream << "        direction LR\n";
    stream << "        attn_norm[\"attn_norm\"]\n";
    stream << "        qkv[\"q/k/v projection\"]\n";
    stream << "        rope[\"RoPE\"]\n";
    stream << "        kv[\"KV cache write/read\"]\n";
    stream << "        attention[\"causal attention\"]\n";
    stream << "        attn_residual[\"attention residual\"]\n";
    stream << "        ffn_norm[\"ffn_norm\"]\n";
    stream << "        gate_up[\"gate/up projection\"]\n";
    stream << "        swiglu[\"SwiGLU\"]\n";
    stream << "        ffn_down[\"ffn down projection\"]\n";
    stream << "        ffn_residual[\"ffn residual\"]\n";
    stream << "        attn_norm --> qkv --> rope --> attention --> attn_residual --> ffn_norm --> gate_up --> swiglu --> ffn_down --> ffn_residual\n";
    stream << "        rope --> kv --> attention\n";
    stream << "    end\n";
    stream << "    module_input --> prompt_batch\n";
    stream << "    module_input --> weight_access\n";
    stream << "    module_input --> runtime_state\n";
    stream << "    prompt_batch --> embedding --> decoder --> final_norm --> lm_head --> logits\n";
    stream << "    weight_access --> embedding\n";
    stream << "    weight_access --> decoder\n";
    stream << "    weight_access --> final_norm\n";
    stream << "    weight_access --> lm_head\n";
    stream << "    runtime_state --> decoder\n";
    stream << "    decoder -. repeated structure .-> block_template\n";
    stream << "    logits --> module_output\n";
    mermaid::write_class_defs(stream);
    stream << "    class module_input,module_output,prompt_batch,weight_access,runtime_state,logits io\n";
    stream << "    class embedding embedding\n";
    stream << "    class decoder,attention,kv,block_template attention\n";
    stream << "    class final_norm,attn_norm,ffn_norm norm\n";
    stream << "    class lm_head,qkv,gate_up,ffn_down linear\n";
    stream << "    class attn_residual,ffn_residual residual\n";
    stream << "    class swiglu activation\n";
    return stream.str();
}

std::string generation_module_mermaid(
    std::string_view graph_name,
    std::string_view module_input,
    std::string_view module_output) {
    std::ostringstream stream;
    stream << "%% " << mermaid::escape(graph_name) << "\n";
    stream << "flowchart TD\n";
    stream << "    module_input([\"input: " << mermaid::escape(module_input) << "\"])\n";
    stream << "    module_output([\"output: " << mermaid::escape(module_output) << "\"])\n";
    stream << "    logits([logits from forward module])\n";
    stream << "    config[\"GenerationConfig<br/>max_new_tokens, sampling, temperature, seed\"]\n";
    stream << "    greedy[\"greedy sampling\"]\n";
    stream << "    stochastic[\"stochastic sampling\"]\n";
    stream << "    next_token[\"next token id\"]\n";
    stream << "    autoregressive[\"append token and continue<br/>until max_new_tokens or EOS\"]\n";
    stream << "    generated_ids([generated token ids])\n";
    stream << "    decode_support([Tokenizer decode support])\n";
    stream << "    generated_text([generated text + full text])\n";
    stream << "    module_input --> logits\n";
    stream << "    module_input --> decode_support\n";
    stream << "    logits --> greedy --> next_token\n";
    stream << "    logits --> stochastic --> next_token\n";
    stream << "    config --> greedy\n";
    stream << "    config --> stochastic\n";
    stream << "    next_token --> autoregressive --> generated_ids\n";
    stream << "    generated_ids --> decode_support --> generated_text\n";
    stream << "    generated_text --> module_output\n";
    mermaid::write_class_defs(stream);
    stream << "    class logits,next_token,generated_ids tensor\n";
    stream << "    class config metadata\n";
    stream << "    class greedy,stochastic,autoregressive activation\n";
    stream << "    class decode_support group\n";
    stream << "    class module_input,module_output,generated_text io\n";
    return stream.str();
}

ops::OperatorContext context_for_device(Device device) {
    return ops::OperatorContext {.device = device};
}

uint32_t resolve_runtime_sequence_length(
    const Internlm2Config &config,
    std::size_t prompt_token_count,
    uint32_t max_new_tokens) {
    const auto requested = static_cast<uint64_t>(prompt_token_count) + static_cast<uint64_t>(max_new_tokens);
    const auto bounded = std::min<uint64_t>(
        config.context_length,
        std::max<uint64_t>(1U, requested));
    return static_cast<uint32_t>(bounded);
}

void require_execution_device(
    const TensorBuffer &tensor,
    const Device &device,
    std::string_view op_name,
    std::string_view arg_name) {
    if (tensor.device() != device) {
        throw GgufError(
            std::string(op_name) + " requires " + std::string(arg_name) +
            " on execution device " + device.to_string() + ", but got " +
            tensor.placement_string());
    }
}

float tensor3_get(
    const TensorBuffer &tensor,
    uint64_t token_index,
    uint64_t head_index,
    uint64_t dim_index,
    uint64_t head_dim) {
    const auto offset = token_index * tensor.cols() + head_index * head_dim + dim_index;
    return tensor.values().at(static_cast<std::size_t>(offset));
}

void tensor3_set(
    TensorBuffer &tensor,
    uint64_t token_index,
    uint64_t head_index,
    uint64_t dim_index,
    uint64_t head_dim,
    float value) {
    const auto offset = token_index * tensor.cols() + head_index * head_dim + dim_index;
    tensor.values().at(static_cast<std::size_t>(offset)) = value;
}

float dot_tensor_row(
    const float *lhs,
    const uint8_t *rhs,
    std::size_t length,
    GgmlType type) {
    float sum = 0.0F;
    if (type == GgmlType::F32) {
        const auto *weights = reinterpret_cast<const float *>(rhs);
        for (std::size_t index = 0; index < length; ++index) {
            sum += lhs[index] * weights[index];
        }
        return sum;
    }
    if (type == GgmlType::F16) {
        const auto *weights = reinterpret_cast<const uint16_t *>(rhs);
        for (std::size_t index = 0; index < length; ++index) {
            sum += lhs[index] * half_to_float(weights[index]);
        }
        return sum;
    }

    throw GgufError("linear projection is not implemented for tensor type " + to_string(type));
}

TensorBuffer linear_project_cpu(
    const GgufTensorReader &weights,
    std::string_view tensor_name,
    const TensorBuffer &input) {
    const auto tensor = weights.require_tensor(tensor_name);
    if (tensor.dimensions.size() != 2U) {
        throw GgufError("linear projection requires a 2D tensor: " + std::string(tensor_name));
    }
    if (input.cols() != tensor.row_width()) {
        throw GgufError("linear projection input width does not match tensor width: " + std::string(tensor_name));
    }

    const auto input_width = static_cast<std::size_t>(input.cols());
    const auto output_width = static_cast<std::size_t>(tensor.row_count());
    const auto bytes_per_row = input_width * ggml_type_size(tensor.type);

    TensorBuffer output({input.rows(), static_cast<uint64_t>(output_width)}, 0.0F, input.device());
    #if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (std::int64_t row = 0; row < static_cast<std::int64_t>(input.rows()); ++row) {
        for (std::int64_t output_index = 0; output_index < static_cast<std::int64_t>(output_width); ++output_index) {
            const auto *weight_row = tensor.data + output_index * bytes_per_row;
            const auto *input_row = input.data() + row * static_cast<std::int64_t>(input.cols());
            output.at(static_cast<uint64_t>(row), static_cast<uint64_t>(output_index)) =
                dot_tensor_row(input_row, weight_row, input_width, tensor.type);
        }
    }

    return output;
}

TensorBuffer linear_project(
    const GgufTensorReader &weights,
    std::string_view tensor_name,
    const TensorBuffer &input,
    ops::OperatorContext context) {
    require_execution_device(input, context.device, "linear_project", "input");
    if (context.device.is_cpu()) {
        return linear_project_cpu(weights, tensor_name, input);
    }

    if (context.device.is_cuda()) {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
        if (cuda_backend_enabled()) {
            return ops::cuda::linear_project(weights.require_tensor(tensor_name), input);
        }
#endif
        auto cpu_input = input.copy_to(Device::cpu());
        auto cpu_output = linear_project_cpu(weights, tensor_name, cpu_input);
        return cpu_output.copy_to(context.device);
    }

    throw GgufError("linear_project native CUDA backend is not implemented yet");
}

void apply_rope_inplace_cpu(
    TensorBuffer &q,
    TensorBuffer &k,
    const Internlm2Config &config,
    uint32_t start_position) {
    const auto head_dim = config.head_dimension();
    if (head_dim % 2U != 0U) {
        throw GgufError("RoPE currently requires an even head dimension");
    }

    auto rotate = [&](TensorBuffer &tensor, uint32_t head_count) {
        #if defined(_OPENMP)
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (std::int64_t token = 0; token < static_cast<std::int64_t>(tensor.rows()); ++token) {
            for (std::int64_t head = 0; head < static_cast<std::int64_t>(head_count); ++head) {
                const auto position = static_cast<float>(start_position + token);
                for (uint32_t pair = 0; pair < head_dim; pair += 2U) {
                    const auto exponent = static_cast<float>(pair) / static_cast<float>(head_dim);
                    const auto angle = position / std::pow(config.rope_freq_base, exponent);
                    const auto cos_value = std::cos(angle);
                    const auto sin_value = std::sin(angle);

                    const auto x0 = tensor3_get(
                        tensor,
                        static_cast<uint64_t>(token),
                        static_cast<uint64_t>(head),
                        pair,
                        head_dim);
                    const auto x1 = tensor3_get(
                        tensor,
                        static_cast<uint64_t>(token),
                        static_cast<uint64_t>(head),
                        pair + 1U,
                        head_dim);
                    tensor3_set(
                        tensor,
                        static_cast<uint64_t>(token),
                        static_cast<uint64_t>(head),
                        pair,
                        head_dim,
                        x0 * cos_value - x1 * sin_value);
                    tensor3_set(
                        tensor,
                        static_cast<uint64_t>(token),
                        static_cast<uint64_t>(head),
                        pair + 1U,
                        head_dim,
                        x0 * sin_value + x1 * cos_value);
                }
            }
        }
    };

    rotate(q, config.attention_head_count);
    rotate(k, config.attention_head_count_kv);
}

void apply_rope_inplace(
    TensorBuffer &q,
    TensorBuffer &k,
    const Internlm2Config &config,
    uint32_t start_position,
    ops::OperatorContext context) {
    require_execution_device(q, context.device, "rope", "q");
    require_execution_device(k, context.device, "rope", "k");
    if (context.device.is_cpu()) {
        apply_rope_inplace_cpu(q, k, config, start_position);
        return;
    }

    if (context.device.is_cuda()) {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
        if (cuda_backend_enabled()) {
            internlm2_cuda::apply_rope_inplace(q, k, config, start_position);
            return;
        }
#endif
        auto cpu_q = q.copy_to(Device::cpu());
        auto cpu_k = k.copy_to(Device::cpu());
        apply_rope_inplace_cpu(cpu_q, cpu_k, config, start_position);
        q = cpu_q.copy_to(context.device);
        k = cpu_k.copy_to(context.device);
        return;
    }

    throw GgufError("rope native CUDA backend is not implemented yet");
}

void write_kv_cache_cpu(
    KvCacheLayer &layer,
    const TensorBuffer &k,
    const TensorBuffer &v,
    uint32_t start_position,
    const Internlm2Config &config) {
    const auto head_dim = config.head_dimension();
    #if defined(_OPENMP)
    #pragma omp parallel for collapse(3) schedule(static)
    #endif
    for (std::int64_t token = 0; token < static_cast<std::int64_t>(k.rows()); ++token) {
        for (std::int64_t head = 0; head < static_cast<std::int64_t>(config.attention_head_count_kv); ++head) {
            for (uint32_t dim = 0; dim < head_dim; ++dim) {
                const auto cache_position = start_position + static_cast<uint32_t>(token);
                tensor3_set(
                    layer.key,
                    cache_position,
                    static_cast<uint64_t>(head),
                    dim,
                    head_dim,
                    tensor3_get(
                        k,
                        static_cast<uint64_t>(token),
                        static_cast<uint64_t>(head),
                        dim,
                        head_dim));
                tensor3_set(
                    layer.value,
                    cache_position,
                    static_cast<uint64_t>(head),
                    dim,
                    head_dim,
                    tensor3_get(
                        v,
                        static_cast<uint64_t>(token),
                        static_cast<uint64_t>(head),
                        dim,
                        head_dim));
            }
        }
    }
}

void write_kv_cache(
    KvCacheLayer &layer,
    const TensorBuffer &k,
    const TensorBuffer &v,
    uint32_t start_position,
    const Internlm2Config &config,
    ops::OperatorContext context) {
    require_execution_device(k, context.device, "write_kv_cache", "k");
    require_execution_device(v, context.device, "write_kv_cache", "v");
    require_execution_device(layer.key, context.device, "write_kv_cache", "layer.key");
    require_execution_device(layer.value, context.device, "write_kv_cache", "layer.value");
    if (context.device.is_cpu()) {
        write_kv_cache_cpu(layer, k, v, start_position, config);
        return;
    }

    if (context.device.is_cuda()) {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
        if (cuda_backend_enabled()) {
            internlm2_cuda::write_kv_cache(layer, k, v, start_position, config);
            return;
        }
#endif
        KvCacheLayer cpu_layer {
            .key = layer.key.copy_to(Device::cpu()),
            .value = layer.value.copy_to(Device::cpu()),
        };
        auto cpu_k = k.copy_to(Device::cpu());
        auto cpu_v = v.copy_to(Device::cpu());
        write_kv_cache_cpu(cpu_layer, cpu_k, cpu_v, start_position, config);
        layer.key = cpu_layer.key.copy_to(context.device);
        layer.value = cpu_layer.value.copy_to(context.device);
        return;
    }

    throw GgufError("write_kv_cache native CUDA backend is not implemented yet");
}

TensorBuffer causal_attention_cpu(
    const TensorBuffer &q,
    const KvCacheLayer &cache,
    uint32_t start_position,
    const Internlm2Config &config) {
    const auto head_dim = config.head_dimension();
    const auto kv_group_size = config.attention_head_count / config.attention_head_count_kv;
    const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    TensorBuffer context({q.rows(), q.cols()}, 0.0F, q.device());
    const auto task_count =
        static_cast<std::int64_t>(q.rows()) * static_cast<std::int64_t>(config.attention_head_count);
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (std::int64_t task = 0; task < task_count; ++task) {
        const auto token = static_cast<uint64_t>(task / static_cast<std::int64_t>(config.attention_head_count));
        const auto head = static_cast<uint32_t>(task % static_cast<std::int64_t>(config.attention_head_count));
        const auto causal_limit = start_position + static_cast<uint32_t>(token);
        const auto kv_head = head / kv_group_size;

        std::vector<float> scores(static_cast<std::size_t>(causal_limit + 1U), 0.0F);
        float max_score = -std::numeric_limits<float>::infinity();
        for (uint32_t past = 0; past <= causal_limit; ++past) {
            float score = 0.0F;
            for (uint32_t dim = 0; dim < head_dim; ++dim) {
                score += tensor3_get(q, token, head, dim, head_dim) *
                         tensor3_get(cache.key, past, kv_head, dim, head_dim);
            }
            score *= scale;
            scores[static_cast<std::size_t>(past)] = score;
            max_score = std::max(max_score, score);
        }

        float denominator = 0.0F;
        for (float &score : scores) {
            score = std::exp(score - max_score);
            denominator += score;
        }

        for (uint32_t dim = 0; dim < head_dim; ++dim) {
            float value = 0.0F;
            for (uint32_t past = 0; past <= causal_limit; ++past) {
                const auto probability = scores[static_cast<std::size_t>(past)] / denominator;
                value += probability * tensor3_get(cache.value, past, kv_head, dim, head_dim);
            }
            tensor3_set(context, token, head, dim, head_dim, value);
        }
    }

    return context;
}

TensorBuffer causal_attention(
    const TensorBuffer &q,
    const KvCacheLayer &cache,
    uint32_t start_position,
    const Internlm2Config &config,
    ops::OperatorContext context) {
    require_execution_device(q, context.device, "causal_attention", "q");
    require_execution_device(cache.key, context.device, "causal_attention", "cache.key");
    require_execution_device(cache.value, context.device, "causal_attention", "cache.value");
    if (context.device.is_cpu()) {
        return causal_attention_cpu(q, cache, start_position, config);
    }

    if (context.device.is_cuda()) {
#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
        if (cuda_backend_enabled()) {
            return internlm2_cuda::causal_attention(q, cache, start_position, config);
        }
#endif
        KvCacheLayer cpu_cache {
            .key = cache.key.copy_to(Device::cpu()),
            .value = cache.value.copy_to(Device::cpu()),
        };
        auto cpu_q = q.copy_to(Device::cpu());
        auto cpu_context = causal_attention_cpu(cpu_q, cpu_cache, start_position, config);
        return cpu_context.copy_to(context.device);
    }

    throw GgufError("causal_attention native CUDA backend is not implemented yet");
}

}  // namespace

Internlm2Config Internlm2Config::from_gguf(const GgufFile &file) {
    Internlm2Config config;
    config.architecture = read_string(
        require_metadata(file, "general.architecture"),
        "general.architecture");
    if (config.architecture != "internlm2") {
        throw GgufError("this model loader only supports general.architecture=internlm2");
    }

    config.model_name = find_metadata(file, "general.name") == nullptr
        ? "internlm2"
        : read_string(*find_metadata(file, "general.name"), "general.name");
    config.context_length = read_u32(
        require_metadata(file, "internlm2.context_length"),
        "internlm2.context_length");
    config.block_count = read_u32(
        require_metadata(file, "internlm2.block_count"),
        "internlm2.block_count");
    config.embedding_length = read_u32(
        require_metadata(file, "internlm2.embedding_length"),
        "internlm2.embedding_length");
    config.feed_forward_length = read_u32(
        require_metadata(file, "internlm2.feed_forward_length"),
        "internlm2.feed_forward_length");
    config.attention_head_count = read_u32(
        require_metadata(file, "internlm2.attention.head_count"),
        "internlm2.attention.head_count");
    config.attention_head_count_kv = read_u32(
        require_metadata(file, "internlm2.attention.head_count_kv"),
        "internlm2.attention.head_count_kv");
    config.rms_norm_epsilon = read_f32(
        require_metadata(file, "internlm2.attention.layer_norm_rms_epsilon"),
        "internlm2.attention.layer_norm_rms_epsilon");
    config.rope_freq_base = read_f32(
        require_metadata(file, "internlm2.rope.freq_base"),
        "internlm2.rope.freq_base");
    config.vocabulary_size = static_cast<uint32_t>(require_metadata(file, "tokenizer.ggml.tokens")
                                                       .as<MetadataValue::Array>()
                                                       .size());

    if (config.attention_head_count == 0U || config.embedding_length == 0U) {
        throw GgufError("invalid internlm2 attention or embedding configuration");
    }
    if (config.embedding_length % config.attention_head_count != 0U) {
        throw GgufError("embedding_length must be divisible by attention_head_count");
    }

    return config;
}

uint32_t Internlm2Config::head_dimension() const {
    return attention_head_count == 0U ? 0U : embedding_length / attention_head_count;
}

Internlm2TensorLayout Internlm2TensorLayout::build(const Internlm2Config &config) {
    Internlm2TensorLayout layout;
    layout.token_embedding = "token_embd.weight";
    layout.output_norm = "output_norm.weight";
    layout.output = "output.weight";
    layout.layers.reserve(config.block_count);

    for (uint32_t index = 0; index < config.block_count; ++index) {
        const auto prefix = "blk." + std::to_string(index);
        layout.layers.push_back(Internlm2LayerNames {
            .attention_norm = prefix + ".attn_norm.weight",
            .attention_q = prefix + ".attn_q.weight",
            .attention_k = prefix + ".attn_k.weight",
            .attention_v = prefix + ".attn_v.weight",
            .attention_output = prefix + ".attn_output.weight",
            .ffn_norm = prefix + ".ffn_norm.weight",
            .ffn_gate = prefix + ".ffn_gate.weight",
            .ffn_up = prefix + ".ffn_up.weight",
            .ffn_down = prefix + ".ffn_down.weight",
        });
    }

    return layout;
}

Internlm2Runtime::Internlm2Runtime(
    const Internlm2Config &config,
    uint32_t max_sequence_length,
    ops::OperatorContext context)
    : max_sequence_length_(max_sequence_length == 0U ? config.context_length : max_sequence_length),
      context_(context) {
    if (max_sequence_length_ == 0U) {
        throw GgufError("internlm2 runtime max sequence length must be greater than zero");
    }

    layers_.reserve(config.block_count);
    for (uint32_t index = 0; index < config.block_count; ++index) {
        layers_.push_back(KvCacheLayer {
            .key = TensorBuffer(
                {max_sequence_length_, config.attention_head_count_kv, config.head_dimension()},
                0.0F,
                context.device),
            .value = TensorBuffer(
                {max_sequence_length_, config.attention_head_count_kv, config.head_dimension()},
                0.0F,
                context.device),
        });
    }
}

void Internlm2Runtime::reset() {
    consumed_tokens_ = 0U;
    for (auto &layer : layers_) {
        std::fill(layer.key.values().begin(), layer.key.values().end(), 0.0F);
        std::fill(layer.value.values().begin(), layer.value.values().end(), 0.0F);
        if (!context_.device.is_cpu()) {
            layer.key.sync_host_to_device();
            layer.value.sync_host_to_device();
        }
    }
}

uint32_t Internlm2Runtime::max_sequence_length() const noexcept {
    return max_sequence_length_;
}

uint32_t Internlm2Runtime::consumed_tokens() const noexcept {
    return consumed_tokens_;
}

const ops::OperatorContext &Internlm2Runtime::context() const noexcept {
    return context_;
}

const std::vector<KvCacheLayer> &Internlm2Runtime::layers() const noexcept {
    return layers_;
}

std::vector<KvCacheLayer> &Internlm2Runtime::layers() noexcept {
    return layers_;
}

void Internlm2Runtime::mark_tokens_consumed(uint32_t count) {
    consumed_tokens_ += count;
    if (consumed_tokens_ > max_sequence_length_) {
        throw GgufError("runtime sequence length exceeded allocated KV cache");
    }
}

std::string Internlm2ExecutionPlan::describe() const {
    std::ostringstream stream;
    stream << "internlm2 execution plan\n";
    for (std::size_t index = 0; index < steps.size(); ++index) {
        const auto &step = steps[index];
        stream << index << ": " << to_string(step.type) << " :: " << step.name;
        if (!step.weights.empty()) {
            stream << " [";
            for (std::size_t weight_index = 0; weight_index < step.weights.size(); ++weight_index) {
                if (weight_index > 0U) {
                    stream << ", ";
                }
                stream << step.weights[weight_index];
            }
            stream << ']';
        }
        if (!step.note.empty()) {
            stream << " -- " << step.note;
        }
        stream << '\n';
    }
    return stream.str();
}

std::string Internlm2ExecutionPlan::to_flow_mermaid(
    std::string_view graph_name) const {
    std::ostringstream stream;
    stream << "%% " << mermaid::escape(graph_name) << "\n";
    stream << "flowchart TD\n";
    stream << "    start([CLI request])\n";
    stream << "    output([generated text and debug previews])\n";

    if (steps.empty()) {
        stream << "    start --> output\n";
        return stream.str();
    }

    const auto specs = build_graph_module_specs_from_plan(steps);
    stream << "    cli[\""
           << mermaid::node_label("Parse CLI Options", "model path, prompt, device, generation flags") << "\"]\n";
    for (const auto &module : specs.modules) {
        stream << "    " << module.id << "[\""
               << module_node_label(module) << "\"]\n";
        stream << "    " << module_detail_node_id(module.id, "flow_detail")
               << "[\"" << mermaid::node_label("flow detail", module.flow_detail) << "\"]\n";
        stream << "    " << module.id << " -.-> "
               << module_detail_node_id(module.id, "flow_detail") << "\n";
    }
    stream << "    start --> cli --> gguf --> prompt --> runtime --> forward --> generation --> output\n";
    stream << "    gguf -. weights .-> forward\n";
    stream << "    prompt -. decode support .-> generation\n";

    stream << "    subgraph experiment_points[Fast prototype extension points]\n";
    stream << "        direction LR\n";
        stream << "        device_choice[\"Device path<br/>cpu / cuda:n\"]\n";
    stream << "        one_block[\"Layer experiments<br/>--run-one-block / --layers\"]\n";
        stream << "        sampler_choice[\"Sampling experiments<br/>greedy / stochastic / temperature / seed\"]\n";
        stream << "        preview_choice[\"Inspection output<br/>embedding/logits previews\"]\n";
    stream << "    end\n";
    stream << "    cli -. configures .-> device_choice\n";
    stream << "    forward -. inspect or limit .-> one_block\n";
    stream << "    generation -. replace strategy .-> sampler_choice\n";
    stream << "    output -. includes .-> preview_choice\n";

    mermaid::write_class_defs(stream);
    stream << "    class start,output io\n";
    stream << "    class cli group\n";
    for (const auto &module : specs.modules) {
        stream << "    class " << module.id << " " << module.class_name << "\n";
        stream << "    class " << module_detail_node_id(module.id, "flow_detail") << " group\n";
    }
    stream << "    class sampler_choice activation\n";
    stream << "    class device_choice,one_block,preview_choice tensor\n";

    return stream.str();
}

std::string Internlm2ExecutionPlan::to_mermaid(std::string_view graph_name) const {
    return to_flow_mermaid(graph_name);
}

Internlm2Model Internlm2Model::load(const std::filesystem::path &path) {
    auto gguf = GgufFile::load(path);
    auto weights = GgufTensorReader::open(gguf);
    auto tokenizer = Tokenizer::from_gguf(gguf);
    auto config = Internlm2Config::from_gguf(gguf);
    auto layout = Internlm2TensorLayout::build(config);

    ensure_tensor_exists(weights, layout.token_embedding);
    ensure_tensor_exists(weights, layout.output_norm);
    ensure_tensor_exists(weights, layout.output);
    for (const auto &layer : layout.layers) {
        ensure_tensor_exists(weights, layer.attention_norm);
        ensure_tensor_exists(weights, layer.attention_q);
        ensure_tensor_exists(weights, layer.attention_k);
        ensure_tensor_exists(weights, layer.attention_v);
        ensure_tensor_exists(weights, layer.attention_output);
        ensure_tensor_exists(weights, layer.ffn_norm);
        ensure_tensor_exists(weights, layer.ffn_gate);
        ensure_tensor_exists(weights, layer.ffn_up);
        ensure_tensor_exists(weights, layer.ffn_down);
    }

    return Internlm2Model(
        std::move(gguf),
        std::move(weights),
        std::move(tokenizer),
        std::move(config),
        std::move(layout));
}

const GgufFile &Internlm2Model::gguf() const noexcept {
    return gguf_;
}

const GgufTensorReader &Internlm2Model::weights() const noexcept {
    return weights_;
}

const Tokenizer &Internlm2Model::tokenizer() const noexcept {
    return tokenizer_;
}

const Internlm2Config &Internlm2Model::config() const noexcept {
    return config_;
}

const Internlm2TensorLayout &Internlm2Model::layout() const noexcept {
    return layout_;
}

PromptBatch Internlm2Model::prepare_prompt(std::string_view text, bool add_bos, bool add_eos) const {
    const auto tokenized = tokenizer_.encode(text, add_bos, add_eos);

    PromptBatch prompt;
    prompt.token_ids = tokenized.token_ids;
    prompt.positions.reserve(prompt.token_ids.size());
    for (std::size_t index = 0; index < prompt.token_ids.size(); ++index) {
        prompt.positions.push_back(static_cast<uint32_t>(index));
    }
    return prompt;
}

Internlm2Runtime Internlm2Model::create_runtime(
    uint32_t max_sequence_length,
    ops::OperatorContext context) const {
    return Internlm2Runtime(config_, max_sequence_length, context);
}

TensorBuffer Internlm2Model::embed_tokens(const std::vector<uint32_t> &token_ids, Device device) const {
    const auto embedding_info = weights_.require_info(layout_.token_embedding);
    if (embedding_info.dimensions.size() != 2U) {
        throw GgufError("token_embd.weight is expected to be a 2D tensor");
    }

    const auto row_width = embedding_info.dimensions[0];
    const auto row_count = embedding_info.dimensions[1];
    for (uint32_t token_id : token_ids) {
        if (token_id >= row_count) {
            throw GgufError("token id exceeds token embedding vocabulary size");
        }
    }

#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
    if (device.is_cuda() && cuda_backend_enabled()) {
        return ops::cuda::embedding_lookup(
            weights_.require_tensor(layout_.token_embedding),
            token_ids,
            device);
    }
#endif

    TensorBuffer hidden_state({static_cast<uint64_t>(token_ids.size()), row_width}, 0.0F, device);
    hidden_state.values() = weights_.read_rows_f32(layout_.token_embedding, token_ids);
    if (!device.is_cpu()) {
        hidden_state.sync_host_to_device();
    }
    return hidden_state;
}

TensorBuffer Internlm2Model::run_prompt_embedding(
    const PromptBatch &prompt,
    Internlm2Runtime &runtime) const {
    if (prompt.token_ids.empty()) {
        return TensorBuffer({0U, config_.embedding_length}, 0.0F, runtime.context().device);
    }
    if (runtime.consumed_tokens() + prompt.token_ids.size() > runtime.max_sequence_length()) {
        throw GgufError("prompt does not fit in allocated internlm2 runtime cache");
    }

    auto hidden_state = embed_tokens(prompt.token_ids, runtime.context().device);
    runtime.mark_tokens_consumed(static_cast<uint32_t>(prompt.token_ids.size()));
    return hidden_state;
}

Internlm2ExecutionPlan Internlm2Model::build_prefill_plan() const {
    Internlm2ExecutionPlan plan;
    plan.steps.push_back({
        .type = Internlm2OpType::Embedding,
        .name = "token_embedding_lookup",
        .weights = {layout_.token_embedding},
        .note = "token ids -> hidden_state[seq, hidden]",
    });

    for (uint32_t index = 0; index < config_.block_count; ++index) {
        const auto &layer = layout_.layers[static_cast<std::size_t>(index)];
        const auto prefix = "layer_" + std::to_string(index);
        plan.steps.push_back({
            .type = Internlm2OpType::RmsNorm,
            .name = prefix + ".attn_norm",
            .weights = {layer.attention_norm},
            .note = "normalize hidden state before attention",
        });
        plan.steps.push_back({
            .type = Internlm2OpType::Linear,
            .name = prefix + ".qkv_projection",
            .weights = {layer.attention_q, layer.attention_k, layer.attention_v},
            .note = "project hidden state into q/k/v",
        });
        plan.steps.push_back({
            .type = Internlm2OpType::Rope,
            .name = prefix + ".rope",
            .weights = {},
            .note = "apply rotary position embedding to q/k",
        });
        plan.steps.push_back({
            .type = Internlm2OpType::Attention,
            .name = prefix + ".attention",
            .weights = {},
            .note = "causal self-attention with KV cache update",
        });
        plan.steps.push_back({
            .type = Internlm2OpType::Linear,
            .name = prefix + ".attn_output_projection",
            .weights = {layer.attention_output},
            .note = "project attention output back to hidden size",
        });
        plan.steps.push_back({
            .type = Internlm2OpType::ResidualAdd,
            .name = prefix + ".attn_residual",
            .weights = {},
            .note = "add attention residual",
        });
        plan.steps.push_back({
            .type = Internlm2OpType::RmsNorm,
            .name = prefix + ".ffn_norm",
            .weights = {layer.ffn_norm},
            .note = "normalize hidden state before MLP",
        });
        plan.steps.push_back({
            .type = Internlm2OpType::Linear,
            .name = prefix + ".ffn_gate_up",
            .weights = {layer.ffn_gate, layer.ffn_up},
            .note = "compute gate and up projections",
        });
        plan.steps.push_back({
            .type = Internlm2OpType::SwiGLU,
            .name = prefix + ".swiglu",
            .weights = {},
            .note = "silu(gate) * up",
        });
        plan.steps.push_back({
            .type = Internlm2OpType::Linear,
            .name = prefix + ".ffn_down_projection",
            .weights = {layer.ffn_down},
            .note = "project MLP output back to hidden size",
        });
        plan.steps.push_back({
            .type = Internlm2OpType::ResidualAdd,
            .name = prefix + ".ffn_residual",
            .weights = {},
            .note = "add MLP residual",
        });
    }

    plan.steps.push_back({
        .type = Internlm2OpType::RmsNorm,
        .name = "output_norm",
        .weights = {layout_.output_norm},
        .note = "normalize final hidden state",
    });
    plan.steps.push_back({
        .type = Internlm2OpType::OutputProjection,
        .name = "lm_head",
        .weights = {layout_.output},
        .note = "project last token hidden state to logits",
    });

    return plan;
}

std::string Internlm2Model::describe_pipeline() const {
    std::ostringstream stream;
    stream << "internlm2 prompt pipeline\n";
    stream << "1. tokenizer -> token ids\n";
    stream << "2. positions -> prompt positions [0..n-1]\n";
    stream << "3. embedding -> " << layout_.token_embedding << '\n';
    stream << "4. decoder blocks -> " << config_.block_count << " layers\n";
    stream << "5. final norm -> " << layout_.output_norm << '\n';
    stream << "6. lm head -> " << layout_.output << '\n';
    stream << "7. sampling -> next token\n";
    stream << "execution plan nodes: " << build_prefill_plan().steps.size() << '\n';
    stream << "implemented now: steps 1-7 with greedy/stochastic sampling, runtime/KV cache allocation, execution plan builder\n";
    stream << "reserved interfaces: top-k/top-p sampling and further performance optimization\n";
    return stream.str();
}

std::string Internlm2Model::to_dataflow_mermaid(std::string_view graph_name) const {
    const auto specs = build_graph_module_specs(
        config_.block_count,
        config_.embedding_length,
        config_.context_length,
        tokenizer_.vocab_size());
    std::ostringstream stream;
    stream << "%% " << mermaid::escape(graph_name) << "\n";
    stream << "flowchart TD\n";
    stream << "    gguf_file[(GGUF model file)]\n";
    stream << "    prompt_text([Prompt text])\n";
    stream << "    device_options([Device and generation options])\n";
    stream << "    output_text([Generated text])\n";
    for (const auto &module : specs.modules) {
        stream << "    " << module.id << "[\""
               << module_node_label(module) << "\"]\n";
        stream << "    " << module_detail_node_id(module.id, "data_detail")
               << "[\"" << mermaid::node_label("data detail", module.data_detail) << "\"]\n";
        stream << "    " << module.id << " -.-> "
               << module_detail_node_id(module.id, "data_detail") << "\n";
    }

    stream << "    subgraph shared_data[Shared data interfaces from module specs]\n";
    stream << "        metadata[\"GGUF loading data<br/>metadata + tensor infos + tensor data\"]\n";
    stream << "        tokenizer[\"Tokenizer<br/>vocab size: " << tokenizer_.vocab_size() << "\"]\n";
    stream << "        config[\"Internlm2Config<br/>layers: " << config_.block_count
           << ", hidden: " << config_.embedding_length
           << ", ctx: " << config_.context_length << "\"]\n";
    stream << "        prompt_batch[\"PromptBatch<br/>token_ids + positions\"]\n";
    stream << "        runtime_state[\"Runtime state<br/>OperatorContext + KV Cache\"]\n";
    stream << "        embedding_tensor[\"embedding TensorBuffer<br/>[seq, " << config_.embedding_length << "]\"]\n";
    stream << "        decoder_blocks[\"Decoder Block x" << config_.block_count
           << "<br/>hidden_state transformed layer by layer\"]\n";
    stream << "        logits[\"logits<br/>[vocab: " << tokenizer_.vocab_size() << "]\"]\n";
    stream << "        sampled_ids[\"generated token ids\"]\n";
    stream << "    end\n";

    stream << "    gguf_file --> gguf\n";
    stream << "    prompt_text --> prompt\n";
    stream << "    device_options --> runtime\n";
    for (const auto &connection : specs.connections) {
        stream << "    " << connection.from << " -->|" << mermaid::escape(connection.data_label)
               << "| " << connection.to << "\n";
    }
    stream << "    generation --> output_text\n";
    stream << "    gguf --> metadata --> tokenizer\n";
    stream << "    metadata --> config\n";
    stream << "    tokenizer --> prompt --> prompt_batch\n";
    stream << "    config --> runtime --> runtime_state\n";
    stream << "    prompt_batch --> embedding_tensor --> decoder_blocks --> logits\n";
    stream << "    runtime_state <--> decoder_blocks\n";
    stream << "    logits --> sampled_ids --> generation\n";
    stream << "    tokenizer --> generation\n";

    mermaid::write_class_defs(stream);
    stream << "    class gguf_file file\n";
    stream << "    class prompt_text,device_options,output_text io\n";
    for (const auto &module : specs.modules) {
        stream << "    class " << module.id << " " << module.class_name << "\n";
        stream << "    class " << module_detail_node_id(module.id, "data_detail") << " group\n";
    }
    stream << "    class metadata,config metadata\n";
    stream << "    class tokenizer,prompt_batch group\n";
    stream << "    class embedding_tensor,decoder_blocks,logits,sampled_ids tensor\n";
    stream << "    class runtime_state runtime\n";
    return stream.str();
}

std::vector<MermaidGraphDocument> Internlm2Model::build_module_graphs() const {
    const auto specs = build_graph_module_specs(
        config_.block_count,
        config_.embedding_length,
        config_.context_length,
        tokenizer_.vocab_size());
    const auto *gguf = find_module_spec(specs, "gguf");
    const auto *prompt = find_module_spec(specs, "prompt");
    const auto *runtime = find_module_spec(specs, "runtime");
    const auto *forward = find_module_spec(specs, "forward");
    const auto *generation = find_module_spec(specs, "generation");
    return {
        {
            .file_name = "00-module-overview.mmd",
            .description = "module overview graph",
            .content = module_overview_mermaid("simple-LLMRF module overview: " + config_.model_name, specs),
        },
        {
            .file_name = gguf == nullptr ? "01-gguf-loading.mmd" : gguf->file_name,
            .description = gguf == nullptr ? "GGUF loading module graph" : gguf->description,
            .content = mermaid::gguf_loading_module(
                "simple-LLMRF GGUF loading module",
                gguf == nullptr ? "model path" : gguf->input,
                gguf == nullptr ? "metadata, tensor reader" : gguf->output),
        },
        {
            .file_name = prompt == nullptr ? "02-prompt-tokenizer.mmd" : prompt->file_name,
            .description = prompt == nullptr ? "prompt and tokenizer module graph" : prompt->description,
            .content = mermaid::prompt_tokenizer_module(
                "simple-LLMRF prompt and tokenizer module",
                prompt == nullptr ? "prompt text + tokenizer" : prompt->input,
                prompt == nullptr ? "PromptBatch" : prompt->output),
        },
        {
            .file_name = runtime == nullptr ? "03-runtime-tensor.mmd" : runtime->file_name,
            .description = runtime == nullptr ? "runtime and tensor module graph" : runtime->description,
            .content = mermaid::runtime_tensor_module(
                "simple-LLMRF runtime and tensor module",
                runtime == nullptr ? "config + device options" : runtime->input,
                runtime == nullptr ? "runtime, KV cache, TensorBuffer placement" : runtime->output,
                config_.block_count,
                config_.embedding_length,
                config_.attention_head_count_kv,
                config_.head_dimension()),
        },
        {
            .file_name = forward == nullptr ? "04-model-forward.mmd" : forward->file_name,
            .description = forward == nullptr ? "model forward module graph" : forward->description,
            .content = forward_module_mermaid(
                "simple-LLMRF InternLM2 forward module",
                forward == nullptr ? "PromptBatch + weights + runtime" : forward->input,
                forward == nullptr ? "hidden state + logits" : forward->output,
                config_),
        },
        {
            .file_name = generation == nullptr ? "05-generation-output.mmd" : generation->file_name,
            .description = generation == nullptr ? "generation output module graph" : generation->description,
            .content = generation_module_mermaid(
                "simple-LLMRF generation output module",
                generation == nullptr ? "logits + tokenizer" : generation->input,
                generation == nullptr ? "generated ids and text" : generation->output),
        },
    };
}

TensorBuffer Internlm2Model::apply_final_norm(
    const TensorBuffer &hidden_state,
    ops::OperatorContext context) const {
    const auto weight_info = weights_.require_info(layout_.output_norm);
    if (weight_info.dimensions.size() != 1U || weight_info.dimensions[0] != config_.embedding_length) {
        throw GgufError("output_norm.weight shape does not match embedding length");
    }

    const auto effective_context =
        context.device == hidden_state.device() ? context : context_for_device(hidden_state.device());
    return ops::rms_norm(
        hidden_state,
        weights_.require_tensor(layout_.output_norm),
        config_.rms_norm_epsilon,
        effective_context);
}

TensorBuffer Internlm2Model::forward_prompt(
    const PromptBatch &prompt,
    Internlm2Runtime &runtime,
    uint32_t layer_count) const {
    const auto embedding = run_prompt_embedding(prompt, runtime);
    return forward_blocks(embedding, runtime, layer_count);
}

TensorBuffer Internlm2Model::forward_layer(
    const TensorBuffer &hidden_state,
    uint32_t layer_index,
    Internlm2Runtime &runtime) const {
    if (layer_index >= config_.block_count) {
        throw GgufError("layer index out of range");
    }
    if (hidden_state.rows() == 0U) {
        return hidden_state;
    }
    if (runtime.consumed_tokens() < hidden_state.rows()) {
        throw GgufError("runtime consumed token count is inconsistent with hidden state length");
    }
    if (hidden_state.device() != runtime.context().device) {
        throw GgufError("forward_layer hidden state device does not match runtime execution device");
    }

    const auto start_position =
        runtime.consumed_tokens() - static_cast<uint32_t>(hidden_state.rows());
    const auto &names = layout_.layers[static_cast<std::size_t>(layer_index)];
    const auto context = runtime.context();

    const auto attn_norm = ops::rms_norm(
        hidden_state,
        weights_.require_tensor(names.attention_norm),
        config_.rms_norm_epsilon,
        context);
    auto q = linear_project(weights_, names.attention_q, attn_norm, context);
    auto k = linear_project(weights_, names.attention_k, attn_norm, context);
    auto v = linear_project(weights_, names.attention_v, attn_norm, context);

    apply_rope_inplace(q, k, config_, start_position, context);
    write_kv_cache(runtime.layers()[layer_index], k, v, start_position, config_, context);

    const auto attention_context =
        causal_attention(q, runtime.layers()[layer_index], start_position, config_, context);
    const auto attention_output =
        linear_project(weights_, names.attention_output, attention_context, context);
    auto hidden = ops::add(hidden_state, attention_output, context);

    const auto ffn_norm = ops::rms_norm(
        hidden,
        weights_.require_tensor(names.ffn_norm),
        config_.rms_norm_epsilon,
        context);
    const auto gate = linear_project(weights_, names.ffn_gate, ffn_norm, context);
    const auto up = linear_project(weights_, names.ffn_up, ffn_norm, context);
    const auto swiglu = ops::multiply(ops::silu(gate, context), up, context);
    const auto down = linear_project(weights_, names.ffn_down, swiglu, context);
    hidden = ops::add(hidden, down, context);

    return hidden;
}

TensorBuffer Internlm2Model::forward_blocks(
    const TensorBuffer &hidden_state,
    Internlm2Runtime &runtime,
    uint32_t layer_count) const {
    if (config_.block_count == 0U) {
        return hidden_state;
    }
    if (hidden_state.device() != runtime.context().device) {
        throw GgufError("forward_blocks hidden state device does not match runtime execution device");
    }

    const auto max_layers = layer_count == 0U
        ? config_.block_count
        : std::min(layer_count, config_.block_count);
    auto current = hidden_state;
    for (uint32_t layer = 0; layer < max_layers; ++layer) {
        current = forward_layer(current, layer, runtime);
    }
    return current;
}

std::vector<float> Internlm2Model::compute_logits(const TensorBuffer &hidden_state) const {
    if (hidden_state.rows() == 0U) {
        return {};
    }

    const auto context = context_for_device(hidden_state.device());
    const auto normalized = apply_final_norm(hidden_state, context);
    const auto output = weights_.require_tensor(layout_.output);
    if (output.dimensions.size() != 2U) {
        throw GgufError("output.weight is expected to be a 2D tensor");
    }
    if (output.row_width() != config_.embedding_length) {
        throw GgufError("output.weight hidden dimension does not match embedding length");
    }

#if defined(SLLMRF_USE_CUDA) && SLLMRF_USE_CUDA
    if (normalized.device().is_cuda() && cuda_backend_enabled()) {
        return ops::cuda::project_last_token_logits(output, normalized);
    }
#endif

    const auto normalized_cpu =
        normalized.device().is_cpu() ? normalized : normalized.copy_to(Device::cpu());
    const auto last_token = normalized_cpu.row(normalized_cpu.rows() - 1U);

    const auto vocab_size = static_cast<std::size_t>(output.row_count());
    const auto hidden_size = static_cast<std::size_t>(output.row_width());
    const auto bytes_per_row = hidden_size * ggml_type_size(output.type);

    std::vector<float> logits(vocab_size, 0.0F);
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (std::int64_t token_index = 0; token_index < static_cast<std::int64_t>(vocab_size); ++token_index) {
        const auto *row_ptr = output.data + token_index * bytes_per_row;
        float dot = 0.0F;

        if (output.type == GgmlType::F32) {
            const auto *weights = reinterpret_cast<const float *>(row_ptr);
            for (std::size_t hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
                dot += last_token[hidden_index] * weights[hidden_index];
            }
        } else if (output.type == GgmlType::F16) {
            const auto *weights = reinterpret_cast<const uint16_t *>(row_ptr);
            for (std::size_t hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
                dot += last_token[hidden_index] * half_to_float(weights[hidden_index]);
            }
        } else {
            throw GgufError("output projection is not implemented for tensor type " + to_string(output.type));
        }

        logits[static_cast<std::size_t>(token_index)] = dot;
    }

    return logits;
}

uint32_t Internlm2Model::sample_greedy(const std::vector<float> &logits) const {
    if (logits.empty()) {
        throw GgufError("cannot sample from empty logits");
    }

    return static_cast<uint32_t>(std::distance(
        logits.begin(),
        std::max_element(logits.begin(), logits.end())));
}

uint32_t Internlm2Model::sample_stochastic(
    const std::vector<float> &logits,
    float temperature,
    uint64_t seed) const {
    std::mt19937_64 rng(seed);
    return sample_stochastic_impl(logits, temperature, rng);
}

GenerationResult Internlm2Model::generate(
    std::string_view prompt,
    const GenerationConfig &config) const {
    GenerationResult result;
    const auto prompt_batch = prepare_prompt(prompt, config.add_bos, false);
    if (config.max_new_tokens == 0U) {
        result.prompt_token_ids = prompt_batch.token_ids;
        result.full_text = std::string(prompt);
        return result;
    }

    auto runtime = create_runtime(
        resolve_runtime_sequence_length(config_, prompt_batch.token_ids.size(), config.max_new_tokens),
        config.execution_context);
    result.prompt_token_ids = prompt_batch.token_ids;

    auto hidden = forward_prompt(prompt_batch, runtime, config.layer_count);
    auto logits = compute_logits(hidden);
    std::mt19937_64 stochastic_rng;
    if (config.sampling_strategy == SamplingStrategy::Stochastic) {
        uint64_t seed = config.seed;
        if (!config.use_seed) {
            std::random_device device;
            seed = (static_cast<uint64_t>(device()) << 32U) ^ static_cast<uint64_t>(device());
        }
        stochastic_rng.seed(seed);
    }

    for (uint32_t step = 0; step < config.max_new_tokens; ++step) {
        const auto next_token = config.sampling_strategy == SamplingStrategy::Stochastic
            ? sample_stochastic_impl(logits, config.temperature, stochastic_rng)
            : sample_greedy(logits);
        if (config.stop_at_eos && next_token == tokenizer_.config().eos_token_id) {
            break;
        }

        result.generated_token_ids.push_back(next_token);
        PromptBatch next_batch;
        next_batch.token_ids = {next_token};
        next_batch.positions = {runtime.consumed_tokens()};
        hidden = forward_prompt(next_batch, runtime, config.layer_count);
        logits = compute_logits(hidden);
    }

    result.generated_text = tokenizer_.decode(result.generated_token_ids);
    result.full_text = std::string(prompt);
    result.full_text += result.generated_text;
    return result;
}

Internlm2Model::Internlm2Model(
    GgufFile gguf,
    GgufTensorReader weights,
    Tokenizer tokenizer,
    Internlm2Config config,
    Internlm2TensorLayout layout)
    : gguf_(std::move(gguf)),
      weights_(std::move(weights)),
      tokenizer_(std::move(tokenizer)),
      config_(std::move(config)),
      layout_(std::move(layout)) {}

}  // namespace sllmrf
