#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "sllmrf/gguf.h"
#include "sllmrf/operators.h"
#include "sllmrf/tensor.h"
#include "sllmrf/tokenizer.h"

namespace sllmrf {

struct Internlm2Config {
    std::string architecture;
    std::string model_name;
    uint32_t context_length {0};
    uint32_t block_count {0};
    uint32_t embedding_length {0};
    uint32_t feed_forward_length {0};
    uint32_t attention_head_count {0};
    uint32_t attention_head_count_kv {0};
    float rms_norm_epsilon {0.0F};
    float rope_freq_base {0.0F};
    uint32_t vocabulary_size {0};

    [[nodiscard]] static Internlm2Config from_gguf(const GgufFile &file);
    [[nodiscard]] uint32_t head_dimension() const;
};

struct Internlm2LayerNames {
    std::string attention_norm;
    std::string attention_q;
    std::string attention_k;
    std::string attention_v;
    std::string attention_output;
    std::string ffn_norm;
    std::string ffn_gate;
    std::string ffn_up;
    std::string ffn_down;
};

struct Internlm2TensorLayout {
    std::string token_embedding;
    std::vector<Internlm2LayerNames> layers;
    std::string output_norm;
    std::string output;

    [[nodiscard]] static Internlm2TensorLayout build(const Internlm2Config &config);
};

struct PromptBatch {
    std::vector<uint32_t> token_ids;
    std::vector<uint32_t> positions;
};

struct KvCacheLayer {
    TensorBuffer key;
    TensorBuffer value;
};

enum class Internlm2OpType {
    Embedding,
    RmsNorm,
    Linear,
    Rope,
    Attention,
    ResidualAdd,
    SwiGLU,
    OutputProjection,
};

struct Internlm2PlanStep {
    Internlm2OpType type {Internlm2OpType::Embedding};
    std::string name;
    std::vector<std::string> weights;
    std::string note;
};

struct Internlm2ExecutionPlan {
    std::vector<Internlm2PlanStep> steps;

    [[nodiscard]] std::string describe() const;
};

enum class SamplingStrategy {
    Greedy,
    Stochastic,
};

struct GenerationConfig {
    uint32_t max_new_tokens {1};
    bool add_bos {true};
    bool stop_at_eos {true};
    uint32_t layer_count {0};
    ops::OperatorContext execution_context {ops::OperatorContext::cpu()};
    SamplingStrategy sampling_strategy {SamplingStrategy::Greedy};
    float temperature {1.0F};
    uint64_t seed {0U};
    bool use_seed {false};
};

struct GenerationResult {
    std::vector<uint32_t> prompt_token_ids;
    std::vector<uint32_t> generated_token_ids;
    std::string generated_text;
    std::string full_text;
};

class Internlm2Runtime {
public:
    Internlm2Runtime() = default;
    Internlm2Runtime(
        const Internlm2Config &config,
        uint32_t max_sequence_length,
        ops::OperatorContext context = ops::OperatorContext::cpu());

    void reset();

    [[nodiscard]] uint32_t max_sequence_length() const noexcept;
    [[nodiscard]] uint32_t consumed_tokens() const noexcept;
    [[nodiscard]] const ops::OperatorContext &context() const noexcept;
    [[nodiscard]] const std::vector<KvCacheLayer> &layers() const noexcept;
    [[nodiscard]] std::vector<KvCacheLayer> &layers() noexcept;

    void mark_tokens_consumed(uint32_t count);

private:
    uint32_t max_sequence_length_ {0};
    uint32_t consumed_tokens_ {0};
    ops::OperatorContext context_ {ops::OperatorContext::cpu()};
    std::vector<KvCacheLayer> layers_;
};

class Internlm2Model {
public:
    [[nodiscard]] static Internlm2Model load(const std::filesystem::path &path);

    [[nodiscard]] const GgufFile &gguf() const noexcept;
    [[nodiscard]] const GgufTensorReader &weights() const noexcept;
    [[nodiscard]] const Tokenizer &tokenizer() const noexcept;
    [[nodiscard]] const Internlm2Config &config() const noexcept;
    [[nodiscard]] const Internlm2TensorLayout &layout() const noexcept;

    [[nodiscard]] PromptBatch prepare_prompt(
        std::string_view text,
        bool add_bos = true,
        bool add_eos = false) const;
    [[nodiscard]] Internlm2Runtime create_runtime(
        uint32_t max_sequence_length = 0,
        ops::OperatorContext context = ops::OperatorContext::cpu()) const;
    [[nodiscard]] TensorBuffer embed_tokens(
        const std::vector<uint32_t> &token_ids,
        Device device = Device::cpu()) const;
    [[nodiscard]] TensorBuffer run_prompt_embedding(
        const PromptBatch &prompt,
        Internlm2Runtime &runtime) const;
    [[nodiscard]] Internlm2ExecutionPlan build_prefill_plan() const;
    [[nodiscard]] std::string describe_pipeline() const;
    [[nodiscard]] TensorBuffer apply_final_norm(
        const TensorBuffer &hidden_state,
        ops::OperatorContext context = ops::OperatorContext::cpu()) const;
    [[nodiscard]] TensorBuffer forward_prompt(
        const PromptBatch &prompt,
        Internlm2Runtime &runtime,
        uint32_t layer_count = 0U) const;
    [[nodiscard]] TensorBuffer forward_layer(
        const TensorBuffer &hidden_state,
        uint32_t layer_index,
        Internlm2Runtime &runtime) const;

    [[nodiscard]] TensorBuffer forward_blocks(
        const TensorBuffer &hidden_state,
        Internlm2Runtime &runtime,
        uint32_t layer_count = 0U) const;
    [[nodiscard]] std::vector<float> compute_logits(const TensorBuffer &hidden_state) const;
    [[nodiscard]] uint32_t sample_greedy(const std::vector<float> &logits) const;
    [[nodiscard]] uint32_t sample_stochastic(
        const std::vector<float> &logits,
        float temperature,
        uint64_t seed) const;
    [[nodiscard]] GenerationResult generate(
        std::string_view prompt,
        const GenerationConfig &config = {}) const;

private:
    Internlm2Model(
        GgufFile gguf,
        GgufTensorReader weights,
        Tokenizer tokenizer,
        Internlm2Config config,
        Internlm2TensorLayout layout);

    GgufFile gguf_;
    GgufTensorReader weights_;
    Tokenizer tokenizer_;
    Internlm2Config config_;
    Internlm2TensorLayout layout_;
};

}  // namespace sllmrf
