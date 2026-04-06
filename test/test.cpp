#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sllmrf.h"

namespace {

using sllmrf::GgufFile;
using sllmrf::GgmlType;
using sllmrf::Internlm2Model;
using sllmrf::MetadataValueType;
using sllmrf::TensorBuffer;
using sllmrf::Tokenizer;

struct TensorSpec {
    std::string name;
    std::vector<uint64_t> dimensions;
    GgmlType type;
    std::vector<float> values;
    uint64_t offset {0};
};

void write_u32(std::ofstream &stream, uint32_t value) {
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_u64(std::ofstream &stream, uint64_t value) {
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_i32(std::ofstream &stream, int32_t value) {
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_f32(std::ofstream &stream, float value) {
    stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_zero_padding(std::ofstream &stream, std::size_t byte_count) {
    std::vector<char> zeros(byte_count, '\0');
    stream.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
}

void write_bool(std::ofstream &stream, bool value) {
    const uint8_t raw = value ? 1U : 0U;
    stream.write(reinterpret_cast<const char *>(&raw), sizeof(raw));
}

void write_string(std::ofstream &stream, const std::string &value) {
    write_u64(stream, static_cast<uint64_t>(value.size()));
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_metadata_key(std::ofstream &stream, const std::string &key) {
    write_string(stream, key);
}

void write_metadata_string(std::ofstream &stream, const std::string &key, const std::string &value) {
    write_metadata_key(stream, key);
    write_u32(stream, static_cast<uint32_t>(MetadataValueType::String));
    write_string(stream, value);
}

void write_metadata_bool(std::ofstream &stream, const std::string &key, bool value) {
    write_metadata_key(stream, key);
    write_u32(stream, static_cast<uint32_t>(MetadataValueType::Bool));
    write_bool(stream, value);
}

void write_metadata_u32(std::ofstream &stream, const std::string &key, uint32_t value) {
    write_metadata_key(stream, key);
    write_u32(stream, static_cast<uint32_t>(MetadataValueType::UInt32));
    write_u32(stream, value);
}

void write_metadata_string_array(
    std::ofstream &stream,
    const std::string &key,
    const std::vector<std::string> &values) {
    write_metadata_key(stream, key);
    write_u32(stream, static_cast<uint32_t>(MetadataValueType::Array));
    write_u32(stream, static_cast<uint32_t>(MetadataValueType::String));
    write_u64(stream, static_cast<uint64_t>(values.size()));
    for (const auto &value : values) {
        write_string(stream, value);
    }
}

void write_metadata_float_array(
    std::ofstream &stream,
    const std::string &key,
    const std::vector<float> &values) {
    write_metadata_key(stream, key);
    write_u32(stream, static_cast<uint32_t>(MetadataValueType::Array));
    write_u32(stream, static_cast<uint32_t>(MetadataValueType::Float32));
    write_u64(stream, static_cast<uint64_t>(values.size()));
    for (float value : values) {
        write_f32(stream, value);
    }
}

void write_metadata_i32_array(
    std::ofstream &stream,
    const std::string &key,
    const std::vector<int32_t> &values) {
    write_metadata_key(stream, key);
    write_u32(stream, static_cast<uint32_t>(MetadataValueType::Array));
    write_u32(stream, static_cast<uint32_t>(MetadataValueType::Int32));
    write_u64(stream, static_cast<uint64_t>(values.size()));
    for (int32_t value : values) {
        write_i32(stream, value);
    }
}

void write_tensor_info(std::ofstream &stream, const TensorSpec &tensor) {
    write_string(stream, tensor.name);
    write_u32(stream, static_cast<uint32_t>(tensor.dimensions.size()));
    for (uint64_t dimension : tensor.dimensions) {
        write_u64(stream, dimension);
    }
    write_u32(stream, static_cast<uint32_t>(tensor.type));
    write_u64(stream, tensor.offset);
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return ((value + alignment - 1U) / alignment) * alignment;
}

std::filesystem::path create_fixture() {
    const auto path = std::filesystem::temp_directory_path() / "sllmrf-fixture.gguf";
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("failed to create GGUF fixture");
    }

    constexpr uint32_t kMagic = 0x46554747U;
    constexpr uint32_t kVersion = 3U;
    constexpr uint64_t kTensorCount = 12U;
    constexpr uint64_t kMetadataCount = 18U;
    constexpr std::size_t kAlignment = 32U;

    std::vector<TensorSpec> tensors = {
        {
            .name = "token_embd.weight",
            .dimensions = {2U, 5U},
            .type = GgmlType::F32,
            .values = {0.0F, 0.0F, 1.0F, 1.5F, 2.0F, 2.5F, 3.0F, 3.5F, 4.0F, 4.5F},
        },
        {.name = "blk.0.attn_norm.weight", .dimensions = {2U}, .type = GgmlType::F32, .values = {1.0F, 1.0F}},
        {.name = "blk.0.attn_q.weight", .dimensions = {2U, 2U}, .type = GgmlType::F32, .values = {1.0F, 0.0F, 0.0F, 1.0F}},
        {.name = "blk.0.attn_k.weight", .dimensions = {2U, 2U}, .type = GgmlType::F32, .values = {1.0F, 0.0F, 0.0F, 1.0F}},
        {.name = "blk.0.attn_v.weight", .dimensions = {2U, 2U}, .type = GgmlType::F32, .values = {1.0F, 0.0F, 0.0F, 1.0F}},
        {.name = "blk.0.attn_output.weight", .dimensions = {2U, 2U}, .type = GgmlType::F32, .values = {1.0F, 0.0F, 0.0F, 1.0F}},
        {.name = "blk.0.ffn_norm.weight", .dimensions = {2U}, .type = GgmlType::F32, .values = {1.0F, 1.0F}},
        {.name = "blk.0.ffn_gate.weight", .dimensions = {2U, 4U}, .type = GgmlType::F32, .values = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F, 0.8F}},
        {.name = "blk.0.ffn_up.weight", .dimensions = {2U, 4U}, .type = GgmlType::F32, .values = {0.8F, 0.7F, 0.6F, 0.5F, 0.4F, 0.3F, 0.2F, 0.1F}},
        {.name = "blk.0.ffn_down.weight", .dimensions = {4U, 2U}, .type = GgmlType::F32, .values = {0.9F, 0.0F, 0.0F, 0.9F, 0.2F, 0.0F, 0.0F, 0.2F}},
        {.name = "output_norm.weight", .dimensions = {2U}, .type = GgmlType::F32, .values = {1.0F, 1.0F}},
        {.name = "output.weight", .dimensions = {2U, 5U}, .type = GgmlType::F32, .values = {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 0.5F, 0.5F}},
    };

    std::size_t current_offset = 0U;
    for (auto &tensor : tensors) {
        current_offset = align_up(current_offset, kAlignment);
        tensor.offset = static_cast<uint64_t>(current_offset);
        current_offset += tensor.values.size() * sizeof(float);
    }

    write_u32(stream, kMagic);
    write_u32(stream, kVersion);
    write_u64(stream, kTensorCount);
    write_u64(stream, kMetadataCount);

    write_metadata_string(stream, "general.architecture", "internlm2");
    write_metadata_string(stream, "general.name", "fixture-internlm2");
    write_metadata_u32(stream, "internlm2.context_length", 16U);
    write_metadata_u32(stream, "internlm2.block_count", 1U);
    write_metadata_u32(stream, "internlm2.embedding_length", 2U);
    write_metadata_u32(stream, "internlm2.feed_forward_length", 4U);
    write_metadata_u32(stream, "internlm2.attention.head_count", 1U);
    write_metadata_u32(stream, "internlm2.attention.head_count_kv", 1U);
    write_metadata_key(stream, "internlm2.attention.layer_norm_rms_epsilon");
    write_u32(stream, static_cast<uint32_t>(MetadataValueType::Float32));
    write_f32(stream, 1.0e-5F);
    write_metadata_key(stream, "internlm2.rope.freq_base");
    write_u32(stream, static_cast<uint32_t>(MetadataValueType::Float32));
    write_f32(stream, 10000.0F);
    write_metadata_string(stream, "tokenizer.ggml.model", "sentencepiece");
    write_metadata_string_array(
        stream,
        "tokenizer.ggml.tokens",
        {"<unk>", "\xE2\x96\x81hello", "\xE2\x96\x81world", "!", "<s>"});
    write_metadata_float_array(stream, "tokenizer.ggml.scores", {0.0F, 3.0F, 2.0F, 0.5F, 0.0F});
    write_metadata_i32_array(stream, "tokenizer.ggml.token_type", {2, 1, 1, 1, 3});
    write_metadata_bool(stream, "tokenizer.ggml.add_space_prefix", true);
    write_metadata_bool(stream, "tokenizer.ggml.add_bos_token", true);
    write_metadata_u32(stream, "tokenizer.ggml.bos_token_id", 4U);
    write_metadata_u32(stream, "tokenizer.ggml.unknown_token_id", 0U);

    for (const auto &tensor : tensors) {
        write_tensor_info(stream, tensor);
    }

    const auto current_size = static_cast<std::size_t>(stream.tellp());
    const auto aligned_size = align_up(current_size, kAlignment);
    if (aligned_size > current_size) {
        write_zero_padding(stream, aligned_size - current_size);
    }

    std::size_t written_tensor_bytes = 0U;
    for (const auto &tensor : tensors) {
        const auto target_offset = static_cast<std::size_t>(tensor.offset);
        if (target_offset > written_tensor_bytes) {
            write_zero_padding(stream, target_offset - written_tensor_bytes);
            written_tensor_bytes = target_offset;
        }
        for (float value : tensor.values) {
            write_f32(stream, value);
        }
        written_tensor_bytes += tensor.values.size() * sizeof(float);
    }

    return path;
}

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_near(float lhs, float rhs, float epsilon, const std::string &message) {
    if (std::fabs(lhs - rhs) > epsilon) {
        std::ostringstream stream;
        stream << message << " lhs=" << lhs << " rhs=" << rhs;
        throw std::runtime_error(stream.str());
    }
}

void run_fixture_test() {
    const auto fixture = create_fixture();
    const auto file = GgufFile::load(fixture);
    expect(file.version() == 3U, "fixture version mismatch");
    expect(file.tensor_count() == 12U, "fixture tensor count mismatch");
    expect(file.metadata().size() == 18U, "fixture metadata count mismatch");

    const auto tokenizer = Tokenizer::from_gguf(file);
    const auto result = tokenizer.encode("hello world!");

    expect(result.token_ids.size() == 4U, "unexpected token count");
    expect(result.token_ids[0] == 4U, "unexpected bos token");
    expect(result.token_ids[1] == 1U, "unexpected first word token");
    expect(result.token_ids[2] == 2U, "unexpected second word token");
    expect(result.token_ids[3] == 3U, "unexpected punctuation token");
    expect(tokenizer.decode(result.token_ids) == "hello world!", "decode mismatch");

    const auto model = Internlm2Model::load(fixture);
    expect(model.config().architecture == "internlm2", "architecture mismatch");
    expect(model.config().block_count == 1U, "block count mismatch");
    expect(model.config().embedding_length == 2U, "embedding length mismatch");
    expect(model.layout().layers.size() == 1U, "layer layout mismatch");

    auto runtime = model.create_runtime();
    const auto prompt = model.prepare_prompt("hello world!");
    expect(prompt.positions.size() == prompt.token_ids.size(), "positions size mismatch");

    const auto embedding = model.run_prompt_embedding(prompt, runtime);
    expect(embedding.shape().size() == 2U, "embedding rank mismatch");
    expect(embedding.shape()[0] == 4U, "embedding row count mismatch");
    expect(embedding.shape()[1] == 2U, "embedding width mismatch");
    expect(runtime.consumed_tokens() == 4U, "runtime consumed token mismatch");

    const std::vector<float> expected_embedding = {4.0F, 4.5F, 1.0F, 1.5F, 2.0F, 2.5F, 3.0F, 3.5F};
    expect(embedding.values() == expected_embedding, "embedding values mismatch");

    const auto pipeline = model.describe_pipeline();
    expect(pipeline.find("execution plan nodes") != std::string::npos, "pipeline description mismatch");

    const auto plan = model.build_prefill_plan();
    expect(!plan.steps.empty(), "execution plan should not be empty");
    expect(plan.steps.front().type == sllmrf::Internlm2OpType::Embedding, "first plan step mismatch");
    expect(plan.steps.back().type == sllmrf::Internlm2OpType::OutputProjection, "last plan step mismatch");
    expect(plan.describe().find("layer_0.qkv_projection") != std::string::npos, "plan description mismatch");

    const TensorBuffer rms_input({1U, 2U}, {3.0F, 4.0F});
    const auto rms_output = sllmrf::ops::rms_norm(rms_input, {1.0F, 1.0F}, 1.0e-5F);
    expect_near(rms_output.at(0, 0), 0.848527F, 1.0e-4F, "rms_norm first value mismatch");
    expect_near(rms_output.at(0, 1), 1.131370F, 1.0e-4F, "rms_norm second value mismatch");

    const auto normalized = model.apply_final_norm(embedding);
    expect(normalized.shape() == embedding.shape(), "final norm shape mismatch");

    const auto logits = model.compute_logits(embedding);
    expect(logits.size() == 5U, "fixture logits size mismatch");
    expect_near(logits[0], 0.0F, 1.0e-4F, "logits[0] mismatch");
    expect_near(logits[1], 0.920575F, 1.0e-3F, "logits[1] mismatch");
    expect_near(logits[2], 1.073971F, 1.0e-3F, "logits[2] mismatch");
    expect_near(logits[3], 1.994546F, 1.0e-3F, "logits[3] mismatch");
    expect_near(logits[4], 0.997273F, 1.0e-3F, "logits[4] mismatch");

    const auto block_output = model.forward_layer(embedding, 0U, runtime);
    expect(block_output.shape() == embedding.shape(), "block output shape mismatch");
    expect_near(block_output.at(0, 0), 5.500523F, 1.0e-4F, "block output[0,0] mismatch");
    expect_near(block_output.at(0, 1), 5.681656F, 1.0e-4F, "block output[0,1] mismatch");
    expect_near(block_output.at(1, 0), 2.373720F, 1.0e-4F, "block output[1,0] mismatch");
    expect_near(block_output.at(1, 1), 2.760564F, 1.0e-4F, "block output[1,1] mismatch");
    expect_near(block_output.at(2, 0), 3.407767F, 1.0e-4F, "block output[2,0] mismatch");
    expect_near(block_output.at(2, 1), 3.748364F, 1.0e-4F, "block output[2,1] mismatch");
    expect_near(block_output.at(3, 0), 4.456122F, 1.0e-4F, "block output[3,0] mismatch");
    expect_near(block_output.at(3, 1), 4.716015F, 1.0e-4F, "block output[3,1] mismatch");

    const auto block_logits = model.compute_logits(block_output);
    expect(block_logits.size() == 5U, "block logits size mismatch");
    expect_near(block_logits[0], 0.0F, 1.0e-4F, "block logits[0] mismatch");
    expect_near(block_logits[1], 0.971275F, 1.0e-3F, "block logits[1] mismatch");
    expect_near(block_logits[2], 1.027922F, 1.0e-3F, "block logits[2] mismatch");
    expect_near(block_logits[3], 1.999197F, 1.0e-3F, "block logits[3] mismatch");
    expect_near(block_logits[4], 0.999599F, 1.0e-3F, "block logits[4] mismatch");
    expect(model.sample_greedy(block_logits) == 3U, "fixture greedy sampling mismatch");
    expect(
        model.sample_stochastic({-1000.0F, -1000.0F, 1000.0F}, 1.0F, 20260406U) == 2U,
        "fixture stochastic sampling mismatch");

    expect_near(runtime.layers()[0].key.at(0, 0), 0.939552F, 1.0e-4F, "kv key cache first token mismatch");
    expect_near(runtime.layers()[0].value.at(0, 0), 0.939552F, 1.0e-4F, "kv value cache first token mismatch");
    expect_near(runtime.layers()[0].key.at(3, 1), -0.933124F, 1.0e-4F, "kv key cache last token mismatch");
    expect_near(runtime.layers()[0].value.at(3, 1), 1.073750F, 1.0e-4F, "kv value cache last token mismatch");

    auto full_runtime = model.create_runtime();
    const auto full_hidden = model.forward_prompt(prompt, full_runtime);
    expect(full_hidden.shape() == block_output.shape(), "full hidden shape mismatch");
    for (std::size_t index = 0; index < full_hidden.values().size(); ++index) {
        expect_near(full_hidden.values()[index], block_output.values()[index], 1.0e-5F, "full rollout mismatch");
    }

    const auto generated = model.generate(
        "hello world!",
        sllmrf::GenerationConfig {
            .max_new_tokens = 3U,
            .add_bos = true,
            .stop_at_eos = true,
            .layer_count = 0U,
        });
    expect(generated.generated_token_ids.size() == 3U, "fixture generation size mismatch");
    expect(generated.generated_token_ids[0] == 3U, "fixture generation token mismatch");
    expect(generated.generated_token_ids[1] == 3U, "fixture generation token[1] mismatch");
    expect(generated.generated_token_ids[2] == 3U, "fixture generation token[2] mismatch");
    expect(generated.generated_text == "!!!", "fixture generation text mismatch");
    expect(generated.full_text == "hello world!!!!", "fixture full text mismatch");

    const auto stochastic_generated_a = model.generate(
        "hello world!",
        sllmrf::GenerationConfig {
            .max_new_tokens = 3U,
            .add_bos = true,
            .stop_at_eos = true,
            .layer_count = 0U,
            .sampling_strategy = sllmrf::SamplingStrategy::Stochastic,
            .temperature = 1.0F,
            .seed = 7U,
            .use_seed = true,
        });
    const auto stochastic_generated_b = model.generate(
        "hello world!",
        sllmrf::GenerationConfig {
            .max_new_tokens = 3U,
            .add_bos = true,
            .stop_at_eos = true,
            .layer_count = 0U,
            .sampling_strategy = sllmrf::SamplingStrategy::Stochastic,
            .temperature = 1.0F,
            .seed = 7U,
            .use_seed = true,
        });
    expect(
        stochastic_generated_a.generated_token_ids == stochastic_generated_b.generated_token_ids,
        "fixture seeded stochastic generation should be repeatable");
}

void run_optional_model_smoke_test() {
    const auto model_path =
        std::filesystem::path(SLLMRF_PROJECT_ROOT) / "models" / "internlm2-1_8-F16.gguf";
    if (!std::filesystem::exists(model_path)) {
        std::cout << "[skip] local model not found: " << model_path << '\n';
        return;
    }

    const auto model = Internlm2Model::load(model_path);
    expect(model.gguf().tensor_count() > 0U, "real model should contain tensors");
    expect(model.config().architecture == "internlm2", "real model architecture mismatch");

    auto runtime = model.create_runtime(64U);
    const auto prompt = model.prepare_prompt("Hello world");
    expect(!prompt.token_ids.empty(), "real model tokenizer returned no tokens");
    expect(prompt.positions.size() == prompt.token_ids.size(), "real model prompt positions mismatch");

    const auto embedding = model.run_prompt_embedding(prompt, runtime);
    expect(embedding.shape().size() == 2U, "real model embedding rank mismatch");
    expect(embedding.shape()[0] == prompt.token_ids.size(), "real model embedding sequence mismatch");
    expect(embedding.shape()[1] == model.config().embedding_length, "real model embedding width mismatch");
    expect(runtime.layers().size() == model.config().block_count, "real model kv cache layer mismatch");
    expect(!model.build_prefill_plan().steps.empty(), "real model execution plan should not be empty");

    const auto generated = model.generate(
        "Hello world",
        sllmrf::GenerationConfig {
            .max_new_tokens = 5U,
            .add_bos = true,
            .stop_at_eos = true,
            .layer_count = 0U,
        });
    expect(generated.generated_token_ids.size() <= 5U, "real model generation size mismatch");
    expect(generated.generated_text == model.tokenizer().decode(generated.generated_token_ids), "real model generation decode mismatch");
    expect(generated.full_text == std::string("Hello world") + generated.generated_text, "real model full text mismatch");

    const auto stochastic_generated_a = model.generate(
        "Hello world",
        sllmrf::GenerationConfig {
            .max_new_tokens = 5U,
            .add_bos = true,
            .stop_at_eos = true,
            .layer_count = 0U,
            .sampling_strategy = sllmrf::SamplingStrategy::Stochastic,
            .temperature = 0.8F,
            .seed = 20260406U,
            .use_seed = true,
        });
    const auto stochastic_generated_b = model.generate(
        "Hello world",
        sllmrf::GenerationConfig {
            .max_new_tokens = 5U,
            .add_bos = true,
            .stop_at_eos = true,
            .layer_count = 0U,
            .sampling_strategy = sllmrf::SamplingStrategy::Stochastic,
            .temperature = 0.8F,
            .seed = 20260406U,
            .use_seed = true,
        });
    expect(
        stochastic_generated_a.generated_token_ids == stochastic_generated_b.generated_token_ids,
        "real model seeded stochastic generation should be repeatable");
    expect(
        stochastic_generated_a.generated_text ==
            model.tokenizer().decode(stochastic_generated_a.generated_token_ids),
        "real model stochastic generation decode mismatch");
}

}  // namespace

int main() {
    try {
        run_fixture_test();
        run_optional_model_smoke_test();
        std::cout << "all tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
