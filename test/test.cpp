#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
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
using sllmrf::Device;

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
    std::cout << (condition ? "[PASS] " : "[FAIL] ") << message << '\n';
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string format_value(const std::string &value) {
    return '"' + value + '"';
}

std::string format_value(const char *value) {
    return value == nullptr ? "<null>" : format_value(std::string(value));
}

std::string format_value(bool value) {
    return value ? "true" : "false";
}

std::string format_value(const Device &value) {
    return value.to_string();
}

template <typename T>
std::string format_value(const std::vector<T> &values) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0U) {
            stream << ", ";
        }
        stream << values[index];
    }
    stream << ']';
    return stream.str();
}

template <typename T>
std::string format_value(const T &value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

void print_section(const std::string &name) {
    std::cout << "\n== " << name << " ==\n";
}

template <typename Actual, typename Expected>
void expect_equal(
    const Actual &actual,
    const Expected &expected,
    const std::string &message) {
    const bool condition = actual == expected;
    std::cout << (condition ? "[PASS] " : "[FAIL] ") << message;
    if (condition) {
        std::cout << " = " << format_value(actual);
    } else {
        std::cout
            << " | expected=" << format_value(expected)
            << " actual=" << format_value(actual);
    }
    std::cout << '\n';
    if (!condition) {
        std::ostringstream stream;
        stream << message
               << " expected=" << format_value(expected)
               << " actual=" << format_value(actual);
        throw std::runtime_error(stream.str());
    }
}

void expect_near(float lhs, float rhs, float epsilon, const std::string &message) {
    const bool condition = std::fabs(lhs - rhs) <= epsilon;
    std::cout << (condition ? "[PASS] " : "[FAIL] ") << message;
    if (condition) {
        std::cout << " | actual=" << std::fixed << std::setprecision(6) << lhs;
    } else {
        std::cout
            << " | expected=" << std::fixed << std::setprecision(6) << rhs
            << " actual=" << lhs
            << " epsilon=" << epsilon;
    }
    std::cout << '\n';
    if (!condition) {
        std::ostringstream stream;
        stream << message
               << " expected=" << rhs
               << " actual=" << lhs
               << " epsilon=" << epsilon;
        throw std::runtime_error(stream.str());
    }
}

void run_fixture_test() {
    print_section("Fixture Setup");
    const auto fixture = create_fixture();
    std::cout << "fixture gguf path=" << fixture << '\n';
    const auto file = GgufFile::load(fixture);
    expect_equal(file.version(), 3U, "fixture version");
    expect_equal(file.tensor_count(), 12U, "fixture tensor count");
    expect_equal(file.metadata().size(), std::size_t {18U}, "fixture metadata count");

    print_section("Fixture Tokenizer");
    const auto tokenizer = Tokenizer::from_gguf(file);
    const auto result = tokenizer.encode("hello world!");
    std::cout << "tokenizer output token_ids=" << format_value(result.token_ids) << '\n';
    expect_equal(result.token_ids.size(), std::size_t {4U}, "token count");
    expect_equal(result.token_ids[0], 4U, "bos token id");
    expect_equal(result.token_ids[1], 1U, "first word token id");
    expect_equal(result.token_ids[2], 2U, "second word token id");
    expect_equal(result.token_ids[3], 3U, "punctuation token id");
    expect_equal(tokenizer.decode(result.token_ids), std::string("hello world!"), "decode text");

    print_section("Fixture Model Load");
    const auto model = Internlm2Model::load(fixture);
    expect_equal(model.config().architecture, std::string("internlm2"), "model architecture");
    expect_equal(model.config().block_count, 1U, "block count");
    expect_equal(model.config().embedding_length, 2U, "embedding length");
    expect_equal(model.layout().layers.size(), std::size_t {1U}, "layer layout size");

    print_section("Fixture Prompt And Embedding");
    auto runtime = model.create_runtime();
    const auto prompt = model.prepare_prompt("hello world!");
    std::cout << "prompt positions=" << format_value(prompt.positions) << '\n';
    expect_equal(prompt.positions.size(), prompt.token_ids.size(), "positions size matches token size");

    const auto embedding = model.run_prompt_embedding(prompt, runtime);
    expect_equal(embedding.shape().size(), std::size_t {2U}, "embedding rank");
    expect_equal(embedding.shape()[0], 4U, "embedding row count");
    expect_equal(embedding.shape()[1], 2U, "embedding width");
    expect_equal(runtime.consumed_tokens(), 4U, "runtime consumed tokens");
    std::cout << "embedding shape=" << embedding.shape_string() << '\n';

    const std::vector<float> expected_embedding = {4.0F, 4.5F, 1.0F, 1.5F, 2.0F, 2.5F, 3.0F, 3.5F};
    expect_near(embedding.values()[0], expected_embedding[0], 1.0e-6F, "embedding first value");
    expect_near(embedding.values().back(), expected_embedding.back(), 1.0e-6F, "embedding last value");

    print_section("Fixture Pipeline Description");
    const auto pipeline = model.describe_pipeline();
    expect(pipeline.find("execution plan nodes") != std::string::npos, "pipeline description available");

    const auto plan = model.build_prefill_plan();
    expect(!plan.steps.empty(), "execution plan is not empty");
    expect_equal(static_cast<int>(plan.steps.front().type), static_cast<int>(sllmrf::Internlm2OpType::Embedding), "first plan step type");
    expect_equal(static_cast<int>(plan.steps.back().type), static_cast<int>(sllmrf::Internlm2OpType::OutputProjection), "last plan step type");
    expect(plan.describe().find("layer_0.qkv_projection") != std::string::npos, "plan contains qkv projection");

    print_section("Fixture Operators");
    const TensorBuffer rms_input({1U, 2U}, {3.0F, 4.0F});
    const auto rms_output = sllmrf::ops::rms_norm(rms_input, {1.0F, 1.0F}, 1.0e-5F);
    expect_near(rms_output.at(0, 0), 0.848527F, 1.0e-4F, "rms_norm first value mismatch");
    expect_near(rms_output.at(0, 1), 1.131370F, 1.0e-4F, "rms_norm second value mismatch");
    const auto rms_weight_tensor = model.weights().require_tensor(model.layout().output_norm);
    const auto rms_output_from_tensor = sllmrf::ops::rms_norm(rms_input, rms_weight_tensor, 1.0e-5F);
    expect_near(rms_output_from_tensor.at(0, 0), 0.848527F, 1.0e-4F, "tensor rms_norm first value mismatch");
    expect_near(rms_output_from_tensor.at(0, 1), 1.131370F, 1.0e-4F, "tensor rms_norm second value mismatch");
    expect_equal(rms_output.device(), Device::cpu(), "rms_norm output device");
    const auto cuda_backend_is_native = sllmrf::cuda_backend_enabled();
    expect_equal(
        sllmrf::describe_device_backend(Device::cuda(0)),
        std::string(cuda_backend_is_native ? "native cuda backend" : "emulated cuda staging backend"),
        "cuda backend description");

    const TensorBuffer cpu_lhs({1U, 2U}, {1.0F, 2.0F}, Device::cpu());
    const TensorBuffer cpu_rhs({1U, 2U}, {3.0F, 4.0F}, Device::cpu());
    const auto cpu_sum = sllmrf::ops::add(cpu_lhs, cpu_rhs, sllmrf::ops::OperatorContext::cpu());
    expect_equal(cpu_sum.device(), Device::cpu(), "cpu add output device");
    expect_near(cpu_sum.at(0, 0), 4.0F, 1.0e-6F, "cpu add first value mismatch");
    expect_near(cpu_sum.at(0, 1), 6.0F, 1.0e-6F, "cpu add second value mismatch");

    bool rejected_cuda_context_mismatch = false;
    try {
        const auto unused = sllmrf::ops::add(
            cpu_lhs,
            cpu_rhs,
            sllmrf::ops::OperatorContext::cuda(0));
        (void)unused;
    } catch (const sllmrf::GgufError &) {
        rejected_cuda_context_mismatch = true;
    }
    expect(rejected_cuda_context_mismatch, "cpu tensors with cuda execution context should be rejected");

    print_section("Fixture Device Roundtrip");
    auto pseudo_cuda_tensor = cpu_lhs.copy_to(Device::cuda(0));
    expect_equal(pseudo_cuda_tensor.device(), Device::cuda(0), "cuda tensor placement");
    expect(pseudo_cuda_tensor.has_device_allocation(), "cuda tensor has device allocation");
    expect(
        pseudo_cuda_tensor.is_device_allocation_emulated() == !cuda_backend_is_native,
        "cuda tensor allocation mode is correct");
    expect(pseudo_cuda_tensor.device_data() != nullptr, "cuda tensor device pointer exists");
    expect(!pseudo_cuda_tensor.host_dirty(), "fresh cuda tensor host state is clean");
    expect(!pseudo_cuda_tensor.device_dirty(), "fresh cuda tensor device state is clean");
    auto roundtrip_cpu_tensor = pseudo_cuda_tensor.copy_to(Device::cpu());
    expect(roundtrip_cpu_tensor.device() == Device::cpu(), "roundtrip cpu tensor placement");
    expect_near(roundtrip_cpu_tensor.at(0, 0), 1.0F, 1.0e-6F, "roundtrip cpu tensor first value mismatch");
    expect_near(roundtrip_cpu_tensor.at(0, 1), 2.0F, 1.0e-6F, "roundtrip cpu tensor second value mismatch");
    pseudo_cuda_tensor.values()[0] = 9.0F;
    expect(pseudo_cuda_tensor.host_dirty(), "cuda tensor host dirty flag is set after host write");
    pseudo_cuda_tensor.sync_host_to_device();
    expect(!pseudo_cuda_tensor.host_dirty(), "cuda tensor host dirty flag clears after upload");

    auto pseudo_cuda_rhs = cpu_rhs.copy_to(Device::cuda(0));
    const auto cuda_sum = sllmrf::ops::add(
        pseudo_cuda_tensor,
        pseudo_cuda_rhs,
        sllmrf::ops::OperatorContext::cuda(0));
    expect_equal(cuda_sum.device(), Device::cuda(0), "cuda add output placement");
    auto cuda_sum_cpu = cuda_sum.copy_to(Device::cpu());
    expect_near(cuda_sum_cpu.at(0, 0), 12.0F, 1.0e-6F, "cuda add first value mismatch");
    expect_near(cuda_sum_cpu.at(0, 1), 6.0F, 1.0e-6F, "cuda add second value mismatch");

    const auto cuda_silu = sllmrf::ops::silu(
        pseudo_cuda_tensor,
        sllmrf::ops::OperatorContext::cuda(0));
    expect_equal(cuda_silu.device(), Device::cuda(0), "cuda silu output placement");
    auto cuda_silu_cpu = cuda_silu.copy_to(Device::cpu());
    expect_near(cuda_silu_cpu.at(0, 0), 8.998890F, 1.0e-4F, "cuda silu first value mismatch");
    expect_near(cuda_silu_cpu.at(0, 1), 1.761594F, 1.0e-4F, "cuda silu second value mismatch");

    auto cuda_inplace = pseudo_cuda_tensor.copy_to(Device::cuda(0));
    sllmrf::ops::add_inplace(
        cuda_inplace,
        pseudo_cuda_rhs,
        sllmrf::ops::OperatorContext::cuda(0));
    auto cuda_inplace_cpu = cuda_inplace.copy_to(Device::cpu());
    expect_near(cuda_inplace_cpu.at(0, 0), 12.0F, 1.0e-6F, "cuda add_inplace first value mismatch");
    expect_near(cuda_inplace_cpu.at(0, 1), 6.0F, 1.0e-6F, "cuda add_inplace second value mismatch");

    bool rejected_device_mismatch = false;
    try {
        const auto unused = sllmrf::ops::silu(pseudo_cuda_tensor);
        (void)unused;
    } catch (const sllmrf::GgufError &) {
        rejected_device_mismatch = true;
    }
    expect(rejected_device_mismatch, "cuda tensor is rejected by cpu backend");

    print_section("Fixture Logits");
    const auto normalized = model.apply_final_norm(embedding);
    expect_equal(normalized.shape(), embedding.shape(), "final norm shape");
    std::cout << "logits preview uses last token hidden state\n";

    const auto logits = model.compute_logits(embedding);
    expect_equal(logits.size(), std::size_t {5U}, "fixture logits size");
    expect_near(logits[0], 0.0F, 1.0e-4F, "logits[0] mismatch");
    expect_near(logits[1], 0.920575F, 1.0e-3F, "logits[1] mismatch");
    expect_near(logits[2], 1.073971F, 1.0e-3F, "logits[2] mismatch");
    expect_near(logits[3], 1.994546F, 1.0e-3F, "logits[3] mismatch");
    expect_near(logits[4], 0.997273F, 1.0e-3F, "logits[4] mismatch");

    print_section("Fixture Decoder Block");
    const auto block_output = model.forward_layer(embedding, 0U, runtime);
    expect_equal(block_output.shape(), embedding.shape(), "block output shape");
    expect_near(block_output.at(0, 0), 5.500523F, 1.0e-4F, "block output[0,0] mismatch");
    expect_near(block_output.at(0, 1), 5.681656F, 1.0e-4F, "block output[0,1] mismatch");
    expect_near(block_output.at(1, 0), 2.373720F, 1.0e-4F, "block output[1,0] mismatch");
    expect_near(block_output.at(1, 1), 2.760564F, 1.0e-4F, "block output[1,1] mismatch");
    expect_near(block_output.at(2, 0), 3.407767F, 1.0e-4F, "block output[2,0] mismatch");
    expect_near(block_output.at(2, 1), 3.748364F, 1.0e-4F, "block output[2,1] mismatch");
    expect_near(block_output.at(3, 0), 4.456122F, 1.0e-4F, "block output[3,0] mismatch");
    expect_near(block_output.at(3, 1), 4.716015F, 1.0e-4F, "block output[3,1] mismatch");

    const auto block_logits = model.compute_logits(block_output);
    expect_equal(block_logits.size(), std::size_t {5U}, "block logits size");
    expect_near(block_logits[0], 0.0F, 1.0e-4F, "block logits[0] mismatch");
    expect_near(block_logits[1], 0.971275F, 1.0e-3F, "block logits[1] mismatch");
    expect_near(block_logits[2], 1.027922F, 1.0e-3F, "block logits[2] mismatch");
    expect_near(block_logits[3], 1.999197F, 1.0e-3F, "block logits[3] mismatch");
    expect_near(block_logits[4], 0.999599F, 1.0e-3F, "block logits[4] mismatch");
    expect_equal(model.sample_greedy(block_logits), 3U, "fixture greedy sampling");
    expect_equal(
        model.sample_stochastic({-1000.0F, -1000.0F, 1000.0F}, 1.0F, 20260406U),
        2U,
        "fixture stochastic sampling");

    print_section("Fixture KV Cache");
    expect_near(runtime.layers()[0].key.at(0, 0), 0.939552F, 1.0e-4F, "kv key cache first token mismatch");
    expect_near(runtime.layers()[0].value.at(0, 0), 0.939552F, 1.0e-4F, "kv value cache first token mismatch");
    expect_near(runtime.layers()[0].key.at(3, 1), -0.933124F, 1.0e-4F, "kv key cache last token mismatch");
    expect_near(runtime.layers()[0].value.at(3, 1), 1.073750F, 1.0e-4F, "kv value cache last token mismatch");

    auto full_runtime = model.create_runtime();
    const auto full_hidden = model.forward_prompt(prompt, full_runtime);
    expect_equal(full_hidden.shape(), block_output.shape(), "full hidden shape");
    expect_near(full_hidden.values().front(), block_output.values().front(), 1.0e-5F, "full rollout first value");
    expect_near(full_hidden.values().back(), block_output.values().back(), 1.0e-5F, "full rollout last value");

    print_section("Fixture CUDA Consistency");
    auto cuda_runtime = model.create_runtime(0U, sllmrf::ops::OperatorContext::cuda(0));
    const auto cuda_embedding = model.run_prompt_embedding(prompt, cuda_runtime);
    expect_equal(cuda_embedding.device(), Device::cuda(0), "cuda embedding placement");
    auto cuda_embedding_cpu = cuda_embedding.copy_to(Device::cpu());
    for (std::size_t index = 0; index < cuda_embedding_cpu.values().size(); ++index) {
        expect_near(
            cuda_embedding_cpu.values()[index],
            embedding.values()[index],
            1.0e-6F,
            "cuda embedding mismatch");
    }
    const auto cuda_block_output = model.forward_layer(cuda_embedding, 0U, cuda_runtime);
    expect_equal(cuda_block_output.device(), Device::cuda(0), "cuda block output placement");
    auto cuda_block_output_cpu = cuda_block_output.copy_to(Device::cpu());
    for (std::size_t index = 0; index < cuda_block_output_cpu.values().size(); ++index) {
        expect_near(
            cuda_block_output_cpu.values()[index],
            block_output.values()[index],
            1.0e-5F,
            "cuda block output mismatch");
    }
    const auto cuda_block_logits = model.compute_logits(cuda_block_output);
    expect_equal(cuda_block_logits.size(), block_logits.size(), "cuda block logits size");
    expect_near(cuda_block_logits.front(), block_logits.front(), 1.0e-5F, "cuda block logits first value");
    expect_near(cuda_block_logits.back(), block_logits.back(), 1.0e-5F, "cuda block logits last value");

    print_section("Fixture Generation");
    const auto generated = model.generate(
        "hello world!",
        sllmrf::GenerationConfig {
            .max_new_tokens = 3U,
            .add_bos = true,
            .stop_at_eos = true,
            .layer_count = 0U,
        });
    std::cout
        << "generated token_ids=" << format_value(generated.generated_token_ids)
        << " generated_text=" << format_value(generated.generated_text)
        << '\n';
    expect_equal(generated.generated_token_ids.size(), std::size_t {3U}, "fixture generation size");
    expect_equal(generated.generated_token_ids[0], 3U, "fixture generation token[0]");
    expect_equal(generated.generated_token_ids[1], 3U, "fixture generation token[1]");
    expect_equal(generated.generated_token_ids[2], 3U, "fixture generation token[2]");
    expect_equal(generated.generated_text, std::string("!!!"), "fixture generated text");
    expect_equal(generated.full_text, std::string("hello world!!!!"), "fixture full text");

    const auto stochastic_generated_a = model.generate(
        "hello world!",
        sllmrf::GenerationConfig {
            .max_new_tokens = 3U,
            .add_bos = true,
            .stop_at_eos = true,
            .layer_count = 0U,
            .execution_context = sllmrf::ops::OperatorContext::cpu(),
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
            .execution_context = sllmrf::ops::OperatorContext::cpu(),
            .sampling_strategy = sllmrf::SamplingStrategy::Stochastic,
            .temperature = 1.0F,
            .seed = 7U,
            .use_seed = true,
        });
    expect(
        stochastic_generated_a.generated_token_ids == stochastic_generated_b.generated_token_ids,
        "fixture seeded stochastic generation should be repeatable");

    const auto cuda_generated = model.generate(
        "hello world!",
        sllmrf::GenerationConfig {
            .max_new_tokens = 3U,
            .add_bos = true,
            .stop_at_eos = true,
            .layer_count = 0U,
            .execution_context = sllmrf::ops::OperatorContext::cuda(0),
        });
    expect_equal(cuda_generated.generated_token_ids, generated.generated_token_ids, "cuda generation token ids");
    expect_equal(cuda_generated.generated_text, generated.generated_text, "cuda generation text");
}

void run_optional_model_smoke_test() {
    print_section("Real Model Smoke Test");
    const auto model_path =
        std::filesystem::path(SLLMRF_PROJECT_ROOT) / "models" / "internlm2-1_8-F16.gguf";
    if (!std::filesystem::exists(model_path)) {
        std::cout << "[skip] local model not found: " << model_path << '\n';
        return;
    }

    const auto model = Internlm2Model::load(model_path);
    expect(model.gguf().tensor_count() > 0U, "real model should contain tensors");
    expect_equal(model.config().architecture, std::string("internlm2"), "real model architecture");

    auto runtime = model.create_runtime(64U);
    const auto prompt = model.prepare_prompt("Hello world");
    expect(!prompt.token_ids.empty(), "real model tokenizer returned tokens");
    expect(prompt.positions.size() == prompt.token_ids.size(), "real model prompt positions are aligned");

    const auto embedding = model.run_prompt_embedding(prompt, runtime);
    expect_equal(embedding.shape().size(), std::size_t {2U}, "real model embedding rank");
    expect_equal(embedding.shape()[0], static_cast<uint64_t>(prompt.token_ids.size()), "real model embedding sequence");
    expect_equal(embedding.shape()[1], static_cast<uint64_t>(model.config().embedding_length), "real model embedding width");
    expect_equal(runtime.layers().size(), static_cast<std::size_t>(model.config().block_count), "real model kv cache layer count");
    expect(!model.build_prefill_plan().steps.empty(), "real model execution plan is not empty");

    const auto generated = model.generate(
        "Hello world",
        sllmrf::GenerationConfig {
            .max_new_tokens = 5U,
            .add_bos = true,
            .stop_at_eos = true,
            .layer_count = 0U,
            .execution_context = sllmrf::ops::OperatorContext::cpu(),
        });
    expect(generated.generated_token_ids.size() <= 5U, "real model generation length within limit");
    std::cout << "real model generated text=" << format_value(generated.generated_text) << '\n';
    expect_equal(generated.generated_text, model.tokenizer().decode(generated.generated_token_ids), "real model generation decode");
    expect_equal(generated.full_text, std::string("Hello world") + generated.generated_text, "real model full text");

    const auto stochastic_generated_a = model.generate(
        "Hello world",
        sllmrf::GenerationConfig {
            .max_new_tokens = 5U,
            .add_bos = true,
            .stop_at_eos = true,
            .layer_count = 0U,
            .execution_context = sllmrf::ops::OperatorContext::cpu(),
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
            .execution_context = sllmrf::ops::OperatorContext::cpu(),
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
        "real model stochastic generation decode");

    auto cuda_runtime = model.create_runtime(64U, sllmrf::ops::OperatorContext::cuda(0));
    const auto cuda_prompt_hidden = model.forward_prompt(prompt, cuda_runtime);
    expect_equal(cuda_prompt_hidden.device(), Device::cuda(0), "real model cuda rollout placement");
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
