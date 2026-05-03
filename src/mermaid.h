#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>

namespace sllmrf::mermaid {

[[nodiscard]] std::string compact_node_id(std::string_view name);
[[nodiscard]] std::string escape(std::string_view value);
[[nodiscard]] std::string node_label(std::string_view title, std::string_view detail);
void write_class_defs(std::ostringstream &stream);

[[nodiscard]] std::string gguf_loading_module(
    std::string_view graph_name,
    std::string_view module_input,
    std::string_view module_output);
[[nodiscard]] std::string prompt_tokenizer_module(
    std::string_view graph_name,
    std::string_view module_input,
    std::string_view module_output);
[[nodiscard]] std::string runtime_tensor_module(
    std::string_view graph_name,
    std::string_view module_input,
    std::string_view module_output,
    uint32_t block_count,
    uint32_t embedding_length,
    uint32_t attention_head_count_kv,
    uint32_t head_dimension);

}  // namespace sllmrf::mermaid
