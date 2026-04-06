#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "sllmrf/gguf.h"

namespace sllmrf {

enum class VocabularyTokenType : int32_t {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
};

struct TokenizerConfig {
    bool add_space_prefix {false};
    bool add_bos_token {false};
    bool add_eos_token {false};
    uint32_t bos_token_id {0};
    uint32_t eos_token_id {0};
    uint32_t unknown_token_id {0};
    std::optional<uint32_t> padding_token_id;
};

struct TokenPiece {
    uint32_t id {0};
    std::string text;
    float score {0.0F};
    int32_t type {static_cast<int32_t>(VocabularyTokenType::Normal)};
};

struct TokenizationResult {
    std::vector<uint32_t> token_ids;
    std::vector<std::string> token_texts;
};

enum class TokenizerStrategy {
    UnigramScores,
    GreedyLongestMatch,
};

class Tokenizer {
public:
    [[nodiscard]] static Tokenizer from_gguf(const GgufFile &file);

    [[nodiscard]] const std::string &model_family() const noexcept;
    [[nodiscard]] const TokenizerConfig &config() const noexcept;
    [[nodiscard]] const TokenPiece &token(uint32_t id) const;
    [[nodiscard]] std::size_t vocab_size() const noexcept;

    [[nodiscard]] TokenizationResult encode(
        std::string_view text,
        bool add_bos = true,
        bool add_eos = false) const;
    [[nodiscard]] std::string decode(
        const std::vector<uint32_t> &token_ids,
        bool skip_special_tokens = true) const;

private:
    [[nodiscard]] TokenizationResult encode_unigram_scores(
        std::string_view text,
        bool add_bos,
        bool add_eos) const;
    [[nodiscard]] TokenizationResult encode_greedy_longest_match(
        std::string_view text,
        bool add_bos,
        bool add_eos) const;
    [[nodiscard]] std::string normalize(std::string_view text) const;
    [[nodiscard]] std::optional<uint32_t> find_longest_token(
        const std::string &normalized,
        std::size_t position) const;
    [[nodiscard]] std::optional<uint32_t> find_byte_fallback(unsigned char byte) const noexcept;
    [[nodiscard]] bool is_byte_token(const TokenPiece &piece) const noexcept;
    [[nodiscard]] bool is_encodable(const TokenPiece &piece) const noexcept;

    std::string model_family_;
    TokenizerStrategy strategy_ {TokenizerStrategy::UnigramScores};
    TokenizerConfig config_;
    std::vector<TokenPiece> vocab_;
    std::unordered_map<unsigned char, std::vector<uint32_t>> tokens_by_first_byte_;
    std::vector<std::optional<uint32_t>> byte_fallback_ids_;
};

}  // namespace sllmrf
