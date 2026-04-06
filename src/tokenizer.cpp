#include "sllmrf/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace sllmrf {

namespace {

constexpr char kSpaceMarker[] = "\xE2\x96\x81";

const MetadataValue &require_metadata(const GgufFile &file, std::string_view key) {
    const auto *entry = file.find_metadata(key);
    if (entry == nullptr) {
        throw GgufError("missing required tokenizer metadata: " + std::string(key));
    }
    return entry->value;
}

const MetadataValue *find_metadata(const GgufFile &file, std::string_view key) {
    const auto *entry = file.find_metadata(key);
    return entry == nullptr ? nullptr : &entry->value;
}

const MetadataValue::Array &as_array(const MetadataValue &value, std::string_view key) {
    if (!value.is<MetadataValue::Array>()) {
        throw GgufError("metadata is not an array: " + std::string(key));
    }
    return value.as<MetadataValue::Array>();
}

const std::string &as_string(const MetadataValue &value, std::string_view key) {
    if (!value.is<std::string>()) {
        throw GgufError("metadata is not a string: " + std::string(key));
    }
    return value.as<std::string>();
}

bool as_bool(const MetadataValue &value, std::string_view key) {
    if (!value.is<bool>()) {
        throw GgufError("metadata is not a bool: " + std::string(key));
    }
    return value.as<bool>();
}

uint32_t as_u32(const MetadataValue &value, std::string_view key) {
    if (value.is<uint32_t>()) {
        return value.as<uint32_t>();
    }
    if (value.is<uint64_t>()) {
        return static_cast<uint32_t>(value.as<uint64_t>());
    }
    throw GgufError("metadata is not an unsigned integer: " + std::string(key));
}

int32_t as_i32(const MetadataValue &value, std::string_view key) {
    if (!value.is<int32_t>()) {
        throw GgufError("metadata is not an int32: " + std::string(key));
    }
    return value.as<int32_t>();
}

float as_f32(const MetadataValue &value, std::string_view key) {
    if (!value.is<float>()) {
        throw GgufError("metadata is not a float32: " + std::string(key));
    }
    return value.as<float>();
}

std::optional<uint32_t> find_optional_id(const GgufFile &file, std::initializer_list<std::string_view> keys) {
    for (const auto key : keys) {
        const auto *value = find_metadata(file, key);
        if (value != nullptr) {
            return as_u32(*value, key);
        }
    }
    return std::nullopt;
}

}  // namespace

Tokenizer Tokenizer::from_gguf(const GgufFile &file) {
    Tokenizer tokenizer;

    tokenizer.model_family_ = as_string(
        require_metadata(file, "tokenizer.ggml.model"),
        "tokenizer.ggml.model");
    tokenizer.strategy_ = tokenizer.model_family_ == "llama"
        ? TokenizerStrategy::GreedyLongestMatch
        : TokenizerStrategy::UnigramScores;
    tokenizer.byte_fallback_ids_.assign(256U, std::nullopt);

    const auto &tokens = as_array(
        require_metadata(file, "tokenizer.ggml.tokens"),
        "tokenizer.ggml.tokens");
    const auto *scores_metadata = find_metadata(file, "tokenizer.ggml.scores");
    const auto *types_metadata = find_metadata(file, "tokenizer.ggml.token_type");

    const MetadataValue::Array *scores = nullptr;
    if (scores_metadata != nullptr) {
        scores = &as_array(*scores_metadata, "tokenizer.ggml.scores");
    }

    const MetadataValue::Array *token_types = nullptr;
    if (types_metadata != nullptr) {
        token_types = &as_array(*types_metadata, "tokenizer.ggml.token_type");
    }

    if (scores != nullptr && scores->size() != tokens.size()) {
        throw GgufError("tokenizer.ggml.scores size does not match tokenizer.ggml.tokens");
    }
    if (token_types != nullptr && token_types->size() != tokens.size()) {
        throw GgufError("tokenizer.ggml.token_type size does not match tokenizer.ggml.tokens");
    }

    tokenizer.config_.add_space_prefix = find_metadata(file, "tokenizer.ggml.add_space_prefix") != nullptr
        ? as_bool(*find_metadata(file, "tokenizer.ggml.add_space_prefix"), "tokenizer.ggml.add_space_prefix")
        : false;
    tokenizer.config_.add_bos_token = find_metadata(file, "tokenizer.ggml.add_bos_token") != nullptr
        ? as_bool(*find_metadata(file, "tokenizer.ggml.add_bos_token"), "tokenizer.ggml.add_bos_token")
        : false;
    tokenizer.config_.add_eos_token = find_metadata(file, "tokenizer.ggml.add_eos_token") != nullptr
        ? as_bool(*find_metadata(file, "tokenizer.ggml.add_eos_token"), "tokenizer.ggml.add_eos_token")
        : false;
    tokenizer.config_.bos_token_id =
        find_optional_id(file, {"tokenizer.ggml.bos_token_id"}).value_or(0U);
    tokenizer.config_.eos_token_id =
        find_optional_id(file, {"tokenizer.ggml.eos_token_id"}).value_or(0U);
    tokenizer.config_.unknown_token_id =
        find_optional_id(
            file,
            {
                "tokenizer.ggml.unknown_token_id",
                "tokenizer.ggml.unk_token_id",
                "tokenizer.ggml.unkown_token_id",
            })
            .value_or(0U);
    tokenizer.config_.padding_token_id =
        find_optional_id(file, {"tokenizer.ggml.padding_token_id"});

    tokenizer.vocab_.reserve(tokens.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const auto key = "tokenizer.ggml.tokens";
        TokenPiece piece;
        piece.id = static_cast<uint32_t>(index);
        piece.text = as_string(tokens[index], key);
        piece.score = scores == nullptr ? 0.0F : as_f32(scores->at(index), "tokenizer.ggml.scores");
        piece.type = token_types == nullptr
            ? static_cast<int32_t>(VocabularyTokenType::Normal)
            : as_i32(token_types->at(index), "tokenizer.ggml.token_type");

        if (!piece.text.empty()) {
            const auto first_byte = static_cast<unsigned char>(piece.text.front());
            tokenizer.tokens_by_first_byte_[first_byte].push_back(piece.id);
        }
        if (piece.type == static_cast<int32_t>(VocabularyTokenType::Byte) && piece.text.size() == 6U &&
            piece.text.rfind("<0x", 0U) == 0U && piece.text.back() == '>') {
            unsigned int raw = 0U;
            if (std::sscanf(piece.text.c_str(), "<0x%02X>", &raw) == 1U) {
                tokenizer.byte_fallback_ids_[raw] = piece.id;
            }
        }
        tokenizer.vocab_.push_back(std::move(piece));
    }

    for (auto &[first_byte, ids] : tokenizer.tokens_by_first_byte_) {
        (void)first_byte;
        std::sort(
            ids.begin(),
            ids.end(),
            [&tokenizer](uint32_t left, uint32_t right) {
                const auto &lhs = tokenizer.vocab_[left];
                const auto &rhs = tokenizer.vocab_[right];
                if (lhs.text.size() != rhs.text.size()) {
                    return lhs.text.size() > rhs.text.size();
                }
                return lhs.score > rhs.score;
            });
    }

    return tokenizer;
}

const std::string &Tokenizer::model_family() const noexcept {
    return model_family_;
}

const TokenizerConfig &Tokenizer::config() const noexcept {
    return config_;
}

const TokenPiece &Tokenizer::token(uint32_t id) const {
    if (id >= vocab_.size()) {
        throw std::out_of_range("token id out of range");
    }
    return vocab_[id];
}

std::size_t Tokenizer::vocab_size() const noexcept {
    return vocab_.size();
}

TokenizationResult Tokenizer::encode(std::string_view text, bool add_bos, bool add_eos) const {
    switch (strategy_) {
        case TokenizerStrategy::UnigramScores:
            return encode_unigram_scores(text, add_bos, add_eos);
        case TokenizerStrategy::GreedyLongestMatch:
            return encode_greedy_longest_match(text, add_bos, add_eos);
    }

    throw GgufError("unsupported tokenizer strategy");
}

TokenizationResult Tokenizer::encode_unigram_scores(
    std::string_view text,
    bool add_bos,
    bool add_eos) const {
    TokenizationResult result;

    if (add_bos && config_.bos_token_id < vocab_.size()) {
        result.token_ids.push_back(config_.bos_token_id);
    }

    const auto normalized = normalize(text);
    if (!normalized.empty()) {
        constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
        constexpr float kUnknownPenalty = 1000.0F;

        const auto unknown_id = config_.unknown_token_id < vocab_.size() ? config_.unknown_token_id : 0U;
        std::vector<float> best(normalized.size() + 1U, kNegativeInfinity);
        std::vector<int64_t> prev(normalized.size() + 1U, -1);
        std::vector<uint32_t> chosen(normalized.size() + 1U, unknown_id);

        best[0] = 0.0F;

        for (std::size_t position = 0; position < normalized.size(); ++position) {
            if (!std::isfinite(best[position])) {
                continue;
            }

            const auto first_byte = static_cast<unsigned char>(normalized[position]);
            const auto bucket = tokens_by_first_byte_.find(first_byte);
            if (bucket != tokens_by_first_byte_.end()) {
                for (uint32_t token_id : bucket->second) {
                    const auto &piece = vocab_[token_id];
                    if (!is_encodable(piece)) {
                        continue;
                    }
                    if (position + piece.text.size() > normalized.size()) {
                        continue;
                    }
                    if (normalized.compare(position, piece.text.size(), piece.text) != 0) {
                        continue;
                    }

                    const auto next = position + piece.text.size();
                    const auto candidate = best[position] + piece.score;
                    const auto current_span = prev[next] < 0 ? 0U : next - static_cast<std::size_t>(prev[next]);
                    if (candidate > best[next] + 1e-6F ||
                        (std::fabs(candidate - best[next]) <= 1e-6F && piece.text.size() > current_span)) {
                        best[next] = candidate;
                        prev[next] = static_cast<int64_t>(position);
                        chosen[next] = token_id;
                    }
                }
            }

            if (position + 1U <= normalized.size() && unknown_id < vocab_.size()) {
                const auto next = position + 1U;
                const auto candidate = best[position] - kUnknownPenalty;
                if (candidate > best[next]) {
                    best[next] = candidate;
                    prev[next] = static_cast<int64_t>(position);
                    chosen[next] = unknown_id;
                }
            }
        }

        if (prev[normalized.size()] < 0 && normalized.size() > 0U) {
            throw GgufError("failed to encode prompt with current tokenizer");
        }

        std::vector<uint32_t> encoded;
        for (std::size_t cursor = normalized.size(); cursor > 0U;) {
            encoded.push_back(chosen[cursor]);
            cursor = static_cast<std::size_t>(prev[cursor]);
        }
        std::reverse(encoded.begin(), encoded.end());
        result.token_ids.insert(result.token_ids.end(), encoded.begin(), encoded.end());
    }

    if (add_eos && config_.eos_token_id < vocab_.size()) {
        result.token_ids.push_back(config_.eos_token_id);
    }

    result.token_texts.reserve(result.token_ids.size());
    for (uint32_t token_id : result.token_ids) {
        result.token_texts.push_back(token(token_id).text);
    }

    return result;
}

TokenizationResult Tokenizer::encode_greedy_longest_match(
    std::string_view text,
    bool add_bos,
    bool add_eos) const {
    TokenizationResult result;

    if (add_bos && config_.bos_token_id < vocab_.size()) {
        result.token_ids.push_back(config_.bos_token_id);
    }

    const auto normalized = normalize(text);
    for (std::size_t position = 0; position < normalized.size();) {
        const auto token_id = find_longest_token(normalized, position);
        if (token_id.has_value()) {
            result.token_ids.push_back(*token_id);
            position += token(*token_id).text.size();
            continue;
        }

        const auto byte = static_cast<unsigned char>(normalized[position]);
        const auto byte_fallback = find_byte_fallback(byte);
        if (byte_fallback.has_value()) {
            result.token_ids.push_back(*byte_fallback);
            ++position;
            continue;
        }

        result.token_ids.push_back(config_.unknown_token_id);
        ++position;
    }

    if (add_eos && config_.eos_token_id < vocab_.size()) {
        result.token_ids.push_back(config_.eos_token_id);
    }

    result.token_texts.reserve(result.token_ids.size());
    for (uint32_t token_id : result.token_ids) {
        result.token_texts.push_back(token(token_id).text);
    }
    return result;
}

std::string Tokenizer::decode(const std::vector<uint32_t> &token_ids, bool skip_special_tokens) const {
    std::string normalized;
    for (uint32_t token_id : token_ids) {
        const auto &piece = token(token_id);
        if (skip_special_tokens && !is_encodable(piece)) {
            continue;
        }
        if (is_byte_token(piece)) {
            unsigned int raw = 0U;
            if (std::sscanf(piece.text.c_str(), "<0x%02X>", &raw) == 1U) {
                normalized.push_back(static_cast<char>(raw));
                continue;
            }
        }
        normalized += piece.text;
    }

    std::string decoded;
    decoded.reserve(normalized.size());
    for (std::size_t index = 0; index < normalized.size();) {
        if (normalized.compare(index, sizeof(kSpaceMarker) - 1U, kSpaceMarker) == 0) {
            decoded.push_back(' ');
            index += sizeof(kSpaceMarker) - 1U;
        } else {
            decoded.push_back(normalized[index]);
            ++index;
        }
    }

    if (config_.add_space_prefix && !decoded.empty() && decoded.front() == ' ') {
        decoded.erase(decoded.begin());
    }

    return decoded;
}

std::string Tokenizer::normalize(std::string_view text) const {
    std::string normalized(text);
    if (config_.add_space_prefix && !normalized.empty() && normalized.front() != ' ') {
        normalized.insert(normalized.begin(), ' ');
    }

    std::string with_markers;
    with_markers.reserve(normalized.size() * 3U);
    for (char ch : normalized) {
        if (ch == ' ') {
            with_markers += kSpaceMarker;
        } else {
            with_markers.push_back(ch);
        }
    }

    return with_markers;
}

std::optional<uint32_t> Tokenizer::find_longest_token(
    const std::string &normalized,
    std::size_t position) const {
    const auto bucket = tokens_by_first_byte_.find(static_cast<unsigned char>(normalized[position]));
    if (bucket == tokens_by_first_byte_.end()) {
        return std::nullopt;
    }

    for (uint32_t token_id : bucket->second) {
        const auto &piece = vocab_[token_id];
        if (!is_encodable(piece) || is_byte_token(piece)) {
            continue;
        }
        if (position + piece.text.size() > normalized.size()) {
            continue;
        }
        if (normalized.compare(position, piece.text.size(), piece.text) == 0) {
            return token_id;
        }
    }

    return std::nullopt;
}

std::optional<uint32_t> Tokenizer::find_byte_fallback(unsigned char byte) const noexcept {
    if (byte >= byte_fallback_ids_.size()) {
        return std::nullopt;
    }
    return byte_fallback_ids_[byte];
}

bool Tokenizer::is_byte_token(const TokenPiece &piece) const noexcept {
    return piece.type == static_cast<int32_t>(VocabularyTokenType::Byte);
}

bool Tokenizer::is_encodable(const TokenPiece &piece) const noexcept {
    if (piece.text.empty()) {
        return false;
    }

    switch (static_cast<VocabularyTokenType>(piece.type)) {
        case VocabularyTokenType::Normal:
        case VocabularyTokenType::UserDefined:
        case VocabularyTokenType::Byte:
            return true;
        case VocabularyTokenType::Unknown:
        case VocabularyTokenType::Control:
        case VocabularyTokenType::Unused:
            return false;
    }

    return true;
}

}  // namespace sllmrf
