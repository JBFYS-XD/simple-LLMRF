#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace sllmrf {

enum class GgmlType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    Count = 40,
};

enum class MetadataValueType : uint32_t {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
};

class GgufError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class MetadataValue {
public:
    using Array = std::vector<MetadataValue>;
    using Storage = std::variant<
        std::monostate,
        uint8_t,
        int8_t,
        uint16_t,
        int16_t,
        uint32_t,
        int32_t,
        float,
        bool,
        std::string,
        uint64_t,
        int64_t,
        double,
        Array>;

    MetadataValue() = default;
    MetadataValue(MetadataValueType type, Storage storage);

    [[nodiscard]] MetadataValueType type() const noexcept;
    [[nodiscard]] const Storage &storage() const noexcept;

    template <typename T>
    [[nodiscard]] bool is() const {
        return std::holds_alternative<T>(storage_);
    }

    template <typename T>
    [[nodiscard]] const T &as() const {
        return std::get<T>(storage_);
    }

    [[nodiscard]] std::string debug_string(std::size_t max_items = 8) const;

private:
    MetadataValueType type_ {MetadataValueType::UInt8};
    Storage storage_ {std::monostate {}};
};

struct MetadataEntry {
    std::string key;
    MetadataValue value;
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions;
    GgmlType type {GgmlType::F32};
    uint64_t offset {0};

    [[nodiscard]] uint64_t element_count() const noexcept;
};

struct TensorView {
    std::string name;
    std::vector<uint64_t> dimensions;
    GgmlType type {GgmlType::F32};
    uint64_t offset {0};
    std::size_t byte_size {0};
    const uint8_t *data {nullptr};

    [[nodiscard]] uint64_t row_width() const noexcept;
    [[nodiscard]] uint64_t row_count() const noexcept;
};

class GgufFile {
public:
    [[nodiscard]] static GgufFile load(const std::filesystem::path &path);

    [[nodiscard]] const std::filesystem::path &path() const noexcept;
    [[nodiscard]] uint32_t version() const noexcept;
    [[nodiscard]] uint32_t alignment() const noexcept;
    [[nodiscard]] uint64_t tensor_count() const noexcept;
    [[nodiscard]] uint64_t tensor_data_offset() const noexcept;
    [[nodiscard]] const std::vector<MetadataEntry> &metadata() const noexcept;
    [[nodiscard]] const std::vector<TensorInfo> &tensors() const noexcept;
    [[nodiscard]] const MetadataEntry *find_metadata(std::string_view key) const noexcept;
    [[nodiscard]] std::string describe(std::size_t preview_count = 6) const;

private:
    GgufFile(
        std::filesystem::path path,
        uint32_t version,
        uint32_t alignment,
        uint64_t tensor_data_offset,
        std::vector<MetadataEntry> metadata,
        std::vector<TensorInfo> tensors);

    std::filesystem::path path_;
    uint32_t version_ {0};
    uint32_t alignment_ {32};
    uint64_t tensor_data_offset_ {0};
    std::vector<MetadataEntry> metadata_;
    std::vector<TensorInfo> tensors_;
};

class GgufTensorReader {
public:
    [[nodiscard]] static GgufTensorReader open(const GgufFile &file);
    [[nodiscard]] static GgufTensorReader open(const std::filesystem::path &path);

    GgufTensorReader() = default;

    [[nodiscard]] const GgufFile &file() const;
    [[nodiscard]] bool has_tensor(std::string_view name) const noexcept;
    [[nodiscard]] const TensorInfo &require_info(std::string_view name) const;
    [[nodiscard]] TensorView require_tensor(std::string_view name) const;
    [[nodiscard]] std::vector<float> read_tensor_f32(std::string_view name) const;
    [[nodiscard]] std::vector<float> read_rows_f32(
        std::string_view name,
        const std::vector<uint32_t> &row_ids) const;

private:
    struct Impl;

    explicit GgufTensorReader(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;
};

[[nodiscard]] std::string to_string(MetadataValueType type);
[[nodiscard]] std::string to_string(GgmlType type);
[[nodiscard]] std::size_t ggml_type_size(GgmlType type);

}  // namespace sllmrf
