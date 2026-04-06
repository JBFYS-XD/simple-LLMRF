#include "sllmrf/gguf.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sllmrf {

namespace {

constexpr uint32_t kGgufMagic = 0x46554747U;
constexpr uint32_t kDefaultAlignment = 32U;

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path &path) {
        const auto path_string = path.string();
        fd_ = ::open(path_string.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw GgufError("failed to open GGUF file: " + path_string);
        }

        struct stat stat_buffer {};
        if (::fstat(fd_, &stat_buffer) != 0) {
            ::close(fd_);
            throw GgufError("failed to stat GGUF file: " + path_string);
        }

        size_ = static_cast<std::size_t>(stat_buffer.st_size);
        if (size_ == 0U) {
            ::close(fd_);
            throw GgufError("GGUF file is empty: " + path_string);
        }

        mapping_ = static_cast<const uint8_t *>(
            ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (mapping_ == MAP_FAILED) {
            ::close(fd_);
            throw GgufError("failed to mmap GGUF file: " + path_string);
        }
    }

    ~MappedFile() {
        if (mapping_ != nullptr && mapping_ != MAP_FAILED) {
            ::munmap(const_cast<uint8_t *>(mapping_), size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;

    [[nodiscard]] const uint8_t *data() const noexcept {
        return mapping_;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return size_;
    }

private:
    int fd_ {-1};
    std::size_t size_ {0};
    const uint8_t *mapping_ {nullptr};
};

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
        exponent = exponent + (127U - 15U);
        result = sign | (exponent << 23U) | (mantissa << 13U);
    }

    return std::bit_cast<float>(result);
}

class BinaryReader {
public:
    BinaryReader(const uint8_t *begin, std::size_t size) : begin_(begin), current_(begin), end_(begin + size) {}

    template <typename T>
    [[nodiscard]] T read_pod(std::string_view label) {
        require(sizeof(T), label);
        T value {};
        std::memcpy(&value, current_, sizeof(T));
        current_ += sizeof(T);
        return value;
    }

    [[nodiscard]] std::string read_string(std::string_view label) {
        const auto length = read_pod<uint64_t>(label);
        require(static_cast<std::size_t>(length), label);

        std::string value(reinterpret_cast<const char *>(current_), static_cast<std::size_t>(length));
        current_ += static_cast<std::ptrdiff_t>(length);
        return value;
    }

    [[nodiscard]] MetadataValue read_metadata_value(MetadataValueType type) {
        switch (type) {
            case MetadataValueType::UInt8:
                return MetadataValue(type, read_pod<uint8_t>("uint8 metadata value"));
            case MetadataValueType::Int8:
                return MetadataValue(type, read_pod<int8_t>("int8 metadata value"));
            case MetadataValueType::UInt16:
                return MetadataValue(type, read_pod<uint16_t>("uint16 metadata value"));
            case MetadataValueType::Int16:
                return MetadataValue(type, read_pod<int16_t>("int16 metadata value"));
            case MetadataValueType::UInt32:
                return MetadataValue(type, read_pod<uint32_t>("uint32 metadata value"));
            case MetadataValueType::Int32:
                return MetadataValue(type, read_pod<int32_t>("int32 metadata value"));
            case MetadataValueType::Float32:
                return MetadataValue(type, read_pod<float>("float32 metadata value"));
            case MetadataValueType::Bool: {
                const auto raw = read_pod<uint8_t>("bool metadata value");
                if (raw > 1U) {
                    throw GgufError("metadata bool value must be 0 or 1");
                }
                return MetadataValue(type, raw == 1U);
            }
            case MetadataValueType::String:
                return MetadataValue(type, read_string("string metadata value"));
            case MetadataValueType::Array: {
                const auto element_type = static_cast<MetadataValueType>(read_pod<uint32_t>("metadata array type"));
                const auto length = read_pod<uint64_t>("metadata array length");
                MetadataValue::Array values;
                values.reserve(static_cast<std::size_t>(length));
                for (uint64_t index = 0; index < length; ++index) {
                    values.push_back(read_metadata_value(element_type));
                }
                return MetadataValue(type, std::move(values));
            }
            case MetadataValueType::UInt64:
                return MetadataValue(type, read_pod<uint64_t>("uint64 metadata value"));
            case MetadataValueType::Int64:
                return MetadataValue(type, read_pod<int64_t>("int64 metadata value"));
            case MetadataValueType::Float64:
                return MetadataValue(type, read_pod<double>("float64 metadata value"));
        }

        throw GgufError("unsupported metadata value type");
    }

    void align_to(uint32_t alignment) {
        if (alignment == 0U) {
            throw GgufError("GGUF alignment must be greater than zero");
        }

        const auto offset = position();
        const auto aligned = (offset + alignment - 1U) / alignment * alignment;
        if (aligned > offset) {
            require(aligned - offset, "alignment padding");
            current_ += static_cast<std::ptrdiff_t>(aligned - offset);
        }
    }

    [[nodiscard]] std::size_t position() const noexcept {
        return static_cast<std::size_t>(current_ - begin_);
    }

private:
    void require(std::size_t bytes, std::string_view label) const {
        if (current_ + static_cast<std::ptrdiff_t>(bytes) > end_) {
            throw GgufError("unexpected end of file while reading " + std::string(label));
        }
    }

    const uint8_t *begin_ {nullptr};
    const uint8_t *current_ {nullptr};
    const uint8_t *end_ {nullptr};
};

uint32_t read_alignment(const std::vector<MetadataEntry> &metadata) {
    for (const auto &entry : metadata) {
        if (entry.key != "general.alignment") {
            continue;
        }

        if (entry.value.is<uint32_t>()) {
            return entry.value.as<uint32_t>();
        }
        if (entry.value.is<uint64_t>()) {
            return static_cast<uint32_t>(entry.value.as<uint64_t>());
        }
        throw GgufError("general.alignment must be uint32 or uint64");
    }

    return kDefaultAlignment;
}

}  // namespace

MetadataValue::MetadataValue(MetadataValueType type, Storage storage)
    : type_(type), storage_(std::move(storage)) {}

MetadataValueType MetadataValue::type() const noexcept {
    return type_;
}

const MetadataValue::Storage &MetadataValue::storage() const noexcept {
    return storage_;
}

std::string MetadataValue::debug_string(std::size_t max_items) const {
    return std::visit(
        [max_items](const auto &value) -> std::string {
            using ValueType = std::decay_t<decltype(value)>;

            if constexpr (std::is_same_v<ValueType, std::monostate>) {
                return "<empty>";
            } else if constexpr (std::is_same_v<ValueType, bool>) {
                return value ? "true" : "false";
            } else if constexpr (std::is_same_v<ValueType, std::string>) {
                constexpr std::size_t kPreviewLength = 40U;
                if (value.size() <= kPreviewLength) {
                    return "\"" + value + "\"";
                }
                return "\"" + value.substr(0, kPreviewLength) + "...\"";
            } else if constexpr (std::is_same_v<ValueType, Array>) {
                std::ostringstream stream;
                stream << "array[" << value.size() << "](";
                const auto count = std::min(value.size(), max_items);
                for (std::size_t index = 0; index < count; ++index) {
                    if (index > 0U) {
                        stream << ", ";
                    }
                    stream << value[index].debug_string(3);
                }
                if (value.size() > count) {
                    stream << ", ...";
                }
                stream << ')';
                return stream.str();
            } else {
                std::ostringstream stream;
                stream << value;
                return stream.str();
            }
        },
        storage_);
}

uint64_t TensorInfo::element_count() const noexcept {
    uint64_t count = 1U;
    for (uint64_t dimension : dimensions) {
        count *= dimension;
    }
    return count;
}

uint64_t TensorView::row_width() const noexcept {
    return dimensions.empty() ? 0U : dimensions.front();
}

uint64_t TensorView::row_count() const noexcept {
    if (dimensions.size() <= 1U) {
        return dimensions.empty() ? 0U : 1U;
    }

    uint64_t count = 1U;
    for (std::size_t index = 1; index < dimensions.size(); ++index) {
        count *= dimensions[index];
    }
    return count;
}

GgufFile GgufFile::load(const std::filesystem::path &path) {
    const MappedFile mapped_file(path);
    BinaryReader reader(mapped_file.data(), mapped_file.size());

    const auto magic = reader.read_pod<uint32_t>("magic");
    if (magic != kGgufMagic) {
        throw GgufError("invalid GGUF magic number");
    }

    const auto version = reader.read_pod<uint32_t>("version");
    if (version < 2U || version > 3U) {
        throw GgufError("unsupported GGUF version: " + std::to_string(version));
    }

    const auto tensor_count = reader.read_pod<uint64_t>("tensor count");
    const auto metadata_count = reader.read_pod<uint64_t>("metadata count");

    std::vector<MetadataEntry> metadata;
    metadata.reserve(static_cast<std::size_t>(metadata_count));
    for (uint64_t index = 0; index < metadata_count; ++index) {
        const auto key = reader.read_string("metadata key");
        const auto type = static_cast<MetadataValueType>(reader.read_pod<uint32_t>("metadata type"));
        metadata.push_back({key, reader.read_metadata_value(type)});
    }

    const auto alignment = read_alignment(metadata);

    std::vector<TensorInfo> tensors;
    tensors.reserve(static_cast<std::size_t>(tensor_count));
    for (uint64_t index = 0; index < tensor_count; ++index) {
        TensorInfo info;
        info.name = reader.read_string("tensor name");

        const auto dimension_count = reader.read_pod<uint32_t>("tensor rank");
        info.dimensions.reserve(dimension_count);
        for (uint32_t dimension_index = 0; dimension_index < dimension_count; ++dimension_index) {
            info.dimensions.push_back(reader.read_pod<uint64_t>("tensor dimension"));
        }

        info.type = static_cast<GgmlType>(reader.read_pod<uint32_t>("tensor type"));
        info.offset = reader.read_pod<uint64_t>("tensor offset");
        tensors.push_back(std::move(info));
    }

    if (reader.position() < mapped_file.size()) {
        reader.align_to(alignment);
    }

    return GgufFile(
        path,
        version,
        alignment,
        static_cast<uint64_t>(reader.position()),
        std::move(metadata),
        std::move(tensors));
}

const std::filesystem::path &GgufFile::path() const noexcept {
    return path_;
}

uint32_t GgufFile::version() const noexcept {
    return version_;
}

uint32_t GgufFile::alignment() const noexcept {
    return alignment_;
}

uint64_t GgufFile::tensor_count() const noexcept {
    return static_cast<uint64_t>(tensors_.size());
}

uint64_t GgufFile::tensor_data_offset() const noexcept {
    return tensor_data_offset_;
}

const std::vector<MetadataEntry> &GgufFile::metadata() const noexcept {
    return metadata_;
}

const std::vector<TensorInfo> &GgufFile::tensors() const noexcept {
    return tensors_;
}

const MetadataEntry *GgufFile::find_metadata(std::string_view key) const noexcept {
    const auto iter = std::find_if(
        metadata_.begin(),
        metadata_.end(),
        [key](const MetadataEntry &entry) { return entry.key == key; });
    return iter == metadata_.end() ? nullptr : &(*iter);
}

std::string GgufFile::describe(std::size_t preview_count) const {
    std::ostringstream stream;
    stream << "GGUF file: " << path_ << '\n';
    stream << "version: " << version_ << '\n';
    stream << "alignment: " << alignment_ << '\n';
    stream << "metadata entries: " << metadata_.size() << '\n';
    stream << "tensors: " << tensors_.size() << '\n';
    stream << "tensor data offset: " << tensor_data_offset_ << '\n';

    const auto count = std::min(tensors_.size(), preview_count);
    if (count > 0U) {
        stream << "tensor preview:\n";
        for (std::size_t index = 0; index < count; ++index) {
            const auto &tensor = tensors_[index];
            stream << "  - " << tensor.name << " [";
            for (std::size_t dim = 0; dim < tensor.dimensions.size(); ++dim) {
                if (dim > 0U) {
                    stream << " x ";
                }
                stream << tensor.dimensions[dim];
            }
            stream << "] type=" << to_string(tensor.type) << " offset=" << tensor.offset << '\n';
        }
    }

    return stream.str();
}

GgufFile::GgufFile(
    std::filesystem::path path,
    uint32_t version,
    uint32_t alignment,
    uint64_t tensor_data_offset,
    std::vector<MetadataEntry> metadata,
    std::vector<TensorInfo> tensors)
    : path_(std::move(path)),
      version_(version),
      alignment_(alignment),
      tensor_data_offset_(tensor_data_offset),
      metadata_(std::move(metadata)),
      tensors_(std::move(tensors)) {}

std::string to_string(MetadataValueType type) {
    switch (type) {
        case MetadataValueType::UInt8:
            return "uint8";
        case MetadataValueType::Int8:
            return "int8";
        case MetadataValueType::UInt16:
            return "uint16";
        case MetadataValueType::Int16:
            return "int16";
        case MetadataValueType::UInt32:
            return "uint32";
        case MetadataValueType::Int32:
            return "int32";
        case MetadataValueType::Float32:
            return "float32";
        case MetadataValueType::Bool:
            return "bool";
        case MetadataValueType::String:
            return "string";
        case MetadataValueType::Array:
            return "array";
        case MetadataValueType::UInt64:
            return "uint64";
        case MetadataValueType::Int64:
            return "int64";
        case MetadataValueType::Float64:
            return "float64";
    }

    return "unknown";
}

std::string to_string(GgmlType type) {
    switch (type) {
        case GgmlType::F32:
            return "F32";
        case GgmlType::F16:
            return "F16";
        case GgmlType::Q4_0:
            return "Q4_0";
        case GgmlType::Q4_1:
            return "Q4_1";
        case GgmlType::Q5_0:
            return "Q5_0";
        case GgmlType::Q5_1:
            return "Q5_1";
        case GgmlType::Q8_0:
            return "Q8_0";
        case GgmlType::Q8_1:
            return "Q8_1";
        case GgmlType::Q2_K:
            return "Q2_K";
        case GgmlType::Q3_K:
            return "Q3_K";
        case GgmlType::Q4_K:
            return "Q4_K";
        case GgmlType::Q5_K:
            return "Q5_K";
        case GgmlType::Q6_K:
            return "Q6_K";
        case GgmlType::Q8_K:
            return "Q8_K";
        case GgmlType::IQ2_XXS:
            return "IQ2_XXS";
        case GgmlType::IQ2_XS:
            return "IQ2_XS";
        case GgmlType::IQ3_XXS:
            return "IQ3_XXS";
        case GgmlType::IQ1_S:
            return "IQ1_S";
        case GgmlType::IQ4_NL:
            return "IQ4_NL";
        case GgmlType::IQ3_S:
            return "IQ3_S";
        case GgmlType::IQ2_S:
            return "IQ2_S";
        case GgmlType::IQ4_XS:
            return "IQ4_XS";
        case GgmlType::I8:
            return "I8";
        case GgmlType::I16:
            return "I16";
        case GgmlType::I32:
            return "I32";
        case GgmlType::I64:
            return "I64";
        case GgmlType::F64:
            return "F64";
        case GgmlType::IQ1_M:
            return "IQ1_M";
        case GgmlType::BF16:
            return "BF16";
        case GgmlType::TQ1_0:
            return "TQ1_0";
        case GgmlType::TQ2_0:
            return "TQ2_0";
        case GgmlType::MXFP4:
            return "MXFP4";
        case GgmlType::Count:
            return "Count";
    }

    return "unknown";
}

std::size_t ggml_type_size(GgmlType type) {
    switch (type) {
        case GgmlType::F16:
            return 2U;
        case GgmlType::F32:
            return 4U;
        default:
            throw GgufError("ggml type size is not implemented for " + to_string(type));
    }
}

struct GgufTensorReader::Impl {
    explicit Impl(GgufFile file)
        : file(std::move(file)),
          mapped_file(std::make_shared<MappedFile>(this->file.path())) {
        tensor_data_base = mapped_file->data() + this->file.tensor_data_offset();
        tensor_lookup.reserve(this->file.tensors().size());
        for (const auto &tensor : this->file.tensors()) {
            tensor_lookup.emplace(tensor.name, &tensor);
        }
    }

    [[nodiscard]] TensorView make_view(std::string_view name) const {
        const auto *info = require_info(name);
        const auto byte_size = static_cast<std::size_t>(info->element_count()) * ggml_type_size(info->type);
        return TensorView {
            .name = info->name,
            .dimensions = info->dimensions,
            .type = info->type,
            .offset = info->offset,
            .byte_size = byte_size,
            .data = tensor_data_base + info->offset,
        };
    }

    [[nodiscard]] const TensorInfo *find_info(std::string_view name) const noexcept {
        const auto iter = tensor_lookup.find(std::string(name));
        return iter == tensor_lookup.end() ? nullptr : iter->second;
    }

    [[nodiscard]] const TensorInfo *require_info(std::string_view name) const {
        const auto *info = find_info(name);
        if (info == nullptr) {
            throw GgufError("tensor not found: " + std::string(name));
        }
        return info;
    }

    GgufFile file;
    std::shared_ptr<MappedFile> mapped_file;
    const uint8_t *tensor_data_base {nullptr};
    std::unordered_map<std::string, const TensorInfo *> tensor_lookup;
};

GgufTensorReader GgufTensorReader::open(const GgufFile &file) {
    return GgufTensorReader(std::make_shared<Impl>(file));
}

GgufTensorReader GgufTensorReader::open(const std::filesystem::path &path) {
    return GgufTensorReader(std::make_shared<Impl>(GgufFile::load(path)));
}

GgufTensorReader::GgufTensorReader(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

const GgufFile &GgufTensorReader::file() const {
    if (!impl_) {
        throw GgufError("GGUF tensor reader is not initialized");
    }
    return impl_->file;
}

bool GgufTensorReader::has_tensor(std::string_view name) const noexcept {
    return impl_ != nullptr && impl_->find_info(name) != nullptr;
}

const TensorInfo &GgufTensorReader::require_info(std::string_view name) const {
    if (!impl_) {
        throw GgufError("GGUF tensor reader is not initialized");
    }
    return *impl_->require_info(name);
}

TensorView GgufTensorReader::require_tensor(std::string_view name) const {
    if (!impl_) {
        throw GgufError("GGUF tensor reader is not initialized");
    }
    return impl_->make_view(name);
}

std::vector<float> GgufTensorReader::read_tensor_f32(std::string_view name) const {
    const auto view = require_tensor(name);
    const auto element_count =
        static_cast<std::size_t>(std::accumulate(
            view.dimensions.begin(),
            view.dimensions.end(),
            uint64_t {1},
            [](uint64_t lhs, uint64_t rhs) { return lhs * rhs; }));

    std::vector<float> values(element_count, 0.0F);
    if (view.type == GgmlType::F32) {
        std::memcpy(values.data(), view.data, view.byte_size);
        return values;
    }
    if (view.type == GgmlType::F16) {
        const auto *source = reinterpret_cast<const uint16_t *>(view.data);
        for (std::size_t index = 0; index < element_count; ++index) {
            values[index] = half_to_float(source[index]);
        }
        return values;
    }

    throw GgufError("tensor conversion to float32 is not implemented for type " + to_string(view.type));
}

std::vector<float> GgufTensorReader::read_rows_f32(
    std::string_view name,
    const std::vector<uint32_t> &row_ids) const {
    const auto view = require_tensor(name);
    if (view.dimensions.size() != 2U) {
        throw GgufError("row reads require a 2D tensor: " + std::string(name));
    }

    const auto width = static_cast<std::size_t>(view.row_width());
    const auto total_rows = view.row_count();
    const auto element_size = ggml_type_size(view.type);
    const auto row_bytes = width * element_size;

    std::vector<float> values(row_ids.size() * width, 0.0F);
    for (std::size_t row_index = 0; row_index < row_ids.size(); ++row_index) {
        const auto token_row = static_cast<uint64_t>(row_ids[row_index]);
        if (token_row >= total_rows) {
            throw GgufError("row index exceeds tensor shape for " + std::string(name));
        }

        const auto *row_ptr = view.data + token_row * row_bytes;
        auto *destination = values.data() + row_index * width;

        if (view.type == GgmlType::F32) {
            std::memcpy(destination, row_ptr, row_bytes);
            continue;
        }
        if (view.type == GgmlType::F16) {
            const auto *source = reinterpret_cast<const uint16_t *>(row_ptr);
            for (std::size_t col = 0; col < width; ++col) {
                destination[col] = half_to_float(source[col]);
            }
            continue;
        }

        throw GgufError("row read is not implemented for tensor type " + to_string(view.type));
    }

    return values;
}

}  // namespace sllmrf
