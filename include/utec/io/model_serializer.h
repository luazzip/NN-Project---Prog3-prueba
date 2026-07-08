#pragma once

#include "../algebra/tensor_ops.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace utec::tf {

    struct SaveOptions {
        std::unordered_map<std::string, std::string> metadata;
    };

}

namespace utec::tf::io {

    inline constexpr char model_file_magic[4] = {'U', 'T', 'E', 'C'};
    inline constexpr uint32_t model_file_version = 1;

    class ModelWriter {
    public:
        explicit ModelWriter(const std::string& path)
            : file(path, std::ios::binary) {
            if (!file.is_open()) {
                throw std::runtime_error("no se pudo crear el archivo del modelo: " + path);
            }
        }

        void write_magic_and_version() {
            file.write(model_file_magic, sizeof(model_file_magic));
            write_uint32(model_file_version);
        }

        void write_uint32(uint32_t value) {
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }

        void write_string(const std::string& text) {
            write_uint32(static_cast<uint32_t>(text.size()));
            file.write(text.data(), static_cast<std::streamsize>(text.size()));
        }

        void write_shape(const Shape& shape) {
            write_uint32(static_cast<uint32_t>(shape.rank()));
            for (size_t axis = 0; axis < shape.rank(); ++axis) {
                write_uint32(static_cast<uint32_t>(shape[axis]));
            }
        }

        void write_tensor(const Tensor<float>& tensor) {
            write_shape(tensor.shape());
            for (size_t position = 0; position < tensor.size(); ++position) {
                float value = tensor.flat(position);
                file.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }

    private:
        std::ofstream file;
    };

    class ModelReader {
    public:
        explicit ModelReader(const std::string& path)
            : file(path, std::ios::binary) {
            if (!file.is_open()) {
                throw std::runtime_error("no se pudo abrir el archivo del modelo: " + path);
            }
        }

        void check_magic_and_version() {
            char magic[sizeof(model_file_magic)];
            read_raw_bytes(magic, sizeof(magic));
            for (size_t i = 0; i < sizeof(model_file_magic); ++i) {
                if (magic[i] != model_file_magic[i]) {
                    throw std::runtime_error("el archivo del modelo esta corrupto");
                }
            }
            uint32_t version = read_uint32();
            if (version != model_file_version) {
                throw std::runtime_error("version del archivo del modelo no soportada");
            }
        }

        uint32_t read_uint32() {
            uint32_t value = 0;
            read_raw_bytes(reinterpret_cast<char*>(&value), sizeof(value));
            return value;
        }

        std::string read_string() {
            uint32_t length = read_uint32();
            std::string text(length, '\0');
            if (length > 0) {
                read_raw_bytes(text.data(), length);
            }
            return text;
        }

        Shape read_shape() {
            uint32_t rank = read_uint32();
            std::vector<size_t> dimensions;
            for (uint32_t axis = 0; axis < rank; ++axis) {
                dimensions.push_back(static_cast<size_t>(read_uint32()));
            }
            return Shape(dimensions);
        }

        Tensor<float> read_tensor() {
            Shape shape = read_shape();
            Tensor<float> tensor(shape);
            for (size_t position = 0; position < tensor.size(); ++position) {
                float value = 0.0f;
                read_raw_bytes(reinterpret_cast<char*>(&value), sizeof(value));
                tensor.flat(position) = value;
            }
            return tensor;
        }

    private:
        void read_raw_bytes(char* destination, size_t byte_count) {
            file.read(destination, static_cast<std::streamsize>(byte_count));
            if (static_cast<size_t>(file.gcount()) != byte_count) {
                throw std::runtime_error("el archivo del modelo esta incompleto o corrupto");
            }
        }

        std::ifstream file;
    };

}

using namespace utec::tf;
