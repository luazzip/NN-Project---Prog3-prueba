#pragma once

#include "../algebra/tensor_ops.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace utec::tf::io {

    struct Dataset {
        Tensor<float> inputs;
        Tensor<float> targets;
    };

    class DatasetLoader {
    public:
        static Dataset from_csv(const std::string& inputs_path,
                                const std::string& targets_path,
                                const Shape& sample_shape) {
            std::vector<std::vector<float>> input_rows = read_numeric_rows(inputs_path);
            std::vector<std::vector<float>> target_rows = read_numeric_rows(targets_path);

            size_t values_per_sample = sample_shape.total_size();
            validate_input_rows(input_rows, values_per_sample);
            validate_target_rows(target_rows);
            validate_matching_sample_count(input_rows, target_rows);

            Dataset dataset;
            dataset.inputs = build_inputs_tensor(input_rows, sample_shape);
            dataset.targets = build_targets_tensor(target_rows);
            return dataset;
        }

    private:
        static std::vector<std::vector<float>> read_numeric_rows(const std::string& path) {
            std::ifstream file(path);
            if (!file.is_open()) {
                throw std::invalid_argument("no se pudo abrir el archivo: " + path);
            }
            std::vector<std::vector<float>> rows;
            std::string line;
            while (std::getline(file, line)) {
                if (line_is_blank(line)) {
                    continue;
                }
                rows.push_back(parse_numeric_line(line));
            }
            return rows;
        }

        static bool line_is_blank(const std::string& line) {
            for (char character : line) {
                bool is_whitespace = character == ' ' || character == '\t' ||
                                     character == '\r' || character == '\n';
                if (!is_whitespace) {
                    return false;
                }
            }
            return true;
        }

        static std::vector<float> parse_numeric_line(const std::string& line) {
            std::vector<float> values;
            std::stringstream line_stream(line);
            std::string cell;
            while (std::getline(line_stream, cell, ',')) {
                values.push_back(parse_single_value(cell));
            }
            return values;
        }

        static float parse_single_value(const std::string& cell) {
            try {
                return std::stof(cell);
            } catch (const std::exception&) {
                throw std::invalid_argument("el archivo contiene un valor no numerico");
            }
        }

        static void validate_input_rows(const std::vector<std::vector<float>>& input_rows,
                                        size_t values_per_sample) {
            if (input_rows.empty()) {
                throw std::invalid_argument("el archivo de inputs no tiene filas");
            }
            for (const std::vector<float>& row : input_rows) {
                if (row.size() != values_per_sample) {
                    throw std::invalid_argument(
                        "una fila de inputs no coincide con el shape indicado");
                }
            }
        }

        static void validate_target_rows(const std::vector<std::vector<float>>& target_rows) {
            if (target_rows.empty()) {
                throw std::invalid_argument("el archivo de etiquetas no tiene filas");
            }
            size_t columns_in_first_row = target_rows.front().size();
            for (const std::vector<float>& row : target_rows) {
                if (row.size() != columns_in_first_row) {
                    throw std::invalid_argument(
                        "las filas de etiquetas tienen distinta cantidad de columnas");
                }
            }
        }

        static void validate_matching_sample_count(
                const std::vector<std::vector<float>>& input_rows,
                const std::vector<std::vector<float>>& target_rows) {
            if (input_rows.size() != target_rows.size()) {
                throw std::invalid_argument(
                    "inputs y etiquetas tienen distinta cantidad de muestras");
            }
        }

        static Tensor<float> build_inputs_tensor(
                const std::vector<std::vector<float>>& input_rows,
                const Shape& sample_shape) {
            size_t sample_count = input_rows.size();
            size_t values_per_sample = sample_shape.total_size();
            Shape batch_shape = prepend_batch_dimension(sample_shape, sample_count);

            Tensor<float> inputs(batch_shape);
            size_t flat_position = 0;
            for (const std::vector<float>& row : input_rows) {
                for (size_t value_index = 0; value_index < values_per_sample; ++value_index) {
                    inputs.flat(flat_position) = row[value_index];
                    ++flat_position;
                }
            }
            return inputs;
        }

        static Tensor<float> build_targets_tensor(
                const std::vector<std::vector<float>>& target_rows) {
            size_t sample_count = target_rows.size();
            size_t class_count = target_rows.front().size();

            Tensor<float> targets(Shape{sample_count, class_count});
            size_t flat_position = 0;
            for (const std::vector<float>& row : target_rows) {
                for (size_t class_index = 0; class_index < class_count; ++class_index) {
                    targets.flat(flat_position) = row[class_index];
                    ++flat_position;
                }
            }
            return targets;
        }

        static Shape prepend_batch_dimension(const Shape& sample_shape, size_t sample_count) {
            std::vector<size_t> dimensions;
            dimensions.push_back(sample_count);
            for (size_t axis = 0; axis < sample_shape.rank(); ++axis) {
                dimensions.push_back(sample_shape[axis]);
            }
            return Shape(dimensions);
        }
    };

}

using namespace utec::tf;
