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
private:
    static std::vector<float> parse_csv_row(const std::string& line) {
        std::vector<float> values;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            if (cell.empty()) {
                throw std::invalid_argument("celda vacia en csv");
            }
            values.push_back(std::stof(cell));
        }

        if (values.empty()) {
            throw std::invalid_argument("fila vacia en csv");
        }
        return values;
    }

    static std::vector<std::vector<float>> read_csv(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::invalid_argument("no se pudo abrir csv: " + path);
        }

        std::vector<std::vector<float>> rows;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                rows.push_back(parse_csv_row(line));
            }
        }

        if (rows.empty()) {
            throw std::invalid_argument("csv vacio: " + path);
        }
        return rows;
    }

public:
    static Dataset from_csv(const std::string& x_path,
                            const std::string& y_path,
                            const Shape& sample_shape) {
        auto x_rows = read_csv(x_path);
        auto y_rows = read_csv(y_path);

        if (x_rows.size() != y_rows.size()) {
            throw std::invalid_argument("cantidad de inputs y labels incompatible");
        }

        const size_t sample_size = sample_shape.total_size();
        const size_t num_samples = x_rows.size();
        const size_t num_classes = y_rows.front().size();

        if (num_classes == 0) {
            throw std::invalid_argument("labels sin clases");
        }

        for (const auto& row : x_rows) {
            if (row.size() != sample_size) {
                throw std::invalid_argument("fila de input con tamano incorrecto");
            }
        }
        for (const auto& row : y_rows) {
            if (row.size() != num_classes) {
                throw std::invalid_argument("etiquetas mal formadas");
            }
        }

        std::vector<size_t> input_dims;
        input_dims.push_back(num_samples);
        for (size_t i = 0; i < sample_shape.rank(); ++i) {
            input_dims.push_back(sample_shape[i]);
        }

        Tensor<float> inputs{Shape(input_dims)};
        Tensor<float> targets(Shape{num_samples, num_classes});

        for (size_t n = 0; n < num_samples; ++n) {
            for (size_t i = 0; i < sample_size; ++i) {
                inputs.flat(n * sample_size + i) = x_rows[n][i];
            }
            for (size_t c = 0; c < num_classes; ++c) {
                targets(n, c) = y_rows[n][c];
            }
        }

        return Dataset{inputs, targets};
    }
};

}
