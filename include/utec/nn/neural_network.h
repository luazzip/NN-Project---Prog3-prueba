#pragma once

#include "../algebra/tensor_ops.h"
#include "nn_interfaces.h"
#include "nn_activation.h"
#include "nn_ops.h"
#include "nn_dense.h"
#include "nn_convolution.h"
#include "nn_pooling.h"
#include "nn_flatten.h"
#include "nn_loss.h"
#include "nn_optimizer.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace utec::tf {

struct FitOptions {
    int epochs = 1;
    int batch_size = 1;
};

struct SaveOptions {
    std::unordered_map<std::string, std::string> metadata;
};

namespace layers {
    struct Input {
        utec::Shape shape;
        explicit Input(const utec::Shape& s) : shape(s) {}
    };
}

struct EvalResult {
    float loss = 0.0f;
};

class Sequential {
public:
    struct History {
        std::vector<float> loss;
    };

    template <typename L>
    void add(L layer) {
        using D = std::decay_t<L>;
        if constexpr (std::is_same_v<D, layers::Input>) {
            input_shape_ = layer.shape;
            current_shape_ = layer.shape;
            has_input_ = true;
        } else {
            if (!has_input_) {
                throw std::logic_error("debe agregar Input antes de las capas");
            }
            auto ptr = std::make_unique<D>(std::move(layer));
            ptr->build(current_shape_);
            std::string base = ptr->type_name();
            std::string name = base + "_" + std::to_string(type_counts_[base]++);
            current_shape_ = ptr->output_shape();
            names_.push_back(std::move(name));
            layers_.push_back(std::move(ptr));
        }
    }

    void compile(optimizers::SGD optimizer,
                 losses::CategoricalCrossentropy) {
        optimizer_ = optimizer;
        compiled_ = true;
    }

    Tensor<float> predict(const Tensor<float>& x) {
        validate_input(x);

        Tensor<float> a = x;
        for (auto& layer : layers_) {
            a = layer->forward(a);
        }
        return a;
    }

    void backward() {
        if (!compiled_) {
            throw std::logic_error("backward sobre un modelo no compilado");
        }
    }

    Tensor<float> backward(const Tensor<float>& grad) {
        if (!compiled_) {
            throw std::logic_error("backward sobre un modelo no compilado");
        }

        Tensor<float> g = grad;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            g = (*it)->backward(g);
        }

        collect_last_gradients();
        return g;
    }

    EvalResult evaluate(const Tensor<float>& x, const Tensor<float>& y) {
        Tensor<float> pred = predict(x);
        validate_labels(pred, y);
        return EvalResult{losses::CategoricalCrossentropy::loss(pred, y)};
    }

    History fit(const Tensor<float>& x, const Tensor<float>& y,
                FitOptions options) {
        if (!compiled_) {
            throw std::logic_error("fit sobre un modelo no compilado");
        }
        if (options.epochs <= 0 || options.batch_size <= 0) {
            throw std::invalid_argument("epochs y batch_size deben ser > 0");
        }
        if (x.shape().rank() < 2 || x.shape()[0] == 0) {
            throw std::invalid_argument("el dataset no puede estar vacio");
        }

        {
            Tensor<float> pred = predict(x);
            validate_labels(pred, y);
        }

        const size_t n = x.shape()[0];
        const size_t per_x = x.size() / n;
        const size_t per_y = y.size() / n;
        const size_t batch_size = static_cast<size_t>(options.batch_size);

        History history;
        for (int epoch = 0; epoch < options.epochs; ++epoch) {
            float epoch_loss = 0.0f;
            size_t batches = 0;

            for (size_t start = 0; start < n; start += batch_size) {
                const size_t b = std::min(batch_size, n - start);
                Tensor<float> xb = slice_rows(x, start, b, per_x);
                Tensor<float> yb = slice_rows(y, start, b, per_y);

                Tensor<float> pred = predict(xb);
                epoch_loss += losses::CategoricalCrossentropy::loss(pred, yb);
                ++batches;

                Tensor<float> grad = losses::CategoricalCrossentropy::grad(pred, yb);
                for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                    grad = (*it)->backward(grad);
                }
                for (auto& layer : layers_) {
                    for (auto& pr : layer->trainable_parameters()) {
                        optimizer_->update(*pr.first, *pr.second);
                    }
                }
            }

            history.loss.push_back(epoch_loss / static_cast<float>(batches));
        }

        collect_last_gradients();
        return history;
    }

    [[nodiscard]] std::unordered_map<std::string, Tensor<float>> parameters() {
        std::unordered_map<std::string, Tensor<float>> out;
        for (size_t i = 0; i < layers_.size(); ++i) {
            for (auto& kv : layers_[i]->parameters()) {
                out[names_[i] + "/" + kv.first] = kv.second;
            }
        }
        return out;
    }

    [[nodiscard]] std::unordered_map<std::string, Tensor<float>> last_gradients() const {
        return last_gradients_;
    }

    void save(const std::string& path,
              SaveOptions options = SaveOptions{}) const {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("no se pudo crear el archivo del modelo");
        }

        write_string(file, "UTEC_NN_V1");
        write_shape(file, input_shape_);

        write_u32(file, static_cast<uint32_t>(options.metadata.size()));
        for (const auto& kv : options.metadata) {
            write_string(file, kv.first);
            write_string(file, kv.second);
        }

        write_u32(file, static_cast<uint32_t>(layers_.size()));
        for (const auto& layer : layers_) {
            const std::string type = layer->type_name();
            write_string(file, type);

            auto params = layer->parameters();
            write_u32(file, static_cast<uint32_t>(params.size()));
            for (const auto& kv : params) {
                write_string(file, kv.first);
                write_tensor(file, kv.second);
            }
        }
    }

    static Sequential load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("no se pudo abrir el archivo del modelo");
        }

        if (read_string(file) != "UTEC_NN_V1") {
            throw std::runtime_error("archivo de modelo corrupto");
        }

        Sequential model;
        Shape input_shape = read_shape(file);
        model.add(layers::Input(input_shape));

        uint32_t metadata_count = read_u32(file);
        for (uint32_t i = 0; i < metadata_count; ++i) {
            std::string key = read_string(file);
            std::string value = read_string(file);
            model.metadata_[key] = value;
        }

        uint32_t layer_count = read_u32(file);
        for (uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
            std::string type = read_string(file);
            uint32_t param_count = read_u32(file);
            std::unordered_map<std::string, Tensor<float>> params;
            for (uint32_t p = 0; p < param_count; ++p) {
                std::string name = read_string(file);
                params[name] = read_tensor(file);
            }

            if (type == "conv2d") {
                const auto& weights = params.at("weights");
                model.add(layers::Conv2D(
                    static_cast<int>(weights.shape()[3]),
                    {weights.shape()[0], weights.shape()[1]},
                    Activation::Relu
                ));
                assign_parameters(*model.layers_.back(), params);
            } else if (type == "maxpooling2d") {
                model.add(layers::MaxPooling2D({2, 2}));
            } else if (type == "flatten") {
                model.add(layers::Flatten());
            } else if (type == "dense") {
                const auto& weights = params.at("weights");
                model.add(layers::Dense(static_cast<int>(weights.shape()[1]), Activation::Softmax));
                assign_parameters(*model.layers_.back(), params);
            } else {
                throw std::runtime_error("tipo de capa desconocido al cargar el modelo");
            }
        }

        model.compile(optimizers::SGD(0.01f), losses::CategoricalCrossentropy{});
        return model;
    }

    [[nodiscard]] const std::unordered_map<std::string, std::string>& metadata() const {
        return metadata_;
    }

private:
    void validate_input(const Tensor<float>& x) const {
        if (!has_input_) {
            throw std::logic_error("modelo sin capa Input");
        }
        if (x.shape().rank() != input_shape_.rank() + 1) {
            throw std::invalid_argument("input incompatible con la forma esperada");
        }
        for (size_t i = 0; i < input_shape_.rank(); ++i) {
            if (x.shape()[i + 1] != input_shape_[i]) {
                throw std::invalid_argument("input incompatible con la forma esperada");
            }
        }
    }

    void validate_labels(const Tensor<float>& pred,
                         const Tensor<float>& y) const {
        if (!(pred.shape() == y.shape())) {
            throw std::invalid_argument("y_true incompatible con la salida del modelo");
        }
    }

    void collect_last_gradients() {
        last_gradients_.clear();
        for (size_t i = 0; i < layers_.size(); ++i) {
            for (auto& kv : layers_[i]->gradients()) {
                last_gradients_[names_[i] + "/" + kv.first] = kv.second;
            }
        }
    }

    static Tensor<float> slice_rows(const Tensor<float>& t,
                                    size_t start,
                                    size_t batch,
                                    size_t per_row) {
        std::vector<size_t> dims;
        dims.push_back(batch);
        for (size_t i = 1; i < t.shape().rank(); ++i) {
            dims.push_back(t.shape()[i]);
        }

        Tensor<float> out{Shape(dims)};
        for (size_t i = 0; i < batch * per_row; ++i) {
            out.flat(i) = t.flat(start * per_row + i);
        }
        return out;
    }

    static void assign_parameters(layers::Layer& layer,
                                  const std::unordered_map<std::string, Tensor<float>>& params) {
        if (auto* dense = dynamic_cast<layers::Dense*>(&layer)) {
            dense->set_weights(params.at("weights"));
            dense->set_bias(params.at("bias"));
            return;
        }
        if (auto* conv = dynamic_cast<layers::Conv2D*>(&layer)) {
            conv->set_weights(params.at("weights"));
            conv->set_bias(params.at("bias"));
            return;
        }
    }

    static void write_u32(std::ofstream& file, uint32_t value) {
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    static uint32_t read_u32(std::ifstream& file) {
        uint32_t value = 0;
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!file) {
            throw std::runtime_error("archivo de modelo incompleto");
        }
        return value;
    }

    static void write_string(std::ofstream& file, const std::string& text) {
        write_u32(file, static_cast<uint32_t>(text.size()));
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
    }

    static std::string read_string(std::ifstream& file) {
        uint32_t size = read_u32(file);
        std::string text(size, '\0');
        if (size > 0) {
            file.read(text.data(), static_cast<std::streamsize>(size));
            if (!file) {
                throw std::runtime_error("archivo de modelo incompleto");
            }
        }
        return text;
    }

    static void write_shape(std::ofstream& file, const Shape& shape) {
        write_u32(file, static_cast<uint32_t>(shape.rank()));
        for (size_t i = 0; i < shape.rank(); ++i) {
            write_u32(file, static_cast<uint32_t>(shape[i]));
        }
    }

    static Shape read_shape(std::ifstream& file) {
        uint32_t rank = read_u32(file);
        std::vector<size_t> dims;
        for (uint32_t i = 0; i < rank; ++i) {
            dims.push_back(static_cast<size_t>(read_u32(file)));
        }
        return Shape(dims);
    }

    static void write_tensor(std::ofstream& file, const Tensor<float>& tensor) {
        write_shape(file, tensor.shape());
        for (size_t i = 0; i < tensor.size(); ++i) {
            float value = tensor.flat(i);
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    }

    static Tensor<float> read_tensor(std::ifstream& file) {
        Shape shape = read_shape(file);
        Tensor<float> tensor(shape);
        for (size_t i = 0; i < tensor.size(); ++i) {
            float value = 0.0f;
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
            if (!file) {
                throw std::runtime_error("archivo de modelo incompleto");
            }
            tensor.flat(i) = value;
        }
        return tensor;
    }

    std::vector<std::unique_ptr<layers::Layer>> layers_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, int> type_counts_;
    utec::Shape input_shape_;
    utec::Shape current_shape_;
    bool has_input_ = false;
    bool compiled_ = false;
    std::optional<optimizers::SGD> optimizer_;
    std::unordered_map<std::string, Tensor<float>> last_gradients_;
    std::unordered_map<std::string, std::string> metadata_;
};

}

using namespace utec::tf;
