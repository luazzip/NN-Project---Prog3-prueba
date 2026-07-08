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
#include "../io/model_serializer.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace utec::tf {

    struct FitOptions {
        int epochs = 1;
        int batch_size = 1;
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
                auto ptr = std::make_unique<D>(std::move(layer));
                ptr->build(current_shape_);
                std::string base = ptr->type_name();
                std::string name = base + "_" +
                                   std::to_string(type_counts_[base]++);
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
            if (x.size() == 0 || x.shape()[0] == 0) {
                throw std::invalid_argument("el dataset no puede estar vacio");
            }

            {
                Tensor<float> pred = predict(x);
                validate_labels(pred, y);
            }

            size_t n = x.shape()[0];
            size_t per_x = x.size() / n;
            size_t per_y = y.size() / n;
            size_t batch_size = static_cast<size_t>(options.batch_size);

            History history;
            for (int epoch = 0; epoch < options.epochs; ++epoch) {
                float epoch_loss = 0.0f;
                size_t batches = 0;
                for (size_t start = 0; start < n; start += batch_size) {
                    size_t b = std::min(batch_size, n - start);
                    Tensor<float> xb = slice_rows(x, start, b, per_x);
                    Tensor<float> yb = slice_rows(y, start, b, per_y);

                    Tensor<float> pred = predict(xb);
                    epoch_loss +=
                        losses::CategoricalCrossentropy::loss(pred, yb);
                    ++batches;

                    Tensor<float> grad =
                        losses::CategoricalCrossentropy::grad(pred, yb);
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

        [[nodiscard]] std::unordered_map<std::string, Tensor<float>>
        parameters() {
            std::unordered_map<std::string, Tensor<float>> out;
            for (size_t i = 0; i < layers_.size(); ++i) {
                for (auto& kv : layers_[i]->parameters()) {
                    out[names_[i] + "/" + kv.first] = kv.second;
                }
            }
            return out;
        }

        [[nodiscard]] std::unordered_map<std::string, Tensor<float>>
        last_gradients() const {
            return last_gradients_;
        }

        void save(const std::string& path, const SaveOptions& options) {
            io::ModelWriter writer(path);
            writer.write_magic_and_version();
            write_metadata(writer, options.metadata);
            writer.write_shape(input_shape_);

            writer.write_uint32(static_cast<uint32_t>(layers_.size()));
            for (auto& layer_ptr : layers_) {
                write_layer(writer, *layer_ptr);
            }

            metadata_ = options.metadata;
        }

        static Sequential load(const std::string& path) {
            io::ModelReader reader(path);
            reader.check_magic_and_version();

            std::unordered_map<std::string, std::string> metadata = read_metadata(reader);
            Shape input_shape = reader.read_shape();

            Sequential model;
            model.add(layers::Input(input_shape));

            uint32_t layer_count = reader.read_uint32();
            for (uint32_t layer_index = 0; layer_index < layer_count; ++layer_index) {
                read_and_add_layer(reader, model);
            }

            model.metadata_ = metadata;
            return model;
        }

        const std::unordered_map<std::string, std::string>& metadata() const {
            return metadata_;
        }

    private:
        void validate_labels(const Tensor<float>& pred,
                             const Tensor<float>& y) const {
            if (!(pred.shape() == y.shape())) {
                throw std::invalid_argument(
                    "y_true incompatible con la salida del modelo");
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

        static Tensor<float> slice_rows(const Tensor<float>& t, size_t start,
                                        size_t b, size_t per) {
            std::vector<size_t> dims;
            dims.push_back(b);
            for (size_t i = 1; i < t.shape().rank(); ++i) {
                dims.push_back(t.shape()[i]);
            }
            utec::Shape shape(dims);
            Tensor<float> out(shape);
            for (size_t i = 0; i < b * per; ++i) {
                out.flat(i) = t.flat(start * per + i);
            }
            return out;
        }

        static void write_metadata(io::ModelWriter& writer,
                const std::unordered_map<std::string, std::string>& metadata) {
            writer.write_uint32(static_cast<uint32_t>(metadata.size()));
            for (const auto& entry : metadata) {
                writer.write_string(entry.first);
                writer.write_string(entry.second);
            }
        }

        static std::unordered_map<std::string, std::string> read_metadata(
                io::ModelReader& reader) {
            uint32_t entry_count = reader.read_uint32();
            std::unordered_map<std::string, std::string> metadata;
            for (uint32_t entry_index = 0; entry_index < entry_count; ++entry_index) {
                std::string key = reader.read_string();
                std::string value = reader.read_string();
                metadata.emplace(std::move(key), std::move(value));
            }
            return metadata;
        }

        static void write_layer(io::ModelWriter& writer, layers::Layer& layer) {
            writer.write_string(layer.type_name());
            writer.write_uint32(activation_to_code(activation_of(layer)));

            auto parameters = layer.parameters();
            writer.write_uint32(static_cast<uint32_t>(parameters.size()));
            for (auto& named_tensor : parameters) {
                writer.write_string(named_tensor.first);
                writer.write_tensor(named_tensor.second);
            }
        }

        static void read_and_add_layer(io::ModelReader& reader, Sequential& model) {
            std::string type_name = reader.read_string();
            Activation activation = activation_from_code(reader.read_uint32());

            uint32_t parameter_count = reader.read_uint32();
            std::unordered_map<std::string, Tensor<float>> parameters;
            for (uint32_t parameter_index = 0; parameter_index < parameter_count;
                 ++parameter_index) {
                std::string name = reader.read_string();
                parameters.emplace(std::move(name), reader.read_tensor());
            }

            if (type_name == "conv2d") {
                const Tensor<float>& weights = parameters.at("weights");
                size_t kernel_height = weights.shape()[0];
                size_t kernel_width = weights.shape()[1];
                size_t filters = weights.shape()[3];
                model.add(layers::Conv2D(static_cast<int>(filters),
                                         {kernel_height, kernel_width}, activation));
                assign_parameters(*model.layers_.back(), parameters);
            } else if (type_name == "dense") {
                const Tensor<float>& weights = parameters.at("weights");
                size_t units = weights.shape()[1];
                model.add(layers::Dense(static_cast<int>(units), activation));
                assign_parameters(*model.layers_.back(), parameters);
            } else if (type_name == "flatten") {
                model.add(layers::Flatten());
            } else {
                throw std::runtime_error("tipo de capa desconocido al cargar el modelo");
            }
        }

        static void assign_parameters(layers::Layer& layer,
                const std::unordered_map<std::string, Tensor<float>>& parameters) {
            if (auto* dense = dynamic_cast<layers::Dense*>(&layer)) {
                dense->set_weights(parameters.at("weights"));
                dense->set_bias(parameters.at("bias"));
            } else if (auto* conv = dynamic_cast<layers::Conv2D*>(&layer)) {
                conv->set_weights(parameters.at("weights"));
                conv->set_bias(parameters.at("bias"));
            }
        }

        static Activation activation_of(layers::Layer& layer) {
            if (auto* dense = dynamic_cast<layers::Dense*>(&layer)) {
                return dense->activation();
            }
            if (auto* conv = dynamic_cast<layers::Conv2D*>(&layer)) {
                return conv->activation();
            }
            return Activation::Linear;
        }

        static uint32_t activation_to_code(Activation activation) {
            return static_cast<uint32_t>(activation);
        }

        static Activation activation_from_code(uint32_t code) {
            return static_cast<Activation>(code);
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
