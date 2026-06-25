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

        std::vector<std::unique_ptr<layers::Layer>> layers_;
        std::vector<std::string> names_;
        std::unordered_map<std::string, int> type_counts_;
        utec::Shape input_shape_;
        utec::Shape current_shape_;
        bool has_input_ = false;
        bool compiled_ = false;
        std::optional<optimizers::SGD> optimizer_;
        std::unordered_map<std::string, Tensor<float>> last_gradients_;
    };

}
