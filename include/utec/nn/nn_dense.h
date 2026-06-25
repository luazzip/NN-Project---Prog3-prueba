#pragma once
#include "nn_interfaces.h"
#include "nn_activation.h"
#include "nn_ops.h"
#include "../algebra/tensor_ops.h"
#include <random>
#include <stdexcept>

namespace utec::tf::layers {

    class Dense : public Layer {
    public:
        Dense(int units, Activation activation)
            : units_(static_cast<size_t>(units)), activation_(activation) {
            if (units <= 0) {
                throw std::invalid_argument("units debe ser positivo");
            }
        }

        void build(const utec::Shape& input_shape) override {
            in_features_ = input_shape.total_size();
            weights_ = utec::Tensor<float>(utec::Shape{in_features_, units_});
            bias_ = utec::Tensor<float>::zeros(utec::Shape{units_});

            std::mt19937 gen(
                static_cast<unsigned>(1234 + in_features_ * 31 + units_ * 7));
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            for (size_t i = 0; i < weights_.size(); ++i) {
                weights_.flat(i) = dist(gen);
            }

            grad_weights_ = utec::Tensor<float>::zeros(weights_.shape());
            grad_bias_ = utec::Tensor<float>::zeros(bias_.shape());
        }

        utec::Tensor<float> forward(const utec::Tensor<float>& x) override {
            last_input_ = x;
            size_t batch = x.shape()[0];

            utec::Tensor<float> z = matmul(x, weights_);  // {batch, units}
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < units_; ++j) {
                    z.flat(i * units_ + j) += bias_.flat(j);
                }
            }
            last_preact_ = z;
            forward_done_ = true;
            return apply_activation(z, activation_);
        }

        utec::Tensor<float> backward(const utec::Tensor<float>& grad) override {
            if (!forward_done_) {
                throw std::logic_error("backward llamado antes de forward");
            }
            size_t batch = last_input_.shape()[0];

            utec::Tensor<float> dz = grad;
            if (activation_ == Activation::Relu) {
                dz = relu_backward(grad, last_preact_);
            }

            grad_weights_ = utec::Tensor<float>::zeros(weights_.shape());
            for (size_t p = 0; p < in_features_; ++p) {
                for (size_t j = 0; j < units_; ++j) {
                    float acc = 0.0f;
                    for (size_t i = 0; i < batch; ++i) {
                        acc += last_input_.flat(i * in_features_ + p) *
                               dz.flat(i * units_ + j);
                    }
                    grad_weights_.flat(p * units_ + j) = acc;
                }
            }

            grad_bias_ = utec::Tensor<float>::zeros(bias_.shape());
            for (size_t j = 0; j < units_; ++j) {
                float acc = 0.0f;
                for (size_t i = 0; i < batch; ++i) {
                    acc += dz.flat(i * units_ + j);
                }
                grad_bias_.flat(j) = acc;
            }

            utec::Tensor<float> dx(utec::Shape{batch, in_features_});
            for (size_t i = 0; i < batch; ++i) {
                for (size_t p = 0; p < in_features_; ++p) {
                    float acc = 0.0f;
                    for (size_t j = 0; j < units_; ++j) {
                        acc += dz.flat(i * units_ + j) *
                               weights_.flat(p * units_ + j);
                    }
                    dx.flat(i * in_features_ + p) = acc;
                }
            }
            return dx;
        }

        std::unordered_map<std::string, utec::Tensor<float>> parameters() override {
            return {{"weights", weights_}, {"bias", bias_}};
        }
        std::unordered_map<std::string, utec::Tensor<float>> gradients() override {
            return {{"weights", grad_weights_}, {"bias", grad_bias_}};
        }
        std::vector<std::pair<utec::Tensor<float>*, utec::Tensor<float>*>>
        trainable_parameters() override {
            return {{&weights_, &grad_weights_}, {&bias_, &grad_bias_}};
        }

        std::string type_name() const override { return "dense"; }
        utec::Shape output_shape() const override { return utec::Shape{units_}; }
        std::unique_ptr<Layer> clone() const override {
            return std::make_unique<Dense>(*this);
        }

    private:
        size_t units_;
        Activation activation_;
        size_t in_features_ = 0;
        utec::Tensor<float> weights_;
        utec::Tensor<float> bias_;
        utec::Tensor<float> grad_weights_;
        utec::Tensor<float> grad_bias_;
        utec::Tensor<float> last_input_;
        utec::Tensor<float> last_preact_;
        bool forward_done_ = false;
    };

}
