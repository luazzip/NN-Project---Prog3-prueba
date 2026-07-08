#pragma once
#include "nn_interfaces.h"
#include "nn_activation.h"
#include "../algebra/tensor_ops.h"
#include <random>
#include <stdexcept>

namespace utec::tf::layers {

    class Conv2D : public Layer {
    public:
        Conv2D(int filters, std::initializer_list<size_t> kernel_size,
               Activation activation)
            : out_channels_(static_cast<size_t>(filters)), activation_(activation) {
            if (filters <= 0) {
                throw std::invalid_argument("filters debe ser positivo");
            }
            auto it = kernel_size.begin();
            kernel_h_ = *it++;
            kernel_w_ = *it;
            if (kernel_h_ == 0 || kernel_w_ == 0) {
                throw std::invalid_argument("kernel size debe ser > 0");
            }
        }

        void build(const utec::Shape& input_shape) override {
            in_h_ = input_shape[0];
            in_w_ = input_shape[1];
            in_c_ = input_shape[2];
            out_h_ = in_h_ - kernel_h_ + 1;
            out_w_ = in_w_ - kernel_w_ + 1;

            weights_ = utec::Tensor<float>(
                utec::Shape{kernel_h_, kernel_w_, in_c_, out_channels_});
            bias_ = utec::Tensor<float>::zeros(utec::Shape{out_channels_});

            std::mt19937 gen(static_cast<unsigned>(
                987 + kernel_h_ * 7 + kernel_w_ * 13 + in_c_ * 17 +
                out_channels_ * 23));
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            for (size_t i = 0; i < weights_.size(); ++i) {
                weights_.flat(i) = dist(gen);
            }

            grad_weights_ = utec::Tensor<float>::zeros(weights_.shape());
            grad_bias_ = utec::Tensor<float>::zeros(bias_.shape());
        }

        void set_weights(const utec::Tensor<float>& w) { weights_ = w; }
        void set_bias(const utec::Tensor<float>& b) { bias_ = b; }

        Activation activation() const { return activation_; }

        utec::Tensor<float> forward(const utec::Tensor<float>& x) override {
            last_input_ = x;
            size_t batch = x.shape()[0];
            utec::Tensor<float> z(
                utec::Shape{batch, out_h_, out_w_, out_channels_});

            for (size_t n = 0; n < batch; ++n) {
                for (size_t oh = 0; oh < out_h_; ++oh) {
                    for (size_t ow = 0; ow < out_w_; ++ow) {
                        for (size_t co = 0; co < out_channels_; ++co) {
                            float acc = bias_.flat(co);
                            for (size_t kh = 0; kh < kernel_h_; ++kh) {
                                for (size_t kw = 0; kw < kernel_w_; ++kw) {
                                    for (size_t ci = 0; ci < in_c_; ++ci) {
                                        acc += x(n, oh + kh, ow + kw, ci) *
                                               weights_(kh, kw, ci, co);
                                    }
                                }
                            }
                            z(n, oh, ow, co) = acc;
                        }
                    }
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
            utec::Tensor<float> dz = grad;
            if (activation_ == Activation::Relu) {
                dz = relu_backward(grad, last_preact_);
            }

            size_t batch = last_input_.shape()[0];
            grad_weights_ = utec::Tensor<float>::zeros(weights_.shape());
            grad_bias_ = utec::Tensor<float>::zeros(bias_.shape());
            utec::Tensor<float> dx =
                utec::Tensor<float>::zeros(last_input_.shape());

            for (size_t n = 0; n < batch; ++n) {
                for (size_t oh = 0; oh < out_h_; ++oh) {
                    for (size_t ow = 0; ow < out_w_; ++ow) {
                        for (size_t co = 0; co < out_channels_; ++co) {
                            float g = dz(n, oh, ow, co);
                            grad_bias_.flat(co) += g;
                            for (size_t kh = 0; kh < kernel_h_; ++kh) {
                                for (size_t kw = 0; kw < kernel_w_; ++kw) {
                                    for (size_t ci = 0; ci < in_c_; ++ci) {
                                        grad_weights_(kh, kw, ci, co) +=
                                            last_input_(n, oh + kh, ow + kw, ci) * g;
                                        dx(n, oh + kh, ow + kw, ci) +=
                                            weights_(kh, kw, ci, co) * g;
                                    }
                                }
                            }
                        }
                    }
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

        std::string type_name() const override { return "conv2d"; }
        utec::Shape output_shape() const override {
            return utec::Shape{out_h_, out_w_, out_channels_};
        }
        std::unique_ptr<Layer> clone() const override {
            return std::make_unique<Conv2D>(*this);
        }

    private:
        size_t out_channels_;
        Activation activation_;
        size_t kernel_h_ = 0;
        size_t kernel_w_ = 0;
        size_t in_h_ = 0, in_w_ = 0, in_c_ = 0;
        size_t out_h_ = 0, out_w_ = 0;
        utec::Tensor<float> weights_;
        utec::Tensor<float> bias_;
        utec::Tensor<float> grad_weights_;
        utec::Tensor<float> grad_bias_;
        utec::Tensor<float> last_input_;
        utec::Tensor<float> last_preact_;
        bool forward_done_ = false;
    };

}
