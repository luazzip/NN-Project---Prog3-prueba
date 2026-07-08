#pragma once
#include "nn_interfaces.h"
#include "../algebra/tensor_ops.h"
#include <vector>
#include <limits>
#include <stdexcept>

namespace utec::tf::layers {

class MaxPooling2D : public Layer {
private:
    size_t pool_h_ = 0, pool_w_ = 0;
    utec::Shape input_shape_;
    std::vector<size_t> max_indices_;
    size_t out_h_ = 0, out_w_ = 0, out_c_ = 0;
    bool forward_done_ = false;

public:
    MaxPooling2D(std::initializer_list<size_t> pool_size) {
        if (pool_size.size() != 2) {
            throw std::invalid_argument("pool size must have 2 values");
        }

        auto it = pool_size.begin();
        pool_h_ = *it++;
        pool_w_ = *it;

        if (pool_h_ == 0 || pool_w_ == 0) {
            throw std::invalid_argument("pool size must be > 0");
        }
    }

    void build(const utec::Shape& shape) override {
        if (shape.rank() != 3) {
            throw std::invalid_argument("MaxPooling2D expects input shape {H,W,C}");
        }

        if (shape[0] % pool_h_ != 0 || shape[1] % pool_w_ != 0) {
            throw std::invalid_argument("input dims must be divisible by pool size");
        }

        out_h_ = shape[0] / pool_h_;
        out_w_ = shape[1] / pool_w_;
        out_c_ = shape[2];
        forward_done_ = false;
    }

    utec::Tensor<float> forward(const utec::Tensor<float>& x) override {
        if (x.shape().rank() != 4) {
            throw std::invalid_argument("MaxPooling2D forward expects {N,H,W,C}");
        }

        input_shape_ = x.shape();

        size_t N = x.shape()[0];
        size_t H = x.shape()[1];
        size_t W = x.shape()[2];
        size_t C = x.shape()[3];

        if (H % pool_h_ != 0 || W % pool_w_ != 0) {
            throw std::invalid_argument("input dims must be divisible by pool size");
        }

        out_h_ = H / pool_h_;
        out_w_ = W / pool_w_;
        out_c_ = C;

        utec::Tensor<float> out(utec::Shape{N, out_h_, out_w_, C});
        max_indices_.assign(N * out_h_ * out_w_ * C, 0);

        for (size_t n = 0; n < N; ++n)
        for (size_t oh = 0; oh < out_h_; ++oh)
        for (size_t ow = 0; ow < out_w_; ++ow)
        for (size_t c = 0; c < C; ++c) {
            float best_val = -std::numeric_limits<float>::infinity();
            size_t best_idx = 0;

            for (size_t ph = 0; ph < pool_h_; ++ph)
            for (size_t pw = 0; pw < pool_w_; ++pw) {
                size_t ih = oh * pool_h_ + ph;
                size_t iw = ow * pool_w_ + pw;

                float v = x(n, ih, iw, c);

                if (v > best_val) {
                    best_val = v;
                    best_idx = ((n * H + ih) * W + iw) * C + c;
                }
            }

            out(n, oh, ow, c) = best_val;

            size_t out_idx = ((n * out_h_ + oh) * out_w_ + ow) * C + c;
            max_indices_[out_idx] = best_idx;
        }

        forward_done_ = true;
        return out;
    }

    utec::Tensor<float> backward(const utec::Tensor<float>& grad) override {
        if (!forward_done_) {
            throw std::logic_error("backward called before forward");
        }

        if (grad.shape() != utec::Shape{input_shape_[0], out_h_, out_w_, out_c_}) {
            throw std::invalid_argument("MaxPooling2D grad has incompatible shape");
        }

        utec::Tensor<float> dx = utec::Tensor<float>::zeros(input_shape_);

        size_t N = input_shape_[0];
        size_t C = input_shape_[3];
        size_t total = N * out_h_ * out_w_ * C;

        for (size_t i = 0; i < total; ++i) {
            size_t tmp = i;

            size_t c  = tmp % C;
            tmp /= C;

            size_t ow = tmp % out_w_;
            tmp /= out_w_;

            size_t oh = tmp % out_h_;
            tmp /= out_h_;

            size_t n = tmp;

            dx.flat(max_indices_[i]) += grad(n, oh, ow, c);
        }

        return dx;
    }

    std::string type_name() const override {
        return "maxpooling2d";
    }

    utec::Shape output_shape() const override {
        if (out_h_ == 0 || out_w_ == 0 || out_c_ == 0) {
            return {};
        }

        return utec::Shape{out_h_, out_w_, out_c_};
    }

    std::unique_ptr<Layer> clone() const override {
        return std::make_unique<MaxPooling2D>(*this);
    }
};

}

using namespace utec::tf;
