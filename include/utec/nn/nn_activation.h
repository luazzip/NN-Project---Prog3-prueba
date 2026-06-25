#pragma once
#include "../algebra/tensor_ops.h"
#include <algorithm>
#include <cmath>

namespace utec::tf {

    enum class Activation { Linear, Relu, Softmax };
    inline Tensor<float> apply_activation(const Tensor<float>& z, Activation act) {
        if (act == Activation::Linear) {
            return z;
        }

        Tensor<float> out = z;

        if (act == Activation::Relu) {
            for (size_t i = 0; i < out.size(); ++i) {
                float v = z.flat(i);
                out.flat(i) = v > 0.0f ? v : 0.0f;
            }
            return out;
        }

        size_t rows = z.shape()[0];
        size_t cols = z.size() / rows;
        for (size_t r = 0; r < rows; ++r) {
            float max_v = -std::numeric_limits<float>::infinity();
            for (size_t c = 0; c < cols; ++c) {
                max_v = std::max(max_v, z.flat(r * cols + c));
            }
            float sum = 0.0f;
            for (size_t c = 0; c < cols; ++c) {
                float e = std::exp(z.flat(r * cols + c) - max_v);
                out.flat(r * cols + c) = e;
                sum += e;
            }
            for (size_t c = 0; c < cols; ++c) {
                out.flat(r * cols + c) /= sum;
            }
        }
        return out;
    }

    inline Tensor<float> relu_backward(const Tensor<float>& grad,
                                       const Tensor<float>& z) {
        Tensor<float> out = grad;
        for (size_t i = 0; i < out.size(); ++i) {
            out.flat(i) = z.flat(i) > 0.0f ? grad.flat(i) : 0.0f;
        }
        return out;
    }

}
