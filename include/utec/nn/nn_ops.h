#pragma once

#include "../algebra/tensor_ops.h"

namespace utec::tf {
    using ops::Padding;
    using ops::Strides;

    inline Tensor<float> matmul(const Tensor<float>& a,
                                const Tensor<float>& b) {
        return ops::matmul(a, b);
    }

    inline Tensor<float> conv2d(const Tensor<float>& input,
                                const Tensor<float>& kernel) {
        return ops::conv2d(input, kernel);
    }

    inline Tensor<float> conv2d(const Tensor<float>& input,
                                const Tensor<float>& kernel,
                                Strides strides,
                                Padding padding) {
        return ops::conv2d(input, kernel, strides, padding);
    }
}
