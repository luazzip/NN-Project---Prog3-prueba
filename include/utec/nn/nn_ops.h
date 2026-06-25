#pragma once
#include "../algebra/tensor_ops.h"

namespace utec::tf {

    inline Tensor<float> matmul(const Tensor<float>& a, const Tensor<float>& b) {
        size_t m = a.shape()[0];
        size_t k = a.shape()[1];
        size_t n = b.shape()[1];
        Tensor<float> out(Shape{m, n});
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (size_t p = 0; p < k; ++p) {
                    acc += a.flat(i * k + p) * b.flat(p * n + j);
                }
                out.flat(i * n + j) = acc;
            }
        }
        return out;
    }

}
