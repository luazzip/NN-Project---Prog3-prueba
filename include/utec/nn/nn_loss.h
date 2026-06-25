#pragma once
#include "../algebra/tensor_ops.h"
#include <cmath>

namespace utec::tf::losses {

    using utec::Tensor;

    struct CategoricalCrossentropy {
        static float loss(const Tensor<float>& pred, const Tensor<float>& target) {
            size_t batch = pred.shape()[0];
            const float eps = 1e-7f;
            float total = 0.0f;
            for (size_t i = 0; i < pred.size(); ++i) {
                total += -target.flat(i) * std::log(pred.flat(i) + eps);
            }
            return total / static_cast<float>(batch);
        }

        static Tensor<float> grad(const Tensor<float>& pred,
                                  const Tensor<float>& target) {
            size_t batch = pred.shape()[0];
            Tensor<float> g(pred.shape());
            for (size_t i = 0; i < pred.size(); ++i) {
                g.flat(i) = (pred.flat(i) - target.flat(i)) / static_cast<float>(batch);
            }
            return g;
        }
    };

}
