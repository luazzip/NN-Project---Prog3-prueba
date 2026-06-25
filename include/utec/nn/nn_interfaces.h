#pragma once
#include "../algebra/tensor_backend.h"
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace utec::tf::layers {
    using utec::Shape;
    template<typename T> using Tensor = utec::Tensor<T>;

    class Layer {
    public:
        virtual ~Layer() = default;
        virtual void build(const Shape& input_shape) = 0;
        virtual utec::Tensor<float> forward(const utec::Tensor<float>& x) = 0;
        virtual utec::Tensor<float> backward(const utec::Tensor<float>& grad) = 0;
        virtual std::unordered_map<std::string, utec::Tensor<float>> parameters() {
            return {};
        }
        virtual std::unordered_map<std::string, utec::Tensor<float>> gradients() {
            return {};
        }
        virtual std::unique_ptr<Layer> clone() const = 0;

        virtual std::string type_name() const { return "layer"; }

        virtual Shape output_shape() const { return {}; }

        virtual std::vector<std::pair<utec::Tensor<float>*, utec::Tensor<float>*>>
        trainable_parameters() {
            return {};
        }
    };
}