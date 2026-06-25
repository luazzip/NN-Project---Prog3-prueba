#pragma once
#include "nn_interfaces.h"
#include "../algebra/tensor_ops.h"
#include <stdexcept>

namespace utec::tf::layers {
    class Flatten : public Layer {

    public:
        void build(const utec::Shape& shape) override {
            input_shape_ = shape;
            flat_size_ = shape.total_size();
            forward_done_ = false;
        }
        utec::Tensor<float> forward(const utec::Tensor<float>& x) override {
            input_shape_ = x.shape();
            size_t batch = x.shape()[0];
            size_t flat  = x.size() / batch;
            forward_done_ = true;
            return x.reshaped(utec::Shape{batch, flat});
        }
        utec::Tensor<float> backward(const utec::Tensor<float>& grad) override {
            if (!forward_done_)
                throw std::logic_error("backward called before forward");
            if (grad.size() != input_shape_.total_size())
                throw std::invalid_argument("grad size does not match input size");
            return grad.reshaped(input_shape_);
        }
        std::unique_ptr<Layer> clone() const override {
            return std::make_unique<Flatten>(*this);
        }
        std::string type_name() const override { return "flatten"; }
        utec::Shape output_shape() const override { return utec::Shape{flat_size_}; }

    private:
        utec::Shape input_shape_;
        size_t flat_size_ = 0;
        bool forward_done_ = false;
    };
}

using namespace utec::tf;
