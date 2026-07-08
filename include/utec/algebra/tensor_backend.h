#pragma once

#define UTEC_TENSOR_BACKEND_BUILDING
#include "shape.h"
#undef UTEC_TENSOR_BACKEND_BUILDING

#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace utec {

template<typename T>
class Tensor {
private:
    Shape shape_;
    std::vector<T> data_;

    size_t compute_offset(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.rank()) {
            throw std::out_of_range("cantidad incorrecta de indices");
        }

        size_t pos = 0;
        size_t stride = 1;

        for (size_t r = shape_.rank(); r > 0; --r) {
            size_t axis = r - 1;
            if (indices[axis] >= shape_[axis]) {
                throw std::out_of_range("indice fuera de rango");
            }
            pos += indices[axis] * stride;
            stride *= shape_[axis];
        }

        return pos;
    }

public:
    Tensor() = default;

    explicit Tensor(const Shape& s)
        : shape_(s), data_(s.total_size(), T{}) {}

    static Tensor<T> zeros(const Shape& s) {
        return Tensor<T>(s);
    }

    static Tensor<T> ones(const Shape& s) {
        Tensor<T> t(s);
        for (auto& value : t.data_) {
            value = static_cast<T>(1);
        }
        return t;
    }

    static Tensor<T> from_data(const Shape& s, std::initializer_list<T> values) {
        if (values.size() != s.total_size()) {
            throw std::invalid_argument("cantidad de datos incompatible con shape");
        }

        Tensor<T> t(s);
        size_t i = 0;
        for (const auto& value : values) {
            t.data_[i++] = value;
        }
        return t;
    }

    Shape shape() const {
        return shape_;
    }

    size_t rank() const {
        return shape_.rank();
    }

    size_t size() const {
        return data_.size();
    }

    Tensor<T> reshaped(const Shape& new_shape) const {
        if (new_shape.total_size() != data_.size()) {
            throw std::invalid_argument("reshape incompatible");
        }

        Tensor<T> out(new_shape);
        out.data_ = data_;
        return out;
    }

    Tensor<T> reshape(const Shape& new_shape) const {
        return reshaped(new_shape);
    }

    T& flat(size_t i) {
        if (i >= data_.size()) {
            throw std::out_of_range("indice flat fuera de rango");
        }
        return data_.at(i);
    }

    const T& flat(size_t i) const {
        if (i >= data_.size()) {
            throw std::out_of_range("indice flat fuera de rango");
        }
        return data_.at(i);
    }

    Tensor<T> operator+(const Tensor<T>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("suma incompatible");
        }

        Tensor<T> out(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = data_[i] + other.data_[i];
        }
        return out;
    }

    Tensor<T> operator-(const Tensor<T>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("resta incompatible");
        }

        Tensor<T> out(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = data_[i] - other.data_[i];
        }
        return out;
    }

    Tensor<T> operator*(T scalar) const {
        Tensor<T> out(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            out.data_[i] = data_[i] * scalar;
        }
        return out;
    }

    template<typename... Args>
    T& operator()(Args... args) {
        std::vector<size_t> indices{static_cast<size_t>(args)...};
        return data_.at(compute_offset(indices));
    }

    template<typename... Args>
    const T& operator()(Args... args) const {
        std::vector<size_t> indices{static_cast<size_t>(args)...};
        return data_.at(compute_offset(indices));
    }
};

}

namespace utec::tf {
    using Shape = utec::Shape;

    template<typename T>
    using Tensor = utec::Tensor<T>;
}
