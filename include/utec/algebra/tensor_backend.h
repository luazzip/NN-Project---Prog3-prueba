#pragma once
#define UTEC_TENSOR_BACKEND_BUILDING
#include "shape.h"
#undef UTEC_TENSOR_BACKEND_BUILDING
#include <vector>
#include <initializer_list>
#include <cmath>

namespace utec {
    template<typename T>
    class Tensor {
        Shape shape_;
        std::vector<T> data;
    public:
        Tensor() = default;
        Tensor(const Shape& s) : shape_(s), data(s.total_size(), T{}) {}

        static Tensor<T> ones(const Shape& s) {
            Tensor<T> t(s);
            for (auto& x : t.data) x = 1;
            return t;
        }
        static Tensor<T> zeros(const Shape& s) { return Tensor<T>(s); }
        static Tensor<T> from_data(const Shape& s, std::initializer_list<T> v) {
            Tensor<T> t(s);
            t.data = v;
            return t;
        }

        Shape shape() const { return shape_; }
        size_t size() const { return data.size(); }

        Tensor<T> reshaped(const Shape& ns) const {
            Tensor<T> r(ns);
            r.data = data;
            return r;
        }

        T& flat(size_t i) { return data[i]; }
        const T& flat(size_t i) const { return data[i]; }

        template <typename... Idx>
T& operator()(Idx... idx) {
    std::array<size_t, sizeof...(Idx)> indices{static_cast<size_t>(idx)...};

    if (indices.size() != shape_.rank()) {
        throw std::out_of_range("cantidad incorrecta de indices");
    }

    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("indice fuera de rango");
        }
    }

    size_t pos = offset(indices);
    return data.at(pos);
}

template <typename... Idx>
const T& operator()(Idx... idx) const {
    std::array<size_t, sizeof...(Idx)> indices{static_cast<size_t>(idx)...};

    if (indices.size() != shape_.rank()) {
        throw std::out_of_range("cantidad incorrecta de indices");
    }

    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("indice fuera de rango");
        }
    }

    size_t pos = offset(indices);
    return data.at(pos);
}
    };
}

