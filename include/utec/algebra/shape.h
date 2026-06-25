#pragma once
#include <vector>
#include <initializer_list>
#include <ostream>

namespace utec {
    class Shape {
        std::vector<size_t> dims;
    public:
        Shape() = default;
        Shape(std::initializer_list<size_t> values) : dims(values) {}
        explicit Shape(std::vector<size_t> values) : dims(std::move(values)) {}

        size_t rank() const { return dims.size(); }
        size_t total_size() const {
            size_t r = 1;
            for (auto x : dims) r *= x;
            return r;
        }
        size_t operator[](size_t i) const { return dims[i]; }
        bool operator==(const Shape& o) const { return dims == o.dims; }

        friend std::ostream& operator<<(std::ostream& os, const Shape& s) {
            os << "{";
            for (size_t i = 0; i < s.dims.size(); ++i) {
                if (i) os << ", ";
                os << s.dims[i];
            }
            return os << "}";
        }
    };
}

#include "tensor_ops.h"
namespace utec::tf {}
using namespace utec::tf;