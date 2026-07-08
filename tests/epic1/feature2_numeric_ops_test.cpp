#include <cmath>
#include <stdexcept>
#include <string_view>
#include "utec/algebra/tensor_backend.h"
#include "utec/nn/nn_ops.h"

using utec::tf::Shape;
using utec::tf::Tensor;
using namespace utec::tf::ops;

namespace {
void expect(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template <class Exception, class Fn>
void expect_throw(Fn&& fn, std::string_view message) {
    try {
        fn();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}
}

int main() {
    auto a = Tensor<float>::from_data(Shape{2, 2}, {1, 2, 3, 4});
    auto b = Tensor<float>::from_data(Shape{2, 2}, {5, 6, 7, 8});

    auto c = matmul(a, b);

    expect(c.shape() == Shape{2, 2}, "matmul shape incorrecto");
    expect(std::fabs(c(0, 0) - 19.0f) < 1e-6f, "matmul valor incorrecto");

    auto input = Tensor<float>::from_data(
        Shape{1, 4, 4, 1},
        {
            1, 1, 1, 1,
            1, 2, 2, 1,
            1, 2, 2, 1,
            1, 1, 1, 1
        }
    );

    auto kernel = Tensor<float>::from_data(
        Shape{2, 2, 1, 1},
        {
            1, 0,
            0, -1
        }
    );

    auto out = conv2d(input, kernel, Strides{1, 1}, Padding::Valid);

    expect(out.shape() == Shape{1, 3, 3, 1}, "conv2d shape incorrecto");

    expect_throw<std::invalid_argument>(
        [&] {
            auto bad_kernel = Tensor<float>::ones(Shape{3, 3, 2, 1});
            (void)conv2d(input, bad_kernel, Strides{1, 1}, Padding::Valid);
        },
        "conv2d debio fallar por canales incompatibles"
    );

    expect_throw<std::invalid_argument>(
        [&] {
            (void)conv2d(input, kernel, Strides{0, 1}, Padding::Valid);
        },
        "conv2d debio fallar por stride invalido"
    );

    std::puts("EPIC1_FEATURE2_OK");
    return 0;
}
