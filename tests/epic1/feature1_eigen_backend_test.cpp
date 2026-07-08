#include <cmath>
#include <stdexcept>
#include <string_view>
#include "utec/algebra/tensor_backend.h"

using utec::tf::Shape;
using utec::tf::Tensor;

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
    auto x = Tensor<float>::zeros(Shape{2, 3});

    expect(x.rank() == 2, "rank incorrecto");
    expect(x.shape() == Shape{2, 3}, "shape incorrecto");

    x(1, 2) = 7.0f;
    expect(std::fabs(x(1, 2) - 7.0f) < 1e-6f, "acceso por indice fallo");

    auto y = Tensor<float>::ones(Shape{2, 3});
    auto z = x + y;

    expect(std::fabs(z(1, 2) - 8.0f) < 1e-6f, "suma elemento a elemento fallo");

    auto reshaped = z.reshape(Shape{3, 2});
    expect(reshaped.shape() == Shape{3, 2}, "reshape invalido");

    expect_throw<std::invalid_argument>(
        [] {
            (void)Tensor<float>::zeros(Shape{2, 0});
        },
        "shape invalido no lanzo excepcion"
    );

    expect_throw<std::out_of_range>(
        [&] {
            (void)x(8, 0);
        },
        "acceso fuera de rango no lanzo excepcion"
    );

    expect_throw<std::invalid_argument>(
        [&] {
            (void)z.reshape(Shape{5, 2});
        },
        "reshape incompatible no lanzo excepcion"
    );

    std::puts("EPIC1_FEATURE1_OK");
    return 0;
}
