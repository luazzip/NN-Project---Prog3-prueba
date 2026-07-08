#include <stdexcept>
#include <string_view>
#include "utec/nn/neural_network.h"

using namespace utec::tf;

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
    Sequential model;

    model.add(layers::Input(Shape{6, 6, 1}));
    model.add(layers::Conv2D(2, {3, 3}, Activation::Relu));
    model.add(layers::MaxPooling2D({2, 2}));
    model.add(layers::Flatten());
    model.add(layers::Dense(2, Activation::Softmax));

    model.compile(
        optimizers::SGD(0.05f),
        losses::CategoricalCrossentropy{}
    );

    auto x = Tensor<float>::from_data(
        Shape{4, 6, 6, 1},
        {
            1,1,1,0,0,0,
            1,1,1,0,0,0,
            1,1,1,0,0,0,
            1,1,1,0,0,0,
            1,1,1,0,0,0,
            1,1,1,0,0,0,

            0,0,0,1,1,1,
            0,0,0,1,1,1,
            0,0,0,1,1,1,
            0,0,0,1,1,1,
            0,0,0,1,1,1,
            0,0,0,1,1,1,

            1,1,1,0,0,0,
            1,1,1,0,0,0,
            1,1,1,0,0,0,
            1,1,1,0,0,0,
            1,1,1,0,0,0,
            1,1,1,0,0,0,

            0,0,0,1,1,1,
            0,0,0,1,1,1,
            0,0,0,1,1,1,
            0,0,0,1,1,1,
            0,0,0,1,1,1,
            0,0,0,1,1,1
        }
    );

    auto y = Tensor<float>::from_data(
        Shape{4, 2},
        {
            1, 0,
            0, 1,
            1, 0,
            0, 1
        }
    );

    auto params_before = model.parameters();
    auto before = model.evaluate(x, y).loss;

    auto history = model.fit(x, y, FitOptions{.epochs = 30, .batch_size = 2});

    auto after = model.evaluate(x, y).loss;
    auto params_after = model.parameters();

    expect(history.loss.size() == 30, "history incompleto");
    expect(after < before, "la perdida no disminuyo");

    auto gradients = model.last_gradients();

    expect(!gradients.empty(), "no se registraron gradientes");
    expect(gradients.contains("conv2d_0/weights"), "faltan gradientes de conv weights");
    expect(gradients.contains("conv2d_0/bias"), "faltan gradientes de conv bias");
    expect(gradients.contains("dense_0/weights"), "faltan gradientes de dense weights");
    expect(gradients.contains("dense_0/bias"), "faltan gradientes de dense bias");

    expect(
        !allclose(
            params_before.at("conv2d_0/weights"),
            params_after.at("conv2d_0/weights"),
            1e-6f
        ),
        "los weights de Conv2D no se actualizaron"
    );

    expect(
        !allclose(
            params_before.at("conv2d_0/bias"),
            params_after.at("conv2d_0/bias"),
            1e-6f
        ),
        "los bias de Conv2D no se actualizaron"
    );

    expect(
        !allclose(
            params_before.at("dense_0/weights"),
            params_after.at("dense_0/weights"),
            1e-6f
        ),
        "los weights de Dense no se actualizaron"
    );

    expect(
        !allclose(
            params_before.at("dense_0/bias"),
            params_after.at("dense_0/bias"),
            1e-6f
        ),
        "los bias de Dense no se actualizaron"
    );

    expect_throw<std::invalid_argument>(
        [&] {
            auto wrong_y = Tensor<float>::ones(Shape{4, 3});
            (void)model.fit(x, wrong_y, FitOptions{.epochs = 1, .batch_size = 2});
        },
        "fit debio fallar por etiquetas incompatibles"
    );

    expect_throw<std::logic_error>(
        [] {
            Sequential bad_model;
            bad_model.add(layers::Input(Shape{4}));
            bad_model.add(layers::Dense(2, Activation::Softmax));
            (void)bad_model.backward();
        },
        "backward debio fallar en un modelo no compilado"
    );

    std::puts("EPIC1_FEATURE4_OK");
    return 0;
}
