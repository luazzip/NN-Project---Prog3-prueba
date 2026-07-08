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

    model.add(layers::Input(Shape{8, 8, 1}));
    model.add(layers::Conv2D(4, {3, 3}, Activation::Relu));
    model.add(layers::MaxPooling2D({2, 2}));
    model.add(layers::Flatten());
    model.add(layers::Dense(3, Activation::Softmax));

    model.compile(
        optimizers::SGD(0.01f),
        losses::CategoricalCrossentropy{}
    );

    auto batch = Tensor<float>::ones(Shape{2, 8, 8, 1});
    auto pred = model.predict(batch);

    expect(pred.shape() == Shape{2, 3}, "predict shape incorrecto");

    auto params = model.parameters();

    expect(params.contains("conv2d_0/weights"), "faltan weights de Conv2D");
    expect(params.contains("conv2d_0/bias"), "faltan bias de Conv2D");
    expect(params.contains("dense_0/weights"), "faltan weights de Dense");
    expect(params.contains("dense_0/bias"), "faltan bias de Dense");
    expect(params.at("dense_0/bias").shape() == Shape{3}, "bias de Dense con shape incorrecto");

    expect_throw<std::invalid_argument>(
        [&] {
            auto wrong_batch = Tensor<float>::ones(Shape{2, 7, 7, 1});
            (void)model.predict(wrong_batch);
        },
        "predict debio fallar por shape incompatible"
    );

    expect_throw<std::invalid_argument>(
        [] {
            (void)layers::Conv2D(4, {0, 3}, Activation::Relu);
        },
        "Conv2D debio fallar por kernel invalido"
    );

    expect_throw<std::invalid_argument>(
        [] {
            (void)layers::MaxPooling2D({0, 2});
        },
        "MaxPooling2D debio fallar por ventana invalida"
    );

    std::puts("EPIC1_FEATURE3_OK");
    return 0;
}
