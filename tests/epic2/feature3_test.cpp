#include <filesystem>
#include <fstream>
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
  model.add(layers::Conv2D(2, {3, 3}, Activation::Relu));
  model.add(layers::Flatten());
  model.add(layers::Dense(2, Activation::Softmax));
  model.compile(optimizers::SGD(0.01f), losses::CategoricalCrossentropy{});
  auto x = Tensor<float>::ones(Shape{1, 8, 8, 1});
  auto before = model.predict(x);
  auto params_before = model.parameters();
  std::filesystem::create_directories("artifacts");
  const std::filesystem::path path = "artifacts/test_model.bin";
  model.save(path.string(), SaveOptions{
      .metadata = {
          {"input_shape", "8x8x1"},
          {"num_classes", "2"}
      }
  });
  expect(std::filesystem::exists(path), "no se guardo el modelo");
  auto restored = Sequential::load(path.string());
  auto after = restored.predict(x);
  auto params_after = restored.parameters();
  expect(allclose(before, after, 1e-5f), "la serializacion no preservo predicciones");
  expect(allclose(params_before.at("conv2d_0/weights"),
                  params_after.at("conv2d_0/weights"), 1e-5f),
         "los weights no se restauraron correctamente");
  expect(allclose(params_before.at("conv2d_0/bias"),
                  params_after.at("conv2d_0/bias"), 1e-5f),
         "los bias no se restauraron correctamente");
  expect(restored.metadata().at("input_shape") == "8x8x1", "metadata incorrecta");
  expect(restored.metadata().at("num_classes") == "2", "metadata incompleta");
  expect_throw<std::runtime_error>(
      [] { (void)Sequential::load("artifacts/no_existe.bin"); },
      "load debio fallar con archivo inexistente");
  {
    std::ofstream corrupted("artifacts/corrupted.bin");
    corrupted << "contenido-invalido";
  }
  expect_throw<std::runtime_error>(
      [] { (void)Sequential::load("artifacts/corrupted.bin"); },
      "load debio fallar con archivo corrupto");
  std::puts("FEATURE3_OK");
  return 0;
}
