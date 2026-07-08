#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include "utec/apps/PatternClassifier.h"
#include "utec/io/dataset_loader.h"
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
  apps::PatternClassifier app;
  std::filesystem::create_directories("artifacts");
  {
    std::ofstream samples("artifacts/train_x.csv");
    samples << "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\n";
    samples << "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n";
    samples << "1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0\n";
    samples << "0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1\n";
  }
  {
    std::ofstream labels("artifacts/train_y.csv");
    labels << "1,0,0\n";
    labels << "1,0,0\n";
    labels << "0,1,0\n";
    labels << "0,0,1\n";
  }
  auto dataset = io::DatasetLoader::from_csv(
      "artifacts/train_x.csv",
      "artifacts/train_y.csv",
      Shape{4, 4, 1});
  app.load_dataset(dataset.inputs, dataset.targets);
  app.build_default_model(Shape{4, 4, 1}, 3);
  auto report = app.train(FitOptions{.epochs = 3, .batch_size = 4});
  expect(report.epochs_completed == 3, "entrenamiento incompleto");
  auto sample = Tensor<float>::ones(Shape{1, 4, 4, 1});
  auto prediction = app.predict(sample);
  expect(prediction.shape() == Shape{1, 3}, "prediccion con shape incorrecto");
  expect_throw<std::invalid_argument>(
      [&] {
        apps::PatternClassifier empty_app;
        empty_app.build_default_model();
        (void)empty_app.train(FitOptions{.epochs = 1, .batch_size = 1});
      },
      "la aplicacion debio fallar con dataset vacio");
  expect_throw<std::invalid_argument>(
      [&] {
        (void)io::DatasetLoader::from_csv(
            "artifacts/no_existe.csv",
            "artifacts/train_y.csv",
            Shape{4, 4, 1});
      },
      "la lectura de inputs debio fallar con archivo inexistente");
  {
    std::ofstream bad_labels("artifacts/bad_y.csv");
    bad_labels << "1,0\n";
    bad_labels << "0,1,0\n";
  }
  expect_throw<std::invalid_argument>(
      [&] {
        (void)io::DatasetLoader::from_csv(
            "artifacts/train_x.csv",
            "artifacts/bad_y.csv",
            Shape{4, 4, 1});
      },
      "la lectura de inputs debio fallar con etiquetas mal formadas");
  std::puts("FEATURE2_OK");
  return 0;
}
