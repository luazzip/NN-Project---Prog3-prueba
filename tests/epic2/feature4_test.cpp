#include <filesystem>
#include <stdexcept>
#include <string_view>
#include "utec/monitoring/report_exporter.h"
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
  monitoring::ReportExporter exporter;
  exporter.add_run({
      .name = "cnn_baseline",
      .train_loss = {1.2f, 0.8f, 0.5f},
      .val_accuracy = {0.45f, 0.70f, 0.82f},
      .notes = "baseline con Conv2D + Dense"
  });
  exporter.add_run({
      .name = "cnn_augmented",
      .train_loss = {1.1f, 0.7f, 0.4f},
      .val_accuracy = {0.48f, 0.74f, 0.86f},
      .notes = "dataset con ruido y rotaciones leves"
  });
  exporter.write_markdown("docs/reporte_feature4.md");
  exporter.write_csv("docs/reporte_feature4.csv");
  expect(std::filesystem::exists("docs/reporte_feature4.md"), "no se genero markdown");
  expect(std::filesystem::exists("docs/reporte_feature4.csv"), "no se genero csv");
  expect_throw<std::logic_error>(
      [] {
        monitoring::ReportExporter empty;
        empty.write_markdown("docs/empty.md");
      },
      "write_markdown debio fallar sin corridas registradas");
  expect_throw<std::invalid_argument>(
      [] {
        monitoring::ReportExporter invalid;
        invalid.add_run({
            .name = "bad_run",
            .train_loss = {0.9f, 0.5f},
            .val_accuracy = {0.8f},
            .notes = "series inconsistentes"
        });
      },
      "add_run debio fallar con series inconsistentes");
  std::puts("FEATURE4_OK");
  return 0;
}
