#include <cmath>
#include <stdexcept>
#include <string_view>
#include "utec/monitoring/training_monitor.h"
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
  monitoring::TrainingMonitor monitor;
  monitor.on_epoch_end(1, {.loss = 0.91f, .accuracy = 0.60f, .duration_ms = 12});
  monitor.on_epoch_end(2, {.loss = 0.55f, .accuracy = 0.83f, .duration_ms = 10});
  auto history = monitor.history();
  expect(history.size() == 2, "historial incompleto");
  expect(std::fabs(history.back().loss - 0.55f) < 1e-6f, "loss final incorrecta");
  auto csv = monitor.to_csv();
  expect(csv.find("loss") != std::string::npos, "csv sin cabecera");
  expect_throw<std::invalid_argument>(
      [&] { monitor.on_epoch_end(3, {.loss = NAN, .accuracy = 0.2f, .duration_ms = 9}); },
      "monitor debio fallar por valor NaN");
  expect_throw<std::invalid_argument>(
      [&] { monitor.on_epoch_end(2, {.loss = 0.4f, .accuracy = 0.9f, .duration_ms = 8}); },
      "monitor debio fallar por epoca duplicada");
  std::puts("FEATURE1_OK");
  return 0;
}
