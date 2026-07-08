#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace utec::tf::monitoring {

    struct EpochMetrics {
        float loss = 0.0f;
        float accuracy = 0.0f;
        int duration_ms = 0;
    };

    class TrainingMonitor {
    public:
        void on_epoch_end(int epoch_number, const EpochMetrics& metrics) {
            if (metrics_contain_nan(metrics)) {
                throw std::invalid_argument(
                    "las metricas de la epoca contienen un valor NaN");
            }
            if (epoch_already_registered(epoch_number)) {
                throw std::invalid_argument(
                    "la epoca ya fue registrada anteriormente");
            }
            registered_epoch_numbers.insert(epoch_number);
            epoch_numbers_in_order.push_back(epoch_number);
            recorded_metrics.push_back(metrics);
        }

        const std::vector<EpochMetrics>& history() const {
            return recorded_metrics;
        }

        std::string to_csv() const {
            std::ostringstream csv_output;
            csv_output << "epoch,loss,accuracy,duration_ms\n";
            for (size_t position = 0; position < recorded_metrics.size(); ++position) {
                const EpochMetrics& metrics = recorded_metrics[position];
                csv_output << epoch_numbers_in_order[position] << ","
                           << metrics.loss << ","
                           << metrics.accuracy << ","
                           << metrics.duration_ms << "\n";
            }
            return csv_output.str();
        }

    private:
        static bool metrics_contain_nan(const EpochMetrics& metrics) {
            return std::isnan(metrics.loss) || std::isnan(metrics.accuracy);
        }

        bool epoch_already_registered(int epoch_number) const {
            return registered_epoch_numbers.count(epoch_number) != 0;
        }

        std::vector<EpochMetrics> recorded_metrics;
        std::vector<int> epoch_numbers_in_order;
        std::unordered_set<int> registered_epoch_numbers;
    };

}

using namespace utec::tf;
