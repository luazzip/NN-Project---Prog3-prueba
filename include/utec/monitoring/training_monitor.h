#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace utec::tf::monitoring {

struct EpochMetrics {
    float loss = 0.0f;
    float accuracy = 0.0f;
    int duration_ms = 0;
};

struct EpochRecord {
    int epoch = 0;
    float loss = 0.0f;
    float accuracy = 0.0f;
    int duration_ms = 0;
};

class TrainingMonitor {
private:
    std::vector<EpochRecord> records_;

public:
    void on_epoch_end(int epoch, EpochMetrics metrics) {
        if (epoch <= 0) {
            throw std::invalid_argument("epoch debe ser positivo");
        }
        if (!std::isfinite(metrics.loss) || !std::isfinite(metrics.accuracy)) {
            throw std::invalid_argument("metricas invalidas");
        }
        if (metrics.accuracy < 0.0f || metrics.accuracy > 1.0f) {
            throw std::invalid_argument("accuracy fuera de rango");
        }
        if (metrics.duration_ms < 0) {
            throw std::invalid_argument("duration_ms invalido");
        }
        for (const auto& record : records_) {
            if (record.epoch == epoch) {
                throw std::invalid_argument("epoch duplicada");
            }
        }

        records_.push_back(EpochRecord{epoch, metrics.loss, metrics.accuracy, metrics.duration_ms});
    }

    std::vector<EpochRecord> history() const {
        return records_;
    }

    std::string to_csv() const {
        std::ostringstream out;
        out << "epoch,loss,accuracy,duration_ms\n";
        for (const auto& record : records_) {
            out << record.epoch << ','
                << record.loss << ','
                << record.accuracy << ','
                << record.duration_ms << '\n';
        }
        return out.str();
    }
};

}
