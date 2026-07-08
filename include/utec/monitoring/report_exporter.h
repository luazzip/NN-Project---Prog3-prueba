#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace utec::tf::monitoring {

struct ReportRun {
    std::string name;
    std::vector<float> train_loss;
    std::vector<float> val_accuracy;
    std::string notes;
};

class ReportExporter {
private:
    std::vector<ReportRun> runs_;

    static void validate_run(const ReportRun& run) {
        if (run.name.empty()) {
            throw std::invalid_argument("nombre de corrida vacio");
        }
        if (run.train_loss.empty() || run.val_accuracy.empty()) {
            throw std::invalid_argument("series vacias");
        }
        if (run.train_loss.size() != run.val_accuracy.size()) {
            throw std::invalid_argument("series inconsistentes");
        }
    }

    static void ensure_parent_directory(const std::string& path) {
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
    }

public:
    void add_run(const ReportRun& run) {
        validate_run(run);
        runs_.push_back(run);
    }

    void write_markdown(const std::string& path) const {
        if (runs_.empty()) {
            throw std::logic_error("no hay corridas registradas");
        }

        ensure_parent_directory(path);
        std::ofstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("no se pudo crear markdown");
        }

        file << "# Reporte Feature 4\n\n";
        file << "| Corrida | Epocas | Loss final | Accuracy final | Notas |\n";
        file << "|---|---:|---:|---:|---|\n";
        for (const auto& run : runs_) {
            file << "| " << run.name << " | "
                 << run.train_loss.size() << " | "
                 << run.train_loss.back() << " | "
                 << run.val_accuracy.back() << " | "
                 << run.notes << " |\n";
        }
    }

    void write_csv(const std::string& path) const {
        if (runs_.empty()) {
            throw std::logic_error("no hay corridas registradas");
        }

        ensure_parent_directory(path);
        std::ofstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("no se pudo crear csv");
        }

        file << "name,epoch,train_loss,val_accuracy,notes\n";
        for (const auto& run : runs_) {
            for (size_t i = 0; i < run.train_loss.size(); ++i) {
                file << run.name << ','
                     << (i + 1) << ','
                     << run.train_loss[i] << ','
                     << run.val_accuracy[i] << ','
                     << run.notes << '\n';
            }
        }
    }
};

}
