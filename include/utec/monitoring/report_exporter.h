#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace utec::tf::monitoring {

    struct RunResult {
        std::string name;
        std::vector<float> train_loss;
        std::vector<float> val_accuracy;
        std::string notes;
    };

    class ReportExporter {
    public:
        void add_run(const RunResult& run) {
            if (run.train_loss.size() != run.val_accuracy.size()) {
                throw std::invalid_argument(
                    "train_loss y val_accuracy deben tener la misma cantidad de epocas");
            }
            recorded_runs.push_back(run);
        }

        void write_markdown(const std::string& path) const {
            fail_if_there_are_no_runs();
            create_parent_directory(path);
            std::ofstream file(path);
            if (!file.is_open()) {
                throw std::runtime_error("no se pudo crear el archivo markdown: " + path);
            }
            file << build_markdown_report();
        }

        void write_csv(const std::string& path) const {
            fail_if_there_are_no_runs();
            create_parent_directory(path);
            std::ofstream file(path);
            if (!file.is_open()) {
                throw std::runtime_error("no se pudo crear el archivo csv: " + path);
            }
            file << build_csv_report();
        }

    private:
        void fail_if_there_are_no_runs() const {
            if (recorded_runs.empty()) {
                throw std::logic_error("no hay corridas registradas para exportar");
            }
        }

        static void create_parent_directory(const std::string& path) {
            std::filesystem::path parent_directory = std::filesystem::path(path).parent_path();
            if (!parent_directory.empty()) {
                std::filesystem::create_directories(parent_directory);
            }
        }

        std::string build_markdown_report() const {
            std::ostringstream report;
            report << "# Reporte de experimentos\n\n";
            report << "| Corrida | Epocas | Loss final | Accuracy final | Notas |\n";
            report << "| --- | --- | --- | --- | --- |\n";
            for (const RunResult& run : recorded_runs) {
                report << "| " << run.name
                       << " | " << run.train_loss.size()
                       << " | " << final_value(run.train_loss)
                       << " | " << final_value(run.val_accuracy)
                       << " | " << run.notes
                       << " |\n";
            }
            return report.str();
        }

        std::string build_csv_report() const {
            std::ostringstream report;
            report << "name,epochs,final_train_loss,final_val_accuracy,notes\n";
            for (const RunResult& run : recorded_runs) {
                report << run.name << ","
                       << run.train_loss.size() << ","
                       << final_value(run.train_loss) << ","
                       << final_value(run.val_accuracy) << ","
                       << run.notes << "\n";
            }
            return report.str();
        }

        static float final_value(const std::vector<float>& series) {
            if (series.empty()) {
                return 0.0f;
            }
            return series.back();
        }

        std::vector<RunResult> recorded_runs;
    };

}

using namespace utec::tf;
