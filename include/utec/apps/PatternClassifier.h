#pragma once

#include "../nn/neural_network.h"

#include <stdexcept>

namespace utec::tf::apps {

struct TrainingReport {
    int epochs_completed = 0;
    float final_loss = 0.0f;
};

class PatternClassifier {
private:
    Tensor<float> inputs_;
    Tensor<float> targets_;
    Sequential model_;
    bool has_dataset_ = false;
    bool model_ready_ = false;

public:
    void load_dataset(const Tensor<float>& inputs,
                      const Tensor<float>& targets) {
        if (inputs.shape().rank() < 2 || targets.shape().rank() != 2) {
            throw std::invalid_argument("dataset con formas invalidas");
        }
        if (inputs.shape()[0] != targets.shape()[0]) {
            throw std::invalid_argument("inputs y targets con batch incompatible");
        }

        inputs_ = inputs;
        targets_ = targets;
        has_dataset_ = true;
    }

    void build_default_model(const Shape& sample_shape = Shape{4, 4, 1},
                             int num_classes = 3) {
        if (num_classes <= 0) {
            throw std::invalid_argument("num_classes debe ser positivo");
        }
        if (sample_shape.rank() != 3) {
            throw std::invalid_argument("sample_shape debe ser {H,W,C}");
        }
        if (sample_shape[0] < 3 || sample_shape[1] < 3) {
            throw std::invalid_argument("sample_shape demasiado pequeno para Conv2D 3x3");
        }

        model_ = Sequential{};
        model_.add(layers::Input(sample_shape));
        model_.add(layers::Conv2D(4, {3, 3}, Activation::Relu));
        model_.add(layers::Flatten());
        model_.add(layers::Dense(num_classes, Activation::Softmax));
        model_.compile(optimizers::SGD(0.01f), losses::CategoricalCrossentropy{});
        model_ready_ = true;
    }

    TrainingReport train(FitOptions options) {
        if (!has_dataset_) {
            throw std::invalid_argument("dataset vacio");
        }
        if (!model_ready_) {
            build_default_model();
        }

        auto history = model_.fit(inputs_, targets_, options);
        TrainingReport report;
        report.epochs_completed = static_cast<int>(history.loss.size());
        if (!history.loss.empty()) {
            report.final_loss = history.loss.back();
        }
        return report;
    }

    Tensor<float> predict(const Tensor<float>& sample) {
        if (!model_ready_) {
            throw std::logic_error("modelo no construido");
        }
        return model_.predict(sample);
    }

    Sequential& model() {
        return model_;
    }
};

}
