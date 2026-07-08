#pragma once

#include "../nn/neural_network.h"

#include <stdexcept>
#include <utility>

namespace utec::tf::apps {

    struct TrainReport {
        int epochs_completed = 0;
    };

    class PatternClassifier {
    public:
        void load_dataset(const Tensor<float>& inputs, const Tensor<float>& targets) {
            dataset_inputs = inputs;
            dataset_targets = targets;
            dataset_loaded = true;
        }

        void build_default_model(const Shape& sample_shape, int number_of_classes) {
            Sequential fresh_model;
            fresh_model.add(layers::Input(sample_shape));
            fresh_model.add(layers::Conv2D(4, {3, 3}, Activation::Relu));
            fresh_model.add(layers::Flatten());
            fresh_model.add(layers::Dense(number_of_classes, Activation::Softmax));
            fresh_model.compile(optimizers::SGD(0.01f), losses::CategoricalCrossentropy{});

            model = std::move(fresh_model);
            model_built = true;
        }

        void build_default_model() {
            build_default_model(Shape{4, 4, 1}, 3);
        }

        TrainReport train(FitOptions options) {
            if (!model_built) {
                throw std::logic_error("el modelo no ha sido construido");
            }
            if (!dataset_loaded || dataset_inputs.size() == 0) {
                throw std::invalid_argument("no hay un dataset cargado para entrenar");
            }

            Sequential::History history = model.fit(dataset_inputs, dataset_targets, options);

            TrainReport report;
            report.epochs_completed = static_cast<int>(history.loss.size());
            return report;
        }

        Tensor<float> predict(const Tensor<float>& sample) {
            if (!model_built) {
                throw std::logic_error("el modelo no ha sido construido");
            }
            return model.predict(sample);
        }

    private:
        Sequential model;
        Tensor<float> dataset_inputs;
        Tensor<float> dataset_targets;
        bool dataset_loaded = false;
        bool model_built = false;
    };

}

using namespace utec::tf;
