#pragma once
#ifndef RESERVOIR_PREDICTOR_HPP
#define RESERVOIR_PREDICTOR_HPP

#include <vector>
#include <functional>
#include <random>

#include "../math/matrix/matrix_ops.hpp"

template <typename DataType>
class MultiLayerPerceptron {

    using Matrix = Eigen::Matrix< DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix< DataType, 1, Eigen::Dynamic>;

    struct Activation {
        std::function<Vector(const Vector&)> f;
        std::function<Vector(const Vector&)> df;
    };

    std::vector<int> layer_sizes;
    std::vector<Activation> act_functions;

    int num_layers;  // number of layers

    // Parameters
    std::vector<Matrix> W, dW;
    std::vector<Vector> b, db;

    // Forward pass buffers
    std::vector<Matrix> A;   // activations
    std::vector<Matrix> Z;   // pre-activations

    // Backprop buffers
    std::vector<Matrix> delta;

public:

    MultiLayerPerceptron(
        const std::vector<int>& layers,
        const std::vector<Activation>& activations
    ) : layer_sizes(layers), act_functions(activations)
    {
        assert(activations.size() == layers.size() - 1);
        num_layers = layers.size();
        initialize_weights();
    }

    // Forward pass: X is (input_dim × batch_size)
    Matrix forward(const Matrix& X) {
        // The first layer has no activations ( we can jut include them 
        // in any preprocessing we need)
        A[0] = X;

        for (int l = 1; l < num_layers; ++l) {
            // Z[l] = W[l] * A[l-1] + b[l] (broadcasted with eigen)
            Z[l] = (W[l] * A[l - 1]).colwise() + b[l];
            A[l] = act_functions[l - 1].f(Z[l]);
        }
        return A[num_layers - 1];
    }

    void backward(const Matrix& dLoss_dA_L) {
        int batch_size = dLoss_dA_L.cols();

        // Output layer delta
        delta[num_layers - 1] =
            dLoss_dA_L.cwiseProduct(act_functions[num_layers - 2].df(Z[num_layers - 1]));

        // Hidden layers
        for (int l = num_layers - 2; l >= 1; --l) {
            delta[l] =
                (W[l + 1].transpose() * delta[l + 1])
                .cwiseProduct(act_functions[l - 1].df(Z[l]));
        }

        // Gradients
        for (int l = 1; l < num_layers; ++l) {
            dW[l] = (delta[l] * A[l - 1].transpose()) / batch_size;
            db[l] = delta[l].rowwise().mean();
        }
    }

    // SGD update, the batch normalization factor is absorbed by the learning
    // rate as standard.
    void apply_gradients(DataType lr) {
        for (int l = 1; l < num_layers; ++l) {
            W[l] -= lr * dW[l];
            b[l] -= lr * db[l];
        }
    }

    const std::vector<Matrix>& get_weights() const { return W; }

private:

    void initialize_weights() {

        W.resize(num_layers); b.resize(num_layers); dW.resize(num_layers); db.resize(num_layers);
        A.resize(num_layers); Z.resize(num_layers); delta.resize(num_layers);

        std::mt19937 gen(std::random_device{}());

        for (int l = 1; l < num_layers; ++l) {

            int rows = layer_sizes[l];
            int cols = layer_sizes[l - 1];
            float stddev = 1.0f / std::sqrt(layer_sizes[l - 1]);
            std::normal_distribution<DataType> dist(0.0, stddev);

            W[l].resize(rows, cols);
            b[l].resize(rows);

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    W[l](i, j) = dist(gen);

            b[l].setZero();
        }
    }
};


class ReservoirPredictor {

	unsigned int output_size;


};

#endif