#pragma once
#ifndef RESERVOIR_PREDICTOR_HPP
#define RESERVOIR_PREDICTOR_HPP

#include <vector>
#include <functional>
#include <random>

#include "../math/matrix/matrix_ops.hpp"

template <typename DataType>
class MultiLayerPerceptron {

public:

    using Mat = Eigen::Matrix< DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec = Eigen::Matrix< DataType, 1, Eigen::Dynamic>;

    struct Activation {
        std::function<Vec(const Vec&)> f;
        std::function<Vec(const Vec&)> df;
    };

    /*
    // Constructor: layer sizes + activations
    MultiLayerPerceptron(
        const std::vector<int>& layers,
        const std::vector<Activation>& activations
    ): layers(layers), activations(activations)
    {
        initalize_weights();
    }

    // Forward pass
    Vec forward(const Vec& input) {
        a[0] = input;
        for (int l = 1; l < L; ++l) {
            z[l] = W[l] * a[l - 1] + b[l];
            a[l] = activations[l - 1].f(z[l]);  // activation for layer l
        }
        return a[L - 1];
    }

    // Backpropagation: given dLoss/da(L)
    void backward(const Vec& dLoss_daL) {
        delta[L - 1] = dLoss_daL.cwiseProduct(activations[L - 2].df(z[L - 1]));

        for (int l = L - 2; l >= 1; --l) {
            delta[l] = (W[l + 1].transpose() * delta[l + 1])
                .cwiseProduct(activations[l - 1].df(z[l]));
        }

        for (int l = 1; l < L; ++l) {
            dW[l] = delta[l] * a[l - 1].transpose();
            db[l] = delta[l];
        }
    }

    // SGD update
    void apply_gradients(float lr) {
        for (int l = 1; l < L; ++l) {
            W[l] -= lr * dW[l];
            b[l] -= lr * db[l];
        }
    }

    // Accessors
    const std::vector<Mat>& get_weights() const { return W; }

private:
    std::vector<int> layers;
    std::vector<Activation> activations;

    int L;  // number of layers

    std::vector<Mat> W, dW;
    std::vector<Vec> b, db;
    std::vector<Vec> a, z, delta;

    void initalize_weights() {
        L = layers.size();

        W.resize(L);
        b.resize(L);
        dW.resize(L);
        db.resize(L);
        a.resize(L);
        z.resize(L);
        delta.resize(L);

        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 0.1f);

        for (int l = 1; l < L; ++l) {
            W[l] = Mat(layers[l], layers[l - 1]);
            b[l] = Vec(layers[l]);

            for (int i = 0; i < W[l].rows(); ++i)
                for (int j = 0; j < W[l].cols(); ++j)
                    W[l](i, j) = dist(gen);

            b[l].setZero();
        }
    }

    */
};


class ReservoirPredictor {

	unsigned int output_size;


};

#endif