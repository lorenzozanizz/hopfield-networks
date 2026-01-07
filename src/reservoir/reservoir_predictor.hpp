#pragma once
#ifndef RESERVOIR_PREDICTOR_HPP
#define RESERVOIR_PREDICTOR_HPP

#include <vector>
#include <functional>
#include <optional>
#include <random>

// To collect and plot information about the loss functions
#include "../io/plot/plot.hpp"
#include "../io/datasets/data_collection.hpp"
#include "../io/datasets/dataset.hpp"

// To compute the loss gradient automatically.
#include "../math/autograd/variables.hpp"
#include "../math/autograd/functions.hpp"
#include "../math/matrix/matrix_ops.hpp"

template <typename DataType>
class ActivationFunction {
    
public:

    using Matrix = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;

    std::function<Matrix(const Matrix&)> f;
    std::function<Matrix(const Matrix&)> df;
};

template <typename DataType>
class Activations {

public:

    using Matrix = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;

    static const ActivationFunction<DataType> sigmoid;
    static const ActivationFunction<DataType> tanh;
    static const ActivationFunction<DataType> relu;
    static const ActivationFunction<DataType> identity;

};

template <typename DataType>
const ActivationFunction<DataType> Activations<DataType>::sigmoid = {
    // f(x)
    [](const Matrix& x) -> Matrix {
        return (1.0 / (1.0 + (-x.array()).exp())).matrix();
    },
    // df(x)
    [](const Matrix& x) -> Matrix {
        auto s = (1.0 / (1.0 + (-x.array()).exp()));
        return (s * (1.0 - s)).matrix();
    }
};

template <typename DataType>
const ActivationFunction<DataType> Activations<DataType>::identity = {
    // f(x)
    [](const Matrix& x) -> Matrix {
        return x;
    },
    // df(x)
    [](const Matrix& x) -> Matrix {
        return Matrix::Ones(x.rows(), x.cols());
    }
};

template <typename DataType>
const ActivationFunction<DataType> Activations<DataType>::tanh = {
    [](const Matrix& x) -> Matrix {
        return x.array().tanh().matrix();
    },
    [](const Matrix& x) -> Matrix {
        return (1.0 - x.array().tanh().square()).matrix();
    }
};

template <typename DataType>
const ActivationFunction<DataType> Activations<DataType>::relu = {
    [](const Matrix& x) -> Matrix {
        return x.array().max(DataType(0)).matrix();
    },
    [](const Matrix& x) -> Matrix {
        return (x.array() > DataType(0)).template cast<DataType>().matrix();
    }
};

template <typename DataType>
class MultiLayerPerceptron {

    using Activation = ActivationFunction<DataType>;
    using Matrix = Eigen::Matrix< DataType, Eigen::Dynamic, Eigen::Dynamic>;
    // This is a COLUMN vector! otherwise eigen cries
    using Vector = Eigen::Matrix< DataType, Eigen::Dynamic, 1>;

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

    unsigned int input_size() const {
        return layer_sizes[0];
    }
    
    unsigned int output_size() const {
        return layer_sizes[num_layers - 1];
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

    const std::vector<Matrix>& get_weights() const { 
        return W; 
    }

private:

    void initialize_weights() {

        // Resize all the vector with the correct sizes. Note that this does
        // not yet allocate the memory for the eigen matrices!
        W.resize(num_layers); b.resize(num_layers); dW.resize(num_layers); 
        db.resize(num_layers); A.resize(num_layers); Z.resize(num_layers); 
        delta.resize(num_layers);

        std::mt19937 gen(std::random_device{}());

        for (int l = 1; l < num_layers; ++l) {

            int rows = layer_sizes[l];
            int cols = layer_sizes[l - 1];
            // We use a glorot kind initialization strategy so that the weights
            // are zero-mean normal with std_dev given by 1/sqrt(input size)
            float stddev = 2.0f / std::sqrt( rows + cols );
            std::normal_distribution<DataType> dist(0.0, stddev);

            W[l].resize(rows, cols);
            b[l].resize(rows);

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    W[l](i, j) = dist(gen);

            // Zero initialize the bias in this implementation. 
            b[l].setZero();
        }
    }
};


template <typename DataType>
class NetworkTrainer {

    using Matrix = Eigen::Matrix< DataType, Eigen::Dynamic, Eigen::Dynamic>;
    // This is a COLUMN vector! otherwise eigen cries
    using Vector = Eigen::Matrix< DataType, Eigen::Dynamic, 1>;

    using Network = MultiLayerPerceptron<DataType>;
    using LossFunction = autograd::ScalarFunction<DataType>;
    using LossGradient = autograd::VectorFunction<DataType>;
    using LastOutput = autograd::VectorVariable; 

    Network& network;

    LossFunction* loss_function;
    LossGradient gradient_generator;
    autograd::EvalMap<float> loss_eval_map;

    LastOutput layer_variable;
    LastOutput ground_truth_variable;

    bool record_loss;
    bool record_verif_loss;

    // To contain information obtained during the training.
    NamedVectorCollection<DataType> nvc;

public:

    NetworkTrainer( Network& network ): network(network), record_loss(false),
        record_verif_loss(false), loss_function(nullptr) {
        nvc.clear();
    }

    void set_loss_function(LossFunction* loss, LastOutput var, LastOutput ground_truth) {
        loss_function = loss;
        loss_function->derivative(gradient_generator, var);
        layer_variable = var;
        ground_truth_variable = ground_truth;
    }

    void feed_additional_loss_variables(autograd::VectorVariable var, Vector& vector) {
        loss_eval_map.emplace(var, vector);
    }

    using VectorType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using Dataset = VectorDataset< VectorType, VectorType >;
    // Avoid something fancy like a generator expression for the 
    // datasets to avoiding the handling of too much data at once, even though
    // it would be a good idea in principle. 
    void train(
        // The dataset
        unsigned int epochs,
        Dataset& data,
        // An optional verification dataset
        std::optional< std::reference_wrapper<Dataset> > verification_data,
        unsigned int batch_size,
        double lr
    ) {

        using namespace Eigen;
        if (data.size() < batch_size)
            throw std::runtime_error("Batch size too large for the dataset!");
        if (record_verif_loss && !verification_data)
            throw std::runtime_error("Cannot log the verification data, no verification dataset was provided!");

        // Declare all the batch memory here to avoid reallocation
        Matrix batch_input(network.input_size(), batch_size);
        Matrix batch_gradient(network.output_size(), batch_size);
        Matrix batch_output(network.output_size(), batch_size);

        for (unsigned int epoch = 0; epoch < epochs; ++epoch)
        {
            data.shuffle();
            // For each epoch perform the following train iteration. 
            for (auto batch : data.batches(batch_size)) {
                // First we build the batched input matrix (Note that the i iterations are
                // batch local indices, not global indices!)
                for (size_t i = 0; i < batch.size(); ++i) {
                    batch_input.col(i) = batch.x_of(i);
                }
                batch_output = network.forward(batch_input);
                // Then we construct the batched gradient matrix
                for (size_t i = 0; i < batch.size(); ++i) {
                    VectorType out_col = batch_output.col(i);
                    VectorType true_col = batch.y_of(i);
                    loss_eval_map.clear();

                    loss_eval_map.emplace(layer_variable, out_col);
                    loss_eval_map.emplace(ground_truth_variable, true_col);
                    gradient_generator(loss_eval_map, batch_gradient.col(i));
                }
                // We then perform backpropagation 
                network.backward(batch_gradient);

                // And apply the gradient (averaged, so no lr/batch)
                network.apply_gradients( lr );
            }
            // These are logging routines to aid with training

            if (record_loss) {
                notify_loss( compute_loss_over_dataset( data) );
            }
            if (record_verif_loss) {
                notify_verification_loss( compute_loss_over_dataset( *verification_data ) );
            }
        }

        network.apply_gradients(lr);
    }

    void notify_loss(double loss) {

    }

    void notify_verification_loss(double loss) {

    }

    void do_log_loss(bool value) {
        record_loss = value;
        if (value)
            nvc.register_name("Loss");
    }

    void do_log_verification_loss(bool value) {
        record_verif_loss = true;
        if (value)
            nvc.register_name("Verification loss");
    }

    void plot_loss(Plotter& plotter) {
        if (record_loss || record_verif_loss) {

        }
    }

protected:

    double compute_loss_over_dataset(Dataset& data) {
        // Use the current state of the weights to compute the loss
        double total_loss = 0.0;
        const auto n_size = data.size();
        for (int i = 0; i < n_size; ++i) {
            loss_eval_map.clear();
        }
        return 1 / n_size;
    }

};

#endif