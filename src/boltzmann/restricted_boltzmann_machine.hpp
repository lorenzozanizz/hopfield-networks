#pragma once
#ifndef RESTRICTED_BOLTZMANN_MACHINE_HPP
#define RESTRICTED_BOLTZMANN_MACHINE_HPP

#include <cmath>
#include <vector>
#include <random>
#include <vector>
#include <cassert>

#include "../io/io_utils.hpp"
#include "../io/plot/plot.hpp"
#include "../io/datasets/dataset.hpp"
// Import Eigen needed for the operations. 
#include "../math/matrix/matrix_ops.hpp"

#include "boltzmann_logger.hpp"

/**
 * @brief A simple calculator class.
 */
template <typename FloatingType>
class RestrictedBoltzmannMachine {

    using Vector = Eigen::Matrix<FloatingType, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<FloatingType, Eigen::Dynamic, Eigen::Dynamic>;

    using State = Vector;

    int nv, nh;
    Matrix weights;     // nv x nh
    Vector b_v;    // visible biases
    Vector b_h;    // hidden biases

    State hidden;
    State visible;
    double energy;

    // We use this to perform contrastive divergence and the
    // gibb sampling of the state. 
    std::mt19937 rng;
    std::uniform_real_distribution<double> uni;

    std::vector<BoltzmannLogger<FloatingType>*> loggers;

    // These are to be used as buffers when computing the cd-k algorithm,
    // this avoids the overhead of re-allocating the local fields. 

    Matrix visible_local_fields;
    Matrix hidden_local_fields;

public:

    enum class LayerType {
        Hidden,
        Visible
    };

    RestrictedBoltzmannMachine(int num_vis, int num_hidden)
        : nv(num_vis), nh(num_hidden), rng(std::random_device{ }()),
        uni(0.0, 1.0) {
        weights.resize(num_vis, num_hidden);
        b_v.resize(num_vis); b_h.resize(num_hidden);
        
        // Although both the hidden and visible states are fully binary
        // internally, the algorithm is very math-heavy so that converting
        // to a floating type array every time is too expensive. 
        // We just store them into a float vector and convert them 
        // back into binary for logging routines. 
        hidden.resize(num_hidden);
        visible.resize(num_vis);
    }

    unsigned int hidden_size() const {
        return nh;
    }

    unsigned int visible_size() const {
        return nv;
    }

    /**
     * @brief Resize the size of the internal local fields to fit the input batch size
     */
    void resize_local_fields(unsigned int batch_size) {
        hidden_local_fields.resize(nh, batch_size);
        visible_local_fields.resize(nv, batch_size);
    }

    /**
     * @brief Initializes the weight to be small normally distributed
     */
    void initialize_weights(double std) {
        // Randomly initializes the weights. 
        std::normal_distribution<FloatingType> dist(0.0, std);
        for (int i = 0; i < weights.rows(); ++i)
            for (int j = 0; j < weights.cols(); ++j)
                weights(i, j) = dist(rng);
        b_v.setZero();
        b_h.setZero();
    }

    /**
     * @brief A simple calculator class.
     */
    void seed(unsigned long long seed) {
        rng.seed(seed);
    }

    void attach_logger(BoltzmannLogger<FloatingType>* logger) {
        loggers.push_back(logger);
    }

    void detach_logger(BoltzmannLogger<FloatingType>* logger) {
        loggers.erase(
            std::remove(loggers.begin(), loggers.end(), logger), loggers.end());
    }


    template <typename AggregateType>
    // sample hidden from visible
    void sample_hidden(const AggregateType& visible, AggregateType& hidden, bool do_sample, 
        double temperature = 1.0) {
        // We stored the matrix w as a (visible, hidden) matrix so  we need 
        // to transpose w to compute the hidden local fields
        hidden_local_fields.noalias() = (weights.transpose() * visible).colwise() + b_h;
        // Compute the sigmoid. 
        auto& probs = hidden_local_fields;
        probs = (1.0 / (1.0 + (-hidden_local_fields.array() / temperature).exp())).matrix();

        // Eigen optimizes this writing inline. 
        if (do_sample)
            hidden = probs.unaryExpr([this](FloatingType p) {
                return (this->uni(this->rng) < p) ? FloatingType(1.0) : FloatingType(0);
            });
        else
            hidden = probs;
        return;
    }

    template <typename AggregateType>
    // Sample visible units from the hidden units using the 
    // conditional probability p( v | h = s_h ), as described in 
    // Mehlig
    void sample_visible(AggregateType& visible, const AggregateType& hidden, bool do_sample,
        double temperature = 1.0) {
        // We stored the matrix w as a (visible, hidden) matrix so this can
        // be done, vice versa we need to transpose w to compute the hidden local fields
        visible_local_fields.noalias() = (weights * hidden).colwise() + b_v;
        // Compute the sigmoid. 
        auto& probs = visible_local_fields;
        probs = (1.0 / (1.0 + (-visible_local_fields.array() / temperature).exp())).matrix();

        if (do_sample)
            // ------ NOTE: This sampling is safe because eigen never parallelizes this kind of 
            // operations which are memory bound (as per documentation)
            visible = probs.unaryExpr([this](FloatingType p) {
                return (this->uni(this->rng) < p) ? FloatingType(1.0) : FloatingType(0);
            });
        else 
            visible = probs;
        return;
    }


    void clamp_visible(const State& state) {
        // Feed a pattern in the network. 
        visible = state;
    }

    void random_visible(double prob_on = 0.5) {
        for (int i = 0; i < nv; ++i)
            visible(i) = (uni(rng) < prob_on) ? 1.0 : 0;
    }

    void run_cd_k(unsigned int k, bool activate_final = false, double temperature = 1.0) {
        // Run the update on the network for k times, using the marginal
        // conditional distributions. This acts on the hidden and state vectors,
        // updating the notify

        notify_on_run_begin( this->visible );
        for (unsigned int k_it = 0; k_it < k; ++k_it) {
            sample_hidden(this->visible, this->hidden, true, temperature);
            sample_visible(this->visible, this->hidden, (k_it < k-1)? true : activate_final, temperature);
            notify_on_visible_change(this->visible);
        }
        notify_on_run_end();
    }

    Matrix& get_weights() {
        return weights;
    }

    // https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    // CD-k training on a collection of vectorial inputs, WE DO NOT store
    // images as binary states anymore as in hopfield because the computations
    // are much much heavier. We batch the training, unlike Mehlig
    void train_cd(
        int epochs,
        VectorCollection<Vector>& data,
        unsigned int batch_size,
        double lr,
        int k /* K parameter of the contrastive divergence algorithm. */,
        double decay = 1e-4
    ) {
        int num_samples = data.size();

        Matrix batch_visible(nv, batch_size);
        Matrix logging_batch(nv, batch_size);

        Matrix batch_hidden(nh, batch_size);

        Matrix negative_batch_visible(nv, batch_size);
        Matrix negative_batch_hidden(nh, batch_size);

        // Resize the buffered local fields matrices to avoid overhead of reallocation
        this->resize_local_fields(batch_size);

        MultiProgressBar prog_bar(epochs);
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            prog_bar.update(epoch);
            prog_bar.print_intermediate("Magnitude of the weights: " + std::to_string(weights.norm()));
            // For each epoch we need to take into account all patterns
            // in the training. 
            data.shuffle();
            // For each epoch perform the following train iteration. 
            bool first_batch = false;
            for (auto batch : data.batches(batch_size)) {

                // First we build the batched input matrix (Note that the i iterations are
                // batch local indices, not global indices!)
                for (int i = 0; i < batch_size; ++i) {
                    batch_visible.col(i) = batch.x_of(i);
                    negative_batch_visible.col(i) = batch.x_of(i);
                }
                if (!first_batch) {
                    first_batch = true;
                    logging_batch = batch_visible;
                    sample_hidden(logging_batch, batch_hidden, true);
                    sample_visible(logging_batch, batch_hidden, false);

                    auto loss = (batch_visible - logging_batch).norm();
                    prog_bar.print_intermediate("Retrieval loss: " + std::to_string(loss));
                }

                // This is the positive phase, DO  Sample the hidden states!
                sample_hidden(batch_visible, batch_hidden, /* do_sample */ true);

                // NOW we use CD-K to compute the negative phase. 
                negative_batch_hidden = batch_hidden;
                sample_visible(negative_batch_visible, negative_batch_hidden, /* do sample */ false);
                for (int cd_step = 0; cd_step < k; ++cd_step) {
                    sample_hidden(negative_batch_visible, negative_batch_hidden, false);
                    sample_visible(negative_batch_visible, negative_batch_hidden, /* do sample */ false);
                }
                sample_hidden(negative_batch_visible, negative_batch_hidden, false);
                sample_visible(negative_batch_visible, negative_batch_hidden, /* do sample */ true);

                // Apply the positive and negative updates, noting that when we multiply
                // together the hidden and visible states we are effectively accumulating the
                // update over the entire batch and we need to normalize by /batch_size
                weights += lr * (batch_visible * batch_hidden.transpose()) / batch_size;
                b_v += lr * batch_visible.rowwise().mean();
                b_h += lr * batch_hidden.rowwise().mean();

                // negative phase
                weights -= lr * (negative_batch_visible * negative_batch_hidden.transpose()) / batch_size;
                b_v -= lr * negative_batch_visible.rowwise().mean();
                b_h -= lr * negative_batch_hidden.rowwise().mean();
                weights *= (1 - lr * decay);

            }
        }
    }

    void map_into_hidden(
        const VectorCollection<Vector>& data_input,
        VectorCollection<Vector>& data_output,
        unsigned int batch_size
    ) {

        Matrix batch_visible(nv, batch_size);
        Matrix batch_hidden(nh, batch_size);

        // Do not shuffle the dataset, this is not about SGD. 
        for (auto batch : data_input.batches(batch_size)) {

            // First we build the batched input matrix (Note that the i iterations are
            // batch local indices, not global indices!)
            for (int i = 0; i < batch_size; ++i) {
                batch_visible.col(i) = batch.x_of(i);
            }
            sample_hidden(batch_visible, batch_hidden, false);

            for (int i = 0; i < batch_size; ++i) {
                data_output.add_sample(batch_hidden.col(i), batch.id_of(i));
            }
        }

    }

    double compute_energy() {
        // From Mehlig, p. 67
        energy = -visible.dot(weights * hidden) - visible.dot(b_v) - hidden.dot(b_h);
        // Notify the logging function to visualize the energy. 
        return energy;
    }

    void dump_weights(const std::string into) {
        MathOps::save_matrix_binary(into, weights);
    }

    void load_weights(const std::string from) {
        MathOps::load_matrix_binary(from, weights);
    }

    void plot_state(Plotter& p, unsigned int width, unsigned int height) {
        auto ctx = p.context();
        ctx.set_title("State").show_heatmap(visible.data(), width, height, "gray");
    }

    void plot_kernel(Plotter& p, unsigned int hidden_index, unsigned int width, unsigned int height) {
        // This visualizes the kernel learned by the weights as heatmaps
        // with the size of the visible layer, this represent patterns to which
        // "the hidden states are sensible"
        auto ctx = p.context();
        ctx.set_title("Kernel").show_heatmap(weights.col(hidden_index).data(), width, height, "gray");
    }

    void plot_higher_order_kernel(Plotter& p, unsigned int hidden_index, unsigned int width,
        unsigned int height, RestrictedBoltzmannMachine<FloatingType>& machine) {
        // Use this as a weighted linear combination of the kernels of the lower order machine
        auto& lower_weights_mat = machine.get_weights();

        // NOTE: The matrix of the higher order is (visible_higher x hidden_higher )
        // and the matrix of the lower order is (hidden_higher, hidden_lower ) so we take 
        // a mat multiplication. 

        Vector weighted_sum = lower_weights_mat * weights.col(hidden_index);
        auto ctx = p.context();
        ctx.set_title("Higher Kernel").show_heatmap(weighted_sum.data(), width, height, "gray");
    }

    void compute_higher_order_kernels( Matrix& lower_semantic_weights, Matrix& out_location ) {
        // This function computes the higher order kernels for an upper layer in a DBN,
        // extending the above procedure to a matrix .

        // We assume that the lower machine is a machine with a compatible setup for the layer sizes, 
        // meaning that if this-> machine has weights ( hidden x hidden_upper ), the upper machine has weights 
        // ( visible_before x hidden )  
        // the product obtains hidden_upper kernels of size visible_before to be reinterpreted as one wishes. 
        out_location.noalias() = lower_semantic_weights * weights;

    }

    void notify_on_visible_change(const State& new_state) {
        for (auto* o : loggers) o->on_visible_change(new_state);
    }
    void notify_on_run_end() {
        for (auto* o : loggers) o->on_run_end();
    }

    void notify_on_run_begin(const State& initial_state) {
        for (auto* o : loggers) o->on_run_begin(initial_state);
    }

};



#endif