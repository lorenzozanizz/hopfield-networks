#pragma once
#ifndef RESERVOIR_HPP
#define RESERVOIR_HPP

#include <cmath>

#include "reservoir_logger.hpp"
// Import sparse matrix required for the intra-reservoir dynamics.
#include "../math/matrix/matrix_ops.hpp"

// Required for the mapping procedure.
#include "../io/datasets/dataset.hpp"

enum class SamplingType {
	Uniform, 
	Normal
};

template <typename DataType>
class Reservoir {
	
	using Matrix = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
	using Vector = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
	using SparseMatrix = Eigen::SparseMatrix<DataType>;

	// We allow the reservoir state and inputs to be matrices to allow full
	// operation on batches of data for dataset mapping!
	using ReservoirState = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
	using Input = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;

	// Matrix input_weights;
	Matrix input_weights;
	SparseMatrix echo_weights;
	ReservoirState state;
	Input input_state;

	std::vector<ReservoirLogger<DataType>*> loggers;
	
	unsigned int input_dim;
	unsigned int state_dim;

	bool has_input;

public:

	Reservoir(unsigned int input_dim, unsigned int state_dim, unsigned int batch_size = 1)
		: input_dim(input_dim), state_dim(state_dim) {
		input_weights.resize(state_dim, input_dim);
		echo_weights.resize(state_dim, state_dim);

		input_state.resize(input_dim, batch_size);
		state.resize(state_dim, batch_size);
		state.setZero();

		has_input = false;
	}

	void resize(unsigned int input_dim, unsigned int state_dim, unsigned int batch_size) {

		input_state.resize(input_dim, batch_size);
		state.resize(state_dim, batch_size);
		state.setZero();

	}

	void attach_logger(ReservoirLogger<DataType>* logger) {
		loggers.push_back(logger);
	}

	void detach_logger(ReservoirLogger<DataType>* logger) {
		loggers.erase(
			std::remove(loggers.begin(), loggers.end(), logger), loggers.end());
	}

	void initialize_echo_weights(
		double sparsity, // fraction of non-zero entries 
		SamplingType sampling, // Note: I think that this variable is never used in this scope
		double spectral_radius_desired = 0.99,
		unsigned long long seed = 0xcafebabe
		// Note: I would point out in a comment that near 1 it is a long memory, < 1 is fading and > 1 is unstable
	) {
		using Trip = Eigen::Triplet<DataType>;
		std::vector<Trip> triplets_vector;

		// Now fill in the triples vector with a number of values that is
		// in the order of sparsity * state_dim

		static thread_local std::mt19937 gen(seed);
		std::uniform_real_distribution<float> unif(0, 1);
		std::normal_distribution<DataType> norm(0, 1 / std::sqrt(state_dim) );
		

		triplets_vector.reserve(static_cast<long>(state_dim * state_dim * sparsity));
		for (int i = 0; i < state_dim; ++i)
			for (int j = 0; j < state_dim; ++j) {
				// Only put nonzero with probability sparsity
				if (unif(gen) < sparsity)
					triplets_vector.push_back(Trip(i, j, norm(gen)));
			}

		echo_weights.setFromTriplets(triplets_vector.begin(), triplets_vector.end());
		double current_radius = MathOps::sparse_power_method(echo_weights);

		std::cout << "Correction factor: " << spectral_radius_desired / current_radius;
		if (current_radius > 0)
			echo_weights *= (spectral_radius_desired / current_radius);

		// Compress the echo matrix finalizing it, no further additions
		// can be made. 
		echo_weights.makeCompressed(); // Note: I don't understand what this function does and why are we using it
	}
	
	void initialize_input_weights(
		SamplingType mode, 
		double variance = 1.0)
	{
		static thread_local std::mt19937 gen(std::random_device{}());

		if (mode == SamplingType::Uniform) {
			std::uniform_real_distribution<DataType> dist(-1, 1);
			for (int i = 0; i < state_dim; ++i)
				for (int j = 0; j < input_dim; ++j)
					input_weights(i, j) = dist(gen);
		}
		else {
			std::normal_distribution<DataType> dist(0.0, std::sqrt(variance));

			for (int i = 0; i < state_dim; ++i)
				for (int j = 0; j < input_dim; ++j)
					input_weights(i, j) = dist(gen);
		}
	}

	void feed(const Input& u) {
		// Save the value of the next input to be employed in the next
		// run. The value is invalidated at each call of run. 
		assert(u.rows() == static_cast<int>(input_dim)); 
		input_state = u;
		has_input = true;
	}

	void run() {
		ReservoirState pre_activation;
		if (has_input)
			pre_activation = echo_weights * state + input_weights * input_state;
		else
			pre_activation = echo_weights * state;
		// RELU activation function 
		state = pre_activation.array().cwiseMax(0.0f).matrix();
		// Delete the previous input, e.g. this is 1 time use only. 
		has_input = false;
		notify_on_norm_change();
		notify_on_state_change();
	}

	void reset() {
		state.setZero();
	}

	void map(const VectorDataset<Vector, Vector>& input, VectorDataset<Vector, Vector>& output,
		unsigned int batch_size, unsigned int dataset_input_size) {

		const int chunks_amt = static_cast<int>(std::ceil(float(dataset_input_size) / float(input_dim)));
		Matrix chunk = Matrix::Zero(input_dim, batch_size); // zero padded

		for (const auto& batch : input.batches(batch_size)) {
			// Reset the reservoir, setting the initial state to zero. This effectively begins a 
			// new temporal sequence for the batches. 
			reset();
			chunk.setZero();
			for (int c = 0; c < chunks_amt; ++c) {

				// The last chunk may be incomplete. 
				const int start = c * input_dim;
				const int len = std::min(input_dim, dataset_input_size - start);


				for (size_t i = 0; i < batch.size(); ++i) {
					if (len < input_dim)
						chunk.col(i).setZero();
					chunk.col(i).head(len) = batch.x_of(i).segment(start, len);
				}

				feed(chunk);
				run();
			}

			for (int i = 0; i < batch.size(); ++i) {
				output.add_sample(state.col(i), batch.y_of(i), batch.id_of(i));
			}
		}
		
	}

	void begin_run() {
		notify_on_run_begin();
	}

	void end_run() {
		notify_on_run_end();
	}

	ReservoirState& get_state() {
		return state;
	}

	SparseMatrix& get_echo_weights() {
		return echo_weights;
	}

	void notify_on_state_change() {
		for (auto* o : loggers) o->on_state_update( state );
	}

	void notify_on_norm_change() {
		auto norm = state.norm();
		for (auto* o : loggers) o->on_norm_update(norm);
	}

	void notify_on_run_begin() {
		for (auto* o : loggers) o->on_run_begin( state);
	}

	void notify_on_run_end() {
		for (auto* o : loggers) o->on_run_end();
	}

};

#endif