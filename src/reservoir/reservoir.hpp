#pragma once
#ifndef RESERVOIR_HPP
#define RESERVOIR_HPP

#include "reservoir_logger.hpp"
// Import sparse matrix required for the intra-reservoir dynamics.
#include "../math/matrix/matrix_ops.hpp"

#include "../io/datasets/dataset.hpp"

enum class SamplingType {
	Uniform, 
	Normal
};

template <typename DataType>
class Reservoir {
	
	using Matrix = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
	using SparseMatrix = Eigen::SparseMatrix<DataType>;
	using ReservoirState = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
	using Input = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

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

	Reservoir(unsigned int input_dim, unsigned int state_dim)
		: input_dim(input_dim), state_dim(state_dim) {
		input_state.resize(input_dim);
		input_weights.resize(state_dim, input_dim);
		echo_weights.resize(state_dim, state_dim);
		state.resize(state_dim);
		state.setZero();

		has_input = false;
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
		unsigned long long seed = 0xcafebabe,
		double spectral_radius_desired = 0.99 
		// Note: I would point out in a comment that near 1 it is a long memory, < 1 is fading and > 1 is unstable
	) {
		using Trip = Eigen::Triplet<DataType>;
		std::vector<Trip> triplets_vector;

		// Now fill in the triples vector with a number of values that is
		// in the order of sparsity * state_dim

		static thread_local std::mt19937 gen(seed);
		std::uniform_real_distribution<float> unif(0, 1);
		std::normal_distribution<DataType> norm(0, 1);
		

		triplets_vector.reserve(static_cast<long>(state_dim * state_dim * sparsity));
		for (int i = 0; i < state_dim; ++i)
			for (int j = 0; j < state_dim; ++j) {
				// Only put nonzero with probability sparsity
				if (unif(gen) < sparsity)
					triplets_vector.push_back(Trip(i, j, norm(gen)));
			}

		echo_weights.setFromTriplets(triplets_vector.begin(), triplets_vector.end());
		double current_radius = MathOps::sparse_power_method(echo_weights);

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
			for (int i = 0; i < input_dim; ++i)
				for (int j = 0; j < input_dim; ++j)
					input_weights(i, j) = dist(gen);
		}
		else {
			std::normal_distribution<DataType> dist(0.0, std::sqrt(variance));

			for (int i = 0; i < input_dim; ++i)
				for (int j = 0; j < input_dim; ++j)
					input_weights(i, j) = dist(gen);
		}
	}

	void feed(const Input& u) {
		// Save the value of the next input to be employed in the next
		// run. The value is invalidated at each call of run. 
		assert(u.size() == static_cast<int>(input_dim)); 
		input_state = u;
		has_input = true;
	}

	void run(unsigned int steps = 1, bool keep_input = false) {
		ReservoirState pre_activation;
		if (has_input)
			pre_activation = echo_weights * state + input_weights * input_state;
		else
			pre_activation = echo_weights * state;
		state = pre_activation.array().tanh().matrix();
		if (!keep_input) {
			has_input = false;
			input_state.setZero();
		}
		notify_on_norm_change();
		notify_on_state_change();
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