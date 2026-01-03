#pragma once
#ifndef RESERVOIR_HPP
#define RESERVOIR_HPP

#include "reservoir_logger.hpp"
// Import sparse matrix required for the intra-reservoir dynamics.
#include "../math/matrix/matrix_ops.hpp"

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

	std::vector<ReservoirLogger*> loggers;
	
	unsigned int input_dim;
	unsigned int state_dim;

public:

	Reservoir(unsigned int input_dim, unsigned int state_dim)
		: input_dim(input_dim), state_dim(state_dim) {
		input_state.resize(input_dim);
		input_weights.resize(input_dim, input_dim);
		echo_weights.resize(state_dim, state_dim);
		state.resize(state_dim);
	}

	void attach_logger(ReservoirLogger* logger) {
		loggers.push_back(logger);
	}

	void initialize_echo_weights(
		double sparsity, // fraction of non-zero entries 
		SamplingType sampling,
		unsigned long long seed = 0xcafebabe,
		double spectral_radius_desired = 0.99
	) {
		using Trip = Eigen::Triplet<DataType>;
		std::vector<Trip> triplets_vector;

		// Now fill in the triples vector with a number of values that is
		// in the order of sparsity * state_dim

		static thread_local std::mt19937 gen(seed);
		std::uniform_real_distribution<float> unif(0, 1);
		std::normal_distribution<DataType> norm(0, 1);
		
		for (int i = 0; i < state_dim; ++i)
			for (int j = 0; j < state_dim; ++j) {
				// Only put nonzero with probability sparsity
				if (unif(gen) < sparsity)
					triplets_vector.push_back(Trip(i, j, norm(gen)));
			}

		echo_weights.setFromTriplets(triplets_vector);
		double current_radius = MathOps::power_method(echo_weights);

		if (current_radius > 0)
			echo_weights *= (spectral_radius_desired / current_radius);

		// Compress the echo matrix finalizing it, no further additions
		// can be made. 
		echo_weights.makeCompressed();
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


	void feed() {
		// Save the value of the next input to be employed in the next
		// run. The value is invalidated at each call of run. 
	}

	void run(bool keep_input = false) {

		// Notify the loggers that the state has changed.
		notify_on_state_change();
	}

	ReservoirState& get_state() {
		return state;
	}

	SparseMatrix& get_echo_weights() {
		return echo_weights;
	}

	void notify_on_state_change() {
		for (auto* o : loggers) o->on_state_change( );
	}

	void notify_on_run_begin() {
		for (auto* o : loggers) o->on_run_begin( );
	}

};

#endif