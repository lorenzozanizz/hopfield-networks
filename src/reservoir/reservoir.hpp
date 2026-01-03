#pragma once
#ifndef RESERVOIR_HPP
#define RESERVOIR_HPP

#include "reservoir_logger.hpp"
// Import sparse matrix required for the intra-reservoir dynamics.
#include "../math/matrix/matrix_ops.hpp"

template <typename DataType>
class Reservoir {
	
	using SparseMatrix = Eigen::SparseMatrix<DataType>;
	using ReservoirState = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

	// Matrix input_weights;
	SparseMatrix echo_weights;
	ReservoirState state;
	
	std::vector<ReservoirLogger*> loggers;
	
	void attach_logger(ReservoirLogger* logger) {
		loggers.push_back(logger);
	}

	void initialize_echo_weights() {
		// Sparsity degree
	}

	void initialize_input_weights() {

	}

	void feed() {
		// Save the value of the next input to be employed in the next
		// run. The value is invalidated at each call of run. 
	}

	void run(bool keep_input = false) {

		 notify_on_state_change();
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