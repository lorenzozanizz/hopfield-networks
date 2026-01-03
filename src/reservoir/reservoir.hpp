#pragma once
#ifndef RESERVOIR_HPP
#define RESERVOIR_HPP

#include "reservoir_logger.hpp"
// Import sparse matrix required for the intra-reservoir dynamics.
#include "../math/matrix/sparse_matrix.hpp"
#include "../math/matrix/matrix_ops.hpp"

template <typename DataType>
class ReservoirState {

	unsigned int reservoir_size;
	std::vector<DataType> example;

};

class Reservoir {
	
	// Matrix input_weights;
	SparseMatrix echo_weights;
	ReservoirState<float> state;
	
	std::vector<ReservoirLogger*> loggers;
	
	void attach_logger(ReservoirLogger&) {

	}

	void initialize_echo_weights() {

	}

	void initialize_input_weights() {

	}

	void feed() {

	}

	void run() {
		/*
		 notify_on_run_begin();
		 // x(t+1) = tanh( W * x(t) + Win * u(t) ) 
		 Matrix Wx = W * state.x; 
		 Matrix Winu = Win * u; 
		 Matrix x_new(state.reservoir_size, 1); 
		 for (unsigned int i = 0; i < state.reservoir_size; i++) 
			 x_new(i, 0) = std::tanh(Wx(i, 0) + Winu(i, 0)); 
		 state.x = x_new; 
		 */
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