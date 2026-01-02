#pragma once
#ifndef RESERVOIR_HPP
#define RESERVOIR_HPP

#include "reservoir_logger.hpp"
// Import sparse matrix required for the intra-reservoir dynamics.
#include "math/matrix/sparse_matrix.hpp"
#include "math/matrix/matrix_ops.hpp"

class ReservoirState {

	unsigned int reservoir_size;

};

class Reservoir {


	std::vector<ReservoirLogger*> loggers;
	ReservoirState state;
	
	void attach_logger(ReservoirLogger&) {

	}

	void initialize(double sparsity_degree = 0.1) {

	}

	void feed() {

	}

	void run() {

	}

	void notify_on_state_change() {
		for (auto* o : loggers) o->on_state_change(bs);
	}

	void notify_on_run_begin() {
		for (auto* o : loggers) o->on_run_begin(bs);
	}

};

#endif