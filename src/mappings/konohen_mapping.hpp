#pragma once
#ifndef MAPPINGS_KONOHEN_HPP
#define MAPPINGS_KONOHEN_HPP

#include <memory>

class NeighbouringFunction{
	
public:

	// How many neighbouring neurons have non-zero contribution, to optimize the
	// running loop
	unsigned int get_support_size() {

	}

	// Provide an iterator over the neurons in a way that is transparent to the mapping 
	void begin( unsigned int winnning_unit_index ) {
		// this returns an iterator with a .contribution_weight() attribute that is the 
		// result of h( winnning_unit_index, neigh ) for sigma
	}

	void end( unsigned int winning_unit_index ) {

	}

	void evolve_sigma(unsigned int iteration_step) {
		// This evolves sigma according to some schedule the user has to provide for example
		// with a function. generally sigma decreases as the iteration continues.
	}

};

template <typename DataType=float>
class KonohenMap {

	std::vector<std::unique_ptr<DataType*>> weight_vectors;

	unsigned int mapping_cortex_size;
	unsigned int stimulus_cortex_input_size;

	// Allocate the required memory to store the weight vectors for each neurons.
	// allow explicit deallocation for fine grained memory control.
	void allocate() {

	}

	void deallocate() {
		for (const auto& u_ptr : weight_vectors)
			u_ptr.reset();
	}

public:

	KonohenMap(unsigned int cortex_size, unsigned int input_size):
		mapping_cortex_size(cortex_size), stimulus_cortex_input_size(input_size) {

	}

	void initialize(unsigned long long seed) {

	}

	// This function receives a collection of datavectors and has to train the mapping cortex
	void train() {

	}

	unsigned int map() {

	}

};

#endif