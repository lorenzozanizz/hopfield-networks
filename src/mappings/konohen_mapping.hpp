#pragma once
#ifndef MAPPINGS_KONOHEN_HPP
#define MAPPINGS_KONOHEN_HPP

class NeighbouringPolicy {

};

template <typename DataType=float>
class KonohenMap {

	unsigned int mapping_cortex_size;
	unsigned int stimulus_cortex_input_size;

	// Allocate the required memory to store the weight vectors for each neurons.
	// allow explicit deallocation for fine grained memory control.
	void allocate() {

	}

	void deallocate() {
		
	}

public:

	KonohenMap(unsigned int cortex_size, unsigned int input_size):
		mapping_cortex_size(cortex_size), stimulus_cortex_input_size(input_size) {

	}

	void train() {

	}

	unsigned int map() {

	}

};

#endif