#pragma once
#ifndef WEIGHTING_BASE_HPP
#define WEIGHTING_BASE_HPP

#include <functional>
#include <utility>

#include "../network_types.hpp"
#include "../states/binary.hpp"

class WeightingPolicy {


};

class DensePolicy : public WeightingPolicy {

	std::unique_ptr<float[]> weights;
	state_size_t net_size;

public:

	HebbianPolicy(state_size_t size) : net_size(size)
		// Do not allocate yet the weights to allow finegrained control.
	{
	}

	void allocate() {
		// The number of elements to represent for the symmetric weight matrix is
		// n * (n+1) where n is the network size
		weights = std::make_unique<float[]>(net_size * (net_size + 1));
	}

	void deallocate() {
		weights.reset();
	}

	inline float& get(state_index_t i, state_index_t j) {
		if (i > j) std::swap(i, j);  // ensure i <= j, we store the lower triangular parte
		return weights[i * (i + 1) / 2 + j];  // alternative formula	
	}

	inline float& operator()(state_index_t i, state_index_t j) {
		if (i > j) std::swap(i, j); // see above
		return weights[i * (i + 1) / 2 + j];
	}

};

class HebbianPolicy: public DensePolicy {

public:

	void store(BinaryState& bs) {
		unsigned int value = 0;
		const auto one_over_n = 1.0 / net_size;
		for (int i = 0; i < net_size; ++i)
			for (int j = 0; j < net_size; ++j) {
				if (bs.get(i) && bs.get(j) || (!bs.get(i) && !bs.get(j)))
					get(i, j) += one_over_n;
				else get(i, j) -= one_over_n;
			}
		return;
	}

};

class StarkovPolicy: public WeightingPolicy {

public:

	void store(BinaryState& bs) {
		unsigned int value = 0;
		const auto one_over_n = 1.0 / net_size;
		for (int i = 0; i < net_size; ++i)
			for (int j = 0; j < net_size; ++j) {
				if (bs.get(i) && bs.get(j) || (!bs.get(i) && !bs.get(j)))
					get(i, j) += one_over_n;
				else get(i, j) -= one_over_n;
			}
		return;
	}

};

class InducedSparsePolicy: public WeightingPolicy {


};

class MatrixFreePolicy: public WeightingPolicy {

	std::vector<std::reference_wrapper<BinaryState>> images;

	void allocate() {
		// No explicit allocation, we recompute the values on the fly
	}

	void deallocate() {
		// Just delete all the references to the images. NOTE that the user 
		// must ensure that the references DO NOT become stale before the run
		// ends, otherwise we compute garbage (and potentially crash!)
		images.clear();
	}

};


#endif