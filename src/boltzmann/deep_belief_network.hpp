#pragma once
#ifndef DEEP_BELIEF_HPP
#define DEEP_BELIEF_HPP

#include <vector>
#include <functional>
#include <map>

#include "../io/plot/plot.hpp"
#include "../io/datasets/dataset.hpp"
// Import Eigen needed for the operations. 
#include "../math/matrix/matrix_ops.hpp"
// We need autograd to perform fine-tuning when needed.
#include "../math/autograd/variables.hpp"

#include "restricted_boltzmann_machine.hpp"
#include "boltzmann_logger.hpp"

template <typename FloatingType>
class StackedRBM {
	
protected:

	// Keep a reference to all machines, avoid owning the machines directly. 
	std::vector<std::reference_wrapper<
		RestrictedBoltzmannMachine>> machines_stack;

public:

	void visualize_kernels_depthwise(
		Plotter& plotter, 
		unsigned int width, 
		unsigned int height,
		unsigned int from = 0,
		unsigned int to = 0
	) {
		// We need to keep a temporary map containing all the representatives for the hidden
		// units. This is somewhat expensive, as it requires
		// layer_1_size * input_size + layer_2_size* input_size ... = 
		// N * input_size floats! However it is the only way to compute this visualization
		// meaningfully. 

		if (!machines_stack.size())
			return;

		using WeightKernels = Eigen::Matrix<FloatingType, Eigen::Dynamic, Eigen::Dynamic>;
		// Associate to each layer its list of image-fitted kernels. 
		map<unsigned int, WeightKernels> layer_mappings;

		if (!from && !to)
			layer_mappings.reserve(machines_stack.size());
		else layer_mappings.reserve(to - from + 1);

		for (unsigned int layer = to; layer <= to; ++layer) {

			// Compute the new mapping using the mapping of the previous layer. 
			// See the note above this function to see why we do it like this. 
		}

	}

	void unsupervised_train() {
		// Use the references for all the boltzmann machines to pre-train the network. 



	}

	void fine_tune() {

		
		// We use the interface for the backpropagation network trainer provided in the
		// .hpp file: as long as we implement the requested methods, everything is fine. 
		// NOTE: We mark such methods protected and declare the trainer class a friend. 

	}
	
};

#endif