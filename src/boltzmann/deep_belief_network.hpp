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

/**
 * @brief A simple calculator class.
 */
template <typename FloatingType>
class StackedRBM {
	
protected:

	using Vector = Eigen::Matrix<FloatingType, Eigen::Dynamic, 1>;
	using Matrix = Eigen::Matrix<FloatingType, Eigen::Dynamic, Eigen::Dynamic>;

	// Keep a reference to all machines, avoid owning the machines directly. 
	std::vector<std::reference_wrapper<
		RestrictedBoltzmannMachine<FloatingType>>> machines_stack;

public:

	void visualize_kernels_depthwise(
		Plotter& plotter, 
		unsigned int width, 
		unsigned int height,
		std::vector<unsigned int> how_many_per_layer, 
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
		std::map<unsigned int, WeightKernels> layer_mappings;

		if (!from && !to)
			layer_mappings.reserve(machines_stack.size());
		else layer_mappings.reserve(to - from + 1);

		// We need to construct all previous kernels, not just from "from" to "to"
		for (unsigned int layer = 1; layer <= to; ++layer) {

			machines_stack[layer].get().compute_higher_order_kernels(

			);
			// Compute the new mapping using the mapping of the previous layer. 
			// See the note above this function to see why we do it like this. 
		}

		// Now plot just the required kernels. 
		for (unsigned int layer = from; layer <= to; ++layer) {
			
			// Just views over the data, remember eigen has column_wise storage by default!
			std::vector<FloatingType*> kernel_mappings;


			// Get the amount of kernels to plot... 
			const auto amount = how_many_per_layer[layer-from];
			plotter.context().set_title("Kernels of machine of order " + std::to_string(layer))
				.plot_multiple_heatmaps(kernel_mappings, width, height);
		}

	}

	void unsupervised_train(
		int epochs,
		VectorCollection<Vector>& data,
		unsigned int batch_size,
		double lr,
		int k /* K parameter of the contrastive divergence algorithm. */,
		double decay = 1e-4
	) {
		// Use the references for all the boltzmann machines to pre-train the network. 


		// After each network is trained, the dataset is translated into its higher representation. 
		// This unfortunately requires duplicating data, to avoid damagign the input data. 


	}

};

#endif