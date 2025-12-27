#pragma once
#ifndef DENSE_HOPFIELD_NETWORK_HPP
#define DENSE_HOPFIELD_NETWORK_HPP

#include "../network_types.hpp"

// We may have chosen to distinguish networks based on their weighting policy
// polymorphically, but weighting is so critical for network performance
// that compile-time knowledge of the weighting may allow the compiler to 
// optimize more (hopefully!)
template <typename WeightingPolicy>
class DenseHopfieldNetwork {

	// The weighing policy determines how network weights are 
	// compute and stored
	WeightingPolicy policy;



};

#endif // !DENSE_HOPFIELD_NETWORK_HPP