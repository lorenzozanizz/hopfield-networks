#pragma once
#ifndef NETWORKS_NETWORKS_TYPES_HPP
#define NETWORKS_NETWORKS_TYPES_HPP

#include "../math/matrix/matrix_ops.hpp"

// An index for a state vector
typedef unsigned long state_index_t;
// The size of a state vector, e.g. the dimension of an 
// Hopfield (Krotov) network.
typedef unsigned long state_size_t;


// Note that while the activations of the hopfield networks are binary,
// the local fields of the units are continuous and depend continuously
// on the weights. So we cannot use the same machinery for their values. 
// The local field themselves and not only their signs are required for the
// stochastic update policy.

template <typename DataType>
using LocalFields = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

#endif