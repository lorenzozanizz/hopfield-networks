#pragma once
#ifndef HOPFIELD_STOCHASTIC_HPP
#define HOPFIELD_STOCHASTIC_HPP

#include <random>

#include "../scheduling/annealing_scheduler.hpp"


// A great portion of the network is shared with the deterministic version,
// just that in this version we actually compute temperature updates (they don't 
// really make sense in deterministic hopfield networks )
// and updates are now deterministic, requiring stochastic runs.
template <typename WeightingPolicy>
class StochasticHopfieldNetwork {



};

#endif