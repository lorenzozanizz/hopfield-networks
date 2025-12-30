#pragma once
#ifndef DENSE_HOPFIELD_NETWORK_HPP
#define DENSE_HOPFIELD_NETWORK_HPP

#include <algorithm>
#include <vector>

#include "../../io/plot/plot.hpp"

#include "../network_types.hpp"
#include "../logger/logger.hpp"
#include "../weighting/weighting_base.hpp"

enum class UpdatePolicy {
	BatchUpdate,
	OnlineUpdate,
	GroupUpdate
};

struct UpdateConfig {

	UpdatePolicy up;
	unsigned int group_size = 0;

};

// We may have chosen to distinguish networks based on their weighting policy
// polymorphically, but weighting is so critical for network performance
// that compile-time knowledge of the weighting may allow the compiler to 
// optimize more (hopefully!)
template <typename WeightingPolicy>
class DenseHopfieldNetwork {

	// The weighing policy determines how network weights are 
	// compute and stored
	WeightingPolicy policy;

	BinaryState binary_state;
	std::vector<HopfieldLogger*> loggers;

public:

	DenseHopfieldNetwork(state_size_t size) : loggers(1), policy(size), binary_state(size) {
		// By interface allow policies to have lazy allocation
		policy.allocate();
	}

	void detach_logger(HopfieldLogger* logger) {
		// Allow detachment of loggers, user may want to switch loggers when
		// testing multiple run configurations
		loggers.erase(
			std::remove(loggers.begin(), loggers.end(), logger), loggers.end());
	}

	void attach_logger(HopfieldLogger* logger) {
		loggers.push_back(logger);
	}

	// This ...
	void store(BinaryState& bs);

	void run(BinaryState& binary_state, const unsigned long iterations, const UpdateConfig uc) {

	}

	// Now we list a few notify methods to be used in conjunction with
	// the observers (the network loggers). Note that there may be
	// multiple observer with different configurations!

    void notify_state(const std::vector<double>& s) {
        for (auto* o : loggers) o->on_state_update(s);
    }

    void notify_energy(double e) {
        for (auto* o : loggers) o->on_energy_update(e);
    }

    void notify_temperature(double t) {
        for (auto* o : loggers) o->on_temperature_update(t);
    }

    void notify_order_parameter(double m) {
        for (auto* o : loggers) o->on_order_parameter_update(m);
    }

};

template <>
void DenseHopfieldNetwork<HebbianPolicy>::store(BinaryState& bs) {

}

#endif // !DENSE_HOPFIELD_NETWORK_HPP