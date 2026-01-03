#pragma once
#ifndef DENSE_HOPFIELD_NETWORK_HPP
#define DENSE_HOPFIELD_NETWORK_HPP

#include <algorithm>
#include <vector>
#include <functional>
#include <iostream>

#include "../../io/plot/plot.hpp"

#include "../network_types.hpp"
#include "../logger/logger.hpp"
#include "../weighting/weighting_base.hpp"

enum class UpdatePolicy {
	// Flip every unit together
	Synchronous,
	// Flip a unit at a time.
	Asynchronous,
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

protected:

	// The weighing policy determines how network weights are 
	// compute and stored
	WeightingPolicy policy;

	BinaryState binary_state;
	double current_energy;
	std::vector<HopfieldLogger*> loggers;

	// To be used when keeping track of order parameter
	std::vector<std::reference_wrapper<BinaryState>> ref_state;

public:

	DenseHopfieldNetwork(state_size_t size) : policy(size), binary_state(size) {
		// By interface allow policies to have lazy allocation
		policy.allocate();
		current_energy = 0;
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

	WeightingPolicy& weighting_policy() {
		return policy;
	}

	BinaryState& get_state() {
		return binary_state;
	}

	void store(BinaryState& bs) {
		policy.store(bs);
	}

	void set_state_strides(unsigned int stride_y, unsigned int stride_z = 0) {
		binary_state.set_stride_y(stride_y);
		if (stride_z)
			binary_state.set_stride_z(stride_z);
	}

	void set_reference_state(BinaryState& bs) {
		ref_state.push_back(std::reference_wrapper(bs));
	}

	// Use this to avoid computation of quantities which are not required
	// during the execution
	struct ComputationSchedule {
		bool do_energy;
		bool do_temperature;
		bool do_order_parameter;
	};

	// See comment above ^
	ComputationSchedule fix_computation_schedule() {
		ComputationSchedule sched;
		for (auto* o : loggers) {
			if (o->is_interested_in(Event::EnergyChanged))
				sched.do_energy = true;
			if (o->is_interested_in(Event::OrderParameterChanged))
				sched.do_temperature = true;
			if (o->is_interested_in(Event::TemperatureChanged))
				sched.do_order_parameter = true;
		}
		return sched;
	}

	void run(const BinaryState& init_state,
		const unsigned long iterations,
		const UpdateConfig uc // Describe whether we have asyncronous, synchronous or group updates.
	) {
		int it;
		std::vector<state_index_t> update_indexes;

		// Copy the content of the initial state. The strides are left untouched, they represent
		// how the logging routines interpret the data. 
		this->binary_state.copy_content(init_state);

		auto schedule = fix_computation_schedule();
		if (schedule.do_order_parameter && ref_state.size() == 0)
			throw std::runtime_error("The network cant compute the order parameters because no "
				"reference states were specified!");
		notify_on_begin(this->binary_state, iterations);
		for (it = 0; it < iterations; ++it) 
		{
			if (uc.up == UpdatePolicy::Synchronous) {
				// Entrust the weightpolicy to compute the dot values for the states. 
			} else if (uc.up == UpdatePolicy::GroupUpdate) {

				// Stochastically select a subset of the units, then run
			}
			else if (uc.up == UpdatePolicy::Asynchronous) {
				notify_state(std::tuple(it, binary_state.get(it)));
			}

			if (schedule.do_order_parameter)
				compute_order_parameter();
			// This internally calls notify_energy()
			if (schedule.do_energy)
				compute_energy();
		}

		notify_on_end(this->binary_state);

		return;
	}

	void compute_order_parameter() {
		// Compute the order parameter for all reference patterns!
	}

	void compute_energy() {
		current_energy = 1;
		notify_energy(current_energy);
	}

	// Now we list a few notify methods to be used in conjunction with
	// the observers (the network loggers). Note that there may be
	// multiple observer with different configurations!

    void notify_state(const std::vector<std::tuple<state_index_t, unsigned char>>& s) {
        for (auto* o : loggers) o->on_state_update(s);
    }

	void notify_state(const std::tuple<state_index_t, unsigned char> change) {
		for (auto* o : loggers) o->on_state_update(change);
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

	void notify_on_end(const BinaryState& bs) {
		for (auto* o : loggers) o->on_run_end(bs);
	}

	void notify_on_begin(const BinaryState& bs, unsigned int iterations) {
		for (auto* o : loggers) {
			o->on_run_begin(bs, iterations);
		}
	}

};



#endif // !DENSE_HOPFIELD_NETWORK_HPP