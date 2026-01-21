#pragma once
#ifndef HOPFIELD_STOCHASTIC_HPP
#define HOPFIELD_STOCHASTIC_HPP

#include <random>
// For exponentiation
#include <cmath>

#include "annealing_scheduler.hpp"
#include "../network_base.hpp"


// A great portion of the network is shared with the deterministic version,
// just that in this version we actually compute temperature updates (they don't 
// really make sense in deterministic hopfield networks )
// and updates are now deterministic, requiring stochastic runs.
template <typename WeightingPolicy>
class StochasticHopfieldNetwork: public BaseHopfield<WeightingPolicy> {

	using DataType = typename BaseHopfield<WeightingPolicy>::DataType;

	// Store the machinery required to stochastically update the units.
	// e.g. no signum activation function!
	static double boltzmann_probability(DataType value, double temperature) {
		return 1 / (1 + std::exp(-2 * temperature * value));
	}

	signed char compute_value(DataType local_field, double temperature) {
		auto prob = boltzmann_probability(local_field, temperature);
		if (uniform(generator) < prob)
			return true;
		return false;
	}

	// To be used during stochastic extractions. 
	unsigned long long random_seed;
	std::uniform_real_distribution<float> uniform;
	std::mt19937 generator;

public:

	StochasticHopfieldNetwork(state_size_t size) : BaseHopfield<WeightingPolicy>(size),
		generator(std::random_device{ }()), uniform(0,  1)
	{ }

	void seed(unsigned long long seed) {
		generator.seed(seed);
	}

	void run(
		const unsigned long iterations,
		std::unique_ptr<AnnealingScheduler>& temp_sched,
		// Our annealing scheduling policy. 
		const UpdateConfig uc 
		// Describe whether we have asyncronous, synchronous or group updates.
	) {
		// Note that because of c++ we have to preface every base inherited functionality
		// of the network with the this-> prefix
		int it;
		std::vector<state_index_t> update_indexes;
		std::vector<DataType> local_fields_out;

		if (uc.up == UpdatePolicy::Asynchronous) {
			local_fields_out.resize(1);
			update_indexes.resize(1);
		}
		else if (uc.up == UpdatePolicy::GroupUpdate) {
			local_fields_out.resize(uc.group_size);
			update_indexes.resize(uc.group_size);
		}

		auto schedule = this->fix_computation_schedule();
		if (schedule.do_order_parameter && this->ref_state.size() == 0)
			throw std::runtime_error("The network cant compute the order parameters because no "
				"reference states were specified!");

		this->notify_on_begin(this->binary_state, iterations);
		for (it = 1; it <= iterations; ++it)
		{
			// Clear all the previously computed indices for the iteration
			update_indexes.clear();
			local_fields_out.clear();
			if (uc.up == UpdatePolicy::Synchronous) {
				// Entrust the weightpolicy to compute the dot values for the states. 
				this->policy.synch_update(this->binary_state, this->local_fields);
				for (int i = 0; i < this->network_size; ++i)
					this->binary_state.set_value(i, 
						compute_value(this->local_fields[i], temp_sched->get_temp()));

				this->notify_state(this->binary_state);
			}
			else if (uc.up == UpdatePolicy::GroupUpdate) {
				// Stochastically select a subset of the units, then run
				NetUtils::random_subset(update_indexes, this->network_size, uc.group_size);
				this->policy.compute_local_fields(
					this->binary_state,
					update_indexes,
					local_fields_out
				);

				for (int i = 0; i < uc.group_size; ++i)
					this->binary_state.set_value(update_indexes[i],
						compute_value(local_fields_out[i], temp_sched->get_temp()));

				this->notify_state(update_indexes, this->binary_state);
			}
			else if (uc.up == UpdatePolicy::Asynchronous) {
				// Select a single state and act upon it.
				auto state_index = NetUtils::pick_random_state(this->network_size);
				update_indexes.push_back(state_index);
				this->policy.compute_local_fields(
					this->binary_state,
					update_indexes,
					local_fields_out
				);
				// Apply the signum activation function and notify. 
				this->binary_state.set_value(state_index, compute_value(local_fields_out[0], temp_sched->get_temp()));

				this->notify_state(std::tuple(state_index, this->binary_state.get(state_index)));
			}

			if (schedule.do_temperature)
				this->notify_temperature(temp_sched->get_temp());

			if (schedule.do_order_parameter)
				this->compute_order_parameter(it);
			// This internally calls notify_energy()
			if (schedule.do_energy)
				this->compute_energy();

			// Update the temperature schedule. This might change both the temperature and
			// the stabilization iterations. 
			temp_sched->update(it);
		}

		this->notify_on_end(this->binary_state);

		return;
	}

};

#endif