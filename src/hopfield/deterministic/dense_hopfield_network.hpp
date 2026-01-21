#pragma once
#ifndef DENSE_HOPFIELD_NETWORK_HPP
#define DENSE_HOPFIELD_NETWORK_HPP

// Required for the activation function(s)
#include <cmath>

#include "../network_base.hpp"
#include "../weighting/weighting_base.hpp"

// We may have chosen to distinguish networks based on their weighting policy
// polymorphically, but weighting is so critical for network performance
// that compile-time knowledge of the weighting may allow the compiler to 
// optimize more (hopefully!)
template <typename WeightingPolicy>
class DenseHopfieldNetwork: public BaseHopfield<WeightingPolicy> {

	using DataType = typename BaseHopfield<WeightingPolicy>::DataType;

public:

	DenseHopfieldNetwork(state_size_t size) : BaseHopfield<WeightingPolicy>(size) 
	{ }

	void run(
		const unsigned long iterations,
		const UpdateConfig uc 
		// ^Describe whether we have asyncronous, synchronous or group updates.
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
				// Entrust the weight policy to compute the dot values for the states. 
				this->policy.synch_update(this->binary_state, this->local_fields);
				for (int i = 0; i < this->network_size; ++i)
					this->binary_state.set_value(i, MathOps::sgn(this->local_fields[i]) > 0);

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
						MathOps::sgn(local_fields_out[i]) > 0);

				this->notify_state(update_indexes, this->binary_state);
			}
			else if (uc.up == UpdatePolicy::Asynchronous)  {
				// Select a single state and act upon it.
				auto state_index = NetUtils::pick_random_state(this->network_size);
				update_indexes.push_back(state_index);
				this->policy.compute_local_fields(
					this->binary_state,
					update_indexes,
					local_fields_out
				);
				// Apply the signum activation function and notify. 
				this->binary_state.set_value(state_index, MathOps::sgn(local_fields_out[0]) > 0);

				this->notify_state(std::tuple(state_index, this->binary_state.get(state_index)));
			}

			if (schedule.do_order_parameter)
				this->compute_order_parameter(it);
			// This internally calls notify_energy()
			if (schedule.do_energy)
				this->compute_energy();
		}

		this->notify_on_end(this->binary_state);

		return;
	}

};



#endif // !DENSE_HOPFIELD_NETWORK_HPP