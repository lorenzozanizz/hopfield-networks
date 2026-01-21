#pragma once
#ifndef PARALLEL_STOCHASTIC_HOPFIELD_NETWORK_HPP
#define PARALLEL_STOCHASTIC_HOPFIELD_NETWORK_HPP

// Required for the activation function(s)
#include <cmath>

#include "../../math/utilities.hpp"
#include "../network_base.hpp"
#include "../weighting/weighting_base.hpp"

#include "../stochastic/annealing_scheduler.hpp"

template <typename WeightingPolicy>
class ParallelStochasticHopfieldNetwork : public BaseHopfield<WeightingPolicy> {

	using DataType = typename BaseHopfield<WeightingPolicy>::DataType;

	// Store the machinery required to stochastically update the units.
	// e.g. no signum activation function!
	static double boltzmann_probability(DataType value, double temperature) {
		return 1 / (1 + std::exp(-2 * temperature * value));
	}

	signed char compute_value(DataType local_field, double temperature) {
		auto prob = boltzmann_probability(local_field, temperature);
		std::uniform_real_distribution<float> uniform;
		if (uniform(generator) < prob)
			return true;
		return false;
	}

	// The generators need to be local to each thread to prevent data races
	// when sampling!
	static thread_local std::mt19937 generator;

public:

	ParallelStochasticHopfieldNetwork(state_size_t size) : BaseHopfield<WeightingPolicy>(size)
	{ }

	void seed(unsigned long long seed) {
		generator.seed(seed);
	}

	void run(
		unsigned int num_threads,	
		const unsigned long iterations,
		std::unique_ptr<AnnealingScheduler>& temp_sched,
		const UpdateConfig uc
		// ^Describe whether we have asyncronous, synchronous or group updates.
	) {

		// A note for the following: All IO operations need to be operated in 
		// a single threaded fashion in order not to mess with the files!

		int it;
		std::vector<state_index_t> update_indexes;
		std::vector<DataType> local_fields_out;

		if (uc.up == UpdatePolicy::Asynchronous) {
			throw std::runtime_error("Cannot have asynchronous updates in parallel, opt for a group update.");
		}
		else if (uc.up == UpdatePolicy::GroupUpdate) {
			// Bound the amount of valeues to the size of the network. 
			local_fields_out.resize(std::min(this->net_size, uc.group_size * num_threads));
			update_indexes.resize(std::min(this->net_size, num_threads * uc.group_size));
		}

		auto schedule = this->fix_computation_schedule();
		if (schedule.do_order_parameter && this->ref_state.size() == 0)
			throw std::runtime_error("The network cant compute the order parameters because no "
				"reference states were specified!");


		// We ensure by contract that this function will end with the same number of threads: 
		// this value will be written beck when we're done!
		unsigned int previous_num_threads = Utilities::eigen_get_num_threads();
		if (uc.up == UpdatePolicy::Synchronous) {
			Utilities::eigen_set_num_threads(num_threads);
		}

		this->notify_on_begin(this->binary_state, iterations);
		for (it = 1; it <= iterations; ++it)
		{
			update_indexes.clear();
			local_fields_out.clear();

			if (uc.up == UpdatePolicy::Synchronous) {
				// NOTE: This computation is fully parallelized by eigen, which we control
				// by setting the number of threads earlier!
				this->policy.synch_update(this->binary_state, this->local_fields);
				// Here we parallelize on our own: 
				#pragma omp parallel for num_threads(num_threads)
				for (int i = 0; i < this->network_size; ++i)
					this->binary_state.set_value(i, compute_value(this->local_fields[i]), temp_sched->get_temp());

				this->notify_state(this->binary_state);
			}
			else if (uc.up == UpdatePolicy::GroupUpdate) {
				// Stochastically select a subset of the units, then run
				NetUtils::random_subset(update_indexes, this->network_size, uc.group_size);
				this->policy.hint_state(this->binary_state);

				#pragma omp parallel for num_threads(num_threads)
				for (int i = 0; i < num_threads; ++i) {
					local_fields_out[i] = this->policy.compute_local_field(
						this->binary_state,
						update_indexes[i]
					);
				}

				#pragma omp parallel for num_threads(num_threads)
				for (int i = 0; i < uc.group_size; ++i)
					this->binary_state.set_value(update_indexes[i],
						compute_value(this->local_fields_out[i], temp_sched->get_temp() ));

				this->notify_state(update_indexes, this->binary_state);
			}

			if (schedule.do_temperature)
				this->notify_temperature(temp_sched->get_temp());

			if (schedule.do_order_parameter)
				this->compute_order_parameter(it);
			// This internally calls notify_energy()
			if (schedule.do_energy)
				this->compute_energy();


			// Update the temperature schedule. This is COMMON for all threads operating in
			// parallel!
			temp_sched->update(it);
		}

		this->notify_on_end(this->binary_state);

		// Pop the "thread number stack" we had by interface contract.
		if (uc.up == UpdatePolicy::Synchronous) {
			Utilities::eigen_set_num_threads(previous_num_threads);
		}

		return;
	}

};



#endif // !DENSE_HOPFIELD_NETWORK_HPP