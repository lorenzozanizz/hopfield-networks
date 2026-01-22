#pragma once
#ifndef PARALLEL_STOCHASTIC_HOPFIELD_NETWORK_HPP
#define PARALLEL_STOCHASTIC_HOPFIELD_NETWORK_HPP

#include <cmath>
#include <omp.h>

// Required for the activation function(s)
#include "../../math/utilities.hpp"
#include "../network_base.hpp"
#include "../weighting/weighting_base.hpp"

#include "../stochastic/annealing_scheduler.hpp"

/**
 * @brief An implementation of a parallelized stochastic hopfield netowkr. 
 * As per the determinsitic parallel implementation, the parallelism is distinguished
 * on the required update policy for the network. Care must be taken when sampling: the
 * sampler classes are localized per thread and seeded with the main seed when required.
 * For more information on the parallelization choices see the comment in the deterministic
 * implementation.
*/
template <typename WeightingPolicy>
class ParallelStochasticHopfieldNetwork : public BaseHopfield<WeightingPolicy> {

	using DataType = typename BaseHopfield<WeightingPolicy>::DataType;

	// Store the machinery required to stochastically update the units.
	// e.g. no signum activation function!
	static double boltzmann_probability(DataType value, double temperature) {
		return 1 / (1 + std::exp(-2 * temperature * value));
	}

	/**
	 * @brief Compute the value associated with the local field for the given temperature. 
	 * This computes the boltzmann activation function. 
	 * 
	 * @param seed the seed for the generator
	*/
	signed char compute_value(DataType local_field, double temperature) {
		// Each thread gets its own instance of this random-number generator.
		// This avoids data races and guarantees reproducible, independent RNG
		// sequences across threads
		static thread_local std::mt19937 generator;
		auto prob = boltzmann_probability(local_field, temperature);
		std::uniform_real_distribution<float> uniform;
		if (uniform(generator) < prob)
			return true;
		return false;
	}

	unsigned long long last_seed;

public:

	ParallelStochasticHopfieldNetwork(state_size_t size) : BaseHopfield<WeightingPolicy>(size) {	
		last_seed = 0xCAFEBABE;
		seed(last_seed);
	}

	/**
	 * @brief Seed the generator and save the seed
	 * @param seed the seed for the generator
	*/
	void seed(unsigned long long seed) {
		last_seed = seed;
	}


	/**
	 * @brief Run the network on the present binary state, at each step instructing the logger
	 * on the changes that affected the network. Based on the update configuration, various
	 * kind of updates are applied. The temperature is scheduled equally for all threads. 
	 * 
	 * @param num_threads The number of threads
	 * @param iterations The number of iterations
	 * @param temp_sched The scheduler for the temperature
	 * @param uc The configuration on each update, the type of update and the size. 
	*/
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
			local_fields_out.resize(std::min(this->network_size, (state_size_t) uc.group_size * num_threads));
			update_indexes.resize(std::min(this->network_size, (state_size_t) num_threads * uc.group_size));
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
					this->binary_state.set_value(i, compute_value(
						this->local_fields[i], temp_sched->get_temp()));

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
						update_indexes[i], false
					);
				}

				for (int i = 0; i < uc.group_size; ++i)
					this->binary_state.set_value(update_indexes[i],
						compute_value(local_fields_out[i], temp_sched->get_temp() ));

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

		// Notify the logger that the run has ended. This may trigger IO operations such as
		// writing images or gifs. 
		this->notify_on_end(this->binary_state);

		// Pop the "thread number stack" we had by interface contract.
		if (uc.up == UpdatePolicy::Synchronous) {
			Utilities::eigen_set_num_threads(previous_num_threads);
		}

		return;
	}

};

#endif // !DENSE_HOPFIELD_NETWORK_HPP