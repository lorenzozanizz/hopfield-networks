#pragma once
#ifndef NETWORK_BASE_HPP
#define NETWORK_BASE_HPP

#include <algorithm>
#include <vector>
#include <functional>
#include <optional>
#include <iostream>
#include <random>

#include "../io/plot/plot.hpp"

#include "classifier/hopfield_classifier.hpp"
#include "states/binary.hpp"
#include "network_types.hpp"
#include "logger/logger.hpp"

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

// A base class for the hopfield network, containing the machinery neede to have the logger and 
template <typename WeightingPolicy>
class BaseHopfield {

protected:

	// The weighing policy determines how network weights are 
	// compute and stored
	state_size_t network_size;
	WeightingPolicy policy;
	LocalFields<typename WeightingPolicy::Type> local_fields;
	BinaryState binary_state;

	std::vector<HopfieldLogger*> loggers;
	// To be used when keeping track of order parameter
	std::vector<std::reference_wrapper<BinaryState>> ref_state;
	std::vector<double> order_parameter_estimate;
	double current_energy;

	// A classifier that can be used in conjunction with the hopfield network to
	// classify data. 
	HopfieldClassifier* classifier; 

public:

	using DataType = typename WeightingPolicy::Type;

	BaseHopfield(state_size_t size) : policy(size), binary_state(size), network_size(size) {
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

	void add_reference_state(BinaryState& bs) {
		ref_state.push_back(std::reference_wrapper(bs));
		order_parameter_estimate.resize(ref_state.size(), 0.0);
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

	void compute_order_parameter(unsigned int iteration) {
		// Compute the order parameter for all reference patterns!
		int i = 0; 
		for (const auto& ref : ref_state) {
			double current_est = (network_size - (double) StateUtils::hamming_distance(binary_state, ref)) / network_size;
			order_parameter_estimate[i] = order_parameter_estimate[i] + (current_est - order_parameter_estimate[i]) / iteration;
			++i;
		}
		notify_order_parameter(order_parameter_estimate);
	}

	void compute_energy(std::tuple<state_index_t, signed int> flipped) {
		// If a single state is flipped, there is a neat expression for the
		// energy change that is much cheaper. 
		current_energy += policy.energy_flip(binary_state, flipped);
		notify_energy(current_energy);
	}

	void compute_energy() {
		// Recompute the entire energy
		current_energy = policy.energy(binary_state);
		notify_energy(current_energy);
	}

	// Now we list a few notify methods to be used in conjunction with
	// the observers (the network loggers). Note that there may be
	// multiple observer with different configurations!

	void notify_state(const std::tuple<state_index_t, unsigned char>& s) {
		for (auto* o : loggers) o->on_state_update(s);
	}

	void notify_state(const std::vector<state_index_t> indexes, BinaryState& group_change) {
		for (auto* o : loggers) o->on_state_update(indexes, group_change);
	}

	void notify_state(const BinaryState& state) {
		for (auto* o : loggers) o->on_state_update(state);
	}

	void notify_energy(double e) {
		for (auto* o : loggers) o->on_energy_update(e);
	}

	void notify_temperature(double t) {
		for (auto* o : loggers) o->on_temperature_update(t);
	}

	void notify_order_parameter(std::vector<double> ms) {
		for (auto* o : loggers) o->on_order_parameter_update(ms);
	}

	void notify_on_end(const BinaryState& bs) {
		for (auto* o : loggers) o->on_run_end(bs);
	}

	void notify_on_begin(const BinaryState& bs, unsigned int iterations) {
		for (auto* o : loggers) {
			o->on_run_begin(bs, iterations);
		}
	}

	void feed(BinaryState& init_state) {
		// Copy the content of the initial state. The strides are left untouched, they represent
		// how the logging routines interpret the data. 
		this->binary_state.copy_content(init_state);
	}

	using Category = int;

	void attach_classifier(HopfieldClassifier* classif) {
		// Attach a classifier
		classifier = classif;
	}

	// The first element of the pair is the binary state with the highest agreement, 
	// category is its label.
	auto classify() {
		if (classifier == nullptr)
			throw std::runtime_error("No classifier was specified for the Hopfield network!");
		// Classify the network's own state. 
		return classifier->classify(binary_state);
	}

};



namespace NetUtils {

	// Sample a single unit uniformly
	state_index_t pick_random_state(state_size_t size, unsigned long long seed = 0xcafebabe) {
		static thread_local std::mt19937 gen(std::random_device{}());
		std::uniform_int_distribution<int> dist(0, size-1);
		return dist(gen);
	}

	// Sample a subset of the units uniformly
	void random_subset(std::vector<state_index_t>& indexes, state_size_t size, unsigned int group_size,
		unsigned long long seed = 0xcafebabe) {

		static thread_local std::mt19937 rng(std::random_device{}());
		std::uniform_int_distribution<int> dist(0, size - 1);

		indexes.reserve(group_size);

		for (int i = 0; i < group_size; ++i)
			indexes.push_back(dist(rng));
	
		return;
	}

}

#endif