#pragma once
#ifndef WEIGHTING_BASE_HPP
#define WEIGHTING_BASE_HPP

#include <functional>
#include <utility>
#include <stdexcept>

#include "../../math/matrix/matrix_ops.hpp"
#include "../network_types.hpp"
#include "../states/binary.hpp"
//

template <typename DataType>
class WeightingPolicy {

protected:

	unsigned int memorized_amt;

public:

	WeightingPolicy() : memorized_amt(0) {}
	// Make the datatype attribute public for type declarations around the
	// code. 
	using Type = DataType;

	unsigned int number_memorized() const {
		return memorized_amt;
	}

	// Store a new patter in the internal weights.
	virtual void store(BinaryState& bs) = 0;

	// Compute the energy associated with the state bs under the current weighting
	// policy. 
	virtual DataType energy(const BinaryState& bs) = 0;

	// Compute the delta of energy associated with the state bs under a single
	// bit flip in position unit_ind
	virtual DataType energy_flip(const BinaryState& bs, std::tuple<state_index_t, signed char> flip) = 0;

	// Compute the local fields across the entirety of the network. This
	// leaves freedom of implementation for the activation function of the network. 
	virtual void synch_update(BinaryState& bs, LocalFields<DataType>& fields) = 0;

	// Compute the local fields of a limited subgroups of the units
	virtual void compute_local_fields(const BinaryState& bs, std::vector<state_index_t> units,
		std::vector<DataType>& out) = 0;

	// Compute the local fields of a single unit
	virtual DataType compute_local_field(const BinaryState& bs, state_index_t unit, bool overwrite) = 0;

	virtual void hint_state(const BinaryState& bs) = 0;

};

template <typename DataType>
using WeightMatrix = Eigen::Matrix< DataType, Eigen::Dynamic, Eigen::Dynamic>;

template <typename DataType>
class DensePolicy : public WeightingPolicy<DataType> {

protected:

	// We use an intermediate matrix to allow rapid calculations
	// in eigen. 
	LocalFields<DataType> intermediate;

	WeightMatrix<DataType> weights;
	state_size_t net_size;

public:

	DensePolicy(state_size_t size) : WeightingPolicy<DataType>(), net_size(size),
		weights()
		// Do not allocate yet the weights to allow finegrained control.
	{ }

	virtual void allocate() {
		// The number of elements to represent for the symmetric weight matrix is
		// n * (n+1) where n is the network size
		weights.resize(net_size, net_size);
		intermediate.resize(net_size);
	}

	void deallocate() {
		weights.reset();
	}

	inline DataType& get(state_index_t i, state_index_t j) {
		return weights(i, j);
	}

	inline float& operator()(state_index_t i, state_index_t j) {
		if (i > j) std::swap(i, j); // see above
		return weights[i * (i + 1) / 2 + j];
	}

	virtual DataType energy(const BinaryState& bs) override {
		// This is common for all dense policies. 
		for (int i = 0; i < this->net_size; ++i)
			intermediate(i) = (bs.get(i)) ? 1 : -1;
		return -0.5 * intermediate.dot( weights * intermediate );
	}

	virtual DataType energy_flip(const BinaryState& bs, std::tuple<state_index_t, signed char> flip) override {
		for (int i = 0; i < this->net_size; ++i)
			intermediate(i) = (bs.get(i)) ? 1 : -1;
		float hk = weights.row(std::get<0>(flip)).dot(intermediate);
		return 2.0f * std::get<1>(flip) * hk; // energy change
	}

	// Compute the local fields across the entirety of the network. This
	// leaves freedom of implementation for the activation function of the network. 
	virtual void synch_update(BinaryState& bs, LocalFields<DataType>& fields) override {
		for (int i = 0; i < this->net_size; ++i)
			intermediate(i) = (bs.get(i)) ? 1 : -1;
		fields = weights * intermediate;
	}

	// Compute the local fields of a limited subgroups of the units
	virtual void compute_local_fields(const BinaryState& bs, std::vector<state_index_t> units,
		std::vector<DataType>& out) override {
		for (int i = 0; i < this->net_size; ++i)
			intermediate(i) = (bs.get(i)) ? 1 : -1;
		for (int unit_i = 0; unit_i < units.size(); ++unit_i) {
			// Take the column (eigen has columnwise storage)
			const auto& r = weights.col(units[unit_i]);
			out.push_back(r.dot(intermediate));
		}
		return;
	}

	virtual DataType compute_local_field(const BinaryState& bs, state_index_t unit, bool overwrite) override {
		if (overwrite)
			for (int i = 0; i < this->net_size; ++i)
				intermediate(i) = (bs.get(i)) ? 1 : -1;
		return weights.col(unit).dot(intermediate);
	}

	virtual void hint_state(const BinaryState& bs) override {
		for (int i = 0; i < this->net_size; ++i)
			intermediate(i) = (bs.get(i)) ? 1 : -1;
		return;
	}

};

template <typename DataType>
class HebbianPolicy: public DensePolicy<DataType> {

public:

	HebbianPolicy(state_size_t size) : DensePolicy<DataType>(size)
		// Do not allocate yet the weights to allow finegrained control.
	{ }

	virtual void store(BinaryState& bs) override {
		unsigned int value = 0;

		this->memorized_amt++;
		// const auto one_over_n = 1.0 / this->net_size;
		for (int i = 0; i < this->net_size; ++i)
			this->intermediate(i) = (bs.get(i))? DataType(1) : DataType(-1);

		// Note eigen optimizes this.
		this->weights.noalias() += 
			(this->intermediate * this->intermediate.transpose()) / this->net_size;
		this->weights.diagonal().setZero();
		return;
	}

};

template <typename DataType>
class StorkeyPolicy: public DensePolicy<DataType> {

	LocalFields<DataType> fields;

public:
	StorkeyPolicy(state_size_t size) : DensePolicy<DataType>(size), fields(size)
		// Do not allocate yet the weights to allow finegrained control.
	{ }

	virtual void store(BinaryState& bs) override {
		// The local fields need only be computed if some other
		// weight has been stored in precedence.

		this->memorized_amt++;
		const int N = this->net_size;

		// Convert pattern to ±1
		for (int i = 0; i < N; ++i)
			this->intermediate(i) = bs.get(i) ? DataType(1) : DataType(-1);

		auto hebbian_term = this->intermediate * this->intermediate.transpose();
		auto net_inputs = this->weights * this->intermediate;

		auto pre_synaptic = this->intermediate * net_inputs.transpose();

		auto post_synaptic = pre_synaptic.transpose();

		this->weights.noalias() += (hebbian_term) / N;
		this->weights.noalias() += (- pre_synaptic - post_synaptic) / (N * N);
		this->weights.diagonal().setZero();

		return;
	}
};

template <typename DataType>
class MatrixFreePolicy: public WeightingPolicy<DataType>{

	LocalFields<DataType> intermediate;
	// Keep a reference o all the images, O(N_data * k)
	std::vector<std::reference_wrapper<BinaryState>> images;
	state_size_t net_size;

public:

	MatrixFreePolicy(state_size_t size) : WeightingPolicy<DataType>(), net_size(size)
		// Do not allocate yet the weights to allow finegrained control.
	{ }

	void allocate() {
		intermediate.resize(net_size);
		// No explicit allocation, we recompute the values on the fly
	}

	void deallocate() {
		// Just delete all the references to the images. NOTE that the user 
		// must ensure that the references DO NOT become stale before the run
		// ends, otherwise we compute garbage (and potentially crash!)
		images.clear();
	}

	virtual void store(BinaryState& bs) override {
		images.push_back(std::reference_wrapper<BinaryState>(bs));
		this->memorized_amt++;
		return;
	}

	virtual DataType energy(const BinaryState& bs) override {
		throw std::runtime_error("You are attempting to use a batch operation on an online weighting"
		"policy. If you require batch computation, use batch weighting policies instead!");
		return 0.0;
	}

	virtual DataType energy_flip(const BinaryState& bs, std::tuple<state_index_t, signed char> flip) override {
		for (int i = 0; i < this->net_size; ++i)
			intermediate(i) = (bs.get(i)) ? 1 : -1;

		DataType hk = DataType(0);
		// Again, compute the local dot product for the unit flip using the memorized patterns, computing
		// the weights on the fly. 
		unsigned int i = std::get<0>(flip);
		for (unsigned int j = 0; j < net_size; ++j) {
			DataType online_weight = compute_weight(i, j);
			hk += intermediate[j] * online_weight;
		}
		return 2.0f * std::get<1>(flip) * hk; // energy change
	}

	// Compute the local fields across the entirety of the network. This
	// leaves freedom of implementation for the activation function of the network. 
	virtual void synch_update(BinaryState& bs, LocalFields<DataType>& fields) {
		throw std::runtime_error("You are attempting to use a batch operation on an online weighting"
			"policy. If you require batch computation, use batch weighting policies instead!");
	}

	DataType compute_weight(state_index_t i, state_index_t j) {
		DataType online_weight = DataType(0);
		if (i == j)
			return 0.0;
		const DataType one_over_n = DataType(1) / this->net_size;

		for (int state = 0; state < this->memorized_amt; ++state) {
			auto& bs = images[state].get();

			if (bs.get(i) && bs.get(j) || (!bs.get(i) && !bs.get(j)))
				online_weight += one_over_n;
			else online_weight -= one_over_n;
		}
		return online_weight;
	}

	// Compute the local fields of a limited subgroups of the units
	virtual void compute_local_fields(const BinaryState& bs, std::vector<state_index_t> units,
		std::vector<DataType>& out) override {

		for (int i = 0; i < this->net_size; ++i)
			intermediate(i) = (bs.get(i)) ? DataType(1) : DataType(-1);
		for (int unit_i = 0; unit_i < units.size(); ++unit_i) {
			DataType local_field = DataType(0);
			unsigned int i = units[unit_i];
			for (unsigned int j = 0; j < net_size; ++j) {

				DataType online_weight = compute_weight(i, j);
				local_field += intermediate[j] * online_weight; 

			}
			out.push_back(local_field);
		}
	}

	virtual void hint_state(const BinaryState& bs) override {
		for (int i = 0; i < this->net_size; ++i)
			intermediate(i) = (bs.get(i)) ? 1 : -1;
		return;
	}

	virtual DataType compute_local_field(const BinaryState& bs, state_index_t unit, bool overwrite) override {
		if (overwrite)
			for (int i = 0; i < this->net_size; ++i)
				intermediate(i) = (bs.get(i)) ? 1 : -1;
		DataType local_field = DataType(0);
		for (unsigned int j = 0; j < net_size; ++j) {
			DataType online_weight = compute_weight(unit, j);
			local_field += intermediate[j] * online_weight;
		}
		return local_field;
	}
};


#endif