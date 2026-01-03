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
	// Make the datatype attribute public for type declarations around the
	// code. 
	using Type = DataType;

	unsigned int number_memorized() const {
		return memorized_amt;
	}

	// Get the weight joining the two units i and j
	virtual inline DataType& get(state_index_t i, state_index_t j) = 0;


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
	virtual void compute_local_fields(BinaryState& bs, std::vector<state_index_t> units,
		std::vector<DataType>& out) = 0;

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

	DensePolicy(state_size_t size) : net_size(size),
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

	inline DataType& get(state_index_t i, state_index_t j) override {
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
	virtual void synch_update(BinaryState& bs, LocalFields<DataType>& fields) {
		for (int i = 0; i < this->net_size; ++i)
			intermediate(i) = (bs.get(i)) ? 1 : -1;
		fields = weights * intermediate;
	}

	// Compute the local fields of a limited subgroups of the units
	virtual void compute_local_fields(BinaryState& bs, std::vector<state_index_t> units,
		std::vector<DataType>& out) {
		for (int i = 0; i < this->net_size; ++i)
			intermediate(i) = (bs.get(i)) ? 1 : -1;
		for (int unit_i = 0; unit_i < units.size(); ++unit_i) {
			const auto& r = weights.row(units[unit_i]);
			out.push_back(r.dot(intermediate));
		}
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
			this->intermediate(i) = (bs.get(i))? 1 : -1;

		// Note eigen optimizes this.
		this->weights.noalias() += 
			(this->intermediate * this->intermediate.transpose()) / this->net_size;
		this->weights.diagonal().setZero();
		return;
	}

};

template <typename DataType>
class StarkovPolicy: public DensePolicy<DataType> {

	WeightMatrix<DataType> weights;
	// We use an extra vector to compute the local fields
	// before appending a new vector. 
	LocalFields<DataType> fields;

public:

	virtual void allocate() override {
		// The number of elements to represent for the symmetric weight matrix is
		// n * (n+1) where n is the network size
		weights.resize(this->net_size, this->net_size);
		fields.resize(this->net_size);
		this->intermediate.resize(this->net_size);
	}

	StarkovPolicy(state_size_t size) : DensePolicy<DataType>(size)
		// Do not allocate yet the weights to allow finegrained control.
	{ }

	virtual void store(BinaryState& bs) override {
		unsigned int value = 0;




		// The local fields need only be computed if some other
		// weight has been stored in precedence.

		this->memorized_amt++;

		// We use our internal vector to compue the local field BEFORE
		// adding the pattern, this is critical 
		/*		const auto one_over_n = 1.0 / net_size;

		for (int i = 0; i < net_size; ++i)
			for (int j = 0; j < net_size; ++j) {
				if (i == j)
					continue;
				if (bs.get(i) && bs.get(j) || (!bs.get(i) && !bs.get(j)))
					get(i, j) += one_over_n;
				else get(i, j) -= one_over_n;
			}
		// Standard contribution
		for (int i = 0; i < net_size; ++i)
			for (int j = 0; j < net_size; ++j) {
				if (i == j)
					continue;
				if (bs.get(i) && bs.get(j) || (!bs.get(i) && !bs.get(j)))
					get(i, j) += one_over_n;
				else get(i, j) -= one_over_n;
			}
		// Now add starkov-only local contributions:
		*/
		return;
	}

};

template <typename DataType>
class MatrixFreePolicy: public WeightingPolicy<DataType>{

	std::vector<std::reference_wrapper<BinaryState>> images;

	void allocate() {
		// No explicit allocation, we recompute the values on the fly
	}

	void deallocate() {
		// Just delete all the references to the images. NOTE that the user 
		// must ensure that the references DO NOT become stale before the run
		// ends, otherwise we compute garbage (and potentially crash!)
		images.clear();
	}

};


#endif