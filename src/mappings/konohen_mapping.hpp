#pragma once
#ifndef MAPPINGS_KONOHEN_HPP
#define MAPPINGS_KONOHEN_HPP

#include <memory>
#include <iterator>
#include "../math/utilities.hpp"

// Enum that describes which evolving function to use on sigma
enum evolving_function {
	exponential, 
	linear,
	piecewise,
	inverse_time
};

class NeighbouringFunction{

private:
	
	double sigma_0;
	double sigma;
	double tau;
	evolving_function sigma_evolving_function = exponential;
	unsigned int map_width; // max idx storable in the map

	// parameters required for some type of evolving functions
	unsigned int t_max;
	double sigma_1;
	double beta;

public:

	NeighbouringFunction(double sigma_0, double tau, unsigned int map_width, std::string &evolving_func)
	:sigma_0(sigma_0), tau(tau), map_width(map_width){
		this->set_sigma_evolving_function(evolving_func);
		this->sigma = sigma_0;
	}

	double get_sigma() const{
		return sigma;
	}

	void set_map_width(unsigned int value) {
		this->map_width = value;
	}

	double get_map_width() const{
		return map_width;
	}

	double get_sigma_0() const{
		return sigma_0;
	}

	void set_tau(double value) {
		this->tau = value;
	}

	void set_sigma_0(double value) {
		this->sigma_0 = value;
	}

	void set_t_max(int value) {
		this->t_max = value;
	}

	void set_sigma_1(double value) {
		this->sigma_1 = value;

	}

	void set_beta(double value) {
		this->beta = value;
	}

	void set_sigma_evolving_function(std::string const& String) {

		if (String == "exponential") {
			this->sigma_evolving_function = exponential;
		}
		else if (String == "piecewise") {
			this->sigma_evolving_function = piecewise;
		}
		else if (String == "linear") {
			this->sigma_evolving_function = linear;
		}
		else if (String == "inverse_time") {
			this->sigma_evolving_function = inverse_time;
		}

	}

	// How many neighbouring neurons have non-zero contribution, to optimize the
	// running loop
	unsigned int get_support_size() const {
		switch (sigma_evolving_function) {

		case exponential: return 3 * sigma;
			break;

		case piecewise: return 3 * sigma;
			break;

		case linear: return 3 * sigma;			
			break;

		case inverse_time: return 3 * sigma;
			break;

		default:
			return static_cast<unsigned int>(std::ceil(3 * sigma));
		}
	}

	void evolve_sigma(unsigned int iteration_step) {
		// This evolves sigma according to some schedule the user has to provide for example
		// with a function. generally sigma decreases as the iteration continues.
		switch (this->sigma_evolving_function) {

		case exponential: this->sigma = sigma_0 * std::exp(-static_cast<double>(iteration_step) / tau);
			break;

		case linear: this->sigma = sigma_0 * (1 - (static_cast<double>(iteration_step) / t_max));
			break;

		case piecewise: 
			if(iteration_step < t_max/2){
				this->sigma = sigma_0 * std::exp(-static_cast<double>(iteration_step) / tau);
				break;
			}else {
				this->sigma = sigma_1 * std::exp(- (static_cast<double>(iteration_step) - t_max/2) / tau);
				break;
			}

		case inverse_time: this->sigma = sigma_0 * std::pow((1 + static_cast<double>(iteration_step)), -beta);
			break;


		}

		

	}

	// Providing an iterator over the neurons in a way that is transparent to the mapping 
	class Iterator {

		const NeighbouringFunction* owner;
		unsigned int winner_idx;
		unsigned int current_idx;

	private:


	public:


		Iterator(const NeighbouringFunction* o, unsigned int w, unsigned int c)
			: owner(o), current_idx(c), winner_idx(w) {}

		using iterator_category = std::forward_iterator_tag;
		using value_type = Iterator;
		using difference_type = int;
		using pointer = Iterator*;
		using reference = Iterator&;

		Iterator& operator++() {
			++current_idx;
			return *this;
		}

		bool operator!=(const Iterator& other) const {
			return current_idx != other.current_idx;
		}

		unsigned int index() const {
			return (current_idx);
		}

		double contribution_weight() const {
			unsigned int map_width = owner->get_map_width();  
			const double sigma = owner->get_sigma(); 
			// computing the coordinates on the map of the winner neuron from its index
			int winner_x = winner_idx % map_width;
			int winner_y = winner_idx / map_width;
			// computing the coordinates on the map of the current neuron from its index
			int x = current_idx % map_width;
			int y = current_idx / map_width;
			// computing the euclidean distance between current and winner neuron
			double dist = std::sqrt((x - winner_x) * (x - winner_x) + (y - winner_y) * (y - winner_y));
			// returning the neighbourhood function computed on the current neuron
			return (std::exp(- (dist * dist) / (2 * sigma * sigma)));
		}


	};


	Iterator begin(unsigned int winner) const {

		unsigned int support_size = get_support_size();
		if (winner > support_size) {
			return Iterator(this, winner,
				winner - support_size);
		}
		else {
			return Iterator(this, winner, 0);
		}

	}

	Iterator end(unsigned int winner) const {
		unsigned int support_size = get_support_size();
		if (winner + support_size < get_map_width()) {
			return Iterator(this, winner,
				winner + support_size);
		}
		else {
			return Iterator(this, winner, get_map_width());
		}
	}
	

};

template <typename DataType=float>
class KonohenMap {

private:

	std::vector<std::unique_ptr<DataType[]>> weight_vectors;

	unsigned int mapping_cortex_size;
	unsigned int stimulus_cortex_input_size;

	// Allocate the required memory to store the weight vectors for each neurons.
	// allow explicit deallocation for fine grained memory control.
	void allocate() {
		weight_vectors.reserve(mapping_cortex_size);
		for (unsigned int i = 0; i < mapping_cortex_size; ++i) {
			weight_vectors.emplace_back(
				std::make_unique<DataType[]>(stimulus_cortex_input_size)
			);
		}
	}

	void deallocate() {
		// free memory of each weight vector
		for (auto& neuron_ptr : weight_vectors) {
			neuron_ptr.reset(); 
		}
		// clear the vector itself
		weight_vectors.clear(); 
	}

public:

	KonohenMap(unsigned int cortex_size, unsigned int input_size):
		mapping_cortex_size(cortex_size), stimulus_cortex_input_size(input_size) {
		allocate();
	}

	void initialize(unsigned long long seed) {
		std::mt19937 rng(seed);
		std::uniform_real_distribution<DataType> dist(0.0, 1.0);

		// assigning for each neuron's component a random value from 0.0 to 1.0
		for (auto& neuron_ptr : weight_vectors) {
			for (unsigned int j = 0; j < stimulus_cortex_input_size; ++j) {
				neuron_ptr[j] = dist(rng);
			}
		}
	}

	// This function receives a collection of datavectors and has to train the mapping cortex
	void train(const std::vector<std::unique_ptr<DataType[]>>& data,
		unsigned int iterations,
		NeighbouringFunction& nf,
		double learning_rate = 0.1 ) {

		for (unsigned int t = 0; t < iterations; t++) {
			for (const auto& input : data) {

				// find best matching unit
				unsigned int bmu = map(input);  

				// iterate over neighbors of BMU
				for (auto it = nf.begin(bmu); it != nf.end(bmu); ++it) {
					unsigned int idx = it.index();
					double h = it.contribution_weight();

					// update weights of neuron idx
					for (unsigned int j = 0; j < stimulus_cortex_input_size; ++j) {
						weight_vectors[idx][j] += learning_rate * h * (input[j] - weight_vectors[idx][j]);
					}
				}
			}
			// evolve neighborhood radius
			nf.evolve_sigma(t);
		}
	}

	unsigned int map(std::unique_ptr<DataType[]>& x) const {
		unsigned int winner_idx;
		double min_distance, distance;
		min_distance = std::numeric_limits<double>::max();

		for (unsigned int j = 0; j < mapping_cortex_size; ++j) {
			distance = Utilities::euclidean_distance(x, weight_vectors[j], stimulus_cortex_input_size);
			if (distance < min_distance) {
				winner_idx = j;
				min_distance = distance;
			}
		}

		return winner_idx;

	}

};

#endif