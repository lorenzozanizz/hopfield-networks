#pragma once
#ifndef MAPPINGS_KONOHEN_HPP
#define MAPPINGS_KONOHEN_HPP

#include <memory>
#include <iterator>
#include <cmath>
#include <vector>
#include <random>
#include <limits>
#include "../math/utilities.hpp"

// Enum that describes which evolving function to use on sigma
enum evolving_function {
	exponential, 
	linear,
	piecewise,
	inverse_time
};

enum dimension {
	ONE_D,
	TWO_D
};

class NeighbouringFunction{

private:
	
	double sigma_0;
	double sigma;
	double tau;
	evolving_function sigma_evolving_function = exponential;
	unsigned int map_width;
	unsigned int map_height;
	unsigned int support_size;

	// parameters required for some type of evolving functions
	unsigned int t_max;
	double sigma_1;
	double beta;

public:

	NeighbouringFunction(double sigma_0, double tau, unsigned int map_width, unsigned int map_height, std::string &evolving_func)
	:sigma_0(sigma_0), tau(tau), map_width(map_width), map_height(map_height){
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

	void set_map_height(unsigned int value) {
		this->map_height = value;
	}

	double get_map_height() const {
		return map_height;
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
		return support_size;
	}

	void set_support_size(unsigned int size) {
		support_size = size;
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

		// this is the support size that assumes a meaning of radius in 2D
		int support;

		// Saves both the coordinates, in case of 1D we use only the x-coordinate
		unsigned int winner_x = 0;
		unsigned int winner_y = 0;

		unsigned int current_x = 0;
		unsigned int current_y = 0;

		// Specify if we are in a 2D or 1D map
		dimension strategy;


	private:


	public:


		Iterator(const NeighbouringFunction* o, unsigned int w, unsigned int c)
			: owner(o), current_x(c), winner_x(w) {
			strategy = ONE_D;
			support = o->get_support_size();
		}

		Iterator(const NeighbouringFunction* o, unsigned int w_x, unsigned int w_y, unsigned int c_x, unsigned int c_y)
			: owner(o), current_x(c_x), winner_x(w_x), current_y(c_y), winner_y(w_y){
			strategy = TWO_D;
			support = o->get_support_size();
		}

		using iterator_category = std::forward_iterator_tag;
		using value_type = Iterator;
		using difference_type = int;
		using pointer = Iterator*;
		using reference = Iterator&;

		Iterator& operator++() {

			if (strategy == ONE_D) {
				++current_x;
			}
			else if (strategy == TWO_D) {

				// The strategy here is iterating through a square neighbourhood determined by the radius = support. 
				// So we start from (x , y) where x = min(0, winning.x - radius) and y = min(0, winning.y - radius).
				// We keep adding on x, until we reach the right bound min(grid width, winning.x + radius). 
				// When we reach it, we add on y and restart from x. We set the end of the iteration on (x_end, y_end) 
				// where x_end = min(grid width,  winning.x + radius) and y_end = min(grid height,  winning.y + radius).


				int next_x, next_y;
				// checking if the next x will be in the square and checking if the next x will in the map domain
				if (static_cast<int>(current_x) - static_cast<int>(winner_x) < support && current_x < owner->get_map_width() - 1) {

					//std::cout << " \n Distance: (" << static_cast<int>(current_x) - static_cast<int>(winner_x)
						//<< " is less than  " << support << " and current x " << current_x << " is less than " << owner->get_map_width() << "\n";

					// if so, we add on x
					next_x = current_x + 1;
					next_y = current_y;
				}
				else {

					//std::cout << " \n Distance: (" << static_cast<int>(current_x) - static_cast<int>(winner_x)
						//<< " is more than  " << support << " or current x " << current_x << " is more than " << owner->get_map_width() << "\n";

					// if so, we add on x
					next_x = current_x + 1;
					next_y = current_y;

					// if not, we update both y and x
					next_y = current_y + 1;
					next_x = std::max(0, static_cast<int>(winner_x) - static_cast<int>(support));

				}
				 
				current_x = next_x;
				current_y = next_y;
			}
			
			return *this;
		}

		bool operator!=(const Iterator& other) const {
			// in 1D the y-coordinates are always set to 0, so the check is just on the x
			return (current_x != other.current_x || current_y != other.current_y);
		}

		unsigned int index() const {
			return (current_x + current_y * owner->get_map_width());
		}

		unsigned int x_coordinate() const { 
			return (current_x);
		}

		unsigned int y_coordinate() const {
			return (current_y);
		}

		double contribution_weight() const {
			unsigned int map_width = owner->get_map_width();  
			const double sigma = owner->get_sigma(); 

			double dist = Utilities::euclidean_distance_2d(current_x, winner_x, current_y, winner_y);

			// returning the neighbourhood function computed on the current neuron
			return (std::exp(- (dist * dist) / (2 * sigma * sigma)));
		}


	};

	// begin for an iteration in 1D
	Iterator begin(unsigned int winner) const {

		unsigned int map_width = get_map_width();
		unsigned int support_size = get_support_size();

		if (winner > support_size) { 
			return Iterator(this, winner,
				winner - support_size);
		}else {
			return Iterator(this, winner, 0);
		}
	}

	// end for an iteration in 1D
	Iterator end(unsigned int winner) const {

		unsigned int support_size = get_support_size();
		unsigned int map_width = get_map_width();

		
		if (winner + support_size < map_width) {
			return Iterator(this, winner,
				winner + support_size + 1);
		}else {
			return Iterator(this, winner, map_width);
		}
		
	}
		
	// begin of an iteration in 2D
	Iterator begin(unsigned int winner_x, unsigned int winner_y) const {

		unsigned int map_width = get_map_width();
		unsigned int support_size = get_support_size();

		int starting_x;
		int starting_y;

		winner_x < support_size ?  starting_x = 0 : starting_x = winner_x - support_size;
		winner_y < support_size ? starting_y = 0 : starting_y = winner_y - support_size;

		return Iterator(this, winner_x, winner_y, starting_x, starting_y); 
	}

	// end of an iteration in 2D
	Iterator end(unsigned int winner_x, unsigned int winner_y) const {

		unsigned int support_size = get_support_size();
		unsigned int map_width = get_map_width(); 
		unsigned int map_height = get_map_height(); 

		int ending_x;
		int ending_y;

		// basically we are setting the end of our iteration at the min of win.x + dx and max_x on the x-axis and
		// at the min of win.y + dy and max_y on the y-axis.
		(winner_y + support_size < map_height) ? ending_y = winner_y + support_size + 1  : ending_y = map_height;
		winner_x < support_size ? ending_x = 0 : ending_x = winner_x - support_size;

		return Iterator(this, winner_x, winner_y, ending_x, ending_y);
		
		
	}
	

};

template <typename DataType=float>
class KonohenMap {

private:

	std::vector<std::unique_ptr<DataType[]>> weight_vectors;

	unsigned int mapping_cortex_width;
	unsigned int mapping_cortex_height;
	unsigned int stimulus_cortex_input_size;

	dimension dim;

	// Allocate the required memory to store the weight vectors for each neurons.
	// allow explicit deallocation for fine grained memory control.
	void allocate() {
		int mapping_cortex_size = mapping_cortex_width * mapping_cortex_height;
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
		mapping_cortex_width(cortex_size), mapping_cortex_height(1), stimulus_cortex_input_size(input_size) {
		dim = ONE_D;
		allocate();
	}

	KonohenMap(unsigned int cortex_width, unsigned int cortex_height, unsigned int input_size) :
		mapping_cortex_width(cortex_width), mapping_cortex_height(cortex_height), stimulus_cortex_input_size(input_size) {
		dim = TWO_D;
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

	const std::unique_ptr<DataType[]>& get_weights(unsigned int neuron_idx) const {
		return weight_vectors[neuron_idx];
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

				if (dim == ONE_D) {

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
				else if (dim == TWO_D) {
					
					int bmu_x = bmu % mapping_cortex_width;
					int bmu_y = bmu / mapping_cortex_width;



					// iterate over neighbors of BMU
					for (auto it = nf.begin(bmu_x, bmu_y); it != nf.end(bmu_x, bmu_y); ++it) {

						unsigned int idx = it.index();
					
						double h = it.contribution_weight();

						// update weights of neuron idx
						for (unsigned int j = 0; j < stimulus_cortex_input_size; ++j) {
							weight_vectors[idx][j] += learning_rate * h * (input[j] - weight_vectors[idx][j]);
						}

					}

				}
				
			}
			// evolve neighborhood radius
			nf.evolve_sigma(t);
		}
	}

	unsigned int map(const std::unique_ptr<DataType[]>& x) const {
		unsigned int winner_idx;
		double min_distance, distance;
		min_distance = std::numeric_limits<double>::max();

		for (unsigned int j = 0; j < mapping_cortex_height; ++j) {
			for (unsigned int i = 0; i < mapping_cortex_width; ++i) {
				distance = Utilities::euclidean_distance(x, weight_vectors[i + j * mapping_cortex_width], stimulus_cortex_input_size);
				if (distance < min_distance) {
					winner_idx = i + j * mapping_cortex_width;
					min_distance = distance;
				}
			}
		}

		return winner_idx;

	}

};

#endif