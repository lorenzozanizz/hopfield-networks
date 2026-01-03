#pragma once
#ifndef CROSS_TALK_VISUALIZER_HPP
#define CROSS_TALK_VISUALIZER_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../../io/plot/plot.hpp"
#include "../../io/image/images.hpp"
#include "../states/binary.hpp"

class HebbianCrossTalkTermVisualizer {

	std::vector<float> cross_talks;

	// Store a copy of the gnu plotter and the network 
	Plotter& plotter;
	bool did_compute;


public:

	void compute_cross_talk_view(BinaryState& reference, std::initializer_list<BinaryState*> init) {
		// Notice that the notion of cross talk only makes sense when a hebbian 
		// weighting strategy is initialized, as for example in the storkov
		// strategy we cannot clearly have additive contributions to the cross talk.
		std::vector<BinaryState*> refs(init);
		compute_cross_talk_view(reference, refs);
	}
	
	void compute_cross_talk_view(BinaryState& reference, std::vector<BinaryState*> init) {
		float sum_i;
		
		memset(cross_talks.data(), 0, cross_talks.size() * sizeof(float));
		const auto size = reference.get_size();
		for (int i = 0; i < size; ++i) {
			// For each entry of the state we compute its cross talk with all other
			// units in all other pattern.
			sum_i = 0.0;
			for (BinaryState* mu : init) {
				BinaryState& x_mu = *mu;
				const signed char x_mu_i = x_mu.get(i) ? 1 : -1;
				for (int j = 0; j < size; ++j) {
					if (j == i)
						continue;
					const signed char x_mu_j = x_mu.get(j) ? 1 : -1;
					const signed char x_v_j = reference.get(j) ? 1 : -1;
					// If xi==xj their contribution is positive
					sum_i += x_mu_i * x_mu_j * x_v_j;
				}
			}
			sum_i /= -((double)(reference.get(i) ? 1 : -1)) * size;
			if (sum_i < 0.0) sum_i = 0.0;
			cross_talks[i] = sum_i;
		}
		did_compute = true;
	}

	HebbianCrossTalkTermVisualizer(Plotter& plot, unsigned int size)
	: plotter(plot), cross_talks(size), did_compute(false) { }

	void show(unsigned int width, unsigned int height) {
		if (width * height != cross_talks.size())
			throw std::runtime_error("Wrong visualization dimensions for the network");
		if (!did_compute)
			throw std::runtime_error("Cant show the result: computation not done yet!");
		// std::unique_ptr<unsigned char[]> image_buffer(new unsigned char[cross_talks.size()]);
		// compute_heatmap(image_buffer.get(), cross_talks.size());
		auto ctx = plotter.context();
		ctx.set_title("Cross talk for the pattern");
		ctx.show_heatmap(cross_talks.data(), width, height);
	}

	void to_image(const std::string& path, unsigned int width, unsigned int height) {
		if (!did_compute)
			throw std::runtime_error("Cant dump to image: computation not done yet!");
		if (width * height != cross_talks.size())
			throw std::runtime_error("Wrong visualization dimensions for the network");
		auto ctx = plotter.context();
		ctx.set_title("Cross talk for the pattern");
		ctx.redirect_to_image(path).show_heatmap(cross_talks.data(), width, height);
	}
};


#endif