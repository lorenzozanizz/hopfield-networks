#pragma once
#ifndef RESERVOIR_LOGGER_HPP
#define RESERVOIR_LOGGER_HPP

#include "../io/plot/plot.hpp"
#include "../io/image/images.hpp"

class ReservoirLogger {

public:
	
	void set_collect_states(bool value, unsigned int width = 0, unsigned int height = 0) {

	}

	void on_state_change() {
		// Visualize the echo updates of the reservoir as an image.
	}

	void on_run_begin() {

	}

	void on_run_end() {

	}

};

#endif