#pragma once
#ifndef BOLTZMANN_LOGGER_HPP
#define BOLTZMANN_LOGGER_HPP

#include <vector>

#include "../io/plot/plot.hpp"
#include "../io/gif/gif.hpp"

class BoltzmannLogger {

	// A small buffer object to map the reduced memory representation of states
	// to the required png buffer to save the intermediate state of the networks
	// (a bit of memory overhead, but around 40kbs at most)
	struct logger_buffer_t {

		unsigned int width = 0;
		unsigned int height = 0;
		std::unique_ptr<unsigned char[]> write_buffer = nullptr;

	};

	// Interpret the visible states / or the hidden states as units.
	unsigned int visual_width;
	unsigned int visual_height;

	// The evolutions of the hidden states and the visible states may be
	// very different as they represent different aspects of the memory. 
	std::string hidden_gif_save;
	std::string states_gif_save;

	logger_buffer_t log_buf;
	GifWriterIO gio;

	bool record_visible;
	bool record_energy;
	bool record_hidden;

	BoltzmannLogger() : record_visible(false), record_hidden(false),
		record_energy(false) {

	}

	void on_run_begin() {

	}

	void on_hidden_change() {

	}

	void on_energy_change(double new_energy) {

	}

	void on_visible_change() {

	}

	void on_run_end() {

	}

	void do_log_visible(bool value) {
		record_visible = value;
	}

	void do_log_hidden(bool value) {
		record_hidden = value;
	}

};

#endif