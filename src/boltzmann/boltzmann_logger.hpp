#pragma once
#ifndef BOLTZMANN_LOGGER_HPP
#define BOLTZMANN_LOGGER_HPP

#include <vector>

#include "../math/matrix/matrix_ops.hpp"
#include "../io/plot/plot.hpp"
#include "../io/gif/gif.hpp"

template <typename FloatingType>
class BoltzmannLogger {

	using State = Eigen::Matrix<FloatingType, Eigen::Dynamic, 1>;

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
	std::string states_gif_save;

	logger_buffer_t log_buf;
	GifWriterIO gio;

	bool record_visible;

public:

	BoltzmannLogger() : record_visible(false) { }

	// Note: this sets the requirement to save the states collected in the selected .gif
	void set_collect_states(bool value /*Note : I think that you are not using this variable in this scope*/,
		const std::string& into = "states.gif",
		unsigned int width = 0, unsigned int height = 0) {
		record_visible = value;
		states_gif_save = into;
		visual_width = width;
		visual_height = height;
	}

	void on_run_begin(const State& begin_state) {
		if (record_visible) {
			if (states_gif_save.empty())
				throw std::runtime_error("Cannot begin logging the states: no save file was specified.");
			const auto height = visual_width;
			const auto width = visual_height;
			if (height <= 0 || width <= 0)
				throw std::runtime_error("Cannot log the states: width and heights should have been set previously!");

			gio.begin(states_gif_save, width, height, /* delay in 10*ms */ 20);

			this->update_logger_buffer(begin_state, /* reset = */ true);
			gio.write_frame(log_buf.write_buffer.get());
		}
	}

	void on_visible_change(const State& new_state) {
		if (record_visible) {
			if (!log_buf.write_buffer)
				throw std::runtime_error("Attempting to update a non initialized buffer!");
			this->update_logger_buffer(new_state, /*reset=*/ false);
			gio.write_frame(log_buf.write_buffer.get());
		}
	}

	void on_run_end() {
		if (record_visible) {
			// Close the gif stream to finalize the gif file. Assume that the last alteration to
			// the state was logged already when on_state_update was called.
			gio.end();
		}
	}

	void do_log_visible(bool value) {
		record_visible = value;
	}

protected:

	void update_logger_buffer(const State& bs, bool reset = false,
		unsigned int low_value = 0, unsigned int high_val = 255) {
		// Ensure the buffer is initialized properly. The values for the strides are to be
		// specified in the hopfield network containing this state.
		const auto required_width = visual_width;
		const auto required_height = visual_height;
		static_assert(GifWriterIO::required_channels == sizeof(uint32_t));

		// If possible, keep the same buffer!
		if (log_buf.width != required_width || log_buf.height != required_height) {
			// NOTE: The gif-h library unfortunaly only supports rgb png writings, so that
			// we are forced to maintain 4 channels instead of just B&W. 
			log_buf.write_buffer.reset(new unsigned char[required_width * required_height *
				GifWriterIO::required_channels]);
			log_buf.width = required_width;
			log_buf.height = required_height;
		}
		if (reset) {
			std::memset(log_buf.write_buffer.get(), 0, log_buf.height * log_buf.width *
				GifWriterIO::required_channels);
		}

		// Now map the values. 
		// because (0, 0, 0, 0) = black and (255, 255, 255, 255 ) = white we just treat the
		// buffer as an int buffer instead, obtaining better memory accesses
		uint32_t* int_buffer = reinterpret_cast<uint32_t*>(log_buf.write_buffer.get());
		for (int i = 0; i < bs.size(); ++i) {
			uint32_t value = 0x00000000;
			FloatingType v = std::max(FloatingType(0.0), std::min(FloatingType(1.0), bs[i]));
			auto gray = uint32_t(v * 255);
			value = (gray << 24) | (gray << 16) | (gray << 8) | 0x0FF;
			int_buffer[i] = value;
		}
		return;
	}

};

#endif