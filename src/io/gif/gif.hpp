#pragma once
#ifndef IO_GIF_FOOTAGE_HPP
#define IO_GIF_FOOTAGE_HPP

//

#include <cstdint>
#include <string>
#include <vector>

// 
#include "../../../external/gif.h"

class GifWriterIO {

protected:

	// The gifwriter object g can be reinitialized when needed
	GifWriter g;

	// Save this information for repeated calls to write each frame
	// in the gif file. 
	unsigned int width;
	unsigned int height;

	unsigned int time_delay;
	std::string file_name;

	void set_internal(const std::string name, unsigned int w, unsigned int h,
		unsigned int delay) {
		width = w;
		height = h;
		time_delay = delay;
		file_name = name;
	}

public:

	GifWriterIO() { }

	GifWriterIO(const std::string name, unsigned int width, unsigned int height,
		unsigned int delay): width(width), height(height), time_delay(delay), file_name(name)
	{ }

	void begin(const std::string name, unsigned int w, unsigned int h,
		unsigned int delay) {
		// Reinitialized the local state if the gif writer has to be
		// passed around
		set_internal(name, w, h, delay);
		begin();
	}

	void begin() {
		// Assumes that global variables have been already initialized
		GifBegin(&g, file_name.data(), width, height, time_delay);
	}

	void write_frame(unsigned char* raw_data) {
		GifWriteFrame(&g, raw_data, width, height, time_delay);
	}

	void write_frame(std::vector<unsigned char> wrapped_data) {
		// Simply extract the raw data
		this->write_frame(wrapped_data.data());
	}

	void end() {
		GifEnd(&g);
	}

	class WritingContext {
		
		GifWriterIO& gf;
	public:

		WritingContext(GifWriterIO& gif_writer): gf(gif_writer) {
			gf.begin();
		}

		void write(unsigned char* data) {
			gf.write_frame(data);
		}

		~WritingContext() {
			gf.end();
		}

	};

	WritingContext initialize_writing_context() {
		return WritingContext(*this);
	}

	WritingContext initialize_writing_context(std::string file_name, int width, int height, int time_delay) {
		set_internal(file_name, width, height, time_delay);
		return initialize_writing_context();
	}

};

#endif // !IO_GIF_FOOTAGE_HPP