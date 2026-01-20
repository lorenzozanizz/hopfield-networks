
#include "gif.hpp"

#include "../../../external/gif.h"

struct GifWriterInternal {
	::GifWriter writer; // This is the C-struct from gif.h
};

void GifWriterIO::begin() {
	// Assumes that global variables have been already initialized
	if (!g) {
		// Create the implementation
		g.reset(new GifWriterInternal());
	}
	GifBegin(&g.get()->writer, file_name.data(), width, height, time_delay);
}

void GifWriterIO::write_frame(unsigned char* raw_data) {
	GifWriteFrame(&g.get()->writer, raw_data, width, height, time_delay);
}


void GifWriterIO::end() {
	GifEnd(&g.get()->writer);
}

GifWriterIO::~GifWriterIO() = default;

GifWriterIO::GifWriterIO() : g(nullptr) { }

GifWriterIO::GifWriterIO(const std::string name, unsigned int width, unsigned int height,
	unsigned int delay) : width(width), height(height), time_delay(delay), file_name(name),
	g(nullptr)
{ }