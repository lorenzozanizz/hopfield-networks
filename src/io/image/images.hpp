#ifndef IO_IMAGE_IMAGES_HPP
#define IO_IMAGE_IMAGES_HPP

#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>

// This module is a small lightweight wrapper to the stbi header-only library for 
// reading and saving various image formats, see more on
// https://github.com/nothings/stb/blob/master/stb_image.h

#include "../io_utils.hpp"

enum class Channels {

	KeepOriginalChannel = 0,
	Greyscale = 1,
	RGB = 3,
	RGBA = 4

};

class Image {

	// We do not use a unique_ptr because stbi requires that we call
	// their own free (or we could just use custom deallocators but its more complex)
	unsigned char* raw_data;
	bool did_free;

public:

	int width;
	int height;
	// Original channels is how many channels were in the image, while channels is
	// how they were forced by stbi_load 
	int orig_channels;
	int channels;

	Image(unsigned int w, unsigned int h, unsigned int ch, const std::string& file_name,
		Channels force_ch);

	Image(const std::string& file, Channels force_ch);

	inline static void trim(std::string& s) { 
		s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }), s.end()); 
	}
	
	inline unsigned char* data() {
		return raw_data;
	}

	inline unsigned char* view() const {
		return raw_data;
	}

	// Allow explicit deallocation (larger images are somewhat memory heavy, e.g. 100s kb)
	void free();

	~Image() {
		// Deallocate with the stbi free method, stbi manages an internal memory
		// pool so this is required. 
		free();
	}

};

class ImageWriter {
public:

	static void write_png(const char* file_name, unsigned char* raw_data, unsigned int width,
		unsigned int height, const Channels ch);

	inline static void write_png(const std::string& file_name, unsigned char* raw_data, unsigned int width,
		unsigned int height, const Channels ch) {
		write_png(file_name.data(), raw_data, width, height, ch);
	}

	static void write_jpg(const char* file_name, unsigned char* raw_data, unsigned int width,
		unsigned int height, const Channels ch, unsigned int quality = 100);
 
	inline static void write_jpg(const std::string& file_name, unsigned char* raw_data, unsigned int width,
		unsigned int height, const Channels ch, unsigned int quality = 100) {
		write_jpg(file_name.data(), raw_data, width, height, ch, quality);
	}
};

namespace ImageUtils {

	void threshold_binarize(unsigned char* img, unsigned int width, unsigned int height,
		unsigned char threshold = 150);

	void threshold_binarize(Image& img, unsigned char threshold = 150);

	// Niblack thresholding on a grayscale image stored in a raw unsigned char buffer.
	// width, height: image dimensions
	// window: odd window size (e.g., 15)
	// k: Niblack parameter (e.g., -0.2)
	// output: must be preallocated (unsigned char[width * height])
	void niblack_threshold(
		const unsigned char* input,
		unsigned char* output,
		int width,
		int height,
		int window,
		double k
	);

	void background_aware_binarize(unsigned char* img, unsigned int width, unsigned int height);

	void background_aware_binarize(Image& img);

	void niblack_binarize(Image& img, unsigned int window, double k = -0.2);
}

#endif