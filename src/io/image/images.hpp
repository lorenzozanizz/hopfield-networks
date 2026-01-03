#ifndef IO_IMAGE_IMAGES_HPP
#define IO_IMAGE_IMAGES_HPP

#include <stdexcept>
#include <string>

// This module is a small lightweight wrapper to the stbi header-only library for 
// reading and saving various image formats, see more on
// https://github.com/nothings/stb/blob/master/stb_image.h
#define STB_IMAGE_IMPLEMENTATION
#include "../../../external/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../external/stb_image_write.h"

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
		Channels force_ch = Channels::Greyscale) {
		// This constructor explicitly checks that the loaded image matches the 
		// required width and height requirements
		did_free = false;
		raw_data = stbi_load(file_name.c_str(), &width, &height, &orig_channels,
			static_cast<unsigned int>(force_ch));
		channels = static_cast<unsigned int>(force_ch);
		if (!raw_data || w != width || h != height || ch != channels)
			throw std::runtime_error("Failed to load the required image: dimensions mismatch.");
	}

	Image(const std::string& file_name, Channels force_ch) {
		did_free = false;
		raw_data = stbi_load(file_name.c_str(), &width, &height, &orig_channels, static_cast<unsigned int>(force_ch));
		channels = static_cast<unsigned int>(force_ch);
		if (!raw_data || width <= 0 || height <= 0)
			throw std::runtime_error("Could not open the image " + file_name);
	}
	
	unsigned char* data() {
		return raw_data;
	}

	unsigned char* view() const {
		return raw_data;
	}

	// Allow explicit deallocation (larger images are somewhat memory heavy, e.g. 100s kb)
	void free() {
		if (!did_free)
			stbi_image_free(raw_data);
		did_free = true;
	}

	~Image() {
		// Deallocate with the stbi free method, stbi manages an internal memory
		// pool so this is required. 
		free();
	}

};

class ImageWriter {
public:

	static void write_png(const char* file_name, unsigned char* raw_data, unsigned int width,
		unsigned int height, const Channels ch) {

		if (!raw_data)
			throw std::runtime_error("Cannot write a null image.");
		stbi_write_png(file_name, width, height, static_cast<unsigned int>(ch), raw_data,
			/* stride */
		 	width * static_cast<unsigned int>(ch));
	}

	static void write_png(const std::string& file_name, unsigned char* raw_data, unsigned int width,
		unsigned int height, const Channels ch) {
		write_png(file_name.data(), raw_data, width, height, ch);
	}

	static void write_jpg(const char* file_name, unsigned char* raw_data, unsigned int width,
		unsigned int height, const Channels ch, unsigned int quality = 100) {
		if (ch == Channels::RGBA)
			throw std::runtime_error("Cannot write a jpg with a transparency channel.");
		else if (!raw_data)
			throw std::runtime_error("Cannot write a null image.");
		stbi_write_jpg(file_name, width, height, static_cast<unsigned int>(ch), raw_data,
			/* compression_quality! */
			quality);
	}
 
	static void write_jpg(const std::string& file_name, unsigned char* raw_data, unsigned int width,
		unsigned int height, const Channels ch, unsigned int quality = 100) {
		write_jpg(file_name.data(), raw_data, width, height, ch, quality);
	}
};

namespace ImageUtils {

	void force_bw_to_rgb() {

	}

	void force_rgb_to_bw() {
		// Use a standard luminance formula to convert from RGB to black and
		// white. 

	}

	void threshold_binarize(Image& img, unsigned char threshold=150) {
		// Assumes that the image binary is an image with a greyscale channel. 
		if (img.channels != 1)
			throw std::invalid_argument("Failed to binarize the required image: not black and white");
		const unsigned long long dim = img.width * img.height;
		unsigned char* d = img.data();
		for (int i = 0; i < dim; ++i) {
			if (d[i] > threshold)
				d[i] = 255;
			else
				d[i] = 0;
		}
		return; 
	}

	void background_aware_binarize(Image& img) {
		// A simple naive implementation of Otsu's method 
		// https://en.wikipedia.org/wiki/Otsu%27s_method
		// to binarize the image. This is less naive than a threshold binarization. 

		// This is thread safe but uses 4*256 = 1kb of stack 
		/* static */ int hist[256] = {0};

		// Assumes that the image binary is an image with a greyscale channel. 
		if (img.channels != 1)
			throw std::invalid_argument("Failed to binarize the required image: not black and white");

		const unsigned long long dim = img.width * img.height;
		unsigned char* d = img.data();
		// Compute the pixel intensity histogram required for otsu's algorithm
		for (int i = 0; i < dim; ++i) 
			hist[d[i]]++;

		double sum1 = 0, sumB = 0;
		double wB = 0, wF = 0;
		double maximum = 0;
		double level = 0;

		double mF, val;

		for (int i = 1; i < 256; i++)
			sum1 += i * hist[i];

		for (int ii = 0; ii < 256; ii++) {
			wF = dim - wB;
			if (wB > 0 && wF > 0) {
				mF = (sum1 - sumB) / wF;
				val = wB * wF * ((sumB / wB) - mF) * ((sumB / wB) - mF);
				if (val >= maximum) {
					level = ii;
					maximum = val;
				}
			}
			wB += hist[ii];
			sumB += (ii)*hist[ii];
		}
		// Use the computed estimate of Otsu's threshold to binarize the image. 
		threshold_binarize(img, level);
		return;
	}

}

#endif