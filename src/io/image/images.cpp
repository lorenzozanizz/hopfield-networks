#include "images.hpp"

/*



*/
#define STB_IMAGE_IMPLEMENTATION
#include "../../../external/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../external/stb_image_write.h"


Image::Image(unsigned int w, unsigned int h, unsigned int ch, const std::string& file_name,
	Channels force_ch) {
	// This constructor explicitly checks that the loaded image matches the 
	// required width and height requirements
	did_free = false;
	raw_data = stbi_load(file_name.c_str(), &width, &height, &orig_channels,
		static_cast<unsigned int>(force_ch));
	channels = static_cast<unsigned int>(force_ch);
	if (!raw_data || w != width || h != height || ch != channels)
		throw std::runtime_error("Failed to load the required image: dimensions mismatch.");
}

Image::Image(const std::string& file, Channels force_ch) {
	std::string file_name = file;
	trim(file_name);
	did_free = false;
	raw_data = stbi_load(file_name.c_str(), &width, &height, &orig_channels, static_cast<unsigned int>(force_ch));
	channels = static_cast<unsigned int>(force_ch);
	if (!raw_data || width <= 0 || height <= 0)
		throw std::runtime_error("Could not open the image " + file_name);
}

void Image::free() {
	if (!did_free)
		stbi_image_free(raw_data);
	did_free = true;
}

void ImageWriter::write_png(const char* file_name, unsigned char* raw_data, unsigned int width,
	unsigned int height, const Channels ch) {

	if (!raw_data)
		throw std::runtime_error("Cannot write a null image.");
	stbi_write_png(file_name, width, height, static_cast<unsigned int>(ch), raw_data,
		/* stride */
		width * static_cast<unsigned int>(ch));
}

void ImageWriter::write_jpg(const char* file_name, unsigned char* raw_data, unsigned int width,
	unsigned int height, const Channels ch, unsigned int quality) {
	if (ch == Channels::RGBA)
		throw std::runtime_error("Cannot write a jpg with a transparency channel.");
	else if (!raw_data)
		throw std::runtime_error("Cannot write a null image.");
	stbi_write_jpg(file_name, width, height, static_cast<unsigned int>(ch), raw_data,
		/* compression_quality! */
		quality);
}

namespace ImageUtils {

	void threshold_binarize(unsigned char* img, unsigned int width, unsigned int height,
		unsigned char threshold) {
		const unsigned long long dim = width * height;
		unsigned char* d = img;
		for (int i = 0; i < dim; ++i) {
			if (d[i] > threshold)
				d[i] = 255;
			else
				d[i] = 0;
		}
		return;
	}

	void threshold_binarize(Image& img, unsigned char threshold) {
		// Assumes that the image binary is an image with a greyscale channel. 
		if (img.channels != 1)
			throw std::invalid_argument("Failed to binarize the required image: not black and white");
		const unsigned long long dim = img.width * img.height;
		unsigned char* d = img.data();
		threshold_binarize(d, img.width, img.height, threshold);
	}

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
	) {
		int r = window / 2;  // radius

		// Precompute integral images for fast mean and variance
		std::vector<double> integral((width + 1) * (height + 1), 0.0);
		std::vector<double> integral_sq((width + 1) * (height + 1), 0.0);

		auto idx = [&](int x, int y) { return y * (width + 1) + x; };

		// Build integral images
		for (int y = 1; y <= height; ++y) {
			double row_sum = 0.0;
			double row_sum_sq = 0.0;
			for (int x = 1; x <= width; ++x) {
				unsigned char val = input[(y - 1) * width + (x - 1)];
				row_sum += val;
				row_sum_sq += val * val;

				integral[idx(x, y)] = integral[idx(x, y - 1)] + row_sum;
				integral_sq[idx(x, y)] = integral_sq[idx(x, y - 1)] + row_sum_sq;
			}
		}

		// Apply Niblack threshold
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {

				int x1 = std::max(0, x - r);
				int y1 = std::max(0, y - r);
				int x2 = std::min(width - 1, x + r);
				int y2 = std::min(height - 1, y + r);

				// Convert to integral-image coordinates (+1)
				int ix1 = x1 + 1, iy1 = y1 + 1;
				int ix2 = x2 + 1, iy2 = y2 + 1;

				int area = (ix2 - ix1 + 1) * (iy2 - iy1 + 1);

				double sum =
					integral[idx(ix2, iy2)] -
					integral[idx(ix1 - 1, iy2)] -
					integral[idx(ix2, iy1 - 1)] +
					integral[idx(ix1 - 1, iy1 - 1)];

				double sum_sq =
					integral_sq[idx(ix2, iy2)] -
					integral_sq[idx(ix1 - 1, iy2)] -
					integral_sq[idx(ix2, iy1 - 1)] +
					integral_sq[idx(ix1 - 1, iy1 - 1)];

				double mean = sum / area;
				double variance = (sum_sq / area) - (mean * mean);
				double stddev = variance > 0 ? std::sqrt(variance) : 0.0;

				double T = mean + k * stddev;

				unsigned char pixel = input[y * width + x];
				output[y * width + x] = (pixel > T ? 255 : 0);
			}
		}
	}

	void background_aware_binarize(unsigned char* img, unsigned int width, unsigned int height) {
		// A simple naive implementation of Otsu's method 
		// https://en.wikipedia.org/wiki/Otsu%27s_method
		// to binarize the image. This is less naive than a threshold binarization. 

		// This is thread safe but uses 4*256 = 1kb of stack 
		/* static */ int hist[256] = { 0 };

		// Assumes that the image binary is an image with a greyscale channel. 

		const unsigned long long dim = width * height;
		unsigned char* d = img;
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
		threshold_binarize(img, width, height, level);
		return;
	}

	void background_aware_binarize(Image& img) {
		// A simple naive implementation of Otsu's method 
		// https://en.wikipedia.org/wiki/Otsu%27s_method
		// to binarize the image. This is less naive than a threshold binarization. 

		// This is thread safe but uses 4*256 = 1kb of stack 
		/* static */ int hist[256] = { 0 };

		// Assumes that the image binary is an image with a greyscale channel. 
		if (img.channels != 1)
			throw std::invalid_argument("Failed to binarize the required image: not black and white");

		const unsigned long long dim = img.width * img.height;
		unsigned char* d = img.data();
		background_aware_binarize(d, img.width, img.height);
		return;
	}

	void niblack_binarize(Image& img, unsigned int window, double k) {
		std::unique_ptr<unsigned char> input(new unsigned char[img.width * img.height]);
		std::memcpy(input.get(), img.data(), img.width * img.height);

		niblack_threshold(input.get(), img.data(), img.width, img.height, window, k);
	}

}