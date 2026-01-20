#include "binary.hpp"


std::ostream& operator<<(std::ostream& os, const BinaryState& ref)
{
	// Write the binary stream into the output stream. Should be easily read into
	// python (for later purposes *.*)
	os << "State size: " << ref.get_size() << " of values: \n"; // Declare size
	// Dump everything
	const auto stride_z = ref.get_stride_z();
	const auto stride_y = ref.get_stride_y();
	if (stride_z != 0) // 3d image, cannot print this lol 
	{
	}
	else if (stride_y != 0) {
		// 2d Image
		for (int i = 0; i < ref.get_size() / stride_y; ++i) {
			for (int j = 0; j < stride_y; ++j) {
				os << ((ref(i, j)) ? "1" : "0") << " ";
			}
			os << "\n";
		}
	}
	else {
		throw std::runtime_error("3D state printing not supported");
	}
	return os;
}

double BinaryState::agreement_score(const BinaryState& other) const {
	auto distance = StateUtils::hamming_distance(*this, other);
	return std::abs((double)distance - get_size()) / get_size();
}

namespace StateUtils {


	void plot_state(Plotter& p, const BinaryState& bs) {
		{
			auto ctx = p.context();
			const unsigned int height = bs.get_size() / bs.get_stride_y();
			const unsigned int width = bs.get_stride_y();


			ctx.show_binary_image(bs.data(), width, height);
		}
	}

	void perturb_state(BinaryState& bs, float alpha, unsigned long long seed) {
		std::uniform_real_distribution<float> dis(0.0, 1.0);
		std::mt19937 generator(seed);

		double random_real_01 = dis(generator);

		for (unsigned int i = 0; i < bs.get_size(); ++i) {
			if (dis(generator) < alpha)
				bs.flip(i);
		}
		return;
	}

	void write_state_as_byte_array(BinaryState& bs, unsigned char*& raw_data,
		unsigned char low_value, unsigned char high_value) {
		if (raw_data == nullptr) {
			// Allocate the required buffer amount
			raw_data = new unsigned char[bs.get_size()];
		} // Else we assume that the buffer is non null
		// Use the explicit iterator interface of the binarystate class (its a bit slower but
		// anyway)
		auto it = bs.begin();
		memset(raw_data, low_value, bs.get_size());
		for (; it != bs.end(); ++it) {
			raw_data[*it] = high_value;
		}
		return;
	}

	void write_state_as_image(const BinaryState& bs, const std::string& img, const std::string& ext) {
		std::unique_ptr<unsigned char[]> intermediate_buf(new unsigned char[bs.get_size()]);
		memset(intermediate_buf.get(), 0, bs.get_size());

		const auto width = bs.get_stride_y();
		const auto height = bs.get_size() / bs.get_stride_y();

		if (!width || !height)
			throw std::runtime_error("Cannot write to image " + img + ": state has no interpretable stride");

		if (ext == "jpg") {
			ImageWriter::write_jpg(img, intermediate_buf.get(), width, height, Channels::Greyscale);
		}
		else if (ext == "png") {
			ImageWriter::write_png(img, intermediate_buf.get(), width, height, Channels::Greyscale);

		}
	}

	void load_state_from_image(BinaryState& bs, const std::string& img, bool do_binarize) {
		// Force the channels to be binary. 
		Image image(img, Channels::Greyscale);
		if (image.width * image.height != bs.get_size())
			throw std::invalid_argument("Cannot load a binary state from a non matching image.");

		if (do_binarize)
			ImageUtils::background_aware_binarize(image);
		load_state_from_byte_array(bs, image.data(), bs.get_size());
	}

	void load_state_from_image(BinaryState& bs, Image& img, bool do_binarize) {
		if (img.channels != 1)
			throw std::invalid_argument("Cannot load a binary state from a non black and white image.");
		if (img.width * img.height != bs.get_size())
			throw std::invalid_argument("Cannot load a binary state from a non matching image.");

		// Now ascertain that the input is indeed binary, otherwise the call to load_state
		// makes no sense
		if (do_binarize) {
			ImageUtils::background_aware_binarize(img);
		}
		load_state_from_byte_array(bs, img.data(), bs.get_size());
	}

	void load_state_from_byte_array(BinaryState& bs, const unsigned char* raw_data, const state_size_t sz,
		// NOTE: Only the value representing HIGH is required, any other value is considered LOW.
		unsigned char high_value) {

		// Load the state from a byte array of given length. Notice that this function expects 
		// the byte array to have binary values low_value and high_value, so to employ this
		// function to load an image a constrat mapping is first required. 
		if (!bs.get_size()) {
			bs.set_size(sz);
			bs.realloc();
		}
		// Allow to not declare size, but if size was declared before then it must match.
		else if (bs.get_size() != sz) {
			throw std::invalid_argument("Cannot fill a state with a mismatched state size.");
		}
		// Read 8 values at a time to be a bit more performant with reads

		state_index_t i = 0;
		unsigned char byte_buffer = 0x00;

		for (i = 0; i < sz; ++i) {
			if (raw_data[i] == high_value)
				bs.set(i);
			else {
				bs.unset(i);
			}
		}

		return;
	}

	long hamming_distance(const BinaryState& s1, const BinaryState& s2) {
		// From https://dev.to/ggorantala/hamming-distance-kcm we get the
		// single-byte algorithm, apply it to the entire combination
		long distance = 0;
		// Notice that the function get_byte() already handles the edge case
		// by zeroing out non-used bits 
		for (int byte = 0; byte < s1.byte_size() - 1; ++byte) {
			unsigned char xor_val = s1.get_byte(byte) ^ s2.get_byte(byte);
			while (xor_val ^ 0) {
				if (xor_val % 2 == 1)
					distance += 1;
				xor_val >>= 1;
			}
		}
		return distance;
	}
};