#pragma once
#ifndef HOPFIELD_STATES_BINARY_HPP
#define HOPFIELD_STATES_BINARY_HPP

#include <memory>
#include <cstring>
#include <stdexcept>
#include <ctype.h>
#include <cmath>
#include <random>

#include "../network_types.hpp"
#include "../../io/image/images.hpp"
#include "../../io/plot/plot.hpp"
#include "../../io/io_utils.hpp"

// A simple class to contain a binary state, with the obvious optimization of using a 
// binary string to contain the -1/1 pair of values. This reduces storage by a factor 
// of 32 (for ints and floats) and by a factor of 8 for black and white pngs (a single 
// byte represents a bit)
// 
// As an example, imagine the average image png as being ca. 16 kb uncompressed, this 
// encoding brings it down to a more manageable 2 kb (sometimes we may need to store 
// patterns in memory e.g. for matrix free weighting policies!
// 
class BinaryState {

protected:

	using byte = unsigned char;

	std::unique_ptr<byte[]> raw_data;
	state_size_t size;

	// Allow the state to be indexed with different strides, as
	// per user request. This means for example when an access over
	// two indices is requested, the returnet value is ind1 * offset_y + ind2
	state_size_t stride_y;
	state_size_t stride_z;
	// NOTE: stride_y should be read like "Stride to advance elements in y dimension"
	// (like for a 2d array, to read column-wise you have a stride_y of row_size
	// so that stride_y == row_size for 2d arrays)

public:

	BinaryState(): size(0) { }

	BinaryState(state_size_t sz) : size(sz), raw_data(nullptr) {
		raw_data = std::make_unique<byte[]>((sz >> 3) + 1);
		stride_y = stride_z = 0;
	}

	BinaryState& operator=(const BinaryState& ref) {
		raw_data.reset(new unsigned char[ref.byte_size()]);
		memcpy(raw_data.get(), ref.data(), ref.byte_size());

		stride_y = ref.stride_y;
		stride_z = ref.stride_z;
		size = ref.size;

		return *this;
	}

	void copy_content(const BinaryState& ref) {
		if (ref.get_size() != size) {
			throw std::runtime_error("Attempting to copy the content of a state with a different size.");
		}
		raw_data.reset(new unsigned char[ref.byte_size()]);
		memcpy(raw_data.get(), ref.data(), ref.byte_size());
	}

	void set_size(state_size_t sz) {
		size = sz;
	}

	void set_stride_y(state_size_t sz) {
		stride_y = sz;
	}

	void set_stride_z(state_size_t sz) {
		stride_z = sz;
	}

	state_size_t get_stride_y() const {
		return stride_y;
	}

	state_size_t get_stride_z() const {
		return stride_z;
	}

	bool stride_equals(const BinaryState& other) {
		return (other.stride_y == stride_y && other.stride_z == stride_z);
	}

	void realloc(bool initialize = true) {
		// Notice that this implicitly calls the distructor delete[] for
		// the previous raw_data value, explicitly deallocating the old
		// memory.
		raw_data = std::make_unique<byte[]>((size >> 3) + 1);
		if (initialize)
			clear();
	}
	
	void clear() {
		// Clear the entire memory pool
		if (raw_data)
			std::memset(raw_data.get(), 0, byte_size());
	}

	inline int operator()(state_index_t ind_i, state_index_t ind_j, state_index_t ind_k) const {
		// Triple index accessing:
		const auto bt = ind_i * stride_z + ind_j * stride_y + ind_k;
		return get(bt);
	}

	inline int operator()(state_index_t ind_i, state_index_t ind_j) const {
		// Double index accessing:
		const auto bt = ind_i * stride_y + ind_j;
		return get(bt);
	}

	unsigned int byte_size() const {
		return (size >> 3) + 1;
	}

	inline int operator()(state_index_t ind_i) const {
		// Single index accessing:
		return get(ind_i);
	}

	inline byte& get_byte(state_index_t ind) {
		// Non const-version needed for assignment state[byte_no] = x
		return raw_data[ind];
	}

	state_size_t get_size() const {
		return size;
	}

	unsigned char get_byte(state_index_t i) const {
		if (i == (size/8)) {
			unsigned char b = raw_data[i];
			// Notice this bit hack: we get the mask with 1s only in the allowed values
			// by subtracting from the next-possible bit..
			b &= ((unsigned long)1 << (size % 8)) - 1;
			return b;
		}
		return raw_data[i];
	}

	void flip(state_index_t bit_index) {
		raw_data[bit_index >> 3] ^= mask_for((bit_index % 8));
		return;
	}

	inline void set(state_index_t bit_index) {
		raw_data[bit_index >> 3] |= mask_for((bit_index % 8));
		return;
	}

	inline void unset(state_index_t bit_index) {
		raw_data[bit_index >> 3] &= ~mask_for((bit_index % 8));
		return;
	}

	inline void set_value(state_index_t t, bool value) {
		if (value)
			set(t);
		else unset(t);
	}

	inline long get(state_index_t bit_index) const {
		return raw_data[bit_index >> 3] & mask_for(bit_index % 8);
	}

	inline bool high(state_index_t bit_index) const {
		return (raw_data[bit_index >> 3] & mask_for(bit_index % 8)) != 0;
	}
		
	class BinaryStateIterator {

		const BinaryState& bs;
		unsigned long index;

	public:
		BinaryStateIterator(const BinaryState& c, long start)
			: bs(c), index(start) {
			while (index < bs.get_size() && !bs.get(index)) {
				++index;
			}
		}

		// This is a constant iterator, you cannot edit the values
		long operator*() const {
			return index;
		}

		BinaryStateIterator& operator++() {
			do {
				++index;
			} while (index < bs.get_size() && !bs.get(index));
			return *this;
		}

		bool operator!=(const BinaryStateIterator& other) const {
			return index != other.index;
		}

		bool operator<(const BinaryStateIterator& other) const {
			return index < other.index;
		}

		bool operator==(const BinaryStateIterator& other) const {
			return index == other.index;
		}

	};

	BinaryStateIterator begin() const {
		return BinaryStateIterator(*this, 0);
	}

	BinaryStateIterator end() const {
		return BinaryStateIterator(*this, size);
	}

	unsigned char* data() const {
		return raw_data.get();
	}

	protected:

	static constexpr const unsigned long long mask_for(state_index_t b) {
		return ((unsigned long long) 1 << (unsigned long long) b);
	}

	friend std::ostream& operator<<(std::ostream& os, const BinaryState& ref);

};

namespace StateUtils {


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

	void load_state_from_stream(std::istream& input, BinaryState& bs) {
		// Load the state from an input stream (e.g. stdin or a file)
		// Assume a human readable format ('1' and '0' instead of binary string

	}

	void load_state_from_byte_array(BinaryState& bs, const unsigned char* raw_data, const state_size_t sz,
		// NOTE: Only the value representing HIGH is required, any other value is considered LOW.
		unsigned char high_value = 255) {

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

	void load_state_from_image(BinaryState& bs, Image& img, bool do_binarize = false) {
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


	void load_state_from_image(BinaryState& bs, const std::string& img, bool do_binarize = false) {
		// Force the channels to be binary. 
		Image image(img, Channels::Greyscale);
		if (image.width * image.height != bs.get_size())
			throw std::invalid_argument("Cannot load a binary state from a non matching image.");

		if (do_binarize)
			ImageUtils::background_aware_binarize(image);
		load_state_from_byte_array(bs, image.data(), bs.get_size());
	}

	void write_state_as_image(const BinaryState& bs, const std::string& img, const std::string& ext = "png") {
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


	void write_state_as_byte_array(BinaryState& bs, unsigned char*& raw_data,
		unsigned char low_value = 0, unsigned char high_value = 255) {
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

	void perturb_state(BinaryState& bs, float alpha, unsigned long long seed = 0xcafebabe) {
		std::uniform_real_distribution<float> dis(0.0, 1.0);
		std::mt19937 generator(seed);

		double random_real_01 = dis(generator);

		for (unsigned int i = 0; i < bs.get_size(); ++i) {
			if (dis(generator) < alpha)
				bs.flip(i);
		}
		return;
	}

	void plot_state(Plotter& p, const BinaryState& bs) {
		{
			auto ctx = p.context();
			const unsigned int height = bs.get_size() / bs.get_stride_y();
			const unsigned int width = bs.get_stride_y();

		
			ctx.show_binary_image(bs.data(), width, height);
		}
	}

		
} // end namespace StateUtils

std::ostream& operator<<(std::ostream& os, const BinaryState& ref) {
	// Write the binary stream into the output stream. Should be easily read into
	// python (for later purposes *.*)
	os << "State size: " << ref.get_size() << " of values: \n"; // Declare size
	// Dump everything
	const auto stride_z = ref.get_stride_z();
	const auto stride_y = ref.get_stride_y();
	if (stride_z != 0) // 3d image, cannot print this lol 
	{ }
	else if (stride_y != 0) {
		// 2d Image
		for (int i = 0; i < ref.get_size() / stride_y; ++i) {
			for (int j = 0; j < stride_y; ++j) {
				os << ((ref(i, j))? "1" : "0") << " ";
			}
			os << "\n";
		}
	}
	else {
		throw std::runtime_error("3D state printing not supported");
	}
	return os;
}

#endif //!HOPFIELD_STATES_BINARY_HPP