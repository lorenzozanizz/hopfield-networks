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
#include "../../math/matrix/matrix_ops.hpp"

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

	// Copy constructor
	BinaryState(const BinaryState& ref) {
		raw_data.reset(new unsigned char[ref.byte_size()]);
		memcpy(raw_data.get(), ref.data(), ref.byte_size());

		stride_y = ref.stride_y;
		stride_z = ref.stride_z;
		size = ref.size;
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

	inline void set_stride_y(state_size_t sz) {
		stride_y = sz;
	}

	inline void set_stride_z(state_size_t sz) {
		stride_z = sz;
	}

	inline state_size_t get_stride_y() const {
		return stride_y;
	}

	inline state_size_t get_stride_z() const {
		return stride_z;
	}

	inline bool stride_equals(const BinaryState& other) {
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
	
	inline void clear() {
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

	inline unsigned int byte_size() const {
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

	inline state_size_t get_size() const {
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

	inline void flip(state_index_t bit_index) {
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
		inline long operator*() const {
			return index;
		}

		inline BinaryStateIterator& operator++() {
			do {
				++index;
			} while (index < bs.get_size() && !bs.get(index));
			return *this;
		}

		inline bool operator!=(const BinaryStateIterator& other) const {
			return index != other.index;
		}

		inline bool operator<(const BinaryStateIterator& other) const {
			return index < other.index;
		}

		inline bool operator==(const BinaryStateIterator& other) const {
			return index == other.index;
		}

	};

	double agreement_score(const BinaryState& other) const;

	inline BinaryStateIterator begin() const {
		return BinaryStateIterator(*this, 0);
	}

	inline BinaryStateIterator end() const {
		return BinaryStateIterator(*this, size);
	}

	inline unsigned char* data() const {
		return raw_data.get();
	}

	protected:

	inline static constexpr const unsigned long long mask_for(state_index_t b) {
		return ((unsigned long long) 1 << (unsigned long long) b);
	}

	friend std::ostream& operator<<(std::ostream& os, const BinaryState& ref);

};

namespace StateUtils {


	long hamming_distance(const BinaryState& s1, const BinaryState& s2);

	void load_state_from_byte_array(BinaryState& bs, const unsigned char* raw_data, const state_size_t sz,
		// NOTE: Only the value representing HIGH is required, any other value is considered LOW.
		unsigned char high_value = 255);

	void load_state_from_image(BinaryState& bs, Image& img, bool do_binarize = false);

	void load_state_from_image(BinaryState& bs, const std::string& img, bool do_binarize = false);

	void write_state_as_image(const BinaryState& bs, const std::string& img, const std::string& ext = "png");

	void write_state_as_byte_array(BinaryState& bs, unsigned char*& raw_data,
		unsigned char low_value = 0, unsigned char high_value = 255);

	void perturb_state(BinaryState& bs, float alpha, unsigned long long seed = 0xcafebabe);

	void plot_state(Plotter& p, const BinaryState& bs);
		
} // end namespace StateUtils

std::ostream& operator<<(std::ostream& os, const BinaryState& ref);

#endif //!HOPFIELD_STATES_BINARY_HPP