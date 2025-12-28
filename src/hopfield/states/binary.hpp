#pragma once
#ifndef NETWORKS_STATES_BINARY_HPP
#define NETWORKS_STATES_BINARY_HPP

#include <memory>
#include <cstring>
#include <stdexcept>
#include <ctype.h>
#include <cmath>

#include "../network_types.hpp"

namespace hfnets {
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

		BinaryState(state_size_t sz) : size(sz) {
			raw_data = std::make_unique<byte[]>((sz >> 3) + 1);
		}

		void set_size(state_size_t sz) {
			size = sz;
		}

		void realloc() {
			// Notice that this implicitly calls the distructor delete[] for
			// the previous raw_data value, explicitly deallocating the old
			// memory.
			raw_data = std::make_unique<byte[]>((size >> 3) + 1);
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

		inline long get(state_index_t bit_index) const {
			return raw_data[bit_index >> 3] & mask_for(bit_index % 8);
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

		protected:

		static constexpr const unsigned long long mask_for(state_index_t b) {
			return ((unsigned long long) 1 << (unsigned long long) b);
		}

		friend std::ostream& operator<<(std::ostream& os, const BinaryState& ref);

	};

	void load_state_from_stream(std::istream& input, BinaryState& bs) {
		// Load the state from an input stream (e.g. stdin or a file)
		// Assume a human readable format ('1' and '0' instead of binary string

	}

	void load_state_from_byte_array(const unsigned char* raw_data, const state_size_t sz,
		// NOTE: Only the value representing HIGH is required, any other value is considered LOW.
		BinaryState& bs, unsigned char high_value = 255) {

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
		union {
			uint64_t value;
			unsigned char bytes[8];
		} read_along;
		for (; i < ((sz >> 3) << 3); i += 8) {
			read_along.value = *( ((uint64_t*)raw_data) + (i >> 3));
			for (int k = 0; k < 8; ++k)
				if (read_along.bytes[k] == high_value)
					bs.set(i);
				else
					bs.unset(i);
		}
		// Read the remaining bytes (at most 8)
		for (; i < sz; ++i) {
			if (raw_data[i] == high_value)
				bs.set(i);
			else
				bs.unset(i);
		}
		return;
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
	
	void write_state_into_stream(std::ostream& os, const hfnets::BinaryState& ref, unsigned int dimension = 2, bool human_readable = true) {

	}

}

std::ostream& operator<<(std::ostream& os, const hfnets::BinaryState& ref) {
	// Write the binary stream into the output stream. Should be easily read into
	// python (for later purposes *.*)
	os << ref.get_size(); // Declare size
	// Dump everything
	return os;
}

#endif //!NETWORKS_STATES_BINARY_HPP