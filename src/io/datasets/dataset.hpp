#pragma once
#ifndef IO_DATASET_DATASET_HPP
#define IO_DATASET_DATASET_HPP

#include <tuple>
#include <vector>


template <typename XType, typename YType>
class Dataset {

	using index_t = unsigned long long;

public:

	index_t size() {

	}

	YType& y_of(index_t index) const {

	}

	XType& x_of(index_t index) const {

	}

	std::tuple<XTuple, YTuple>& const {

	}

};

namespace DatasetUtils {

	void load_dataset_mnist() {

	}

}


#endif