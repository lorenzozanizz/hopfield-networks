#pragma once
#ifndef UTILITIES_HPP
#define UTILITIES_HPP


namespace Utilities {

	template <typename DataType = float>
	DataType euclidean_distance(std::unique_ptr<DataType[]>& a, std::unique_ptr<DataType[]>& b, int size) {

		DataType result = 0.0;

		for (int i = 0; i < size; i++) {
			result += (a[i] - b[i]) * (a[i] - b[i]);
		}

		return (std::sqrt(result));
	}

}

#endif