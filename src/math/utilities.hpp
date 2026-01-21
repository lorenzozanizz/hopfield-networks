#pragma once
#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <cmath>
#include <thread>
#include <iostream>
#include <vector>

namespace Utilities {

	template <typename DataType = float>
	DataType euclidean_distance(const std::unique_ptr<DataType[]>& a,const std::unique_ptr<DataType[]>& b, int size) {

		DataType result = 0.0;

		for (int i = 0; i < size; i++) {
			result += (a[i] - b[i]) * (a[i] - b[i]);
		}

		return (std::sqrt(result));
	}

	template <typename DataType = float>
	DataType euclidean_distance_1d(DataType a, DataType b) {
		return std::sqrt((a-b)*(a-b));
	}

	template <typename DataType = float>
	DataType euclidean_distance_2d(DataType x_a, DataType x_b, DataType y_a, DataType y_b) {
		DataType result = 0.0;
		result += (x_a - x_b) * (x_a - x_b); 
		result += (y_a - y_b) * (y_a - y_b); 
		return std::sqrt(result);
	}

	void eigen_set_num_threads(int num_threads = -1) {
		if (num_threads ==  -1)
			Eigen::setNbThreads(std::thread::hardware_concurrency());
		else 
			Eigen::setNbThreads(num_threads);
	}

	void eigen_init_parallel(int num_threads = 0) {
		Eigen::initParallel();
		if (num_threads)
			eigen_set_num_threads(num_threads);
	}

	int eigen_get_num_threads() {
		return Eigen::nbThreads();
	}

}

#endif