#pragma once
#ifndef DATASET_REPOSITORY_HPP
#define DATASET_REPOSITORY_HPP

#include <tuple>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream> 
#include <cmath>
#include <fstream>

#include "dataset.hpp"
#include "data_collection.hpp"
#include "../../math/matrix/matrix_ops.hpp"

namespace DatasetRepo {

	using IntVector = Eigen::Matrix<unsigned int, Eigen::Dynamic, 1>;

	void load_mnist_eigen(
		std::string& path,
		unsigned int amount,
		VectorDataset<IntVector, unsigned int>& entries
	) {
		// Load all the vector in the input dataset. The standard file format for
		// the vectordataset is a human-readable list of numbers. 
		entries.reserve(amount);
		
		std::ifstream in(path);
		if (!in)
			std::runtime_error("Could not open the file stream to load the mnist dataset");

		std::string line;
		int file_amount = 0;

		// Read "amount: N" 
		if (std::getline(in, line)) {
			std::istringstream iss(line);
			std::string dummy; // dummy = "amount:" 
			iss >> dummy >> file_amount;
		}

		IntVector vector(28 * 28);
		while (std::getline(in, line)) {
			// Skip away empty lines and comments 
			unsigned int id, label;
			if (line.empty() || line.rfind("#", 0) == 0) continue;
			// --- id --- 
			{
				std::istringstream iss(line);
				std::string dummy;
				iss >> dummy >> id; // dummy = "id:"
			}
			// --- label ---
			std::getline(in, line);
			{
				std::istringstream iss(line);
				std::string dummy;
				iss >> dummy >> label; // dummy = "label:" 
			}
			// --- data --- 
			std::getline(in, line);
			{
				std::istringstream iss(line);
				std::string dummy; iss >> dummy;
				// "data:" 
				int value;
				for (int i = 0; i < 28 * 28; ++i) {
					iss >> value;
					vector(i) = value;
				}
			}
		
			entries.add_sample(vector, label, id);
		}
	}

	void load_mnist_vector(
		std::string& path,
		unsigned int amount,
		VectorDataset<std::vector<unsigned char>, unsigned int>& values
	) {
		values.reserve(amount);

		std::ifstream in(path);
		if (!in)
			std::runtime_error("Could not open the file stream to load the mnist dataset");

		std::string line;
		int file_amount = 0;

		// Read "amount: N" 
		if (std::getline(in, line)) {
			std::istringstream iss(line);
			std::string dummy; // dummy = "amount:" 
			iss >> dummy >> file_amount;
		}

		// This version uses a c++ vector instead of an eigen vector
		std::vector<unsigned char> vector(28 * 28);
		while (std::getline(in, line)) {
			// Skip away empty lines and comments 
			unsigned int id, label;
			if (line.empty() || line.rfind("#", 0) == 0) continue;
			// --- id --- 
			{
				std::istringstream iss(line);
				std::string dummy;
				iss >> dummy >> id; // dummy = "id:"
			}
			// --- label ---
			std::getline(in, line);
			{
				std::istringstream iss(line);
				std::string dummy;
				iss >> dummy >> label; // dummy = "label:" 
			}
			// --- data --- 
			std::getline(in, line);
			{
				std::istringstream iss(line);
				std::string dummy; iss >> dummy;
				// "data:" 
				int value;
				for (int i = 0; i < 28 * 28; ++i) {
					iss >> value;
					vector[i] = value;
				}
			}

			values.add_sample(vector, label, id);
		}
	}

	template <typename FloatingType>
	using FloatVector = Eigen::Matrix<FloatingType, Eigen::Dynamic, 1>;


	template <typename FloatingType>
	void load_sine_time_series(
		unsigned int amount,
		VectorDataset<FloatVector<FloatingType>, FloatVector<FloatingType>>& entries
	) {

		if (amount <= 0)
			throw std::runtime_error("Illegal size for dataset");

		constexpr const auto in_size = 30;
		constexpr const auto out_size = 10;

		FloatVector<FloatingType> in_vector(in_size);
		FloatVector<FloatingType> out_vector(out_size);

		double dt = 0.1;

		for (unsigned int i = 0; i < amount; ++i) {

			for (int j = 0; j < in_size; ++j)
				in_vector(j) = std::sin(i + dt * j);
			for (int j = 0; j < out_size; ++j)
				out_vector(j) = std::sin(i + (dt * (j + in_size) ));

			entries.add_sample(in_vector, out_vector, /* id */i);
		}
		return;
	}


}


#endif