#pragma once
#ifndef DATASET_REPOSITORY_HPP
#define DATASET_REPOSITORY_HPP

#include <tuple>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <fstream>
#include <omp.h>

#include "../image/images.hpp"
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
		std::string path,
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

	template <typename FloatingType>
	void load_mit_bih(
		const std::string& path,
		unsigned int amount,
		VectorDataset<FloatVector<FloatingType>, FloatVector<FloatingType>>& entries) {
		// Load the CV, compute the one-hot representation of classes. 
		// See the Keggle page 
		// https://www.kaggle.com/datasets/josegarciamayen/mit-bih-arrhythmia-dataset-preprocessed?resource=download

		entries.reserve(amount);

		// These are also fixed, see the above link
		constexpr const auto measurement_width = 187;
		constexpr const auto num_categories = 5; 

		std::ifstream file(path);
		std::string line;
		FloatVector<FloatingType> vector(measurement_width);
		FloatVector<FloatingType> one_hot(num_categories);
		unsigned int line_no = 1;

		while (std::getline(file, line)) {
			if (line.empty() || line.rfind("#", 0) == 0) continue;

			one_hot.setZero();

			std::stringstream ss(line);
			std::string cell;
			std::vector<std::string> row;

			while (std::getline(ss, cell, ',')) {
				row.push_back(cell);
			}
			for (int i = 0; i < measurement_width; ++i )
				vector(i) = FloatingType( std::stof(row[i]) );
			one_hot(std::stoi(row[row.size() - 1])) = FloatingType(1.0);

			// Now we have read a CSV line into row, put it into a vector
			// and load it into the dataset. 

			entries.add_sample(vector, one_hot, line_no);
			line_no++;
		}
		return;
	}


	template <typename FloatingType>
	void load_real_faces_128(
		const std::string folder,
		unsigned int amount,
		VectorCollection<FloatVector<FloatingType>>& into
	) {
		// Read the required amount of file names, then read all files. a bit corny,
		// whatever!
		into.reserve(amount);
		using namespace std::filesystem;
		// Path to names.txt
		path names_file = path(folder) / "names.txt";

		std::cout << "file: " << names_file << std::endl;
		std::ifstream file(names_file);
		if (!file.is_open()) {
			throw std::runtime_error("Error: Could not open names file.");
		}

		std::string line;
		int count = 1;

		FloatVector<FloatingType> vector(64 * 64);

		while (count <= amount && std::getline(file, line)) {
			// Trim whitespace
			if (line.empty()) continue;

			// Build full path to the image
			path img_path = path(folder) / line;

			// Load image
			Image image(img_path, Channels::Greyscale);
			ImageUtils::niblack_binarize(image, /* window */ 10);

			auto raw_data = image.data();

			for (int i = 0; i < image.width * image.height; ++i)
				vector(i) = (raw_data[i] > 127)? FloatingType(1.0) : FloatingType(0.0) ;
			into.add_sample(vector, /* id */ count);
			count++;
		}

	}
}


#endif