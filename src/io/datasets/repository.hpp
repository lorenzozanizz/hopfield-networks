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

	template <typename DataType>
	using DataVector = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

	template <typename DataType>
	void load_mnist_eigen(
		const std::string path,
		unsigned int amount,
		VectorDataset<DataVector<DataType>, unsigned int>& entries,
		DataType threshold = 0, DataType low_value = DataType(0),
		DataType high_value = DataType(1)
	) {
		// Load all the vector in the input dataset. The standard file format for
		// the vectordataset is a human-readable list of numbers. 
		entries.reserve(amount);
		
		std::ifstream in(path);
		if (!in)
			throw std::runtime_error("Could not open the file stream to load the mnist dataset");

		std::string line;
		int file_amount = 0;
		unsigned int read_amt = 0;
		unsigned int collected_keys = 0;

		// Read "amount: N" 
		if (std::getline(in, line)) {
			std::istringstream iss(line);
			std::string dummy; // dummy = "amount:" 
			iss >> dummy >> file_amount;
		}

		// Variables to temporarily fill while reading the file
		DataVector<DataType> vector(28 * 28);
		vector.setZero();
		unsigned int id, label;
		std::string dummy;

		while (std::getline(in, line)) {
			// THIS WAS UPDATED TO FIX A BUG IN WHICH LABELS WOULD GET MESSED UP
			// Skip away empty lines and comments 

			if (line.empty() || line.rfind("#", 0) == 0 || line.rfind("\n", 0) == 0)
				continue;
			else if (line.rfind("label", 0) == 0) {
				// --- Label line parsing 
				std::istringstream iss(line);
				iss >> dummy >> label; // dummy = "label:" 
				collected_keys += 1;
			}
			else if (line.rfind("id", 0) == 0) {
				// --- Id line parsing
				std::istringstream iss(line);
				iss >> dummy >> id; // dummy = "id:"
				collected_keys += 1;
			}
			else if (line.rfind("data", 0) == 0) {
				// --- Data line parsing
				collected_keys += 1;
				std::istringstream iss(line);
				iss >> dummy;
				// "data:" 
				DataType value;
				for (int i = 0; i < 28 * 28; ++i) {
					iss >> value;
					if (threshold) {
						if (value > threshold) value = high_value;
						else value = low_value;
					}
					vector(i) = value;
				}
			}

			if (collected_keys == 3) {
				// Reset the MNIST image reading context, to allow for variable order
				// of data storage. 
				collected_keys = 0;

				entries.add_sample(vector, label, id);
				read_amt++;
				if (read_amt >= amount)
					break;
			}
		}

		in.close();
	}

	// Here we ignore the labels, simply storing the data as a collection.
	template <typename DataType>
	void load_mnist_eigen(
		const std::string path,
		unsigned int amount,
		VectorCollection<DataVector<DataType>>& entries,
		DataType threshold = 0, DataType low_value = DataType(0),
		DataType high_value = DataType(1)
	) {
		// Load all the vector in the input dataset. The standard file format for
		// the vectordataset is a human-readable list of numbers. 
		entries.reserve(amount);

		std::ifstream in(path);
		if (!in)
			throw std::runtime_error("Could not open the file stream to load the mnist dataset");

		std::string line;
		int file_amount = 0;
		unsigned int read_amt = 0;
		unsigned int collected_keys = 0;

		// Read "amount: N" 
		if (std::getline(in, line)) {
			std::istringstream iss(line);
			std::string dummy; // dummy = "amount:" 
			iss >> dummy >> file_amount;
		}

		// Variables to temporarily fill while reading the file
		DataVector<DataType> vector(28 * 28);
		vector.setZero();
		unsigned int id, label;
		std::string dummy;

		while (std::getline(in, line)) {
			// THIS WAS UPDATED TO FIX A BUG IN WHICH LABELS WOULD GET MESSED UP
			// Skip away empty lines and comments 

			if (line.empty() || line.rfind("#", 0) == 0 || line.rfind("\n", 0) == 0)
				continue;
			else if (line.rfind("label", 0) == 0) {
				// --- Ignore the label
				collected_keys += 1;
			}
			else if (line.rfind("id", 0) == 0) {
				// --- Id line parsing
				std::istringstream iss(line);
				iss >> dummy >> id; // dummy = "id:"
				collected_keys += 1;
			}
			else if (line.rfind("data", 0) == 0) {
				// --- Data line parsing
				collected_keys += 1;
				std::istringstream iss(line);
				iss >> dummy;
				// "data:" 
				DataType value;
				for (int i = 0; i < 28 * 28; ++i) {
					iss >> value;
					if (threshold) {
						if (value > threshold) value = high_value;
						else value = low_value;
					}
					vector(i) = value;
				}
			}

			if (collected_keys == 3) {
				// Reset the MNIST image reading context, to allow for variable order
				// of data storage. 
				collected_keys = 0;

				entries.add_sample(vector, id);
				read_amt++;
				if (read_amt >= amount)
					break;
			}
		}

		in.close();
	}
	
	void load_mnist_vector(
		std::string path,
		unsigned int amount,
		VectorDataset<std::vector<unsigned char>, unsigned int>& values
	);

	void load_mnist_ten_categories(
		std::string path,
		VectorDataset<std::vector<unsigned char>, unsigned int>& values
	);

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
		VectorDataset<FloatVector<FloatingType>, FloatVector<FloatingType>>& entries,
		unsigned int offset = 0) {
		// Load the CV, compute the one-hot representation of classes. 
		// See the Keggle page 
		// https://www.kaggle.com/datasets/josegarciamayen/mit-bih-arrhythmia-dataset-preprocessed?resource=download

		entries.reserve(amount);

		// These are also fixed, see the above link
		constexpr const auto measurement_width = 187;
		constexpr const auto num_categories = 5; 
		unsigned int read = 0;

		std::ifstream file(path);
		std::string line;
		FloatVector<FloatingType> vector(measurement_width);
		FloatVector<FloatingType> one_hot(num_categories);
		unsigned int line_no = 1;

		// Read the header line once
		std::getline(file, line);

		// Skip over offset lines of the csv. 
		if (offset)
			for (int i = 0; i < offset; ++i)
				std::getline(file, line);
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
			read++;
			if (read >= amount)
				break;
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