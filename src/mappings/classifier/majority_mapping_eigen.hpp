#pragma once
#ifndef MAJORITY_MAPPINGS_EIGEN_HPP
#define MAJORITY_MAPPINGS_EIGEN_HPP

#include <memory>
#include <iterator>
#include <cmath>
#include <vector>
#include <map>
#include <random>
#include <limits>
#include <Eigen/Dense>
#include <stdexcept>
#include "../konohen_mapping_eigen.hpp"
#include "../../io/datasets/dataset_eigen.hpp"
#include "../../io/plot/plot.hpp"
#include "../../math/matrix/matrix_ops.hpp"


template <typename DataType = float>
class MajorityMappingEigen {

private:

	KonohenMapEigen<DataType>& trained_map;
	double threshold; // the percentage of hits required to label a neuron
	std::map<int, std::string>  labels_map; // this maps a label idx to its label
	std::map<int, Eigen::VectorXd> hits_map; // this maps a neuron idx to a vector that keeps track of how many times each label hits that neuron
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> map_neuron_label; // this maps a neuron idx to its label idx 

	const int x_from_idx(int idx) {
		return idx % trained_map.get_map_width();
	}

	const int y_from_idx(int idx) {
		return idx / trained_map.get_map_width(); 
	}

	const int from_xy_to_idx(int x, int y) {
		return x + y * trained_map.get_map_width();
	}

public:

	MajorityMappingEigen(KonohenMapEigen<DataType>& trained_map, double th, std::map<int, std::string> labels_map) :
		trained_map(trained_map), threshold(th), labels_map(labels_map) {
		initialize();
	}

	void initialize() {
		std::map<int, Eigen::VectorXd> hits;
		int size = labels_map.size();
		int num_neurons = trained_map.get_map_width() * trained_map.get_map_height();
		// initializing hits_map, allocating for each neuron a vector of the correct size
		for (int idx = 0; idx < num_neurons; ++idx) {
			Eigen::VectorXd vector = Eigen::VectorXd::Zero(size);
			hits[idx] = vector;
		}
		hits_map = hits;
		Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> M(
			trained_map.get_map_width(), trained_map.get_map_height());
		map_neuron_label = M;

	}

	// Setting a new threshold, i.e. the percentage of hits required to label a neuron
	void set_threshold(double new_th) {
		threshold = new_th;
	}

	// Pass a labeled set of data in order to classify the neurons of the trained map
	void classify(const DatasetEigen<DataType, int>& dataset) { 

		int num_neurons = trained_map.get_map_width() * trained_map.get_map_height();

		std::cout << " classifing..." << "\n";
		std::cout << " dataset size : " << dataset.size() << "\n\n";

		for (size_t i = 0; i < dataset.size(); ++i) {

			// input vector
			const auto& x = dataset.x_of(i);
			// true label
			int label_id = dataset.y_of(i);
			std::cout << "label: " << label_id << "\n";
			// BMU
			int neuron_idx = trained_map.map(x);

			// update hit counter
			for (size_t i = 0; i < dataset.size(); ++i) {

				const auto& x = dataset.x_of(i);
				int label_id = dataset.y_of(i);

				int neuron_idx = trained_map.map(x);

				if (neuron_idx < 0 || neuron_idx >= num_neurons) {
					throw std::out_of_range(
						"Invalid BMU index: " + std::to_string(neuron_idx)
					);
				}

				if (label_id == -1) {
					continue;  // explicitly ignored label
				}

				if (label_id < 0 || label_id >= static_cast<int>(labels_map.size())) {
					throw std::out_of_range(
						"Invalid label id: " + std::to_string(label_id)
					);
				}

				hits_map[neuron_idx](label_id) += 1;
			}

			
		}

		// assign labels to neurons
		int width = trained_map.get_map_width();
		int height = trained_map.get_map_height();

		for (int j = 0; j < height; ++j) {
			for (int i = 0; i < width; ++i) {

				map_neuron_label(i, j) = most_frequent_hit(from_xy_to_idx(i, j));

			}
		}

		for (int j = 0; j < height; ++j) {
			for (int i = 0; i < width; ++i) {

				//std::cout << "label at ( " << i << ", " << j <<" ) is " << map_neuron_label(i, j) << "\n\n";


			}
		}
	
	}

	// Warning! You can call this fuction only after calling classify()!
	std::string label_for(int idx) {
		int label_idx = map_neuron_label(x_from_idx(idx), y_from_idx(idx));
		if (label_idx < 0) return "unlabeled";
		return labels_map[label_idx];
	}

	// Warning! You can call this fuction only after calling classify()!
	std::string label_for(int x, int y) {
		int label_idx = map_neuron_label(x, y);
		if (label_idx < 0) return "unlabeled";
		return labels_map[label_idx];
	}

	// This function takes the index of a neuron and returns the index of the label of its most frequent hits if that is grater
	// than the set threshold. Otherwise is returned -1, that happens if a neuron is hitted by several different data or if it's not hit at all.
	int most_frequent_hit(int idx) {

		Eigen::VectorXd labels = hits_map[idx];
		Eigen::Index maxIdx;

		//std::cout << "labels " << labels << "\n\n";

		int total = labels.sum();
		if (total == 0) return -1;

		//std::cout << "sum " << total << "\n\n";

		float max = labels.maxCoeff(&maxIdx);

		//std::cout << "max  " << max  << " and coefficient "<< maxIdx << "\n\n";

		if (max >= threshold * total) {
			return maxIdx;
		}
		return -1;
	}

	void plot(Plotter& plotter) {

		int width = trained_map.get_map_width();
		int height = trained_map.get_map_height();
		int num_classes = labels_map.size();

		std::vector<int> labels(width * height);

		for (int j = 0; j < height; ++j) {
			for (int i = 0; i < width; ++i) {
				int lbl = map_neuron_label(i, j);
				if (lbl < 0) lbl = -1; //unknown class
				labels[from_xy_to_idx(i,j)] = lbl + 1;
			}
		}
		plotter.context()
			.set_title("SOM Classification Map")
			.show_discrete_categories(labels, trained_map.get_map_width(),
				trained_map.get_map_height(), num_classes + 1,
			/* zero category is black */ true);

		plotter.block();
	}

};
#endif