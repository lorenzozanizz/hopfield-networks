#pragma once
#ifndef MAJORITY_MAPPINGS_HPP
#define MAJORITY_MAPPINGS_HPP

#include <memory>
#include <iterator>
#include <cmath>
#include <vector>
#include <map>
#include <random>
#include <limits>
#include "../konohen_mapping.hpp"
#include "../../io/datasets/dataset.hpp"

template <typename DataType = float>
class MajorityMapping {

private:

	KonohenMap<DataType> trained_map;
	double threshold; // the percentage of hits required to label a neuron
	std::map<int, std::string>  labels_map; // this maps a label idx to its label
	std::map<int, std::vector<int>> hits_map; // this maps a neuron idx to a vector that keeps track of how many times each label hits that neuron
	std::map<int, int> map_neuron_label; // this maps a neuron idx to its label idx 

public:

	MajorityMapping(KonohenMap<DataType>& trained_map, double th, std::map<int, std::string> labels_map) :
		trained_map(trained_map), threshold(th), labels_map(labels_map) {
		initialize();
	}

	void initialize() {
		std::map<int, std::vector<int>> hits;
		int size = labels_map.size();
		int num_neurons = trained_map.get_map_width() * trained_map.get_map_height();
		// initializing hits_map, allocating for each neuron a vector of the correct size
		for (int idx = 0; idx < num_neurons; ++idx) {
			std::vector<int> vector(size, 0);
			hits[idx] = vector;
		}
		hits_map = hits;
	}

	// Setting a new threshold, i.e. the percentage of hits required to label a neuron
	void set_threshold(double new_th) {
		threshold = new_th;
	}

	// Pass a labeled set of data in order to classify the neurons of the trained map
	void classify(const Dataset<std::unique_ptr<DataType[]>, int>& dataset) {

		for (size_t i = 0; i < dataset.size(); ++i) {

			// input vector
			const auto& x = dataset.x_of(i);

			// true label
			int label_id = dataset.y_of(i);

			// BMU
			int neuron_idx = trained_map.map(x);

			// update hit counter
			hits_map[neuron_idx][label_id]++;
		}

		// assign labels to neurons
		int num_neurons = trained_map.get_map_width() * trained_map.get_map_height();
		for (int idx = 0; idx < num_neurons; ++idx) {
			map_neuron_label[idx] = most_frequent_hit(idx);
		}
	}

	// Warning! You can call this fuction only after calling classify()!
	std::string label_for(int neuron_idx) {
		int label_idx = map_neuron_label[neuron_idx];
		if (label_idx < 0) return "unlabeled";
		return labels_map[label_idx];
	}

	// This function takes the index of a neuron and returns the index of the label of its most frequent hits if that is grater
	// than the set threshold. Otherwise is returned -1, that happens if a neuron is hitted by several different data or if it's not hit at all.
	int most_frequent_hit(int idx) {
		const auto& labels = hits_map.at(idx);
		int total = 0;

		for (int c : labels) total += c;
		if (total == 0) return -1;

		for (int i = 0; i < labels.size(); ++i) {
			if (labels[i] >= threshold * total) {
				return i;
			}
		}
		return -1;
	}

};
#endif