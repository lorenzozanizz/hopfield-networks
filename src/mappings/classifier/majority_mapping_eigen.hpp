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
	using DoubleMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
	using IntMatrix = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

	KonohenMapEigen<DataType>& trained_map;
	double threshold; // the percentage of hits required to label a neuron
	std::map<int, std::string>  labels_map; // this maps a label idx to its label
	std::map<int, Eigen::VectorXd> hits_map; // this maps a neuron idx to a vector that keeps track of how many times each label hits that neuron
	IntMatrix map_neuron_label; // this maps a neuron idx to its label idx 

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
	using DoubleVector = Eigen::Matrix<double, Eigen::Dynamic, 1>; 
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

	// Computing the "matches" between the input data and the nodes trained in our Konohen map
	void compute_hits(const VectorDataset<DoubleVector, unsigned int>& dataset,
		int batch_size) {

		int input_size = trained_map.get_input_size();

		for (const auto& batch : dataset.batches(batch_size)) {

			Eigen::MatrixXd X(input_size, batch_size); // rows X cols

			for (int i = 0; i < batch_size; ++i)
				X.col(i) = batch.x_of(i);

			Eigen::VectorXi bmus = trained_map.map(X);

			for (int i = 0; i < batch_size; ++i) {
				int neuron = bmus(i);
				int label = batch.y_of(i);
				hits_map[neuron](label) += 1;
			}
		}

	}

	// Pass a labeled set of data in order to classify the neurons of the trained map
	void classify(const VectorDataset<DoubleVector, unsigned int>& dataset,
		int batch_size) {

		compute_hits(dataset, batch_size);

		// assigning labels to neurons based on which label hit more time every neuron
		int width = trained_map.get_map_width();
		int height = trained_map.get_map_height();  
		for (int j = 0; j < height; ++j) 
			for (int i = 0; i < width; ++i) {
				Plotter plotter;
				// This function implicitly uses our hits map (histogram of the imputations) 
				map_neuron_label(i, j) = most_frequent_hit(from_xy_to_idx(i, j));
				//plotter.context().show_heatmap(trained_map.get_weights(i,j).data(), 28, 28, "gray");
				//std::cout << "this is label : " << map_neuron_label(i, j) << "\n";
				//plotter.block();
			}
				

	}
	/*Eigen::VectorXi map_batch(const Eigen::MatrixXd& batch) const {
    const int B = batch.cols();
    const int N = weight_vectors.cols();

    Eigen::RowVectorXd w_norms = weight_vectors.colwise().squaredNorm();
    Eigen::RowVectorXd x_norms = batch.colwise().squaredNorm();

    Eigen::MatrixXd dist =
        w_norms.transpose().replicate(1, B)
      + x_norms.replicate(N, 1)
      - 2.0 * weight_vectors.transpose() * batch;

    Eigen::VectorXi bmus(B);
    for (int b = 0; b < B; ++b)
        dist.col(b).minCoeff(&bmus(b));

    return bmus;
}
*/
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
		Eigen::VectorXd& labels = hits_map[idx];
		Eigen::Index maxIdx;

		int total = labels.sum();
		if (total == 0) return -1;

		float max = labels.maxCoeff(&maxIdx);

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
				if (lbl < 0) lbl = -1; 
				// Negative labels imply classes which are unknown or which the mapping
				// could not compute
				labels[from_xy_to_idx(i,j)] = lbl + 1;
				// We sum 1 because our plotting function assumes discrete 0, ... N categories
				// so now 0: unknown = BLACK color
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