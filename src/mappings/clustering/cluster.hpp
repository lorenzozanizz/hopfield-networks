#pragma once
#ifndef CLUSTER_HPP 
#define CLUSTER_HPP 

#include <memory>
#include <iterator>
#include <cmath>
#include <vector>
#include <map>
#include <random>
#include <limits>
#include <utility>
#include <queue>
#include <algorithm>
#include "../konohen_mapping_eigen.hpp"
#include "../../io/datasets/dataset_eigen.hpp"
#include "../../math/utilities.hpp"
#include "../../math/matrix/matrix_ops.hpp"

template <typename DataType = float>
class Cluster {

private:
	KonohenMapEigen<DataType>& trained_map;
	size_t u_width, u_heigth, width, heigth;
	double threshold;
	Eigen::MatrixXd u;

	// computing the distance between the weights in the neuron at idx_1 and the neuron at idx_2
	double computing_distance(int idx_1, int idx_2) {
		return (trained_map.get_weights(idx_1) - trained_map.get_weights(idx_2)).norm();
	}

	// Normalize all the elements in the U-matrix
	void normalize() {
		double max = u.maxCoeff();
		if (max == 0.0) return;
		u.array() /= max;

	}

	using DoubleVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
	double cosine_similarity(const DoubleVector x,const DoubleVector y) {
		return std::abs((x.dot(y))/(x.norm()*y.norm()));
	}

public:
	Cluster(KonohenMapEigen<DataType>& trained_map) : trained_map(trained_map) {
		heigth = trained_map.get_map_height();
		width = trained_map.get_map_width();
		u_heigth = (width * heigth);
		u_width = (width * heigth);
		u = Eigen::MatrixXd::Zero(u_heigth, u_width);
	}

	void compute() {

		for (int y = 0; y < u_heigth; ++y) {
			for (int x = 0; x < u_width; ++x) {

				//u(x,y) = cosine_similarity(trained_map.get_weights(x), trained_map.get_weights(y));
				u(x,y) = computing_distance(x, y);

			}
		}

		normalize();

	}

	void plot(Plotter& plotter) {
		std::vector<float> u_f(u.data(), u.data() + u.size());
		plotter.context()
			.set_title("U-Matrix")
			.show_heatmap(u_f.data(), u_width, u_heigth, "grey")
			.set_cblabel("Distance");

		plotter.block();
	}

};
#endif