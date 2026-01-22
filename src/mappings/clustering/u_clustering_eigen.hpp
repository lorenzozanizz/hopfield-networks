#pragma once
#ifndef U_CLUSTERING_EIGEN_HPP 
#define U_CLUSTERING_EIGEN_HPP 

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
class UClusteringEigen {

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

public:
	UClusteringEigen(KonohenMapEigen<DataType>& trained_map) : trained_map(trained_map) {
		heigth = trained_map.get_map_height();
		width = trained_map.get_map_width();
		u_heigth = (2 * heigth - 1);
		u_width = (2 * width - 1);
		u = Eigen::MatrixXd::Zero(u_heigth, u_width); 
	}

	int uidx(int x, int y) const {
		return y * u_width + x;
	};

	int idx(int x, int y) const {
		return y * width + x;
	};

	// return the mean between the 4 "cross" neighbours
	double neighb_mean(int x, int y) {

		int xmin = std::max(0, x - 1);
		int xmax = std::min(static_cast<int>(u_width) - 1, x + 1);
		int ymin = std::max(0, y - 1);
		int ymax = std::min(static_cast<int>(u_heigth) - 1, y + 1);

		return u.block(xmin, ymin,
		xmax - xmin + 1,ymax - ymin + 1).mean();
	}


	void compute() {

		for (int y = 0; y < heigth; ++y) {
			for (int x = 0; x < width; ++x) {

				int i = idx(x, y);
				int j = idx(x + 1, y);
				int k = idx(x, y + 1);
				int z = idx(x + 1, y + 1);

				if (x != width - 1) {
					float d_h = computing_distance(i, j);
					u(2 * x + 1, 2 * y) = d_h; // horizontal distance

				}
				if (y != heigth - 1) {
					float d_v = computing_distance(i, k);
					u(2 * x, 2 * y + 1) = d_v; // vertical distance

				}

				if ((x != width - 1) && (y != heigth - 1)) {
					float d_d = computing_distance(i, z);
					u(2 * x + 1, 2 * y + 1) = d_d; // diagonal distance

				}

				double value = neighb_mean(2 * x, 2 * y);
				u(2 * x, 2 * y) = value;

			}
		}

		normalize();

	}

	void plot(Plotter& plotter) {
		std::vector<float> u_f(u.data(), u.data() + u.size());
		plotter.context()
			.set_title("U-Matrix")
			.show_heatmap(u_f.data(), u_width, u_heigth)
			.set_cblabel("Distance");

		plotter.block();
	}

};
#endif