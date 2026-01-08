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

	double computing_distance(int idx_1, int idx_2) {
		return (trained_map.get_weights(idx_1) - trained_map.get_weights(idx_2)).norm();

	}

	bool inside_map(int x, int y) const {
		return x >= 0 && y >= 0 && x < width && y < heigth;
	}

	// U-matrix value between (x1,y1) and (x2,y2)
	double u_between(int x1, int y1, int x2, int y2) const {
		int ux = x1 + x2;
		int uy = y1 + y2;
		return u(ux, uy);
	}

	void normalize() {
		double max = u.maxCoeff();

		if (max == 0.0) return;
	
		u.array() /= max;
		
	}

public:
	UClusteringEigen(KonohenMapEigen<DataType>& trained_map, double threshold) : trained_map(trained_map), threshold(threshold) {
		heigth = trained_map.get_map_height();
		width = trained_map.get_map_width();
		u_heigth = (2 * heigth - 1);
		u_width = (2 * width - 1);
		u = Eigen::MatrixXd::Zero(u_heigth, u_width); 
	}

	void set_threshold(double new_th) {
		threshold = new_th;
	}

	int uidx(int x, int y) const {
		return y * u_width + x;
	};

	int idx(int x, int y) const {
		return y * width + x;
	};

	double neighb_mean(int x, int y) {
		double sum = 0.0;
		int count = 0;
		for (int dy = -1; dy <= 1; ++dy) {
			for (int dx = -1; dx <= 1; ++dx) {
				int nx = x + dx;
				int ny = y + dy;

				if (nx > 0 && nx < u_width && ny > 0 && ny < u_heigth) {
					sum += u(nx, ny);
					++count;
				}
			}
		}
		return sum / count;
	}

	// WARNING! This is just for some tests
	void print_map() {
		for (int y = 0; y < u_heigth; ++y) {
			for (int x = 0; x < u_width; ++x) {
				std::cout << "(" << x << ", " << y << "): " << u(x, y);
			}
			std::cout << "\n";
		}
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
					//std::cout << "distance between " << "(" << x << ", " << y << ") and " << "(" << x+1 << ", " << y  << ") is " << d_h << "\n";
					u(2 * x + 1, 2 * y) = d_h; // horizontal distance

				}
				if (y != heigth - 1) {
					float d_v = computing_distance(i, k);
					//std::cout << "distance between " << "(" << x << ", " << y << ") and " << "(" << x << ", " << y+1 << ") is " << d_v << "\n";
					u(2 * x, 2 * y + 1) = d_v; // vertical distance
				}

				if ((x != width - 1) && (y != heigth - 1)) {
					float d_d = computing_distance(i, z);
					//std::cout << "distance between " << "(" << x << ", " << y << ") and " << "(" << x+1 << ", " << y + 1 << ") is " << d_d << "\n";
					u(2 * x + 1, 2 * y + 1) = d_d; // diagonal distance
				}

				u(2 * x, 2 * y) = neighb_mean(2 * x, 2 * y);

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