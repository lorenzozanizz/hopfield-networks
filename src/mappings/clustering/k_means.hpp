#pragma once
#ifndef  K_MEANS_HPP 
#define  K_MEANS_HPP  

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

class KMeans {
public:
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXi;
    using IndexVector = Eigen::VectorXi;

    KMeans(int k, int max_iters = 10, double tol = 1e-9)
        : K(k), max_iters(max_iters), tol(tol) {}

    void fit(const KonohenMapEigen<double>& X) {
        init_random(X);

        for (int iter = 0; iter < max_iters; ++iter) {
            assign_labels(X);
            double shift = update_centroids(X);

            if (shift < tol) {
                std::cout << "Converged at iter " << iter << "\n";
                break;
            }
        }
    }


    void plot(Plotter& plotter) {
        std::vector<int> u_f(labels_.data(), labels_.data() + labels_.size());
        plotter.context()
            .set_title("K-Means")
            //.show_heatmap(u_f.data(), W, H, "grey")
            .show_discrete_categories(u_f, W, H, 10)
            .set_cblabel("Distance");

        plotter.block();
    }


    const IndexVector& labels() const { return labels_; }
    const Matrix& centroids() const { return centroids_; }

private:

    int K;
    int max_iters;
    double tol;
    int H;
    int W;

    Matrix centroids_;   // D × K
    IndexVector labels_; // N

    void init_random(const KonohenMapEigen<double>& X) {

        //Plotter plotter;

        int N = X.get_map_width() * X.get_map_height();
        W = X.get_map_width();
        H = X.get_map_height();
        int D = X.get_input_size();

        centroids_.resize(D, K);
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> dist(0, N - 1);

        for (int k = 0; k < K; ++k) {
            unsigned int idx = dist(gen);    // sample a neuron index
            centroids_.col(k) = X.get_weights(idx);
            //plotter.context().show_heatmap(X.get_weights(idx).data(), 28, 28, "gray");
            //std::cout << "this is label : " << k << "\n";
            //plotter.block();
        }

    }

    void assign_labels(const KonohenMapEigen<double>& X) { 

       //Plotter plotter;

        int N = X.get_map_width() * X.get_map_height();
        int D = X.get_input_size();

        labels_.resize(N);

        Eigen::VectorXd x_norms = X.get_all_weights().colwise().squaredNorm();
        Eigen::VectorXd c_norms = centroids_.colwise().squaredNorm(); 

        // Compute distance matrix: K × N
        Matrix distances = (-2.0 * centroids_.transpose() * X.get_all_weights()).colwise() + c_norms;
        distances.rowwise() += x_norms.transpose();

        // Assign labels
        for (int i = 0; i < N; ++i) {
            distances.col(i).minCoeff(&labels_(i));
            //plotter.context().show_heatmap(X.get_weights(i).data(), 28, 28, "gray");
            //std::cout << "label : " << labels_(i) << "\n";
            //plotter.block();
        }
    }


    double update_centroids(const KonohenMapEigen<double>& X) {

        int N = X.get_map_width() * X.get_map_height();
        int D = X.get_input_size();

        Matrix new_centroids_ = Matrix::Zero(D, K);
        Eigen::VectorXi counts = Eigen::VectorXi::Zero(K);

        for (int i = 0; i < N; ++i) {
            new_centroids_.col(labels_(i)) += X.get_all_weights().col(i);
            counts(labels_(i))++;
        }

        for (int k = 0; k < K; ++k) {
            if (counts(k) > 0)
                new_centroids_.col(k) /= counts(k);
            else
                new_centroids_.col(k) = centroids_.col(k); // avoid empty cluster
        }

        double shift = (centroids_ - new_centroids_).norm();
        centroids_ = std::move(new_centroids_);
        return shift;
    }

};
#endif