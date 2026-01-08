#pragma once
#ifndef MATRIX_OPS_HPP
#define MATRIX_OPS_HPP

#include <stdexcept>
#include <iostream>
#include <cmath>
#include <fstream>


#define EIGEN_USE_THREADS 
#undef EIGEN_DONT_PARALLELIZE
#define EIGEN_USE_OPENMP
#include <Eigen/Core>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Sparse>

namespace MathOps {
	
	using namespace Eigen;

    template <typename DenseMatrixType>
    double power_method(
        const DenseMatrixType& A,
        int max_iters = 50,
        double tol = 1e-7
    ) {
        const int n = A.rows();
        VectorXd b = VectorXd::Random(n);   // initial guess
        b.normalize();

        double lambda_old = 0.0;

        for (int iter = 0; iter < max_iters; ++iter) {
            VectorXd Ab = A * b;

            double norm = Ab.norm();
            // If the matrix has a non-trivial null space we are blocked in there forever
            // so the algorithm cannot terminate.
            if (norm == 0.0)
                throw std::runtime_error("A has a zero eigenvalue; power method failed.");

            // Compute the versor and the new estimate of lambda
            VectorXd b_new = Ab;
            b_new /= norm;
            double lambda = b_new.dot(A * b_new);

            // Check convergence with step size criterion.
            if (std::abs(lambda - lambda_old) < tol) {
                return lambda;
            }

            b = b_new;
            lambda_old = lambda;
        }

        // If we reach here, we did not converge! 
        return lambda_old;
    }

    template <typename DataType>
    double sparse_power_method(
        const SparseMatrix<DataType>& A,
        int max_iters = 50,
        double tol = 1e-7
    ) {
        using Vector = Matrix<DataType, Eigen::Dynamic, 1>;

        const int n = A.rows();
        Vector b = Vector::Random(n);   // initial guess
        b.normalize();

        double lambda_old = 0.0;

        for (int iter = 0; iter < max_iters; ++iter) {
            Vector Ab = A * b;
            
            double norm = Ab.norm();
            // If the matrix has a non-trivial null space we are blocked in there forever
            // so the algorithm cannot terminate.
            if (norm == 0.0)
                throw std::runtime_error("A has a zero eigenvalue; power method failed.");

            // Compute the versor and the new estimate of lambda
            Ab /= norm;
            // This is required because of the 
            double lambda = Ab.dot(A * Ab);

            // Check convergence with step size criterion.
            if (std::abs(lambda - lambda_old) < tol) {
                return lambda;
            }

            b = Ab;
            lambda_old = lambda;
           
        }

        // If we reach here, we did not converge! 
        return lambda_old;
    }

    template <typename VectorType>
    float lp_norm(const VectorType& v, float p) { 
        return std::pow(v.array().abs().pow(p).sum(), 1.0f / p);
    }

    template <typename T>
    int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }

    template <typename MatrixType>
    void save_matrix_binary(const std::string& filename, const MatrixType& M) {
        std::ofstream out(filename, std::ios::binary);
        if (!out) throw std::runtime_error("Could not open file for writing");

        int rows = M.rows();
        int cols = M.cols();

        out.write(reinterpret_cast<char*>(&rows), sizeof(int));
        out.write(reinterpret_cast<char*>(&cols), sizeof(int));
        out.write(reinterpret_cast<const char*>(M.data()),
            sizeof(typename MatrixType::Scalar) * rows * cols);
    }

    template <typename MatrixType>
    void load_matrix_binary(const std::string& filename, MatrixType& M) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) throw std::runtime_error("Could not open file for reading");

        int rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(int));
        in.read(reinterpret_cast<char*>(&cols), sizeof(int));

        M.resize(rows, cols);
        in.read(reinterpret_cast<char*>(M.data()),
            sizeof(typename MatrixType::Scalar) * rows * cols);

        return;
    }
}

#endif