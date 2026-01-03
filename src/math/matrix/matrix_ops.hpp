#pragma once
#ifndef MATRIX_OPS_HPP
#define MATRIX_OPS_HPP

#include <stdexcept>
#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/Sparse>

namespace MathOps {
	
	using namespace Eigen;

    template <typename MatrixType>
    std::pair<double, VectorXd> power_method(
        const MatrixType& A,
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
            VectorXd b_new = Ab / norm;
            double lambda = b_new.dot(A * b_new);

            // Check convergence with step size criterion.
            if (std::abs(lambda - lambda_old) < tol) {
                return { lambda, b_new };
            }

            b = b_new;
            lambda_old = lambda;
        }

        // If we reach here, we did not converge! 
        return { lambda_old, b };
    }

    template <typename VectorType>
    float lp_norm(const VectorType& v, float p) { 
        return std::pow(v.array().abs().pow(p).sum(), 1.0f / p);
    }

    template <typename T>
    int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }

}

#endif