#pragma once
#ifndef IO_DATASET_DATASET_EIGEN_HPP
#define IO_DATASET_DATASET_EIGEN_HPP

#include <tuple>
#include <vector>
#include <functional>
#include "../../math/matrix/matrix_ops.hpp"

template <typename DataType, typename YType>
class DatasetEigen {

    public:
        using Vector = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

        DatasetEigen(size_t capacity) {
            x_data.reserve(capacity);
            y_data.reserve(capacity);
        }

        void add_sample(const Vector& x, YType label) {
            x_data.push_back(x);
            y_data.push_back(label);
        }

        const Vector& x_of(size_t i) const {
            return x_data[i];
        }

        const YType& y_of(size_t i) const {
            return y_data[i];
        }

        size_t size() const { return x_data.size(); }

    private:
        std::vector<Vector> x_data;
        std::vector<YType>  y_data;

};


#endif