#pragma once
#ifndef IO_DATASET_DATASET_HPP
#define IO_DATASET_DATASET_HPP

#include <tuple>
#include <vector>


template <typename DataType, typename YType>
class Dataset {

private:
    std::vector<std::unique_ptr<DataType[]>> x_data;
    std::vector<YType> y_data;
    size_t input_size;

public:
    using index_t = std::size_t;

    Dataset(size_t input_size)
        : input_size(input_size) {}

    void add_sample(const DataType* x, YType label) {
        auto ptr = std::make_unique<DataType[]>(input_size);
        for (size_t i = 0; i < input_size; ++i)
            ptr[i] = x[i];

        x_data.push_back(std::move(ptr));
        y_data.push_back(label);
    }

    index_t size() const {
        return x_data.size();
    }

    const std::unique_ptr<DataType[]>& x_of(index_t i) const {
        return x_data.at(i);
    }

    const YType& y_of(index_t i) const {
        return y_data.at(i);
    }
};


#endif