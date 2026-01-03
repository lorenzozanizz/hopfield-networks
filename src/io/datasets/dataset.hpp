#pragma once
#ifndef IO_DATASET_DATASET_HPP
#define IO_DATASET_DATASET_HPP

#include <tuple>
#include <vector>


template <typename XType, typename YType>
class Dataset {

public:

    using index_t = std::size_t;

    virtual index_t size() const = 0;
    virtual const XType& x_of(index_t index) const = 0;
    virtual const YType& y_of(index_t index) const = 0;

    virtual ~Dataset() = default;

};



#endif