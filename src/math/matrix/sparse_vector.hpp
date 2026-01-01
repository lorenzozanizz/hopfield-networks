#ifndef SPARSE_VECTOR_HPP
#define SPARSE_VECTOR_HPP

#include <unordered_map>

class SparseVector {
private:

    const std::size_t dim;
    std::unordered_map<std::size_t, double> data;

    std::size_t index(std::size_t idx) const {
        if (idx >= dim) {
            throw std::out_of_range("SparseVector index out of range");
        }
        return idx;
    }

public:

    SparseVector(std::size_t dim) : dim(dim) {} 

    double get(std::size_t idx) const {
        auto it = data.find(index(idx));
        return (it != data.end()) ? it->second : 0.0;
    }

    void set(std::size_t idx, double value) {
        data.insert_or_assign(index(idx), value);
    }

    std::size_t size() const { return this->dim; }
  
    std::size_t nonZeroCount() { return this->data.size(); }

    SparseVector operator*(const double x) {

        SparseVector result(this->size());

        for (const auto& [idxA, valA] : data) {

            result.set(idxA, valA * x);

        }

        return result;
    }

};

#endif // SPARSE_VECTOR_HPP