#pragma once
#ifndef IO_DATASET_DATASET_HPP
#define IO_DATASET_DATASET_HPP

#include <tuple>
#include <vector>
#include <functional>

template <typename DataType, typename YType>
class Dataset {

private:
    std::vector<std::unique_ptr<DataType[]>> x_data;
    std::vector<YType> y_data;
    size_t input_size;

public:
    using index_t = std::size_t;

    Dataset(size_t input_size) : input_size(input_size) {}

    void add_sample(const DataType* x, YType label) {
        auto ptr = std::make_unique<DataType[]>(input_size);
        std::copy(x, x + input_size, ptr.get());
        x_data.push_back(std::move(ptr));
        y_data.push_back(label);
    }

    index_t size() const { return x_data.size(); }

    const std::unique_ptr<DataType[]>& x_of(index_t i) const {
        return x_data.at(i);
    }

    const YType& y_of(index_t i) const {
        return y_data.at(i);
    }
};

template <typename XType, typename YType>
class VectorDataset {

    using DataID = unsigned long;

private:

    // Associate each value, label pair with a unique id to aid
    // in visualization and categorization. 
    
    std::vector<XType> x_data;
    std::vector<YType> y_data;
    std::vector<DataID> id_data;

    // To shuffle and to extract data.
    std::vector<size_t> permutation;

    size_t input_size;

public:

    using index_t = std::size_t;

    VectorDataset(size_t input_size) : input_size(input_size) {
        // Resize all the data vectors
        x_data.reserve(input_size); y_data.reserve(input_size); id_data.reserve(input_size);
        // Initialize the permutation
        permutation.resize(input_size);
        std::iota(permutation.begin(), permutation.end(), 0);
    }

    void reserve(size_t input_size) {
        x_data.reserve(input_size);
        y_data.reserve(input_size);
        id_data.reserve(input_size);
    }

    void add_sample(const XType& x, const YType& label, const DataID id) {
        x_data.push_back(std::move(x));
        y_data.push_back(std::move(label));
        id_data.push_back(id);
    }

    index_t size() const { return x_data.size(); }

    void shuffle() { 
        // Shuffle the internal permutation to simulate a shuffling of the data. 
        static thread_local std::mt19937 rng(std::random_device{}());
        std::shuffle(permutation.begin(), permutation.end(), rng); 
    }

    const XType& x_of(index_t i) const {
        return x_data.at(i);
    }

    const YType& y_of(index_t i) const {
        return y_data.at(i);
    }

    const DataID id_of(index_t i) const {
        return id_data.at(i);
    }

    class BatchView { 

        const VectorDataset& parent;
        size_t start;
        size_t count; 

    public: 

        BatchView(const VectorDataset& p, size_t s, size_t c) :
            parent(p), start(s), count(c) { }
        
        size_t size() const { 
            return count; 
        } 
        
        const XType& x_of(size_t i) const { 
            return parent.x_of(parent.permutation[start + i]);
        } 
        
        const YType& y_of(size_t i) const { 
            return parent.y_of(parent.permutation[start + i]);
        } 
        
        DataID id_of(size_t i) const { 
            return parent.id_of(parent.permutation[start + i]); 
        } 
    };

    class BatchIterator { 

        const VectorDataset* parent;
        size_t batch_size;
        size_t batch_index;

    public: 
        using value_type = BatchView;
        
        BatchIterator(const VectorDataset* p, size_t bs, size_t idx)
            : parent(p), batch_size(bs), batch_index(idx) { }
        
        bool operator!=(const BatchIterator& other) const { 
            return batch_index != other.batch_index; 
        } 
        
        BatchView operator*() const { 
            size_t start = batch_index * batch_size;
            size_t remaining = parent->size() - start; 
            size_t count = std::min(batch_size, remaining);
            return BatchView(*parent, start, count);
        } 
        
        BatchIterator& operator++() { ++batch_index; return *this; } 
    
    };

    struct BatchRange { 

        BatchIterator b; 
        BatchIterator e; 

        BatchIterator begin() const { 
            return b; 
        } 
        
        BatchIterator end() const { 
            return e; 
        } 
    };

    BatchRange batches(size_t batch_size) const {
        size_t num_batches = (input_size + batch_size - 1) / batch_size;
        return BatchRange{ 
            BatchIterator(this, batch_size, 0),
            BatchIterator(this, batch_size, num_batches) 
        }; 
    
    }

};

template <typename XType, typename YType>
class ReadOnlyDataView {

    using DataID = unsigned long;
    using index_t = std::size_t;

private:

    // Associate each value, label pair with a unique id to aid
    // in visualization and categorization. 

    std::vector<std::reference_wrapper<XType>> x_data;
    std::vector<std::reference_wrapper<YType>> y_data;
    std::vector<std::reference_wrapper<DataID>> id_data;

    size_t view_size;

public:

    size_t size() const {
        return view_size;
    }

    const XType& x_of(index_t i) const {
        return x_data.at(i).get();
    }

    const YType& y_of(index_t i) const {
        return y_data.at(i).get();
    }

    const DataID id_of(index_t i) const {
        return id_data.at(i);
    }

    template <typename X, typename Y>
    friend ReadOnlyDataView<XType, YType> random_subset_view(unsigned int subset_size);

};

namespace DatasetUtils {

    template <typename XType, typename YType>
    ReadOnlyDataView<XType, YType> random_subset_view(VectorDataset<XType, YType>& dataset,
        unsigned int subset_size ) {
        

    }

}

#endif