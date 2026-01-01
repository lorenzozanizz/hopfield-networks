#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include "sparse_vector.hpp"
#include <unordered_map>

class SparseMatrix {
private:

    const std::size_t rows;
    const std::size_t cols;

    std::unordered_map<std::size_t, double> data;

    std::size_t index(std::size_t r, std::size_t c) const {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("SparseMatrix index out of range");
        }
        return r * cols + c;
    }

public:

    SparseMatrix(std::size_t r, std::size_t c) : rows(r), cols(c) {}

    double get(std::size_t r, std::size_t c) const {
        auto it = data.find(index(r, c));
        return (it != data.end()) ? it->second : 0.0;
    }

    void set(std::size_t r, std::size_t c, double value) {
        data.insert_or_assign(index(r, c), value);
    }

    friend std::ostream& operator<<(std::ostream& os, const SparseMatrix& mat) {
        os << "SparseMatrix (" << mat.rows << " x " << mat.cols << ")\n";

        for (const auto& [idx, value] : mat.data) {
            std::size_t i = idx / mat.cols;
            std::size_t j = idx % mat.cols;
            os << "(" << i << ", " << j << ") = " << value << "\n";
        }

        return os;
    }

    void set(std::size_t idx, double value) {
        data.insert_or_assign(idx, value);
    }

    std::size_t numRows() const { return this->rows; }
    std::size_t numCols() const { return this->cols; }

    std::size_t nonZeroCount() { return data.size(); }

    SparseMatrix operator*(const SparseMatrix& other) {
        if (this->cols != other.rows) {
            throw std::invalid_argument("Matrix size mismatch");
        }

        SparseMatrix result(this->rows, other.cols);

        for (const auto& [idxA, valA] : data) {
            std::size_t row = idxA / cols; // row of the element
            std::size_t col = idxA % cols; // column of the element

            for (std::size_t j = 0; j < other.cols; ++j) {
                double valB = other.get(col, j);
                if (valB != 0.0) {
                    result.set(row, j, result.get(row, j) + valA * valB);
                }
            }
        }

        return result;
    }

    SparseMatrix operator*(const double x) {

        SparseMatrix result(this->rows, this->cols);

        for (const auto& [idxA, valA] : data) {
            
            result.set(idxA, valA * x); 
            
        }

        return result;
    }
   
    SparseVector operator*(const std::vector<double>& other) {
        if (this->cols != other.size()) {
            throw std::invalid_argument("Matrix size mismatch");
        }

        SparseVector result(other.size());

        for (const auto& [idxA, valA] : data) {
            std::size_t row = idxA / cols; // row of the element
            std::size_t col = idxA % cols; // column of the element
            double valB = other[col];
            if (valB != 0.0) {
                result.set(row , result.get(row) + valA * valB);
            }
        }

        return result;
    }

    SparseVector operator*(const SparseVector& other) {
        if (this->cols != other.size()) {
            throw std::invalid_argument("Matrix size mismatch");
        }

        SparseVector result(other.size());

        for (const auto& [idxA, valA] : data) {
            std::size_t row = idxA / cols; // row of the element
            std::size_t col = idxA % cols; // column of the element
            //std::cout << "reading " << valA << " in position (" << row << ", " << col << ")" << std::endl;
            double valB = other.get(col);
            //std::cout << "reading " << valB << " in position " << col << std::endl;
            if (valB != 0.0) {
                //std::cout << "setting " << result.get(row) + valA * valB << " in position " << row << std::endl;
                result.set(row, result.get(row) + valA * valB);
            }
        }

        return result;
    }

    
   

    double product_row_vector(size_t row, const SparseVector& other) {
        if (this->cols != other.size()) {
            throw std::invalid_argument("Matrix size mismatch");
        }

        double result;

        for (const auto& [idxA, valA] : data) {
            std::size_t r = idxA / cols; // row of the element
            if (row != r) {
                continue;
            }
            std::size_t col = idxA % cols; // column of the element
            double valB = other.get(col);
            if (valB != 0.0) {
                result += valA*valB;
            }
        }

        return result;
    }


    

    double product_row_vector(size_t row, const std::vector<double>& other) {

        if (this->cols != other.size()) {
            throw std::invalid_argument("Matrix size mismatch");
        }

        double result;

        for (const auto& [idxA, valA] : data) {
            std::size_t r = idxA / cols; // row of the element
            if (row != r) {
                continue;
            }
            std::size_t col = idxA % cols; // column of the element
            double valB = other[col];
            if (valB != 0.0) {
                result += valA * valB;
            }
        }

        return result;
    }

};

#endif // SPARSE_MATRIX_HPP