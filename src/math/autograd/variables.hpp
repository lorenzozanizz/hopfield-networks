#pragma once
#ifndef MATH_AUTOGRAD_VARIABLES_HPP
#define MATH_AUTOGRAD_VARIABLES_HPP

typedef unsigned long dim_t;

template <typename FloatType = float>
class Var {
public:

	// Get the dimension for a variables
	dim_t dimension() const = 0;

};

template <dim_t Dim, typename FloatType = float>
class StaticVar: Var<FloatType> {

public:

	constexpr dim_t dimension() const override {
		return Dim;
	}

};

template <typename FloatType>
class RuntimeVar: Var<FloatType> {

public:

	dim_t dimension() const override {
		return dimension;
	}

protected:

	const dim_t dimension;
};


#endif