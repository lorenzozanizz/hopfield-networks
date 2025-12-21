#include "variables.hpp"

template <typename FloatType = float>
class Function {

	const dim_t dimension;
public:

};

template <typename FloatType = float>
class ScalarFunction : public Function<FloatType> {

};

template <typename FloatType = float>
class VectorFunction : public Function<FloatType> {

	VectorFunction(dim_t dim) { }

};

