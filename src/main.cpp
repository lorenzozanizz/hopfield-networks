#include <cstdio>

#include "networks/states/binary.hpp"
#include "math/autograd/variables.hpp"

#include "io/plot/plot.hpp"

enum NeighbouringStrategy {
	OneDNeighbouring,
	TwoDNeighbouring,
	// Treat the 
	ThreeDNeighbouring
};

enum Stochasticity {
	Deterministic,
	Stochastic
};

// Just create the folder...
int main() {
	
	using namespace autograd;
	using namespace autograd::Ops;
	using namespace autograd::Funcs;
	
	Function func(1); // a scalar function
	auto g = func.generator();
	VectorVariable u = g.create_vector_variable(140);
	const auto expr = g.exponential(g.sum(u, 1.0));


}
	