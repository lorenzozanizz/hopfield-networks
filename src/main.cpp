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
	
	Function func(1); // a scalar function
	auto& g = func.generator();
	auto u = g.create_vector_variable(140);
	const auto expr = g.exponential(g.sum(u, 1.0));


}
	