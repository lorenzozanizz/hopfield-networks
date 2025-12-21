#include <cstdio>

#include "networks/states/binary.hpp"
#include "math/autograd/variables.hpp"

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

	VectorVariable u = create_vector_variable(140);
	const auto func = exponential(sum(u, 1.0));

}
	