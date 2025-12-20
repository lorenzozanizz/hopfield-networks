#include <cstdio>
#include <cblas.h>
#include "io/plot/plot.hpp"
#include "math/autograd/functions.hpp"
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
}
	