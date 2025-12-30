#pragma once
#ifndef WEIGHTING_BASE_HPP
#define WEIGHTING_BASE_HPP

#include <functional>

class WeightingPolicy {


};


class HebbianPolicy: public WeightingPolicy {

public:

	HebbianPolicy(unsigned int p) {

	}

	void allocate() {

	}

};

class StarkovPolicy: public WeightingPolicy {

};

class InducedSparsePolicy: public WeightingPolicy {

};

class MatrixFreePolicy: public WeightingPolicy {

};


#endif