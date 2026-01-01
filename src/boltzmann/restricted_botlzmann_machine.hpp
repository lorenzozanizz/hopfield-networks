#pragma once
#ifndef RESTRICTED_BOLTZMANN_MACHINE_HPP
#define RESTRICTED_BOLTZMANN_MACHINE_HPP

#include <cmath>


// An implementation of 
class RestrictedBoltzmannMachine {

	unsigned int hidden_units_amt;
	unsigned int visible_units_amt;

public:

	unsigned int visible_size() const {
		return visible_units_amt;
	}

	unsigned int hidden_size() const {
		return hidden_units_amt;
	}

	void train() {

	}

	void sample( unsigned int cd_steps = 25) {

	}

};


#endif