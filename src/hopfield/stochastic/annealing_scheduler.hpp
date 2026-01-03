#pragma once
#ifndef HOPFIELD_SCHEDULING_HPP
#define HOPFIELD_SCHEDULING_HPP

// Scheduling policies to be used in conjunction with stochastic hopfield
// networks.

class AnnealingScheduler {

public:

	virtual void update(unsigned long it) = 0;
	virtual double get_temp() const = 0;
	virtual unsigned int get_stabilization_its() const = 0;

};

class ConstantScheduler : public AnnealingScheduler {

};

class LinearScheduler : public AnnealingScheduler {

};

class CustomScheduler : public AnnealingScheduler {

	std::function<double(unsigned long i)> temp_scheduler;
	double temp;

	void update(unsigned long it) {
		temp = temp_scheduler(it);
	}

	double get_temp()  const override {
		return temp;
	}

	unsigned int get_stabilization_its()  const override {
		return 1;
	}

};

#endif