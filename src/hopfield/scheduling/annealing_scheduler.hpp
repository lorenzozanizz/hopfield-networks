#pragma once
#ifndef HOPFIELD_SCHEDULING_HPP
#define HOPFIELD_SCHEDULING_HPP

class AnnealingScheduler {

public:

	virtual update(unsigned long it) = 0;
	virtual get_temp() const = 0;
	virtual get_stabilization_its() const = 0;

};

class ConstantScheduler : public AnnealingScheduler {

};

class LinearScheduler : public AnnealingScheduler {

};

class CustomScheduler : public  {

	std::function<double(unsigned long i)> temp_scheduler;
	double temp;



}

#endif