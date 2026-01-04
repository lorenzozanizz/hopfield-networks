#pragma once
#ifndef HOPFIELD_SCHEDULING_HPP
#define HOPFIELD_SCHEDULING_HPP

#include <functional>
#include <algorithm>

// Scheduling policies to be used in conjunction with stochastic hopfield
// networks.
// Base class for annealing schedules
class AnnealingScheduler {
public:

    virtual void update(unsigned long it) = 0;
    virtual double get_temp() const = 0;
};

class ConstantScheduler : public AnnealingScheduler {
    double temp;

public:
    explicit ConstantScheduler(double t) : temp(t) {}

    void update(unsigned long) override {
        // nothing to update
    }

    double get_temp() const override {
        return temp;
    }
};

class LinearScheduler : public AnnealingScheduler {
    double t0;
    double t1;
    unsigned long max_iter;
    double temp;

public:
    LinearScheduler(double start, double end, unsigned long max_it)
        : t0(start), t1(end), max_iter(max_it), temp(start) {
    }

    void update(unsigned long it) override {
        if (max_iter == 0 || it > max_iter) {
            temp = t1;
            return;
        }

        double alpha = std::min<double>(1.0, double(it) / double(max_iter));
        temp = t0 + (t1 - t0) * alpha;
    }

    double get_temp() const override {
        return temp;
    }
};

class CustomScheduler : public AnnealingScheduler {
    std::function<double(unsigned long)> temp_scheduler;
    double temp;

public:
    explicit CustomScheduler(std::function<double(unsigned long)> f)
        : temp_scheduler(std::move(f)), temp(0.0) {
    }

    void update(unsigned long it) override {
        temp = temp_scheduler(it);
    }

    double get_temp() const override {
        return temp;
    }
};

#endif