#pragma once
#ifndef TIMING_HPP
#define TIMING_HPP

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <unordered_map>

class SegmentTimer {

public:
    using clock = std::chrono::high_resolution_clock;

    struct Segment {
        double total_ms = 0.0;
        clock::time_point start_time;
        bool running = false;
    };

    // Start a named segment
    void start(const std::string& name);

    void stop(const std::string& name);

    // NOTE: This needs to be here due to the use of the auto keyword. 
    auto get_reset(const std::string& name) {
        auto it = segments.find(name);
        if (it == segments.end() || !it->second.running)
            return 0.0;
        auto now = clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(now - it->second.start_time).count();
        it->second.total_ms += elapsed;
        it->second.start_time = now;
        return elapsed;
    }

    // RAII helper for scoped timing, to be used when opening a { } scope and 
    // to avoid the need to explicitly stop the timer. 
    class Scoped {
    public:
        Scoped(SegmentTimer& t, const std::string& name)
            : timer(t), seg_name(name) {
            timer.start(seg_name);
        }
        ~Scoped() {
            timer.stop(seg_name);
        }
    private:
        SegmentTimer& timer;
        std::string seg_name;
    };

    Scoped scoped(const std::string& name) {
        return Scoped(*this, name);
    }

    // Print a simple visualization of the collected data, where each named category is
    // visualized alongside with its time. 
    void print(std::ostream& os = std::cout) const;

private:
    std::unordered_map<std::string, Segment> segments;
};

#endif //! TIMING_HPP