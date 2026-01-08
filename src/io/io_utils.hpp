#pragma once
#ifndef IO_UTILS_HPP
#define IO_UTILS_HPP

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include "../utils/timing.hpp"

class MultiProgressBar {

    size_t total_steps;
    size_t width;

    size_t printed_lines = 0;
    std::vector<std::string> intermediate_lines;

    bool did_start;
    SegmentTimer timer;

public:

    MultiProgressBar(const size_t total, const size_t bar_width = 25)
        : total_steps(total), width(bar_width), did_start(false) {
    }

    // Add a line that will appear below the progress bar
    void print_intermediate(const std::string& line) {
        intermediate_lines.push_back(line);
        std::cout << line << std::endl;
    }

    // Update the progress bar and redraw everything
    void update(size_t current) {
        // Move cursor up to overwrite previous block
        if (!did_start) {
            timer.start("progbar");
            std::cout << std::endl;
        }
        auto size = intermediate_lines.size();
        if (did_start)
            size += 1;
        if (size)
            std::cout << "\033[" << size << "A";

        print_main_bar(current);
        did_start = true;

        std::cout.flush();

        // Clear intermediate lines for next cycle
        intermediate_lines.clear();
    }

protected:

    void print_main_bar(size_t current) {
        double ratio = double(current) / double(total_steps);
        size_t filled = size_t(ratio * width);

        std::cout << "[";
        for (size_t i = 0; i < width; ++i)
            std::cout << (i < filled ? "#" : "-");
        std::cout << "] ";

        std::cout << std::setw(3) << int(ratio * 100) << "%";
        if (did_start)
            std::cout << " ETA: " << (timer.get_reset("progbar") / 1000) * (total_steps - current) << " sec." << std::endl;
        else std::cout << std::endl;
    }

};


#endif