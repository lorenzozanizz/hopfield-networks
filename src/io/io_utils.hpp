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
    inline void print_intermediate(const std::string& line) {
        intermediate_lines.push_back(line);
        std::cout << line << std::endl;
    }

    // Update the progress bar and redraw everything
    void update(size_t current);

protected:

    void print_main_bar(size_t current);

};


#endif