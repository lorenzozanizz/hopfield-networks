#include "io_utils.hpp"



void MultiProgressBar::update(size_t current) {
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


void MultiProgressBar::print_main_bar(size_t current) {
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