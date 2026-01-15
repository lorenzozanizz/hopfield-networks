#include "timing.hpp"

void SegmentTimer::start(const std::string& name) {
    auto& seg = segments[name];
    if (!seg.running) {
        seg.running = true;
        seg.start_time = clock::now();
    }
}

void SegmentTimer::stop(const std::string& name) {
    auto it = segments.find(name);
    if (it == segments.end() || !it->second.running)
        return;

    // Just take the current time and subtract, nothing fancy. 
    auto now = clock::now();
    auto elapsed = std::chrono::duration<double, std::milli>(now - it->second.start_time).count();
    it->second.total_ms += elapsed;
    it->second.running = false;
}

void SegmentTimer::print(std::ostream& os) const {
    os << "\n=== Timing Summary ===\n";
    double total = 0.0;
    for (const auto& [name, seg] : segments)
        total += seg.total_ms;

    for (const auto& [name, seg] : segments) {
        double pct = (total > 0.0) ? (seg.total_ms / total * 100.0) : 0.0;
        os << std::setw(20) << std::left << name << " : "
            << std::setw(10) << std::right << seg.total_ms << " ms  "
            << std::fixed << std::setprecision(1)
            << "(" << pct << "%)\n";
    }
    os << "=======================\n";
}