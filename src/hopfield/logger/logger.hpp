#pragma once
#ifndef LOGGER_LOGGER_HPP
#define LOGGER_LOGGER_HPP

#include <iostream>
#include <fstream>
#include <vector>

class HopfieldLogger {


protected:

    std::vector<std::string> write_buffer;

    std::ostream* out;
    std::ofstream* f_out;

    bool record_states;
    bool record_energy;
    bool record_temperature;
    bool record_order;

    bool buffered;
    bool close_after;

public:

    HopfieldLogger() : write_buffer(10), record_states(false), record_energy(false),
        record_temperature(false), record_order(false), out(nullptr), close_after(false)
    { }

    void set_recording_stream(std::ostream& stream) {
        // Save the recording stream. 
        out = &stream;
        f_out = nullptr;
        close_after = false;
    }

    void set_recording_stream(std::ofstream& stream, bool close_aft = true) {
        f_out = &stream;
        out = nullptr;
        close_after = close_aft;
    }

    void set_buffered(bool v) { buffered = v; }

    void collect_states(bool v, const std::string& into = "states.gif") { record_states = v; }
    
    void collect_energy(bool v) { record_energy = v; }
    
    void collect_temperature(bool v) { record_temperature = v; }

    void collect_order_parameter(bool v) { record_order = v; }


    void finally_plot_data(bool val) {

    }

    void finally_write_last_state_png(bool val, const std::string & = "last.png") {

    }
    
    void on_state_update(const std::vector<double>& s) {
        if (record_states) {

        }
    }

    void on_energy_update(double e) {
        if (record_energy)
            e = 0.0;
    }

    void on_temperature_update(double t) {
        if (record_temperature)
            t = 0.0;
    }

    void on_order_parameter_update(double m) {
        if (record_order)
            m = 0;
    }

    void on_run_end() {
        

        if (close_after)
            f_out->close();
    }

    void on_run_begin() {

    }

};



#endif