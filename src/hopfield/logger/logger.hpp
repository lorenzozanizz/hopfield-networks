#pragma once
#ifndef LOGGER_LOGGER_HPP
#define LOGGER_LOGGER_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <tuple>
#include <stdexcept>

#include "../../io/plot/plot.hpp"
#include "../../io/gif/gif.hpp"
#include "../../io/datasets/data_collection.hpp"

#include "../states/binary.hpp"
#include "../network_types.hpp"

enum class Event {
    OrderParameterChanged,
    TemperatureChanged,
    EnergyChanged,
    StateChanged
};


// A small buffer object to map the reduced memory representation of states
// to the required png buffer to save the intermediate state of the networks
// (a bit of memory overhead, but around 40kbs at most)
struct logger_buffer_t {
    
    unsigned int width =  0;
    unsigned int height = 0;
    std::unique_ptr<unsigned char[]> write_buffer = nullptr;

};

class HopfieldLogger {


protected:

    std::vector<std::string> write_buffer;

    // Utilities to save the state of the network in time and 
    // mapping the state to a binary png.
    std::string states_gif_save;
    logger_buffer_t log_buf;
    GifWriterIO gio;

    std::string last_state_save;

    // Output streams, a file or a general stream (to be 
    // distinguished to allow auto closing of write files)
    std::ofstream* f_out;
    NamedVectorCollection<float> nvc;
    Plotter* plotter;

    // Flags which memorize which values need to be saved for
    // the logger
    bool record_states;
    bool record_energy;
    bool record_temperature;
    bool record_order;
    bool finally_plot;
    bool finally_save_img;

    // Flags related to buffered outputs and output streams
    bool buffered;
    bool close_after;

public:

    HopfieldLogger(Plotter* plot) : write_buffer(10), plotter(plot), record_states(false), record_energy(false),
        record_temperature(false), record_order(false), f_out(nullptr), close_after(false)
    { }

    HopfieldLogger() : write_buffer(10), record_states(false), record_energy(false),
        record_temperature(false), record_order(false), f_out(nullptr), close_after(false)
    { }

    void set_plotter(Plotter* plot) {
        plotter = plot;
    }

    void set_recording_stream(std::ofstream& stream, bool close_aft = true) {
        f_out = &stream;
        close_after = close_aft;
    }

    void set_buffered(bool v) { buffered = v; }

    void set_collect_states(bool v, const std::string& into = "states.gif") { 
        record_states = v; 
        states_gif_save = into;
    }
    
    void set_collect_energy(bool v) { record_energy = v; }
    
    void set_collect_temperature(bool v) { record_temperature = v; }

    void set_collect_order_parameter(bool v) { record_order = v; }

    void finally_plot_data(bool val) {
        finally_plot = true;
    }

    void finally_write_last_state_png(bool val, const std::string & save = "last.png") {
        finally_save_img = true;
        last_state_save = save;
    }
    
    void on_state_update(const std::tuple<state_index_t, unsigned char> online_change) {
        if (record_states) {
            // Flip the data on this object's owned buffer! We already have the heights and
            // widths for the buffer that were initialized in on_run_begin
            if (!log_buf.write_buffer)
                throw std::runtime_error("Attempting to update a non initialized buffer!");
            (reinterpret_cast<uint32_t*>(log_buf.write_buffer.get()))[std::get<0>(online_change)] 
                = (std::get<1>(online_change)) ? 0xFFFFFFFF : 0;
            gio.write_frame(log_buf.write_buffer.get());
        }
    }

    void on_state_update(const std::vector<std::tuple<state_index_t, unsigned char>> group_change) {
        if (record_states) {
            if (!log_buf.write_buffer)
                throw std::runtime_error("Attempting to update a non initialized buffer!");
            for (const auto& tup : group_change) {
                (reinterpret_cast<uint32_t*>(log_buf.write_buffer.get()))[std::get<0>(tup)]
                    = (std::get<1>(tup)) ? 0xFFFFFFFF : 0;
                gio.write_frame(log_buf.write_buffer.get());
            }
        }
    }

    void on_state_update(const BinaryState& new_state) {
        if (record_states) {
            if (!log_buf.write_buffer)
                throw std::runtime_error("Attempting to update a non initialized buffer!");
            this->update_logger_buffer(new_state, /*reset=*/ false);
        }
    }

    void on_energy_update(double e) {
        if (record_energy)
            nvc.append_for("Energy", e);
    }

    void on_temperature_update(double t) {
        if (record_temperature)
            nvc.append_for("Temperature", t);
    }

    void on_order_parameter_update(double m) {
        if (record_order)
            nvc.append_for("Order parameter", m);
    }

    void on_run_end(const BinaryState& final_state) {
        
        // If the user required the stream to be closed, close it
        if (f_out != nullptr) {
            DataUtils::dump_named_data_to_file(*f_out, nvc);
            if (close_after)
                f_out->close();
        }
        if (record_states) {
            // Close the gif stream to finalize the gif file. Assume that the last alteration to
            // the state was logged already when on_state_update was called.
            gio.end();
        }
        if (finally_save_img) {
            StateUtils::write_state_as_image(final_state, last_state_save,
                (last_state_save.find("png") != std::string::npos) ? "png" : "jpg");
        }
        if (finally_plot) {
            // Now plot the data using our plot routines.
            // We use the data we collected internally in the nvc instead of reading 
            // the same file we dumped from memory. 
            for (const auto& pair : nvc) {
                if (!pair.second.size())
                    continue;
                auto ctx = plotter->context();
                ctx.set_title("Evolution of " + pair.first).
                    set_x_label("Iteration steps").plot_sequence(pair.second);
            }
        }
    }

    void on_run_begin(const BinaryState& begin_state, unsigned int iterations) {
        if (record_states) {
            if (states_gif_save.empty())
                throw std::runtime_error("Cannot begin logging the states: no save file was specified.");
            const auto height = begin_state.get_size() / begin_state.get_stride_y();
            const auto width = begin_state.get_stride_y();
            gio.begin(states_gif_save, width, height, /* delay in 10*ms */ 10);

            this->update_logger_buffer(begin_state, /* reset = */ true);
            gio.write_frame(log_buf.write_buffer.get());
        }
        if (record_energy || record_order || record_temperature) {
            // Ensure the file stream is ok
            // Set up the named data collection
            this->nvc.clear();

            if (record_energy)
                nvc.register_name("Energy", /* reserve*/ iterations);
            if (record_temperature)
                nvc.register_name("Temperature", /* reserve */ iterations);
            // TEMPORARY: For each pattern we associate an order parameter! important
            if (record_order)
                nvc.register_name("Order parameter", /* reserve */ iterations);
        }
    }

protected:
    void update_logger_buffer(const BinaryState& bs, bool reset = false, 
        unsigned int low_value = 0, unsigned int high_val = 255) {
        // Ensure the buffer is initialized properly. The values for the strides are to be
        // specified in the hopfield network containing this state.
        const auto required_width = bs.get_stride_y();
        const auto required_height = bs.get_size() / required_width;
        static_assert(GifWriterIO::required_channels == sizeof(uint32_t));

        // If possible, keep the same buffer!
        if (log_buf.width != required_width || log_buf.height != required_height) {
            // NOTE: The gif-h library unfortunaly only supports rgb png writings, so that
            // we are forced to maintain 4 channels instead of just B&W. 
            log_buf.write_buffer.reset( new unsigned char [ required_width * required_height * 
                GifWriterIO::required_channels ] );
            log_buf.width = required_width;
            log_buf.height = required_height;
        }
        if (reset) {
            std::memset(log_buf.write_buffer.get(), 0, log_buf.height * log_buf.width * 
                GifWriterIO::required_channels);
        }
        // Now map the values. 
        // because (0, 0, 0, 0) = black and (255, 255, 255, 255 ) = white we just treat the
        // buffer as an int buffer instead, obtaining better memory accesses
        uint32_t* int_buffer = reinterpret_cast<uint32_t*>(log_buf.write_buffer.get());
        for (int i = 0; i < bs.get_size(); ++i) {
            if (bs.get(i))
                int_buffer[i] = 0xFFFFFFFF;
            else
                int_buffer[i] = 0x00000000;
        }
        return;
    }

};



#endif