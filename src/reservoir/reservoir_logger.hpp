#pragma once
#ifndef RESERVOIR_LOGGER_HPP
#define RESERVOIR_LOGGER_HPP

#include "../math/matrix/matrix_ops.hpp"

#include "../io/gif/gif.hpp"
#include "../io/plot/plot.hpp"
#include "../io/image/images.hpp"

template <typename DataType>
class ReservoirLogger {
    
    // A small buffer object to map the reduced memory representation of states
    // to the required png buffer to save the intermediate state of the networks
    // (a bit of memory overhead, but around 40kbs at most)
    struct logger_buffer_t {

        unsigned int width = 0;
        unsigned int height = 0;
        std::unique_ptr<unsigned char[]> write_buffer = nullptr;

    };

    unsigned int visual_width;
    unsigned int visual_height;

	std::string states_gif_save;
	logger_buffer_t log_buf;
	GifWriterIO gio;

	using ReservoirState = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

	NamedVectorCollection<double> nvc;
    Plotter* plotter;

	bool record_norm;
	bool record_state;

    bool do_plot;

public:

    ReservoirLogger() : record_norm(false), record_state(false) { }
	
    void assign_plotter(Plotter* plot) {
        plotter = plot;
    }

	void set_collect_states(bool value, 
		const std::string& into = "states.gif",
		unsigned int width = 0, unsigned int height = 0) {
		record_state = true;
		states_gif_save = into;
        visual_width = width;
        visual_height = height;

	}
	
	void set_collect_norm(bool value) {
		record_norm = value;
	}

	void on_state_update(const ReservoirState& new_state) {
        if (record_state) {
            if (!log_buf.write_buffer)
                throw std::runtime_error("Attempting to update a non initialized buffer!");
            this->update_logger_buffer(new_state, /*reset=*/ false);
            gio.write_frame(log_buf.write_buffer.get());
        }
	}

    void finally_plot(bool value) {
        do_plot = value;
    }

	void on_norm_update(double norm) {
		if (record_norm) {
            nvc.append_for("Reservoir norm", norm);
		}
	}

    void on_run_begin(const ReservoirState& begin_state) {

        if (record_state) {
            if (states_gif_save.empty())
                throw std::runtime_error("Cannot begin logging the states: no save file was specified.");
            const auto height = visual_width;
            const auto width = visual_height;
            if (height <= 0 || width <= 0)
                throw std::runtime_error("Cannot log the states: width and heights should have been set previously!");
            
            gio.begin(states_gif_save, width, height, /* delay in 10*ms */ 20);

            this->update_logger_buffer(begin_state, /* reset = */ true);
            gio.write_frame(log_buf.write_buffer.get());
        }
        if (record_norm) {
            // Ensure the file stream is ok
            // Set up the named data collection
            this->nvc.clear();

            nvc.register_name("Reservoir norm", /* reserve*/ 25);
        }
    }

    void on_run_end() {
        if (record_state) {
            // Close the gif stream to finalize the gif file. Assume that the last alteration to
            // the state was logged already when on_state_update was called.
            gio.end();
        }
        if (do_plot) {
            // Plot the norm of the reservoir and all the rest.
            std::cout << "hey!";
            for (const auto& pair : nvc) {
                std::cout << "hey!";
                if (!pair.second.size())
                    continue;
                std::cout << " continue";
                auto ctx = plotter->context();
                ctx.set_title("Evolution of " + pair.first).
                    set_x_label("Iteration steps").plot_sequence(pair.second);
            }
        }
	}

protected:

    void update_logger_buffer(const ReservoirState& state, bool reset = false) {
        // Ensure the buffer is initialized properly. The values for the strides are to be
        // specified in the hopfield network containing this state.
        const auto required_width = visual_width;
        const auto required_height = visual_height;
        static_assert(GifWriterIO::required_channels == sizeof(uint32_t));

        // If possible, keep the same buffer!
        if (log_buf.width != required_width || log_buf.height != required_height) {
            // NOTE: The gif-h library unfortunaly only supports rgb png writings, so that
            // we are forced to maintain 4 channels instead of just B&W. 
            log_buf.write_buffer.reset(new unsigned char[required_width * required_height *
                GifWriterIO::required_channels]);
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

        auto max = state.maxCoeff();
        auto min = state.minCoeff();
        // Cap the values in the 0, 255 range when writing. 
        uint32_t* int_buffer = reinterpret_cast<uint32_t*>(log_buf.write_buffer.get());
        for (int i = 0; i < state.size(); ++i) {
            auto value = state(i);
            unsigned char v = static_cast<unsigned char>((value - min) / (max - min) * 255);
            uint32_t rgba = 0x0000000;
            rgba = (uint32_t(v) << 24) | // R 
                (uint32_t(v) << 16) | // G 
                (uint32_t(v) << 8) | // B 
                255; 
            int_buffer[i] = rgba;
        }
        return;
    }



};

#endif