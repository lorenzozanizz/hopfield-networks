#pragma once
#ifndef IO_PLOT_PLOT_HPP
#define IO_PLOT_PLOT_HPP

#include <iostream>
#include <string>
#include <chrono>
#include <thread>

#include <tuple>

// This section requires explicit support for gnuplot and cannot be compiled
// without it. (CMAKE will assert this define if gnuplot is found)
#ifndef COMPILE_WITH_GNUPLOT
#error "You are attempting to include a gnuplot wrapper which requires\n\
	gnuplot. No installation was found"
static_assert(false, "You are attempting to include a gnuplot wrapper which " 
	"requires gnuplot. No installation was found")
#else
// Header only library for gnuplot interface
#include "gnuplot_wrapper.hpp"
// Will now provide a basic wrapper to GNUPlot functionalities. We mainly
// need simple 2d plotting to monitor energy and order parameters during
// the execution of the networks, nothing fancy. s

#include "../image/images.hpp"
#include "../../io/io_utils.hpp"

class Plotter {

	GnuplotPipe pipe;
	unsigned int plot_id;

public:

	inline static const std::string temp_file = "gnuplot_temp.dat";

	Plotter(): pipe() {
		plot_id = 1;
		pipe.set_on_close([this]() {
			for (int i = 1; i < this->plot_id; ++i) {
				this->pipe.send_line("set terminal wxt " + std::to_string(i) + " close");
			}
		});
		// On my installation of linux wsl the default installation of gnuplot
		// reports errors when attempting to write intermediate files with the default
		// terminal. Use an alternative terminal which does not give such an error 
		if (!pipe.success())
			throw std::runtime_error("Failed to open a pipe to gnuplot: could not plot.");
		// pipe.send_line("set term wxt");
	}

	// Basically, to plot we have
	class PlottingContext {

		GnuplotPipe& pipe;

	public:

		// Receive a pipeline from the external plotter object
		PlottingContext(GnuplotPipe& p): pipe(p) { }

		void redirect_to_image(const std::string& file_name) {
			// set output "plot.png"
		}

		void begin_multi_line_plot() {

		}

		void plot_2d() {

		}

		void show_image(const Image& image) {
			// NOTE: This is not thread safe! ensure that plotting happens on a single
			// master thread at all times. 
			const unsigned char* buffer = image.view();
			pipe.send_line("set yrange [*:*] reverse");
			if (image.channels == 1) {
				// We simply handle a greyscale image writing an intermediate and using
				// gnuplot map view.
				std::ofstream out(Plotter::temp_file);
				for (int y = 0; y < image.height; y++) {
					for (int x = 0; x < image.width; x++) {
						out << (int)buffer[y * image.width + x] << " ";
					} out << "\n";
				} out.close();
				// pipe.send_line("set view map");
				// pipe.send_line("plot 'gnuplot_temp.dat' matrix with image");

				auto raw_pipe = pipe.raw();
				fprintf(raw_pipe, "plot '-' binary array=(%d,%d) format='%%uchar' with image\n", image.width, image.height);
				fwrite(buffer, 1, image.width * image.height, raw_pipe);
			}
			else {

			}

		}

		void show_binary_image(unsigned char *data, int width, int height) {
			auto raw_pipe = pipe.raw();
			pipe.send_line("set yrange [*:*] reverse");
			fprintf(raw_pipe, "plot '-' binary array=(%d,%d) format='%%uchar' with image\n", width, height);

			for (int bit = 0; bit < width*height; ++bit) {
				int byte_index = bit / 8;
				int bit_index = bit % 8;

				bool bit_value = (data[byte_index] >> bit_index) & 1;
				if (bit_value) fprintf(raw_pipe, "%c", 255);
				else fprintf(raw_pipe, "%c", 0);
			}
		}

		~PlottingContext() { }

	};

	PlottingContext context() {
		pipe.send_line("pause 0");
		pipe.send_line("set term wxt " + std::to_string(plot_id++) + " persist");
		return PlottingContext(pipe);
	}

	void set_terminal(std::string terminal) {
		pipe.send_line("set term " + terminal);
	}

	void read_gp_file(const std::string) {

	}

	~Plotter() {
		// We do not close explicitly the pipe in this version to avoid
		// double freeing
		// pipe.close();
	}

	void block() {
		// First we ensure that gnuplot has really flushed all our data:
		pipe.flush_send_end_of_data(0);
		std::string v;
		do {
			std::cin >> v;
			// Disallow spin locking...
			std::this_thread::sleep_for(std::chrono::milliseconds(300));
		} while (!(v == "continue") && !(v == "clear"));
	}
protected:


	// Utility function to write the stream of data in the gnuplot
	// pipe
	void write_x_y_pairs() {

	}

}; // ! class Plotter


#endif // !COMPILE_WITH_GNU
#endif // !IO_PLOT_PLOT_HPP