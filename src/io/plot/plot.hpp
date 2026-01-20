#pragma once
#ifndef IO_PLOT_PLOT_HPP
#define IO_PLOT_PLOT_HPP

#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <tuple>

// This section requires explicit support for gnuplot and cannot be compiled
// without it. (CMAKE will assert this define if gnuplot is found)
#ifndef COMPILE_WITH_GNUPLOT
#ifndef WARNED_GNUPLOT_YET
#define WARNED_GNUPLOT_YET
#error "You are attempting to include a gnuplot wrapper which requires\n\
	gnuplot. No installation was found"
static_assert(false, "You are attempting to include a gnuplot wrapper which " 
	"requires gnuplot. No installation was found")

#endif // !WARNED_GNUPLOT_YET
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


	private:

		std::string random_color() {
			static std::mt19937 rng(std::random_device{}());
			static std::uniform_int_distribution<int> dist(0, 255);

			std::ostringstream oss;
			oss << "#"
				<< std::hex << std::setw(2) << std::setfill('0') << dist(rng)
				<< std::setw(2) << std::setfill('0') << dist(rng)
				<< std::setw(2) << std::setfill('0') << dist(rng);

			return oss.str();
		}

	public:

		// Receive a pipeline from the external plotter object
		PlottingContext(GnuplotPipe& p): pipe(p) { 
			// Reset all the previous context (titles, flipping ys, ...)
			pipe.send_line("reset");
		}

		PlottingContext& redirect_to_image(const std::string& file_name) {
			pipe.send_line("set output '" + file_name + "'; set term png;");
			return *this;
		}

		PlottingContext& begin_multi_line_plot() {
			return *this;
		}

		PlottingContext& set_title(const std::string& plot_title) {
			pipe.send_line("set title '" + plot_title + "'");
			return *this;
		}

		PlottingContext& set_x_label(const std::string& label) {
			pipe.send_line("set xlabel '" + label + "'");
			return *this;
		}

		PlottingContext& set_y_label(const std::string& label) {
			pipe.send_line("set ylabel '" + label + "'");
			return *this;
		}

		template <typename XType, typename YType>
		PlottingContext& plot_2d(const std::vector<std::tuple<XType, YType>>& xs) {
			const auto size = xs.size();
			pipe.send_line("plot '-' using 1:2 with lines");
			for (int i = 0; i < size; ++i) {
				auto& pair = xs[i];
				pipe.send_line(std::to_string(std::get<0>(pair)) + " " + std::to_string(std::get<1>(pair)));
			}
			pipe.send_line("e");
			return *this;
		}

		template <typename XType, typename YType>
		PlottingContext& plot_2d(const std::vector<XType>& xs, const std::vector<YType>& ys) {
			if (xs.size() != ys.size()) {
				throw std::runtime_error("Cannot plot mismatch dimensions: x,y ->" +
					std::to_string(xs.size()) + ", " + std::to_string(ys.size()));
			}
			const auto size = xs.size();
			pipe.send_line("plot '-' using 1:2 with lines");
			for (int i = 0; i < size; ++i) {
				pipe.send_line(std::to_string(xs[i]) + " " + std::to_string(ys[i]));
			}
			pipe.send_line("e");
			return *this;
		}

		template <typename YType>
		PlottingContext& plot_sequence(const std::vector<YType>& ys) {
			pipe.send_line("plot '-' using 0:1 with lines");
			for (int i = 0; i < ys.size(); ++i)
				pipe.send_line(std::to_string(ys[i]));
			pipe.send_line("e");
			return *this;
		}

		PlottingContext& write_image(const std::string& path, unsigned char* data, int width, int height, int channels) {
			this->redirect_to_image(path);
			this->show_image(data, width, height, channels);
			return *this;
		}

		PlottingContext& show_image(const unsigned char* buffer, int width, int height, int channels) {

			pipe.send_line("set yrange [*:*] reverse");
			if (channels == 1) {
				// We simply handle a greyscale image writing an intermediate and using
				// gnuplot map view.
				auto raw_pipe = pipe.raw();
				fprintf(raw_pipe, "plot '-' binary array=(%d,%d) format='%%uchar' with image\n", width, height);
				fwrite(buffer, 1, width * height, raw_pipe);
			}
			else {

			}
			return *this;
		}

		template <typename FloatingType>
		PlottingContext& plot_multiple_heatmaps(
			std::vector<FloatingType*> buffers, unsigned int plot_width, unsigned int plot_height,
			unsigned int width, unsigned int height
		) {
			pipe.send_line("set multiplot layout " + 
				std::to_string(width) + "," + std::to_string(height));
			pipe.send_line("unset key");
			pipe.send_line("unset colorbox");
			pipe.send_line("set size square");
			pipe.send_line("set pm3d map");

			// Set the grayscale palettes. 
			pipe.send_line("set palette rgbformula -7,2,-7");

			auto raw_pipe = pipe.raw();
			for (int plot_i = 0; plot_i < plot_height; ++plot_i)
				for (int plot_j = 0; plot_j < plot_width; ++plot_j) {
					
					auto& buffer = buffers[plot_i * plot_width + plot_j];

					pipe.send_line("splot '-' matrix with image");
					for (int i = 0; i < height; ++i) {
						for (int j = 0; j < width; ++j)
							if constexpr (std::is_same_v<FloatingType, double>) 
								fprintf(raw_pipe, "%lf ", buffer[i * width + j]);
							else if constexpr (std::is_same_v<FloatingType, float>) 
								fprintf(raw_pipe, "%f ", buffer[i * width + j]);
						fprintf(raw_pipe, "\n");
					}
					pipe.send_line("e");
				}
			return *this;
		}

		PlottingContext& show_heatmap(const float* buffer, int width, int height, 
			std::string palette = "") {

			pipe.send_line("set yrange [*:*] reverse");
			pipe.send_line("unset key");
			pipe.send_line("set view map");
			if (!palette.empty())
				pipe.send_line("set palette " + palette);
			else
				pipe.send_line("set palette rgbformula -7,2,-7");


			auto raw_pipe = pipe.raw();
			pipe.send_line("splot '-' matrix with image");
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width; ++j)
					fprintf(raw_pipe, "%f ", buffer[i * width + j]);
				fprintf(raw_pipe, "\n");
			}
			pipe.send_line("e");
			return *this;
		}

		PlottingContext& show_heatmap(const double* buffer, int width, int height,
			std::string palette = "") {

			pipe.send_line("set yrange [*:*] reverse");
			pipe.send_line("unset key");
			pipe.send_line("set view map");
			if (!palette.empty())
				pipe.send_line("set palette " + palette);
			else 
				pipe.send_line("set palette rgbformula -7,2,-7");


			auto raw_pipe = pipe.raw();
			pipe.send_line("splot '-' matrix with image");
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width; ++j)
					fprintf(raw_pipe, "%lf ", buffer[i * width + j]);
				fprintf(raw_pipe, "\n");
			}
			pipe.send_line("e");
			return *this;
		}

		PlottingContext& show_discrete_categories(std::vector<int> buffer, int width, int height,
			int num_categories, bool zero_black=false) {

			pipe.send_line("set yrange [*:*] reverse");
			pipe.send_line("unset key");
			pipe.send_line("set view map");
			pipe.send_line("set palette maxcolors " + std::to_string(num_categories));
			pipe.send_line("set palette defined ( \\");
			for (int i = 0; i < num_categories - 1; ++i) {
				if (i == 0 && zero_black)
					pipe.send_line("0  \"#000000\", \\");
				else
					pipe.send_line(std::to_string(i) + " \"" + random_color() + "\", \\");
			}
			pipe.send_line(std::to_string(num_categories-1) + " \"" + random_color() + "\")");
			std::cout << "set cbtics 0,1," + std::to_string(num_categories) << std::endl;
			pipe.send_line("set cbtics 0,1," + std::to_string(num_categories));
			auto raw_pipe = pipe.raw();
			pipe.send_line("plot '-' matrix with image");
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width; ++j)
					fprintf(raw_pipe, "%d ", buffer[i * width + j]);
				fprintf(raw_pipe, "\n");
			}
			pipe.send_line("e");
			return *this;
		}

		PlottingContext& set_cblabel(const std::string& val) {
			pipe.send_line("set cblabel '" + val + "'");
			return *this;
		}

		PlottingContext& unset_cb_ticks(const std::string& val) {
			pipe.send_line("unset cbtics");
			return *this;
		}

		PlottingContext& show_image(const Image& image) {
			return this->show_image(image.view(), image.width, image.height, image.channels);
		}

		PlottingContext& show_binary_image(unsigned char *data, int width, int height) {
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
			return *this;
		}

	
		PlottingContext& plot_multiple_sequence(std::vector<std::vector<double>> values) {
			const unsigned long inner_size = values[0].size();
			if (inner_size == 1) {
				pipe.send_line("plot '-' using 0:1 with lines title 'Line'");
				for (int j = 0; j < values.size(); ++j)
					pipe.send_line(std::to_string(values[j][0]));
				pipe.send_line("e");
			}
			else {
				// At least 2, so send a comma
				pipe.send_line("plot '-' using 0:1 with lines title 'Line 0', \\");
				for (int i = 1; i < inner_size - 1; ++i)
					pipe.send_line(" '-' using 0:1 with lines, \\");
				if (inner_size >= 2)
					pipe.send_line(" '-' using 0:1 with lines");
				for (int i = 0; i < inner_size; ++i) {
					for (int j = 0; j < values.size(); ++j)
						pipe.send_line(std::to_string(values[j][i]));
					pipe.send_line("e");
				}
			}
			return *this;
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
		std::cout << "\x1b[0;33m" << 
			"\n> ( PLOTTER NOTE ) Call to Plotter.block() issued: to continue, write 'continue' or 'clear' to close plots." << 
			"\x1b[0m" << std::endl;
		do {
			std::cin >> v;
			// Make spin locking a bit loose...
			std::this_thread::sleep_for(std::chrono::milliseconds(300));
		} while (!(v == "continue") && !(v == "clear"));

		clear_all_plots();

	}
protected:

	void clear_all_plots() {
		for (int i = 1; i < this->plot_id; ++i) {
			this->pipe.send_line("set terminal wxt " + std::to_string(i) + " close");
		}
	}


	// Utility function to write the stream of data in the gnuplot
	// pipe
	void write_x_y_pairs() {

	}

}; // ! class Plotter


#endif // !COMPILE_WITH_GNU
#endif // !IO_PLOT_PLOT_HPP