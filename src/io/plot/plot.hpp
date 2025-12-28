#pragma once
#ifndef IO_PLOT_PLOT_HPP
#define IO_PLOT_PLOT_HPP

#include <string>
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


class Plotter {

public:

	// When a plotter is created, the gnuplot pipe is explicitly created. 
	Plotter() { }
	~Plotter() { }

	// Basically, to plot we have
	class PlottingContext {

		// Receive a pipeline from the external plotter object
		PlottingContext() {

		}

		void redirect_to_image() {
			// set output "plot.png"
		}

		void begin_multi_line_plot() {

		}

		void conclude_multi_line_plot() {

		}

		void plot_2d() {

		}

		void plot_expr(std::string) {

		}

		~PlottingContext() {


		}

	};

	void read_gp_file(const std::string) {

	}

protected:

	// Utility function to write the stream of data in the gnuplot
	// pipe
	void write_x_y_pairs() {

	}

}; // ! class Plotter


#endif // !COMPILE_WITH_GNU
#endif // !IO_PLOT_PLOT_HPP