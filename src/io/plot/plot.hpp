#pragma once
#ifndef IO_PLOT_PLOT_HPP
#define IO_PLOT_PLOT_HPP

// This section requires explicit support for gnuplot and cannot be compiled
// without it. (CMAKE will assert this define if gnuplot is found)
#ifndef COMPILE_WITH_GNUPLOT
#error "You are attempting to include a gnuplot wrapper which requires\n\
	gnuplot. No installation was found"
static_assert(false, "You are attempting to include a gnuplot wrapper which " 
	"requires gnuplot. No installation was found")
#else
// Header only library for gnuplot interface
// #include "../../../external/gnuplot-iostream.h"
// Will now provide a basic wrapper to GNUPlot functionalities. We mainly
// need simple 2d plotting to monitor energy and order parameters during
// the execution of the networks, nothing fancy. s

// Tell MSVC to not warn about using fopen.
// http://stackoverflow.com/a/4805353/1048959
#if defined(_MSC_VER) && _MSC_VER >= 1400
#pragma warning(disable:4996)
#endif

class Plotter {




}; // ! class Plotter


#endif // !COMPILE_WITH_GNU
#endif // !IO_PLOT_PLOT_HPP