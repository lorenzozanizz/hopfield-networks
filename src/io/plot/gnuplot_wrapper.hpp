#ifndef GNUPLOT_WRAPPER_HPP
#define GNUPLOT_WRAPPER_HPP
#pragma once

// A light wrapper to gnuplot loosely inspired from https://github.com/martinruenz/gnuplot-cpp
// the more standard gnuplut_iostream.h was discarded in favor of a lightweight 
// alternative to avoid the forced dependency on Boost filesystem and system.
// 
// Tell MSVC to not warn about using fopen.
// http://stackoverflow.com/a/4805353/1048959
#if defined(_MSC_VER) && _MSC_VER >= 1400
#pragma warning(disable:4996)
#endif

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <fstream>

class GnuplotPipe {

protected:

    FILE* pipe;
    std::vector<std::string> buffer;

    // Delete copies to avoid messing with pipes
    GnuplotPipe(GnuplotPipe const&) = delete;
    void operator=(GnuplotPipe const&) = delete;

public:
    
    GnuplotPipe(bool persist = true) {
        pipe = popen(persist ? "gnuplot -persist" : "gnuplot", "w");
        // Do not explicitly throw exceptions, instead let the users check explicitly
        // for pipe healt with pipe.success();
        // if (!pipe)
        //     throw std::runtime_error("Failed to create the gnuplot pipe")
    }

    ~GnuplotPipe() {
        close();
    }

    bool success() {
        return pipe != nullptr;
    }

    void close() {
        if (pipe) 
            pclose(pipe);
    }

    void send_line(const std::string& text, bool use_buffer = false) {
        if (!pipe) return;
        if (use_buffer)
            buffer.push_back(text + "\n");
        else
            fputs((text + "\n").c_str(), pipe);
    }

    void flush_send_end_of_data(unsigned repeat_buffer = 1) {
        if (!pipe) return;
        for (unsigned i = 0; i < repeat_buffer; i++) {
            for (auto& line : buffer) fputs(line.c_str(), pipe);
            fputs("e\n", pipe);
        }
        fflush(pipe);
        buffer.clear();
    }

    void send_newline() {
        send_line("\n", !buffer.empty());
    }

    void redirect_to_file(const std::string& file_name) {
        std::ofstream file_out(file_name);
        for (auto& line : buffer)
            file_out << line;
        file_out.close();
    }

};

#endif //! GNUPLOT_WRAPPER_HPP