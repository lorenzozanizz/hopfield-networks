cmake_minimum_required(VERSION 3.12.0)

# Set the required gfortran path explicitly, somehow does not work
# with an export command.. whatever
find_program(GFORTRAN_EXECUTABLE NAMES gfortran REQUIRED) 
set(CMAKE_Fortran_COMPILER ${GFORTRAN_EXECUTABLE})

project(project_title LANGUAGES CXX C Fortran)

find_program(GNUPLOT_EXECUTABLE gnuplot)

if (GNUPLOT_EXECUTABLE)
    message(STATUS "Found gnuplot: ${GNUPLOT_EXECUTABLE}")
    # Require boost to link against the gnuplot header
    find_package(Boost REQUIRED)

    add_compile_definitions(COMPILE_WITH_GNU)
else()
    message(STATUS "Gnuplot not found")
endif()

add_executable(main "src/main.cpp")
