cmake_minimum_required(VERSION 3.12.0)

# Set the required gfortran path explicitly, somehow does not work
# with an export command.. whatever
find_program(GFORTRAN_EXECUTABLE NAMES gfortran REQUIRED) 
set(CMAKE_Fortran_COMPILER ${GFORTRAN_EXECUTABLE})

project(project_title LANGUAGES CXX C Fortran)

add_executable(main "src/main.cpp")
# add_executable(prog "src/math/autograd/math.f90")

find_program(GNUPLOT_EXECUTABLE gnuplot)

if (GNUPLOT_EXECUTABLE)
    message(STATUS "Found gnuplot: ${GNUPLOT_EXECUTABLE}")
    # Require boost to link against the gnuplot header
    # find_package(Boost REQUIRED COMPONENTS filesystem system)

    add_library(gnuplot_io INTERFACE)
    target_include_directories(gnuplot_io INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external ) 
    #target_link_libraries(gnuplot_io INTERFACE
    #    Boost::filesystem 
    #    Boost::system
    #) 

    # target_link_libraries(main PRIVATE gnuplot_io)

    add_compile_definitions(COMPILE_WITH_GNUPLOT)
else()
    message(STATUS "Gnuplot not found")
endif()

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED) 
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(main PRIVATE Eigen3::Eigen)
