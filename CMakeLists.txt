cmake_minimum_required(VERSION 3.12.0)

# Set the required gfortran path explicitly, somehow does not work
# with an export command.. whatever
find_program(GFORTRAN_EXECUTABLE NAMES gfortran REQUIRED) 
set(CMAKE_Fortran_COMPILER ${GFORTRAN_EXECUTABLE})

project(project_title LANGUAGES CXX C Fortran)
find_package(BLAS REQUIRED)

add_executable(main "src/main.cpp")
