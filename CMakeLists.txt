cmake_minimum_required(VERSION 3.12.0)

project(project_title LANGUAGES CXX C)

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

# Merge all the cpp files in a single executable library. 
add_library(executables 

    # --------- UTILS ---------
    "src/utils/timing.cpp"
    # --------- HOPFIELD ---------
    "src/hopfield/classifier/hopfield_classifier.cpp"
    "src/hopfield/states/binary.cpp"
    # --------- IO ROUTINES ---------
    "src/io/image/images.cpp"
    "src/io/gif/gif.cpp"
    "src/io/io_utils.cpp" 
    "src/io/datasets/repository.cpp"
)
target_include_directories(executables PUBLIC "${CMAKE_SOURCE_DIR}/src")

target_link_libraries(executables PRIVATE Eigen3::Eigen)
target_link_libraries(executables PRIVATE OpenMP::OpenMP_CXX)

add_executable(main "src/main.cpp")

# Link all the executables for the application. 
target_link_libraries(main PRIVATE executables)

target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(main PRIVATE Eigen3::Eigen)

add_executable(hopfield "examples/hopfield/hopfield.cpp")
add_executable(boltzmann "examples/boltzmann_machines/boltzmann.cpp")
add_executable(plot "examples/plot_wrapper/plot.cpp")
add_executable(konohen "examples/konohen/konohen.cpp")
add_executable(reservoir "examples/reservoir/reservoir.cpp")
add_executable(blastest "examples/blas/blas.cpp")
    
add_library(example_common INTERFACE)

target_compile_options(example_common INTERFACE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
)

# Disable debug checks
target_compile_definitions(example_common INTERFACE
    $<$<CONFIG:Release>:NDEBUG>
)

target_link_libraries(example_common
    INTERFACE
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
        executables
)

target_link_libraries(blastest PRIVATE example_common)
target_link_libraries(hopfield PRIVATE example_common)
target_link_libraries(reservoir PRIVATE example_common)
target_link_libraries(konohen PRIVATE example_common)
target_link_libraries(boltzmann PRIVATE example_common)
target_link_libraries(plot PRIVATE example_common)
