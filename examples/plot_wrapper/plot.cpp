
#include <vector>
#include <cmath>
#include <tuple>

#include "io/plot/plot.hpp" 

int main() {

    // This opens the pipe to the Gnuplot process.
    Plotter plotter;

    // --- EXAMPLE 1: 2D LINE PLOT (Energy Convergence Simulation) ---
    // We generate a synthetic "Energy" curve: E(t) = exp(-t/10) + noise
    std::vector<double> time;
    std::vector<double> energy;
    for (int t = 0; t < 50; ++t) {
        time.push_back(t);
        energy.push_back(std::exp(-static_cast<double>(t) / 10.0) + (std::rand() % 100) * 0.001);
    }

    // Use context() to open a new window (WXT terminal)
    plotter.context()
        .set_title("Neural Network Energy Convergence")
        .set_x_label("Iteration")
        .set_y_label("Energy Value")
        .plot_2d(time, energy); // Plots the X-Y pair vectors

    // --- EXAMPLE 2: HEATMAP (Weight Matrix / State Visualization) ---
    // We generate a 20x20 interference pattern (Sine wave interference)
    const int width = 20;
    const int height = 20;
    std::vector<float> heatmap_data(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            heatmap_data[y * width + x] = std::sin(x * 0.5f) * std::cos(y * 0.5f);
        }
    }

    // Display as a heatmap using a grayscale-to-blue palette
    plotter.context()
        .set_title("Weight Matrix Heatmap")
        .show_heatmap(heatmap_data.data(), width, height, "grey");

    // --- EXAMPLE 3: MULTIPLE SEQUENCES (Order Parameters) ---
    // Visualizing how multiple stored patterns overlap during retrieval
    std::vector<std::vector<double>> sequences(30, std::vector<double>(3));
    for (int i = 0; i < 30; ++i) {
        sequences[i][0] = std::tanh(i * 0.1);    // Pattern 1 overlap
        sequences[i][1] = 1.0 - std::tanh(i * 0.1); // Pattern 2 overlap
        sequences[i][2] = 0.2 * std::sin(i * 0.5);  // Noise overlap
    }

    plotter.context()
        .set_title("Overlap (Order Parameters)")
        .set_x_label("Timestep")
        .plot_multiple_sequence(sequences);

    // --- EXAMPLE 4: DISCRETE CATEGORIES (Classification Map) ---
    // Imagine a grid where each cell is classified into 4 categories
    std::vector<int> categories(10 * 10);
    for (int i = 0; i < 100; ++i) {
        categories[i] = i % 4; // Cycles through 4 distinct colors
    }

    plotter.context()
        .set_title("Pattern Classification Grid")
        .show_discrete_categories(categories, 10, 10, 4);

    // Since Gnuplot runs in a separate process, the program would exit
    // and close the plots immediately. .block() waits for user input.
    // Type 'continue' in the console to finish.
    plotter.block();


    return 0;
}