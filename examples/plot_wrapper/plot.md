### Plotter Example

This example demonstrates the Plotter class (interfacing with Gnuplot) for creating various visualizations relevant to neural network analysis, such as energy convergence, weight matrices, order parameters, and classification grids.

Key steps include:
- Generating and plotting a 2D line for simulated energy convergence (exponential decay with noise).
- Creating and displaying a 20x20 heatmap of a sine-cosine interference pattern using a grayscale palette.
- Plotting multiple sequences (3 lines) representing pattern overlaps over timesteps.
- Showing a 10x10 grid of discrete categories (cycling through 4 colors) as a classification map.
- Using `plotter.block()` to pause execution and keep plots visible until user input.

The example integrates components from the `io/plot/` directory, showcasing 2D plotting, heatmaps, multi-line plots, and categorical visualizations for debugging and analysis in machine learning contexts.