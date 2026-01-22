### Konohen Mapping Example

This example showcases the use of Konohen (Kohonen-like) self-organizing maps for processing the MNIST dataset.
It loads a subset of MNIST images as vectors, initializes a Konohen map (10x10 grid) for the 28x28 pixel inputs, and trains it over 100 iterations using an gaussian neighboring function with a learning rate of 0.10.

Key steps include:
 - Loading and preparing the MNIST dataset (4000 samples).
 - Configuring the map with parameters like initial sigma (3.0), tau (10.0), and evolving function ("exponential"). 
 - Training the map on the dataset. 
 - Creating a label map for digits 0-9 and classifying using a MajorityMappingEigen classifier with a 0.6 threshold. 
 - Plotting the classification results, heatmap visualizations of mapping weights every 9 units to see what the mapping has learned to associate with the input cortex 
 - Exploring clustering: Computing a distance U-Matrix, performing U-Clustering, and K-Means on the unit weights (10 clusters, 50 iterations). 
 - Plotting all clustering results for analysis.

The example uses Eigen for matrix operations, demonstrating topological mapping, classification, and clustering on the learned map.

This code integrates components from `mappings/`, `io/`, and `datasets/` to create the mappings and load the required MNIST Dataset. 