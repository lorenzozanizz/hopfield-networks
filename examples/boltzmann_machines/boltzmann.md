### Restricted Boltzmann Machines Example

This example demonstrates the use of Restricted Boltzmann Machines (RBMs) stacked into a Deep Belief Network (DBN) for unsupervised learning on the MNIST dataset.

Key steps include: 
- Initializing a Plotter gnuplot interface and three RBMs with visible/hidden sizes: 784 -> 529 (23x23), 529 -> 361 (19x19), 361 -> 225 (15x15), and stacking them into a StackedRBM. 
- Loading and binarizing 9000 MNIST samples (threshold 127, scaled to 0.0-1.0). 
- Setting Eigen parallelism to 6 threads. 
- Training the DBN unsupervised over 40 iterations with mini-batch size 300, learning rate 0.02, and Contrastive Divergence (CD-K) with k=13. 
- Visualizing 9 kernels per layer depthwise as 3x3 grids (28x28 for first layer). 
- Attaching a BoltzmannLogger to the first RBM, generating a random visible state (sparsity 0.15), running CD-K=5, and logging state transitions to a GIF.

The example integrates components from `boltzmann/`, `io/`, `datasets/`, and `math/` directories, showcasing feature learning, kernel visualization, and generative sampling in Boltzmann machines