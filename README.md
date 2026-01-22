# Hopfield/Krotov Networks

This project explores a collection of neural models spanning unsupervised representation learning, associative memory, probabilistic generative modeling, and reservoir computing.
The implemented architectures include Self–Organizing Maps for topology–preserving embedding, Hopfield and Boltzmann networks for energy–based associative memory, deep generative stacks based on Restricted Boltzmann Machines, multilayer perceptrons trained by backpropagation, and reservoir networks for temporal feature extraction.

Together, these models illustrate complementary principles of neural computation—competition, attractor dynamics, stochastic sampling, gradient–based optimization, and dynamical state encoding—providing a unified experimental framework for studying learning, representation, and memory in neural systems.

---

##  Build & Compile Instructions

### Prerequisites
- **GCC** compiler with **OpenMP** support.
- **Eigen** and **GNUPlot** C++ libraries.

### Installation
To install and run, simply clone the repository and create a /build folder inside the project. 
Cmake will first ensure that a matching OpenMP implementation before building the project code.


```bash
git clone https://github.com/lorenzozanizz/hopfield-networks
cd hopfield-networks
mkdir build
cd build
cmake ..
make
```

---
## Examples
To run the examples some dataset are required, available at the following [link](https://zenodo.org/records/18337022).
Download both the datasets and add them in the ```/build``` folder.

### Hopefield Networks on MNIST
This program implements and compares deterministic and stochastic Hopfield networks for associative memory, denoising, and pattern classification on binarized MNIST digits. A small set of representative patterns is stored using different learning rules, and the network dynamics are visualised through logging of states, energy, temperature, and order parameters. The denoising capability is demonstrated by reconstructing noisy inputs, while a simple attractor–based classifier evaluates retrieval performance through a confusion matrix and cross–talk visualisation.
#### Setting Parameters
The network size is fixed by the image resolution (```MNIST_SIZE``` x ```MNIST_SIZE```), determining memory capacity and computational cost. The choice of learning rule (Hebbian or Storkey) controls interference and stability. Update policy and group size regulate convergence speed and synchrony. In the stochastic case, the annealing schedule parameters (initial/final temperature and iterations) govern exploration versus convergence. Noise level in the input tests robustness, while the classifier confidence threshold trades off rejection rate and accuracy. The number of iterations sets reconstruction depth and classification reliability.
#### Running the example
In order to run this example, after following the instructions written in the **Installation** section, in the ```\build ``` folder you have to execute ```/.hopefield```

### Self–Organizing Map on MNIST
This example trains a Kohonen Self–Organizing Map on the MNIST dataset to obtain a topology–preserving embedding of handwritten digits.
After unsupervised training, neurons are labeled by majority voting, and the resulting map is analyzed using U–Matrix methods and K–Means clustering to visualize and study the emergent class structure in the learned representation.

#### Setting Parameters
The map size (```map_width```, ```map_height```) controls the resolution of the representation: larger maps capture finer structure but require more data and training iterations.
The neighborhood width ```sigma0 ``` should initially be large enough to enforce global ordering, then gradually decrease (controlled by ```tau``` and the chosen decay function) to allow local refinement.
The learning rate determines the magnitude of weight updates and is typically set in the range 0.05–0.2 for stable convergence.
The number of iterations should scale with both dataset size and map size to ensure full self–organization.
The classification threshold in the majority map regulates label confidence: higher values yield more reliable but sparser neuron labeling.

#### Running the example
In order to run this example, after following the instructions written in the **Installation** section, in the ```\build ``` folder you have to execute ```/.kohonen```

### Reservoir on ECG signals
This program implements a reservoir computing pipeline for nonlinear time–series representation and classification on ECG signals. A randomly initialised echo–state reservoir generates a high–dimensional dynamical embedding of sliding input windows, whose internal activity is optionally monitored through logging and visualisation. The resulting embedded representations are then fed to a deep multi–layer perceptron trained with cross–entropy loss to perform multi–class heartbeat classification, with training and verification phases used to assess convergence and generalisation.
#### Setting Parameters
The reservoir configuration is defined by the input window size (```input_size```) and the internal state dimension (```reservoir_size```), which control temporal resolution and embedding richness. The sparsity and spectral radius of the recurrent weights regulate memory depth and stability of the dynamics, while the input weight distribution sets the excitation scale. Batch size and thread count determine computational efficiency during mapping. The classifier architecture (```units```) specifies model capacity, and the choice of activations and loss function matches the categorical output structure. Training epochs, batch size, and learning rate jointly govern optimisation speed, numerical stability, and final classification accuracy.
#### Running the example
In order to run this example, after following the instructions written in the **Installation** section, in the ```\build ``` folder you have to execute ```/.reservoir```

### MLP Predictor
This program trains and evaluates a reservoir–MLP predictor on a synthetic sine time series. A reservoir‐based feature mapping is combined with a multi–layer perceptron trained by backpropagation using automatic differentiation and a mean–squared error loss. Training and verification datasets are generated from phase–shifted sine sequences, allowing quantitative monitoring of convergence and generalisation through loss curves and final verification error.

#### Setting Parameters
The vector ```size``` defines the temporal window of each sample, while ```NUM_SAMPLES``` controls the length of the training and verification sequences. The MLP architecture is specified by ```units```, determining the number of neurons per layer, and by the choice of activation functions (```relu``` for hidden layers, ```identity``` for the output). The batch size and learning rate (```BATCH_SIZE```, ```alpha```) regulate the stochastic gradient descent dynamics, whereas the number of epochs determines the training duration. These parameters jointly control model capacity, stability of optimisation, and prediction accuracy on the verification set.
#### Running the example
In order to run this example, after following the instructions written in the **Installation** section, in the ```\build ``` folder you have to execute ```/.mlp```

### Boltzmann Machines on MNIST
This program implements a Deep Belief Network composed of stacked Restricted Boltzmann Machines and trains it in an unsupervised manner on binarized MNIST data. The network learns a hierarchy of latent representations through layer–wise Contrastive Divergence, and the learned filters are visualized across depths to illustrate the progressive emergence of structured features. Additional sampling experiments highlight the generative capability of the lowest RBM by reconstructing and synthesizing patterns from random initial states.
### Setting Parameters
The visible and hidden layer sizes define the spatial resolution and abstraction level of each representation stage. The weight initialization scale controls early training stability. Training iterations and mini–batch size determine convergence speed and noise in gradient estimates. The learning rate governs the update magnitude, while the CD–k parameter sets the trade–off between computational cost and accuracy of the divergence approximation. The number of kernels visualized per layer reflects the desired qualitative inspection depth, and the sampling sparsity and temperature regulate diversity and smoothness in generated patterns.
#### Running the example
In order to run this example, after following the instructions written in the **Installation** section, in the ```\build ``` folder you have to execute ```/.boltzmann```

## Externals
This project uses the public domain header libraries [stb_image.h](https://github.com/nothings/stb) and 
[stb_image_write.h](https://github.com/nothings/stb) to read and write images. 
Additionally the tracked history of states for certain networks can be exported to gifs 
through a light wrapper to the [gif.h](https://github.com/charlietangora/gif-h) header-only
public domain library gif-h. No ownership of the external headers is implied. Additionally we make use of the
[gnuplot-iostream.h](https://github.com/dstahlke/gnuplot-iostream) interface to access the gnuplot subroutines
for plotting.

## References

<a id="1">[1]</a> 
Ramsauer H., Schafl B. (2021). 
Hopfield networks is all you need. 

<a id="1">[2]</a> 
Krotov D., Hopfield J. (2016). 
Dense Associative Memory for Pattern Recognition
30th Conference on Neural Information Processing Systems (NIPS 2016)

<a id="1">[3]</a> 
Bernhard Mehlig (2021). 
Machine learning with neural networks, (Ch. 2-4)
Department of Physics, University of Gothenburg

<a id="1">[4]</a> 
Santos S., Niculae V., McNamee D., Martins A.F.T. (2024). 
Sparse and Structured Hopfield Networks

<a id="1">[5]</a> 
Kanter I., Sompolinsky H. (1987). 
Associative recall of memory without errors
Department of Physics, Bar-Ilan University, Israel

<a id="1">[6]</a> 
Ratschek H., Rokne J. (1999). 
Exact computation of the sign of a finite sum
Applied Mathematics and Computation 99 (1999) 99-127

