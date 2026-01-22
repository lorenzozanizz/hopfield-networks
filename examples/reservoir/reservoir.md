### Reservoir Computing Example

This example demonstrates the application of reservoir computing for nonlinear embedding of ECG time series data from the MIT-BIH Arrhythmia Dataset, followed by training a MultiLayerPerceptron (MLP) classifier for beat classification.

Key steps include:
- Initializing a Reservoir with input size 25 and reservoir size 625, setting echo weights (uniform sampling, spectral radius 0.5) and input weights (normal sampling).
- Attaching a ReservoirLogger to collect state norms and visualize states as a 25x25 GIF, and testing with a sinusoidal input over 20 iterations.
- Loading 15000 training and 3000 verification ECG samples from "mitbih_combined_records.csv".
- Mapping the datasets to nonlinear reservoir embeddings (batch size 1000, sequence length 187) using Eigen parallelism (5 threads) and timing the process.
- Setting up an MLP with layer sizes {625, 2000, 1000, 5} and activations {ReLU, ReLU, Identity}.
- Defining Softmax Cross-Entropy loss for 5-class classification (Normal beat, Supraventricular ectopic, Ventricular ectopic, Fusion, Unknown).
- Training the MLP for 80 epochs with batch size 250 and learning rate 0.001, logging training/verification losses, plotting the loss curve, and reporting initial/final verification losses.

The example integrates components from `reservoir/`, `math/`, `io/`, `datasets/`, `utils/`, and `io/plot/` directories, showcasing chaotic nonlinear representations for sequence classification in signal processing.