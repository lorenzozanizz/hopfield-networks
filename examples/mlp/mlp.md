### Multilayer Perceptron Module

This example demonstrates training a MultiLayerPerceptron (MLP) for time series prediction on sine wave data, using autograd for automatic differentiation and optimization.

Key steps include: 
- Defining an MSE loss function as the squared L2 norm between predictions and ground truth.
- Loading 2000 samples of sine time series data for training and another 2000 for verification, with an output sequence size of 100. 
- Visualizing an example of the input sine function and corresponding ground truth. 
- Initializing an MLP with layer sizes {30, 100, 100, 100} and activations {ReLU, ReLU, Identity}. 
- Setting up a NetworkTrainer with the loss function, enabling loss logging for training and verification. 
- Computing and printing initial verification loss. 
- Training for 50 epochs with batch size 5 and learning rate 0.005, using the training and verification datasets. 
- Printing final verification loss and plotting the training loss curve.

The example integrates components from `math/`, `io/`, `datasets/`, and `reservoir/` directories, illustrating neural network training for sequential data prediction tasks and the use the the small autograd component to compute L2 loss.