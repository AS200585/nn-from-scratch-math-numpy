# **Implementation of Neural Network from Scratch using Numpy**

A from-scratch implementation of a 2-layer neural network for recognizing handwritten digits using the MNIST dataset following Samson Zhang's version. 

**Overview**

This neural network classifies handwritten digits (0-9) from 28x28 pixel grayscale images. The implementation includes:
- 2-layer fully connected neural network
- ReLU activation function for the hidden layer
- Softmax activation function for the output layer
- Cross-entropy loss with gradient descent optimization
- Training and validation accuracy tracking
- Prediction visualization capabilities

**Architecture**

- Input Layer: 784 neurons (28×28 flattened pixels)
- Hidden Layer: 256 neurons with ReLU activation
- Output Layer: 10 neurons with softmax activation (one for each digit 0-9)

**Key Functions**

Core Neural Network Functions:-
1. init_params(): Initialize weights and biases with small random values
2. forward_prop(): Forward propagation through the network
3. back_prop(): Backpropagation to compute gradients
4. gradient_descent(): Main training loop with parameter updates

Activation Functions:-
1. ReLU(): Rectified Linear Unit activation
2. softmax(): Softmax activation for probability distribution
3. derieve_ReLU(): Derivative of ReLU for backpropagation

Utility Functions:-
1. one_hot(): Convert labels to one-hot encoded format
2. get_predictions(): Extract predicted classes from output probabilities
3. get_accuracy(): Calculate prediction accuracy
4. make_predictions(): Generate predictions for new data
5. test_predictions(): Visualize individual predictions

**Performance**

The model achieves approximately 86.3% accuracy on the validation set after 1,500 training iterations with a learning rate of 0.01.

**Getting Started**

- Clone the repository
- Install dependencies: pip install numpy pandas matplotlib
- Download MNIST dataset and save as train.csv
- Run the training script
- Test predictions on individual samples
