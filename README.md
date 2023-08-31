# 2 Quick 2 Drawious - Web-based Doodle Prediction

## Overview

This is a web-based application that simulates Google's Quick Draw! game. Users can doodle in a canvas area, and the application will predict what the doodle represents in real-time using a pre-trained ONNX machine learning model (using PyTorch and the Python code in this repo). The application also features a countdown timer and randomly selects a label for the user to draw, adding an interactive challenge.

## Technologies Used

- HTML5 (for markup)
- JavaScript (for client-side logic)
- ONNX (Open Neural Network Exchange for the ML model)
- CSS3 (for styling)

## Features

- **Canvas for Drawing**: Provides a designated canvas area for doodling.
- **Real-time Prediction**: Predicts what the user is drawing in real-time and displays the predicted label.
- **Random Labeling**: Generates a random label for the user to draw within a set time.
- **Timer Countdown**: Displays a countdown timer, setting a time limit for drawing each label.

## Prerequisites

- A modern web browser that supports HTML5, CSS3, and JavaScript.

## Installation & Setup

1. Clone the repository to your local machine - e.g.

    ```bash
    git clone https://github.com/simpetre/quick-draw.git
    ```

2. Navigate to the project directory.

3. Open `index.html` in your web browser.

## How to Use

1. The application sets a timer and provides a random label for you to draw.
2. Start doodling in the canvas area.
3. The application continuously predicts and displays what it believes you are drawing.
4. When the timer runs out, a new random label is generated for you to draw.

## Custom Styling

The application uses CSS3 for custom styling. Refer to the `styles.css` file for more information.

## Neural Network Architecture

### ConvNet Model

The layer-wise breakdown of the model used in this project follows:
The architecture can be summarized as follows:

- Input: [28x28] Grayscale Image
- Conv1: 32 filters, [3x3] kernel, stride 1, padding 1 -> ReLU -> Max Pooling [2x2]
- Conv2: 64 filters, [3x3] kernel, stride 1, padding 1 -> ReLU -> Max Pooling [2x2]
- Flatten
- FC1: 128 output units -> ReLU
- FC2: 345 output units

This architecture provides a good balance between computational efficiency and model effectiveness for the QuickDraw dataset.

#### Input
- The input to the network is a grayscale image with dimensions [28x28].

#### Convolutional Layers
- **Conv1**: The first convolutional layer has 32 filters of kernel size [3x3], with stride 1 and padding 1.
  - Activation Function: ReLU (Rectified Linear Unit)
  - Max-Pooling: [2x2] window with stride 2
  
- **Conv2**: The second convolutional layer has 64 filters of kernel size [3x3], with stride 1 and padding 1.
  - Activation Function: ReLU (Rectified Linear Unit)
  - Max-Pooling: [2x2] window with stride 2
  
#### Fully Connected Layers
- After the convolutional layers, the tensor is flattened to prepare it for fully connected layers.
- **FC1**: The first fully connected layer has 128 output units.
  - Activation Function: ReLU (Rectified Linear Unit)
  
- **FC2**: The second fully connected layer has 345 output units, corresponding to the 345 classes of the QuickDraw dataset.

#### Output
- The output is a 345-dimensional tensor where the value at each index i represents the model's confidence that the input image belongs to class i.

### Activation Functions
- ReLU (Rectified Linear Unit) is used as the activation function in all layers except the output layer.

### Loss Function
- Cross Entropy Loss is used as the loss function to train the model.

### Optimizer
- Stochastic Gradient Descent (SGD) with a learning rate of 0.001 and momentum of 0.9.

## Scripts

The JavaScript code handles client-side logic, including event handling and real-time doodle prediction. See the `main.js` file for details.

## Contributions

Feel free to contribute to this project by submitting a Pull Request.
