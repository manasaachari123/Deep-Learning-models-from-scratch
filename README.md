
# Implementing Deep Learning Models from Scratch

### [Jan 2024 - May 2024]

This repository contains implementations of various deep learning models built from scratch without relying on high-level deep learning libraries such as TensorFlow or PyTorch. The project demonstrates fundamental concepts of neural networks and their advanced variants, implemented in Python using only basic libraries like NumPy.

## Models Implemented:
1. **Perceptron Learning Algorithm**  
   - A basic building block of neural networks that learns linear decision boundaries.

2. **Feed-Forward Neural Network (FFNN)**  
   - A multi-layer neural network implemented with backpropagation for learning.

3. **Recurrent Neural Networks (RNN)**  
   - Sequential data processing using recurrent connections to capture temporal dependencies.

4. **Long Short-Term Memory (LSTM)**  
   - An advanced version of RNN, designed to better capture long-term dependencies by mitigating vanishing gradient issues.

5. **Gated Recurrent Unit (GRU)**  
   - A simpler alternative to LSTM, achieving similar results with fewer parameters.

6. **Autoencoder**  
   - An unsupervised learning model for encoding input data into a lower-dimensional representation and reconstructing it back.

7. **Variational Autoencoder (VAE)**  
   - A probabilistic approach to generating new data similar to a given dataset by learning latent space representations.

8. **Convolutional Neural Network (CNN)**  
   - A deep learning model designed for visual data, used for object detection on the CIFAR-10 dataset.

9. **Vision Transformer (ViT)**  
   - A transformer-based architecture for object detection, handling image data as sequences of image patches.

## Dataset:
The models have been trained and tested on the **CIFAR-10** dataset, which consists of 60,000 32x32 color images in 10 classes.

## Installation:
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/deep-learning-from-scratch.git
    ```
2. Navigate to the project directory:
    ```bash
    cd deep-learning-from-scratch
    ```

## Usage:
Each model is located in its own directory, and you can run any model by navigating to its folder and running the respective script.

For example, to run the **CNN model** on the CIFAR-10 dataset:
```bash
Run each of the ipynb file for different models implemented from scratch
```

Similarly, other models can be executed by navigating to their directories and running the associated scripts.

## Contributing:
Feel free to contribute to this project by opening issues or submitting pull requests.
