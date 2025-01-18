# Machine Learning Final Project - Botanist

## Introduction
In this project, I developed a classification model to classify leaf images into 38 different classes, representing various plant breeds and potential diseases. We utilized Convolutional Neural Networks (CNNs) because of their effectiveness in image classification problems. PyTorch was chosen over TensorFlow due to its popularity in the industry.

## Model Structure
I designed a CNN model called `LeafCNN`:
- Four convolutional layers with increasing filter sizes (32 → 64 → 128 → 256).
- Each convolutional layer is followed by:
  - A ReLU activation function.
  - A max-pooling layer.
- A flatten layer to convert a 3D tensor into a 1D tensor.
- A fully connected layer with 512 units.
- Two Dropout layers with a dropout probability of 0.5 to prevent overfitting.
- An output layer with 38 units for classification.

<figure>
  <img
  src="/Submission/cnn.png"
  alt="The architecture diagram of LeafCNN">
  <figcaption>Figure 1: The architecture diagram of LeafCNN</figcaption>
</figure>

## Training Flow

### Data Processing
- The dataset included 50,000 images labeled across 38 classes.
- The data was split into:
  - 45,000 training images.
  - 5,000 testing images.
- Data augmentation techniques (flipping and rotating images) were applied to increase diversity.
- Images were converted into tensors for compatibility with the neural network.

### Training Loop
- For each epoch:
  - Gradients were cleared.
  - A forward pass generated predictions.
  - The loss function measured prediction accuracy.
  - A backward pass calculated gradients of the loss.
  - The optimizer updated model parameters.
- The training used:
  - PyTorch’s Adam optimizer with a learning rate of 0.001.
  - Cross-entropy loss function.

## Parameter Optimization

### Number of Convolutional Layers
- Different numbers of layers were tested.
- A model with four layers achieved the best accuracy and lowest cross-entropy loss.

<figure>
  <img
    src="/Submission/diff_conv_layers.png"
    alt="The comparison of different numbers of convolutional layers"
    width="427" height="273">
  <figcaption>Figure 2: The comparison of different numbers of convolutional layers.</figcaption>
</figure>

### Number of Epochs
- Models trained for 30 epochs had the highest accuracy.
- Models trained for 20 epochs had the lowest loss, likely due to overfitting.

<figure>
  <img
    src="/Submission/diff_epoch.png"
    alt="The comparison of different numbers of epochs"
    width="427" height="273">
  <figcaption>Figure 3: The comparison of different numbers of epochs.</figcaption>
</figure>

### Batch Size
- A batch size of 64 resulted in the best accuracy and lowest cross-entropy loss.

<figure>
  <img
    src="/Submission/diff_epoch.png"
    alt="The comparison of different batch sizes"
    width="427" height="273">
  <figcaption>Figure 4: The comparison of different batch sizes.</figcaption>
</figure>

### Data Augmentation
- Surprisingly, models trained without data augmentation had better performance.
- The dataset's inherent diversity and representativeness likely made augmentation unnecessary.

<figure>
  <img
    src="/Submission/diff_data_augment.png"
    alt="The comparison of different combinations of data augmentation."
    width="533" height="273">
<figcaption>Figure 5: The comparison of different combinations of data augmentation.</figcaption><br>
    
## Summary

#### - The best configuration:
  - Four convolutional layers.
  - 30 epochs.
  - Batch size of 64.
  - No data augmentation.

#### - Using the full dataset of 50,000 images, the final model achieved an accuracy of **98.43%**.

#### - These results demonstrate the importance of hyperparameter tuning and iterative experimentation in achieving high-performing models.

