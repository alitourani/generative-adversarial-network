# Generative Adversarial Network (GAN) for Retrieving Data

This repository contains a system that 

## Prerequisites
### 1) Local machine implementation
If you want to run the code locally, you need to install below **Python 3.3+** packages to run the project. I have been using Anaconda for this project, which included Python 3.7.4 as default.
- Keras v.2.3
	- keras-base
	- keras-gpu
	- keras-preprocessing
- OpenCV v.4.2
	- opencv
	py-opencv
	libopencv
- Numpy v.1.18
- CUDA Toolkit v.10.1

**Note:** since the training process is computationally intensive, I highly recommend to use GPU-enabled version of TensorFlow/Keras. You can easily check if you are using GPU by running the below command:

    # Solution 1:
    from keras import backend as K
    K.tensorflow_backend._get_available_gpus()
    
    # Solution 2:
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

### 2) Using Google Colaboratory
The other way to run the project is using Google Colaboratory which I highly recommend. [This link](https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c "This link") provides some simple steps to get started with Google colab.

## Usage
The project uses **the cars Dataset** by Stanford University for training and testing purposes. The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe. Get access to the mentioned dataset images and DevKit through this [link](https://ai.stanford.edu/~jkrause/cars/car_dataset.html "link").
Assuming that you are using Google Colab, you can simply use the below command to dowload dataset into your Google Drive and unzip it:
    # Download the dataset into the desired directory
    !wget -P "/content/drive/My Drive/Sample/Directory/" http://imagenet.stanford.edu/internal/car196/cars_train.tgz

## References
 [1] J. Krause, M. Stark, J. Deng, and L. Fei-Fei, **3D Object Representations for Fine-Grained Categorization**, *4th IEEE Workshop on 3D Representation and Recognition*, at ICCV 2013 (3dRR-13). Sydney, Australia, 2013.
