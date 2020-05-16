# Generative Adversarial Network (GAN) for Retrieving Data

This repository contains a system that 

## Prerequisites
You need to install below **Python 3.3+** packages to run the project. I have been using Anaconda for this project, which included Python 3.7.4 as default.
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

## Usage
The project uses **the cars dataset** by Stanford University for training and testing purposes.
