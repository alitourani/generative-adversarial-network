import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.layers import Activation,BatchNormalization,Dense,Flatten,Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
from numpy import asarray
from keras.preprocessing.image import load_img, img_to_array
import glob

inputImages = "/content/drive/Dataset/cars_train/"
outputImages = "/content/drive/Output-New/"
images = glob.glob(inputImages + "*.jpg")

imageNameCounter = 0
alpha_value = 0.001
Xtrain = []
imageRows = 128
imageColumns = 128
channels = 3
counter = 0
imageShape = (imageRows, imageColumns, channels)
zdim=128

# Generating Overall Architecture of GAN - Generator
def generator(zdim):
    model = Sequential()
    model.add(Dense(512*int(imageRows/4)*int(imageRows/4),input_dim=zdim))
    model.add(Reshape((int(imageRows/4),int(imageRows/4),512)))
    model.add(Conv2DTranspose(256,kernel_size=3,strides=2,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha_value))
    model.add(Conv2DTranspose(128,kernel_size=3,strides=1,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha_value))   
    model.add(Conv2DTranspose(channels,kernel_size=3,strides=2,padding='same'))
    model.add(Activation('tanh'))
    return model

# Generating Overall Architecture of GAN - Discriminator
def discriminator(imageShape):
    model=Sequential()
    model.add(Conv2D(64,kernel_size=3,strides=2,input_shape=imageShape,padding='same'))
    model.add(LeakyReLU(alpha=alpha_value))    
    model.add(Conv2D(128,kernel_size=3,strides=2,input_shape=imageShape,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha_value))
    model.add(Conv2D(256,kernel_size=3,strides=2,input_shape=imageShape,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha_value))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    return model

# Adverserial Network
def buildGAN(gen,dis):
    model = Sequential()
    model.add(gen)
    model.add(dis)
    return model

dis_v = discriminator(imageShape)
dis_v.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

gen_v = generator(zdim)
dis_v.trainable=False
gan_v = buildGAN(gen_v,dis_v)
gan_v.compile(loss='binary_crossentropy',optimizer=Adam())

losses=[]
accuracies=[]
iteration_checks=[]

def train(iterations,batch_size,interval):
    global Xtrain
    global counter
    global imageNameCounter
    i=0
    for img in images:
        image = load_img(img)
        image = img_to_array(image)
        image = cv2.resize(image, (imageRows, imageColumns))
        image = image / np.max(image)
        image = image.reshape(imageRows, imageColumns, channels)
        image = np.expand_dims(np.array(image, dtype=float), axis=0)
        if i == 0:
            Xtrain = image
            i = i + 1
        else:
            Xtrain = np.append(Xtrain, image, axis=0)
    number_data = Xtrain.shape[0]
    real = np.ones((batch_size,1))
    fake = np.zeros((batch_size, 1))

    for iteration in range(iterations):
        if counter>=number_data-batch_size:
            counter=0
        imgs = Xtrain[counter:counter+batch_size]
        counter = counter + 1
        z=np.random.normal(0,1,(batch_size,zdim))
        gen_imgs = gen_v.predict(z)
        dloss_real = dis_v.train_on_batch(imgs,real)
        dloss_fake = dis_v.train_on_batch(gen_imgs, fake)
        dloss,accuracy = 0.5 * np.add(dloss_real,dloss_fake)
        z = np.random.normal(0, 1, (batch_size, zdim))
        gloss = gan_v.train_on_batch(z,real)

        if (iteration+1) % interval == 0:
            losses.append((dloss,gloss))
            accuracies.append(100.0*accuracy)
            iteration_checks.append(iteration+1)
            print("%d [D loss: %f , acc: %.2f] [G loss: %f]" %
                  (iteration+1,dloss,100.0*accuracy,gloss))
            z = np.random.normal(0, 1, (16, zdim))
            gen_imgs = gen_v.predict(z)
            gen_imgs = (0.5 * gen_imgs + 0.5)*255
            for gi in range(16):
                imageNameCounter = imageNameCounter + 1
                cv2.imwrite(outputImages + str(imageNameCounter)+".jpg",gen_imgs[gi,:,:,:])

train(20000,20,1000)