import keras
import cv2
from keras import layers
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import glob

dimensions = 32
height = 32
width = 32
channels = 3

inputImages = "/content/drive/My Drive/cars_train/"
images = glob.glob(inputImages + "*.jpg")

# Generating Overall Architecture of GAN - Generator
inputGenerator = keras.Input(shape=(dimensions,))
x = layers.Dense(128 * int(height/2) * int(width/2))(inputGenerator)
x = layers.LeakyReLU()(x)
x = layers.Reshape((int(height/2) , int(width/2), 128))(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(inputGenerator, x)
print("Summary of the generator architecture:")
generator.summary()

# Generating Overall Architecture of GAN - Discriminator
inputDiscriminator = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(inputDiscriminator)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)
discriminator = keras.models.Model(inputDiscriminator, x)
print("Summary of the discriminator architecture:")
discriminator.summary()
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# Adverserial Network
discriminator.trainable = False
gan_input = keras.Input(shape=(dimensions,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)
gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
print("Summary of the Adverserial Network:")
gan.summary()