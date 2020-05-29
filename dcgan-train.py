import os
from keras.preprocessing import image
import random

outputImages="/content/drive/My Drive/Output/"
trainList=[]
i=0
start = 0
iterations = 10000
batchSize = 200
counter = 0
numberOfTrainingImages = 1500

# Reducing the number of training samples
trainingImages = random.sample(images, numberOfTrainingImages)
print("Randomly selecting ", len(trainingImages), " images from ", len(images), " samples of dataset ...")

for m in trainingImages:
  counter += 1
  if counter % 100 == 0:
    print("Analyzing image #", counter)
  tempImage = load_img(m)
  tempImage = img_to_array(tempImage)
  tempImage = cv2.resize(tempImage, (height, width))
  tempImage = tempImage / np.max(tempImage)
  tempImage = tempImage.reshape(height, width, 3)
  tempImage = np.expand_dims(np.array(tempImage, dtype=float), axis=0)
  if i == 0:
    trainList = tempImage
    i = i + 1
  else:
    trainList = np.append(trainList, tempImage, axis=0)

counter=0

for step in range(iterations):
  counter += 1
  print("Iteration #", counter)
  latentVectors = np.random.normal(size=(batchSize, dimensions))
  generatedList = generator.predict(latentVectors)
  stop = start + batchSize
  realImages = trainList[start: stop]
  combinedImages = np.concatenate([generatedList, realImages])
  labels = np.concatenate([np.ones((batchSize, 1)), np.zeros((batchSize, 1))])
  labels += 0.05 * np.random.random(labels.shape)
  discriminatorLoss = discriminator.train_on_batch(combinedImages, labels)
  latentVectors = np.random.normal(size=(batchSize, dimensions))
  misleadingTargets = np.zeros((batchSize, 1))
  adversialLoss = gan.train_on_batch(latentVectors, misleadingTargets)
  start += batchSize
  if start > len(trainList) - batchSize:
    start = 0
  if step % 100 == 0:
    gan.save_weights('gan.h5')
    print('Discriminator loss: ', discriminatorLoss)
    print('Adversarial loss:', adversialLoss)
    img = image.array_to_img(generatedList[0] * 255., scale=False)
    img.save(os.path.join(outputImages, 'Generated_' + str(step) + '.png'))
    img = image.array_to_img(realImages[0] * 255., scale=False)
    img.save(os.path.join(outputImages, 'Real_' + str(step) + '.png'))

print("Training process has been finished! :)")