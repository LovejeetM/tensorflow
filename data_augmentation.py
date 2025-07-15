import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
import os

# CIFAR-10 dataset for training images
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(x_train[i])
    plt.axis('off')
plt.show()

image = Image.new('RGB', (224, 224), color = (255, 255, 255))

draw = ImageDraw.Draw(image)
draw.rectangle([(50, 50), (174, 174)], fill=(255, 0, 0))

image.save('sample.jpg')

img_path = 'sample.jpg' 
img = load_img(img_path) 
x = img_to_array(img) 
x = np.expand_dims(x, axis=0) 

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(batch[0].astype('uint8'))
    i += 1
    if i % 4 == 0:
        break

plt.show()




datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    samplewise_center=True,
    samplewise_std_normalization=True
)

datagen.fit(x)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(batch[0].astype('uint8'))
    i += 1
    if i % 4 == 0:
        break

plt.show()


def add_random_noise(image):
    noise = np.random.normal(0, 0.1, image.shape)
    return image + noise

# Instance of ImageDataGenerator with the custom augmentation
datagen = ImageDataGenerator(preprocessing_function=add_random_noise)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(batch[0].astype('uint8'))
    i += 1
    if i % 4 == 0:
        break

plt.show()

plt.figure(figsize=(10, 10))
for i, batch in enumerate(datagen.flow(x, batch_size=1)):
    if i >= 4:  
        break
    plt.subplot(2, 2, i+1)
    plt.imshow(batch[0].astype('uint8'))
plt.show()


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

path1 = "sample_images"
images = []

for file in os.listdir(path1):
    if file.lower().endswith("jpg"):
        imgpath = os.path.join(path1, file)
        img = load_img(imgpath)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        i = 0
        for b in datagen.flow(x, batch_size=1):
            plt.figure(i)
            imgplot = plt.imshow(b[0].astype('uint8'))
            i += 1
            if i % 4 == 0:
                break
        plt.show()






def add_random(image):
    dice_throw1 = np.random.randint(1, 7, size = image.shape)
    dice_throw2 = np.random.randint(1, 7, size = image.shape)
    return image + dice_throw1 + dice_throw2

# Instance of ImageDataGenerator with the custom augmentation
datagen = ImageDataGenerator(preprocessing_function=add_random) 

path1 = "sample_images"
images = []

for file in os.listdir(path1):
    if file.lower().endswith("jpg"):
        imgpath = os.path.join(path1, file)
        img = load_img(imgpath)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        i = 0
        for b in datagen.flow(x, batch_size=1):
            plt.figure(i)
            imgplot = plt.imshow(b[0].astype('uint8'))
            i += 1
            if i % 4 == 0:
                break
        plt.show()
        