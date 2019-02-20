from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from net import Model
import numpy as np
#import prepareData
import random
import keras
import h5py
import cv2
data = []
labels = []

weights_period = 10
EPOCHS = 20
INIT_LR = 0.0001
image_size = 96
BS = 32


trainPath = "D:/DeepLearning/histopathologic/Dataset/train/"
testPath = "D:/DeepLearning/histopathologic/Dataset/test/"
validPath = "D:/DeepLearning/histopathologic/Dataset/validation/"

# 
from keras.applications import VGG16
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
for layer in vgg_conv.layers[:]:
    layer.trainable = False
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
from keras import models
from keras import layers
from keras import optimizers
model = models.Sequential()
model.add(vgg_conv)
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

#model = Model.build(image_size,image_size,3,2)
model.summary()
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.05,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")

valAug = ImageDataGenerator(rescale=1 / 255.0)

trainGen = trainAug.flow_from_directory(
	trainPath,
	class_mode="categorical",
	target_size=(image_size, image_size),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

valGen = valAug.flow_from_directory(
	validPath,
	class_mode="categorical",
	target_size=(image_size, image_size),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max',period=weights_period)
earlystop = EarlyStopping(monitor='val_loss', patience=5)
callbacks_list = [checkpoint,earlystop]
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

history = model.fit_generator(
      trainGen,
      steps_per_epoch=trainGen.samples/trainGen.batch_size ,
      epochs=20,
      validation_data=valGen	,
      validation_steps=valGen.samples/valGen.batch_size,
      verbose=1,callbacks=callbacks_list)

model.save('final_model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("plot.png")
plt.show()
