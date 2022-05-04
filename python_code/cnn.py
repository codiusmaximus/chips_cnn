import sys
sys.path.append('/m100/home/userinternal/bagarwal/.conda/envs/dsdl/lib/python3.8/site-packages')
import numpy as np
import pandas as pd
import seaborn as sns
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import time
import tensorflow.keras.preprocessing as tkp

np.random.seed(1234)
tf.random.set_seed(1234)

# load the data
train_data_path = '/m100_work/cin_staff/bagarwal/chips/prepared_data_extra_images_500/Train'
test_data_path = '/m100_work/cin_staff/bagarwal/chips/prepared_data_extra_images_500/Test'

plot_path = '/m100_work/cin_staff/bagarwal/chips_git'  #### < change this once

# num_classes = len(np.unique(y))
num_classes = 2
reg = tf.keras.regularizers.l2(l2=0.0015)
do = 0.5

batch_size = 64
epochs = 100

###

img_height = 500
img_width = 500
#########################################
# X_train = tkp.image_dataset_from_directory(
#  train_data_path,
#  validation_split=0.15,
#  subset="training",
#  seed=123,
#  image_size=(img_height, img_width),
#  batch_size = batch_size)
#
# X_val = tkp.image_dataset_from_directory(
#  train_data_path,
#  validation_split=0.15,
#  subset="validation",
#  seed=123,
#  image_size=(img_height, img_width),
#  batch_size = batch_size)
#
# train_dataset = X_train.cache()
# val_dataset = X_val.cache()
##########################################

img_gen=tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.15)
train_dataset=img_gen.flow_from_directory(directory= train_data_path,target_size=(img_height,img_width),
          color_mode='rgb',batch_size=batch_size,class_mode='sparse',
          seed=123, shuffle=True, subset='training')
val_dataset=img_gen.flow_from_directory(directory= train_data_path,target_size=(img_height,img_width),
          color_mode='rgb', batch_size=batch_size,class_mode='sparse',
          seed=123, shuffle=True, subset='validation')

##########################################

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3,padding='same', activation='relu', kernel_regularizer=reg,input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=reg),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=reg),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=reg),
    layers.MaxPooling2D(),
    layers.Dropout(do),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=reg),
    layers.Dense(num_classes, activation='softmax')
])

base_lr = 0.0005
opt = keras.optimizers.Adam(learning_rate=base_lr)

model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
model.summary()

callbacks = [keras.callbacks.ModelCheckpoint('./../checkpoints/nogpu-checkpoint-{epoch}.h5', save_best_only=True)]

start_t =time.time()

history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    verbose=2,
    callbacks = callbacks
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(plot_path+'/plots/cnn.png',dpi=150)
plt.close()

end_t = time.time()

print(f'with NO GPU(s) it took {end_t-start_t} secs.')
