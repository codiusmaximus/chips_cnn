import sys

sys.path.append('/m100/home/userinternal/bagarwal/.conda/envs/dsdl/lib/python3.8/site-packages')
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow.keras.preprocessing as tkp
###
import horovod.tensorflow.keras as hvd
import math
import time

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
###

print(f'found {hvd.size()} GPU device(s)')

# load the data
train_data_path = '/m100_work/cin_staff/bagarwal/chips/prepared_data_extra_images_500/Train'
test_data_path = '/m100_work/cin_staff/bagarwal/chips/prepared_data_extra_images_500/Test'

plot_path = '/m100_work/cin_staff/bagarwal/chips_git'  #### < change this once

# some model params
batch_size = 64
base_lr = 0.0005
epochs = 50

# no scaling of batch size and lr necessary here
scaled_lr = base_lr

###

img_height = 500
img_width = 500
split = .15

img_gen=tf.keras.preprocessing.image.ImageDataGenerator(validation_split=split)
train_dataset=img_gen.flow_from_directory(directory= train_data_path,target_size=(img_height,img_width),
          color_mode='rgb',batch_size=batch_size,class_mode='sparse',
          seed=123, shuffle=True, subset='training')
val_dataset=img_gen.flow_from_directory(directory= train_data_path,target_size=(img_height,img_width),
          color_mode='rgb', batch_size=batch_size,class_mode='sparse',
          seed=123, shuffle=True, subset='validation')

total_train_batches = train_dataset.samples//batch_size
total_val_batches = val_dataset.samples//batch_size

print(f'total training batches:', total_train_batches)
print(f'total val batches:', total_val_batches)

###

num_classes = 2
reg = tf.keras.regularizers.l2(l2=0.0015)
do = 0.5

opt = keras.optimizers.Adam(learning_rate=scaled_lr)
# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1, average_aggregated_gradients=True)

model = Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=reg,),
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

model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'], experimental_run_tf_function=False)
model.summary()

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    # hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=10, verbose=1),
]
# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./../checkpoints/checkpoint-{epoch}.h5', save_best_only=True))

if hvd.rank() == 0:
    start_t = time.time()

print('training now >>>>>>>>>>>>>>>>>>')

history = model.fit(
    train_dataset,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_epoch=total_train_batches//hvd.size(), #if you use the data generator
    validation_data=val_dataset,
    validation_steps=total_val_batches//hvd.size(), #if you use the data generator
    verbose=2 if hvd.rank() == 0 else 0,
    callbacks=callbacks,
)

if hvd.rank() == 0:

    end_t = time.time()
    print(f'with {hvd.size()} GPU(s) it took {end_t-start_t} secs. with gen')

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
    plt.title('Training and Validation Loss (gen)')
    plt.savefig(plot_path + '/plots/cnn_horovod_'+str(hvd.size())+'_'+str(time.localtime().tm_hour)+str(time.localtime().tm_min)+str(time.localtime().tm_sec)+'.png',dpi=150)
    plt.close()
