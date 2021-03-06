import os
# Ignore tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session
from keras.optimizers import SGD
from keras.applications import InceptionV3
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from keras import initializers, regularizers

import constants
import callbacks
import generators

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

print("Clearing session")
clear_session()

width = constants.WIDTH
height = constants.HEIGHT

print("Load model from previous run")
model = load_model("../models/denseNet121." + str(width) + "x" + str(height) + ".h5")

# Unlock some layers in denseNet121
for layer in model.layers[:34]:
   layer.trainable = False
for layer in model.layers[34:]:
   layer.trainable = True

print(model.summary())

# Load previous best checkpoint
weights_file = "../models/weights.best_denseNet121." + str(width) + "x" + str(height) + ".hdf5"
if(os.path.exists(weights_file)):
    print("load weight file:", weights_file)
    model.load_weights(weights_file)

print("Get callbacks")
# Get callbacks
cb = callbacks.get_callbacks(weights_file)

print("Compile the model")
# Complie the model
model.compile(loss = "categorical_crossentropy", optimizer = SGD(momentum = 0.9), metrics = ["acc"])

# Get data generator
train_gen, valid_gen = generators.get_generators(width, height)

print("Start fitting")
# execute fitting on main thread
model_output = model.fit_generator(train_gen, steps_per_epoch = constants.STEPS,
                                   epochs = 20, verbose = 1, callbacks = cb,
                                   validation_data = valid_gen,
                                   validation_steps = constants.VALIDATION_STEPS,
                                   workers = 0, use_multiprocessing = True,
                                   shuffle = True, initial_epoch = 0)

# Save the result
print("Save the model")
model.save("../models/denseNet121." + str(width) + "x" + str(height) + ".h5")
