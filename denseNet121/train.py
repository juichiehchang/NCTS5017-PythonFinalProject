import os
# Ignore tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session
from keras.optimizers import SGD
from keras.applications.densenet import DenseNet121
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from keras import regularizers
from keras.initializers import he_normal

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

# Convolution base layer
base_layer = DenseNet121(
    # image net weight
    weights = "imagenet",
    # self defined size
    include_top = False,
    input_shape = (height, width, 3)
    )
    
# Fix the weights in the denseNet layer
base_layer.trainable = False

#print(base_layer.summary())

# DenseNet121 as first layer
denseNet_out = base_layer.output
# Average pooling with 7x7 kernel
pooled = AveragePooling2D(pool_size = (7, 7))(denseNet_out)
# Drop 40%
dropped1 = Dropout(0.4)(pooled)
# Flatten the image
flattened = Flatten()(dropped1)
# First dense layer with l2 regularizer to avoid overfitting
dense1 = Dense(256, activation = "relu", kernel_initializer = he_normal(seed = None),
               kernel_regularizer = regularizers.l2(0.0005))(flattened)
# Drop 50%
dropped2 = Dropout(0.5)(dense1)
# Second dense layer with l2 regularizer to avoid overfitting
dense2 = Dense(128, activation = "relu", kernel_initializer = he_normal(seed = None),
               kernel_regularizer = regularizers.l2(0.0005))(dropped2)
# Drop 50%
dropped3 = Dropout(0.5)(dense2)
# Glorot uniform initializer of softmax output layer
output = Dense(constants.NUM_CLASSES, kernel_initializer = "glorot_uniform",
               activation = "softmax")(dropped3)

# Construct the model
model = Model(inputs = base_layer.input, outputs = output)

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
                                   epochs = 16, verbose = 1, callbacks = cb,
                                   validation_data = valid_gen,
                                   validation_steps = constants.VALIDATION_STEPS,
                                   workers = 0, use_multiprocessing = True,
                                   shuffle = True, initial_epoch = 8)

# Save the result
print("Save the model")
model.save("../models/denseNet121." + str(width) + "x" + str(height) + ".h5")
