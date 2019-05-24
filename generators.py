from keras.preprocessing.image import ImageDataGenerator
import os
import constants

# Image data generator for testing data
train_gen = ImageDataGenerator(
    rotation_range = 30, 
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.25,
    zoom_range = 0.2, 
    rescale = 1.0/255
    )

# Image data generator for validation data
valid_gen = ImageDataGenerator(
    rescale = 1.0/255
    )

# Path to training set and testing set
train_dir = os.path.join(constants.BASE_DIR, "train")
test_dir = os.path.join(constants.BASE_DIR, "test")

# Get the generators for training and validation
def get_generators(width, height):

    training_generator = train_gen.flow_from_directory(
        train_dir,
        target_size = (width, height),
        class_mode = "categorical",
        batch_size = constants.GENERATOR_BATCH_SIZE
        )

    validation_generator = valid_gen.flow_from_directory(
        test_dir,
        target_size = (width, height),
        class_mode = "categorical",
        batch_size = constants.GENERATOR_BATCH_SIZE
        )

    return [training_generator, validation_generator]
        
