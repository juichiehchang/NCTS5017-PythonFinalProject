import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras

def load_image(img_path, size):
    image = keras.preprocessing.image.load_img(img_path, target_size = size)
    image = keras.preprocessing.image.img_to_array(image)
    image /= 255
    return np.expand_dims(image, axis=0)

class predictor():

    model = None
    categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
    size = None
    
    def __init__(self, model_path):
        if('inceptionV3' in model_path):
            self.size = (299, 299)
        self.model = keras.models.load_model(model_path)
       
    def predict(self, img_path):
        
        image = load_image(img_path, self.size)
        prediction = self.model.predict(image)
        print(prediction)
        return self.categories[np.argmax(prediction)]

if __name__ == '__main__':
    print('\n Enter path for the keras weights, leave empty to use "../models/inceptionV3.299x299.h5" \n')
    weights_path = input().strip()
    if not weights_path:
        weights_path = "../models/inceptionV3.299x299.h5"

    p = predictor(weights_path)

    print('\n Enter the path to the image\n')
    image_path = input().strip()

    print(p.predict(image_path))
    
