import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras

def load_image(img_path, size):
    # load image from the given path
    image = keras.preprocessing.image.load_img(img_path, target_size = size)
    # convert image into array
    image = keras.preprocessing.image.img_to_array(image)
    # divide by max intensity
    image /= 255
    # add an extra dimension
    return np.expand_dims(image, axis=0)

class predictor():

    model = None
    categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
    size = None
    
    def __init__(self, model_path):
        
        # inveptionV3 default image size
        if('inceptionV3' in model_path):
            self.size = (299, 299)
            
        # denseNet121 default image size
        elif('denseNet121' in model_path):
            self.size = (224, 224)

        # load model from the given path
        self.model = keras.models.load_model(model_path)
       
    def predict_from_path(self, img_path):
        
        # load image and cut it into the given size
        image = load_image(img_path, self.size)
        # get the softmax result predicted by the model
        prediction = self.model.predict(image)
        
        #print(prediction)
        
        # return the category with highest softmax possibility
        return self.categories[np.argmax(prediction)]

    def predict_from_array(self, img):

        # divide by max intensity
        image = img/255
        # get the softmax result predicted by the model
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        # return the category with highed softmax possibility
        return self.categories[np.argmax(prediction)]

if __name__ == '__main__':
    
    print('\n Enter path for the weights file, default is "../models/inceptionV3.299x299.h5" \n')

    # path to the weights file
    weights_path = input().strip()
    
    # load inceptionV3 model as default
    if not weights_path:
        weights_path = "../models/inceptionV3.299x299.h5"

    # initialize predictor
    p = predictor(weights_path)

    print('\n Enter the path to the image\n')
    # path to the image
    image_path = input().strip()

    # print the predicted result
    print(p.predict_from_path(image_path))
    
