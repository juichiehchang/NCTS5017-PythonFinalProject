import predict
import cv2

weights_path = "../../inceptionV3.299x299.h5"

p = predict.predictor(weights_path)

#print('\n Enter the path to the image\n')
#image_path = input().strip()
#print(p.predict_from_path(image_path))

img = cv2.imread("../../pythonFL/data/test/drawings/0A2FD005-76FF-4C5A-8C81-8179010ED1BB.jpg")
img = cv2.resize(img, (299, 299))

print(p.predict_from_array(img))
