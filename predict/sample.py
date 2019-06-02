import predict

print('\n Enter path for the keras weights, leave empty to use "../models/inceptionV3.299x299.h5" \n\
')
weights_path = input().strip()
if not weights_path:
    weights_path = "../models/inceptionV3.299x299.h5"

p = predict.predictor(weights_path)

print('\n Enter the path to the image\n')
image_path = input().strip()

print(p.predict(image_path))
