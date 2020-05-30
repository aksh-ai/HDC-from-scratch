import pickle
import cv2
import numpy as np 

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_copy = img.copy()
    img = cv2.resize(img, (28, 28))
    img = img/255.0
    img = img.reshape(-1, 784)
    return img, cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)

image_path = input('Enter the images path: ')

if image_path:
    image, original = preprocess_image(image_path)

    model = pickle.load(open('models/MNIST.dat', 'rb'))
    
    prediction = model.predict(image)

    print(f"Predicted Digit: {prediction[0]}")

    cv2.putText(original, 'Predicted Digit {}'.format(str(prediction[0])), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2) # B, G, R

    cv2.imshow('HDC', original)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

else:
    print('user has not entered a valid path')
