from keras.models import load_model
from keras.models import model_from_json
from PIL import Image
from PIL import Image
import numpy as np
import os
import cv2

#loading models
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

#Creating function to convert input img to np array
def convert(img):
    img1 = cv2.imread(img)
    img = Image.fromarray(img1, 'RGB')
    image = img.resize((50, 50))
    return (np.array(image))

#Creating function to analyze cell label and return Paractized or Uninfected
def cell_name(label):
    if label == 0:
        return ("Paracitized")
    if label == 1:
        return ("Uninfected")

#Creating function to perform all analysis on input image, including running it through the other functions
def predict(file):
    print ("Evaluating...please wait")
    ar = convert(file)
    ar = ar/255
    label = 1
    a = []
    a.append(ar)
    a = np.array(a)
    score = loaded_model.predict(a, verbose=1)
    print (score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    Cell=cell_name(label_index)
    return ("The predicted cell is "+Cell.lower()+" with accuracy = "+str(acc))

#Testing the model on an image in the "test" directory
#Model returned the correct answer (Uninfected) with an probability of 99% even though multiple cells were present in the image
print(predict("test/test1.jpeg"))