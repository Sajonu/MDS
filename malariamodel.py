from PIL import Image
import numpy as np
import os
import cv2
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import numpy as np
from keras.models import load_model
import os

data=[]
labels=[]
#iterating through Parasitized and Uninfected directories to convert to form that can be used in neural network
Parasitized=os.listdir("cell_images/Parasitized/")
for cell in Parasitized:
    try:
        image=cv2.imread("cell_images/Parasitized/"+a)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        data.append(np.array(size_image))
        labels.append(0)
    except AttributeError:
        print("")

Uninfected=os.listdir("cell_images/Uninfected/")
for cell1 in Uninfected:
    try:
        image=cv2.imread("cell_images/Uninfected/"+b)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        data.append(np.array(size_image))
        labels.append(1)
    except AttributeError:
        print("")

Cells=np.array(data)
labels=np.array(labels)

np.save("Cells", Cells)
np.save("labels", labels)
Cells=np.load("Cells.npy")
labels=np.load("labels.npy")

s=np.arange(Cells.shape[0])
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]

numClasses = len(np.unique(labels))

#Splitting training and test data
(x_train,x_test)=Cells[(int)(0.1*len(Cells)):],Cells[:(int)(0.1*len(Cells))]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

lenTrain = len(x_train)
lenTest = len(x_test)

(y_train,y_test)=labels[(int)(0.1*len(Cells)):],labels[:(int)(0.1*len(Cells))]
y_train=keras.utils.to_categorical(y_train,numClasses)
y_test=keras.utils.to_categorical(y_test,numClasses)

#Initizaling model (neural network)
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2,activation="softmax"))
model.summary()

#compiling and training model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.fit(x_train,y_train,batch_size=50,epochs=40,verbose=1)

#testing model and printing accuracy
practical_test = model.evaluate(x_test, y_test, verbose = 1)
print ("Accuracy is " , practical_test[1])

#saving model to json so it can be imported in the future
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")