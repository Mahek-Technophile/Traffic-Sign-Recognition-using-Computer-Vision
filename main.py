#Importing Libraries
#Importing Libraries
import numpy as np

import matplotlib.pyplot as plt         
#Matplotlib is a cross-platform, data visualization and graphical plotting library it offers a viable open source alternative to MATLAB.
#matplotlib.pyplot is a module in matplotlib used for plotting graphs and visualizations. It provides functions for creating various plots
#like line plots, scatter plots, histograms, etc.
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


path = "Dataset"
labelFile = 'label.csv'
batch_size_val=32
epochs_val =10
imageDimensions =(32,32,3)
testRatio = 0.2
validationRatio = 0.2

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


path = "Dataset"
labelFile = 'label.csv'


count = 0
images = []
classNo = []
myList = os.listdir(path)

# Get total number of classes (subdirectories in the dataset)
print("Total Classes Detected:", len(myList))
noOfClasses = 43
print("Importing Classes.....")

# Image size (adjust as needed)
image_size = (32, 32)  # Resize to 32x32 pixels

#  from 0 to 42
for count in range(0, 43):
    folder_path = os.path.join(path, str(count))
    

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist, skipping...")
        continue
    
    myPicList = os.listdir(folder_path)
    
    # Loop through each image in the folder
    for y in myPicList:
        curImg = cv2.imread(os.path.join(folder_path, y))
        
     
        if curImg is None:
            print(f"Failed to load image: {y}, skipping...")
            continue
        
    
        curImg = cv2.resize(curImg, image_size)
        
       
        images.append(curImg)
        classNo.append(count)
    
    print(count, end=" ")

print(" ")

# Convert images and classNo to NumPy arrays
images = np.array(images)
classNo = np.array(classNo)

# Split the data into training, testing, and validation sets
testRatio = 0.2
validationRatio = 0.2

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

# Print the shapes of the datasets
print("Data Shapes")
print("Train", X_train.shape, y_train.shape)
print("Validation", X_validation.shape, y_validation.shape)
print("Test", X_test.shape, y_test.shape)

# Load the label file 
labelFile = 'labels.csv'
data = pd.read_csv(labelFile)
print("Data shape:", data.shape, "Type:", type(data))

# Preprocessing functions for image normalization
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)  # Convert to grayscale
    img = equalize(img)    # Equalize the histogram
    img = img / 255        # Normalize pixel values
    return img

# Apply preprocessing to the images
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# Reshape the data for CNN input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print("Preprocessing complete!")

dataGen= ImageDataGenerator(width_shift_range=0.1,   
                            height_shift_range=0.1,
                            zoom_range=0.2,  
                            shear_range=0.1,  
                            rotation_range=10)  
dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=20)
X_batch,y_batch = next(batches)
 

y_train = to_categorical(y_train,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
print('done')


def myModel():
    model= Sequential()
    model.add((Conv2D(60,(5,5),input_shape=(imageDimensions[0],imageDimensions[1],1),activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(60, (5,5), activation='relu')))
    model.add(MaxPooling2D(pool_size=(2,2)))
 
    model.add((Conv2D(30, (3,3),activation='relu')))
    model.add((Conv2D(30, (3,3), activation='relu')))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
 
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax')) 
    model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
   
     
    return model
    

model = myModel()
print(model.summary())
history=model.fit(dataGen.flow(X_train,y_train,batch_size=32),steps_per_epoch=len(X_train)//32,epochs=epochs_val,validation_data=(X_validation,y_validation),shuffle=True)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score =model.evaluate(X_test,y_test,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])
 
model.save("model.keras")
print('done')
