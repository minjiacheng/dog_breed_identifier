# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:18:35 2018

@author: A53445
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications import xception
from keras.applications import inception_v3
import pickle
from time import time
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

#import labels
NUM_CLASSES = 120
labels = pd.read_csv('./data/labels.csv')
selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)

#extract bottleneck
inception_bottleneck = inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
#load model
logreg = pickle.load(open('logreg_model.sav', 'rb'))
# define ResNet50 model for dog detector
ResNet50_model = ResNet50(weights='imagenet')

# pre-process data for dog detector
img_width, img_height = 224, 224
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(img_width, img_height))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
### returns "True" if a dog is detected in the image stored at img_path### ret 
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

# to make new predictions, run code below this line
start = time()

img_path = './images/shih_tzu.jpg' #put image path here 
img = image.load_img(img_path, target_size=(299, 299))
img = image.img_to_array(img)
img_prep = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
imgX = xception_bottleneck.predict(img_prep, batch_size=32, verbose=1)
imgI  = inception_bottleneck.predict(img_prep, batch_size=32, verbose=1)
img_stack = np.hstack([imgX, imgI])
prediction = logreg.predict(img_stack)

#plot image and prediction
fig, ax = plt.subplots(figsize=(5,5))
ax.imshow(img / 255.)
breed = selected_breed_list[int(prediction)]
ax.text(10, 250, 'Prediction: %s' % breed, color='k', backgroundcolor='g', alpha=0.8)
ax.axis('off')
plt.show()

if dog_detector(img_path):
    print('Woof woof!')
    print('You look like a ', breed) 
else:
    print('Hello unidentified creature!')
    print('If you were a dog, you\'d be a ', breed)

end = time()
print("time taken for the prediction is: ", end-start, "s")