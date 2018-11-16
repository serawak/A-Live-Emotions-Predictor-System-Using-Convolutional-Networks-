import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import plotly.plotly as py
import plotly.tools as tls
tls.set_credentials_file(username='agsmilinas', api_key='jZ9bGuDnMF46Eq4UcQKn')
from keras.models import Sequential, Model,load_model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Activation, Dropout, GlobalAveragePooling2D
import imutils
from PIL import Image
from resizeimage import resizeimage
import os
import time

#importing best model
model = load_model("/Users/agsmilinas/Desktop/Proyecto_Ingenieria_Alejandro_Salinas_Humberto_Poblano/CNN/model6.h5")
#LIVE DATA EXTRACT
faceCascade = cv2.CascadeClassifier('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/opencv_python-3.4.3.18.dist-info/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

ar = 0
k = cv2.waitKey(30) & 0xff

while True:
    ret, img = cap.read()
    #img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
        #Save just the rectangle faces in SubRecFaces
        sub_face = img[y:y+h, x:x+w]
        FaceFileName = "/Users/agsmilinas/Desktop/Proyecto_Ingenieria_Alejandro_Salinas_Humberto_Poblano/CNN/"+"la buena " + str(ar) + ".jpg"
        cv2.imwrite(FaceFileName, sub_face)
    # Show the image
        cv2.imshow('ITESM Research by Alejandro Salinas ',   img)
        key = cv2.waitKey(10)
        cv2.imshow('video',img)
        k = cv2.waitKey(30) & 0xff
    if (k == 27): # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
#CSV DATOS
def data_load(emotion):
    lines=[]
    df = pd.read_csv("/Users/agsmilinas/Desktop/Proyecto_Ingenieria_Alejandro_Salinas_Humberto_Poblano/CNN/datos_normalized_CNN.csv")
    lines = df[emotion]
    images=[]
    image_path ="/Users/agsmilinas/Desktop/Proyecto_Ingenieria_Alejandro_Salinas_Humberto_Poblano/CNN/resized_images_2/"
    for i in range(0,5651):
        image=cv2.imread(image_path+str(i)+".jpg")
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#conversion de canales
        images.append(image)
#       print("Appending image: "+str(i))
    return(images,lines)
        

#DATOS_JOY
datos_joy = data_load("Joy Evidence")
data_images, data_labels =[],[]
for image, line in zip(datos_joy[0], datos_joy[1]):
    data_images.append(image)
    data_labels.append(line)
    # loop over the rotation angles again, this time ensuring
    # no part of the image is cut off
    if line < 0.9:
        rotated1 = imutils.rotate(image,30)
        data_images.append(rotated1)
        data_labels.append(line)
        rotated2 = imutils.rotate(image,60)
        data_images.append(rotated2)
        data_labels.append(line)

#DATOS_SADNESS
datos_sad = data_load("Sadness Evidence")
data_images2, data_labels2 =[],[]
for image, line in zip(datos_sad[0], datos_sad[1]):
    data_images2.append(image)
    data_labels2.append(line)
    if line < 0.5:
        rotated1 = imutils.rotate(image,30)
        data_images2.append(rotated1)
        data_labels2.append(line)
    if line < 0.3:
        rotated2 = imutils.rotate(image,60)
        data_images2.append(rotated2)
        data_labels2.append(line)
    if line < 0.2:
        rotated3 = imutils.rotate(image,45)
        data_images2.append(rotated3)
        data_labels2.append(line)
        rotated4 = imutils.rotate(image,15)
        data_images2.append(rotated4)
        data_labels2.append(line)
    if line < 0.15:
        rotated3 = imutils.rotate(image,45)
        data_images2.append(rotated3)
        data_labels2.append(line)
        rotated4 = imutils.rotate(image,15)
        data_images2.append(rotated4)
        data_labels2.append(line)

#DATOS FEAR
datos_fear = data_load("Fear Evidence")
data_images3, data_labels3 =[],[]
for image, line in zip(datos_fear[0], datos_fear[1]):
    data_images3.append(image)
    data_labels3.append(line)
    if line < 0.5:
        rotated1 = imutils.rotate(image,30)
        data_images3.append(rotated1)
        data_labels3.append(line)
    if line < 0.3:
        rotated2 = imutils.rotate(image,60)
        data_images3.append(rotated2)
        data_labels3.append(line)
    if line < 0.2:
        rotated3 = imutils.rotate(image,45)
        data_images3.append(rotated3)
        data_labels3.append(line)
        rotated4 = imutils.rotate(image,15)
        data_images3.append(rotated4)
        data_labels3.append(line)
    if line < 0.15:
        rotated3 = imutils.rotate(image,45)
        data_images3.append(rotated3)
        data_labels3.append(line)
        rotated4 = imutils.rotate(image,15)
        data_images3.append(rotated4)
        data_labels3.append(line)

#DATOS SURPRISE
datos_surprise = data_load("Surprise Evidence")
data_images4, data_labels4 =[],[]
for image, line in zip(datos_surprise[0], datos_surprise[1]):
    data_images4.append(image)
    data_labels4.append(line)
    #loop over the rotation angles again, this time ensuring
    #no part of the image is cut off
    if (line < 0.45 and line > 0.15):
        rotated1 = imutils.rotate(image,30)
        data_images4.append(rotated1)
        data_labels4.append(line)
        rotated2 = imutils.rotate(image,60)
        data_images4.append(rotated2)
        data_labels4.append(line)
    
#DATOS ANGER
datos_anger = data_load("Anger Evidence")
data_images5, data_labels5 =[],[]
for image, line in zip(datos_anger[0], datos_anger[1]):
    data_images5.append(image)
    data_labels5.append(line)
    # loop over the rotation angles again, this time ensuring
    # no part of the image is cut off
    if line < 0.75:
        rotated1 = imutils.rotate(image,30)
        data_images5.append(rotated1)
        data_labels5.append(line)
        rotated2 = imutils.rotate(image,60)
        data_images5.append(rotated2)
        data_labels5.append(line)

    
#Training and Testing Data 
X_train_1=np.array(data_images)
print("Joy total:", X_train_1.shape)
y_train_1=np.array(data_labels)
X_train_joy, X_test_joy , y_train_joy, y_test_joy = train_test_split(X_train_1, y_train_1, test_size=0.2, random_state=4)
print("Joy Data ready")

X_train_2=np.array(data_images2)
print("Sadness total:", X_train_2.shape)
y_train_2=np.array(data_labels2)
X_train_sad, X_test_sad, y_train_sad, y_test_sad = train_test_split(X_train_2, y_train_2, test_size=0.2, random_state=4)
print("Sadness Data ready")

X_train_3=np.array(data_images3)
print("Fear total:", X_train_3.shape)
y_train_3=np.array(data_labels3)
X_train_fear, X_test_fear, y_train_fear, y_test_fear = train_test_split(X_train_3, y_train_3, test_size=0.2, random_state=4)
print("Fear Data ready")

X_train_4=np.array(data_images4)
print("Surprise total:", X_train_4.shape)
y_train_4=np.array(data_labels4)
X_train_surprise, X_test_surprise, y_train_surprise, y_test_surprise = train_test_split(X_train_4, y_train_4, test_size=0.2, random_state=4)
print("Surprise Data ready")

X_train_5=np.array(data_images5)
print("Anger total:", X_train_5.shape)
y_train_5=np.array(data_labels5)
X_train_anger, X_test_anger, y_train_anger, y_test_anger = train_test_split(X_train_5, y_train_5, test_size=0.2, random_state=4)
print("Anger Data ready")

img = Image.open("/Users/agsmilinas/Desktop/Proyecto_Ingenieria_Alejandro_Salinas_Humberto_Poblano/CNN/la buena 0.jpg")
cover = img.resize((100, 120), Image.ANTIALIAS) 
x_prob = np.array(cover)
plt.imshow(img)
plt.show()
print(x_prob.shape)

def CNN_ARQ(X_train, X_test, y_train, y_test,x2):
        #ARQUITECTURA DE LA RED NEURONAL CONVOLUCIONAL
        model=Sequential()
        model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(120,100,3)))
        model.add(Cropping2D(cropping=((70,25),(0,0))))
        model.add(Conv2D(32, (5, 5), activation="relu", strides=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(50, (8, 8), activation="relu", strides=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(58, (8, 8), activation="relu", strides=(2, 2),data_format='channels_first'))
        model.add(Dropout(0.2))
        model.add(Conv2D(74,(5,5),activation="relu"))
        model.add(Dropout(0.3))
        model.add(Conv2D(74,(3,3),activation="relu"))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(128,activation="elu"))
        model.add(Dropout(0.4))
        model.add(Dense(64,activation="elu"))
        model.add(Dropout(0.4))
        model.add(Dense(16,activation="elu"))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam',metrics=['mae'])
        history_object=model.fit(X_train, y_train, shuffle=True,nb_epoch=1,verbose=1,validation_data=(X_test, y_test))
        print('done')
        score = model.evaluate(X_test, y_test, verbose=0) #Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.
        x3 = x2.reshape((-1, 120, 100, 3))
        y_prob = model.predict(x3)
        print("y_prob: ",y_prob)
        return int(y_prob*100)


joy_prediction = CNN_ARQ(X_train_joy, X_test_joy , y_train_joy, y_test_joy,x_prob)
sad_prediction = CNN_ARQ(X_train_sad, X_test_sad, y_train_sad, y_test_sad,x_prob)
fear_prediction = CNN_ARQ(X_train_fear, X_test_fear, y_train_fear, y_test_fear,x_prob )
surprise_prediction = CNN_ARQ(X_train_surprise, X_test_surprise, y_train_surprise, y_test_surprise,x_prob)
anger_prediction = CNN_ARQ(X_train_anger, X_test_anger, y_train_anger, y_test_anger,x_prob)

# Make fake dataset
height = [joy_prediction,sad_prediction,fear_prediction,sad_prediction,anger_prediction]
bars = ('Alegría',"Tristeza","Miedo","Sorpresa","Enojo")
y_pos = np.arange(len(bars))
 
# Create horizontal bars
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'g']
plt.barh(y_pos, height,color = colors)
 
# Create names on the y-axis
plt.yticks(y_pos, bars)
 
# Show graphic
plt.show()
