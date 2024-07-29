# %%
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


# %%
imag= cv2.imread("archive/train/happy/Training_10019449.jpg")

# %%
imag.shape

# %%
plt.imshow(imag)

# %%
datadirectory="archive/train/"
classes=["angry","disgust","fear","happy","neutral","sad","surprise"]


# %%
for category in classes:
    path=os.path.join(datadirectory,category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path, img))
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break

# %%
img_size=224
new_array=cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show()


# %%
trainingData=[]
def create_training_data():
    for category in classes:
        path=os.path.join(datadirectory,category)
        class_num= classes.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path, img))
                new_array=cv2.resize(img_array,(img_size,img_size))
                trainingData.append([new_array,class_num])
                
            except Exception as e:
                pass

# %%
create_training_data()

# %%
print(len(trainingData))

# %%
import random
random.shuffle(trainingData)

# %%
x=[]
y=[]
count=0

for features,lables in trainingData[:10000]:
    x.append(features)
    y.append(lables)
        

x=np.array(x).reshape(-1,img_size,img_size,3) #converting to 4 dimensions

# %%
img_array

# %%
x.shape

# %%
x=x/255.0 #normalising the data so dividing by max value possible in the array

# %%
len(y)

# %%
from tensorflow import keras
from tensorflow.keras import layers

# %%
model=tf.keras.applications.MobileNetV2() #pre trained model


# %%
model.summary()

# %%
base_input=model.layers[0].input

# %%
base_output= model.layers[-2].output

# %%
base_output

# %%
final_output= layers.Dense(128)(base_output)
final_output=layers.Activation('relu')(final_output)
final_output=layers.Dense(64)(final_output)
final_output=layers.Activation('relu')(final_output)
final_output=layers.Dense(7,activation='softmax')(final_output)

# %%
new_model=keras.Model(inputs=base_input, outputs=final_output)

# %%
new_model.summary()

# %%
new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=["accuracy"])

# %%
X=np.array(x[:1000])
Y=np.array(y[:1000])

# %%
new_model.fit(X,Y,epochs=10)

# %%
# X=np.array(x[:1500])
# Y=np.array(y[:1500])
# new_model.fit(X,Y,epochs=10)

# %%
new_model.save('mod_my_model_94p90.h5')

# %%
new_model=tf.keras.models.load_model('mod_my_model_94p90.h5')

# %%
frame=cv2.imread("download.jpeg")

# %%
frame.shape

# %%
plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

# %%
faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# %%
gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# %%
faces= faceCascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in faces:
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=frame[y:y+h,x:x+w]
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    facess= faceCascade.detectMultiScale(roi_gray)
    if(len(facess)==0):
        print("face not detected")
    else:
        for(ex,ey,ew,eh) in facess:
            face_roi= roi_color[ey:ey+eh,ex:ex+ew]

# %%
plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

# %%
plt.imshow(cv2.cvtColor(face_roi,cv2.COLOR_BGR2RGB))

# %%
final_image=cv2.resize(face_roi,(224,224))
final_image= np.expand_dims(final_image,axis=0)
final_image=final_image/255.0

# %%
# resized_image = cv2.resize(face_roi, (224, 224))
predictions=new_model.predict(final_image)
predictions



