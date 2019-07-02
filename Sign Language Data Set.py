#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('C:\\Analytics\\Deep Learning\\CNN\\signlanguage digit dataset')


# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
import keras
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D,Flatten,Dropout,MaxPooling2D


# In[13]:


X = np.load('X.npy')
y = np.load('Y.npy')
print('The dataset is loaded')


# 1.1 Helper Function:show_model_history

# In[14]:


def show_model_history(modelHistory, model_name):
    history=pd.DataFrame()
    history["Train Loss"]=modelHistory.history['loss']
    history["Validation Loss"]=modelHistory.history['val_loss']
    history["Train Accuracy"]=modelHistory.history['acc']
    history["Validation Accuracy"]=modelHistory.history['val_acc']
    
    fig, axarr=plt.subplots(nrows=2, ncols=1 ,figsize=(12,8))
    axarr[0].set_title("History of Loss in Train and Validation Datasets")
    history[["Train Loss", "Validation Loss"]].plot(ax=axarr[0])
    axarr[1].set_title("History of Accuracy in Train and Validation Datasets")
    history[["Train Accuracy", "Validation Accuracy"]].plot(ax=axarr[1]) 
    plt.suptitle(" Convulutional Model {} Loss and Accuracy in Train and Validation Datasets".format(model_name))
    plt.show()


# 1.2 Helper Function:evaluate_conv_model

# In[15]:


def evaluate_conv_model(model, model_name, X, y, epochs=100,
                        optimizer=optimizers.RMSprop(lr=0.0001), callbacks=None):
    X_conv=X.reshape(X.shape[0], X.shape[1], X.shape[2],1)
    print("[INFO]:X_conv.shape:",X_conv.shape)
    print("[INFO]:Convolutional Model {} created...".format(model_name))
    X_train, X_test, y_train, y_test=train_test_split(X_conv,y,
                                                      stratify=y,
                                                     test_size=0.3,
                                                     random_state=42)
    
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    print("[INFO]:Convolutional Model {} compiled...".format(model_name))
    
    print("[INFO]:Convolutional Model {} training....".format(model_name))
    modelHistory=model.fit(X_train, y_train, 
             validation_data=(X_test, y_test),
             callbacks=callbacks,
             epochs=epochs,
             verbose=0
                          )
    print("[INFO]:Convolutional Model {} trained....".format(model_name))

    scores=model.evaluate(X_test, y_test, verbose=0)
    
    print("[INFO]:Train Accuracy:{:.3f}".format(modelHistory.history['acc'][-1]))
    print("[INFO]:Validation Accuracy:{:.3f}".format(modelHistory.history["val_acc"][-1]))
    
    show_model_history(modelHistory=modelHistory, model_name=model_name)


# 1.3 Helper Function: show_image_classes

# In[16]:


def decode_OneHotEncoding(label):
    label_new=list()
    for target in label:
        label_new.append(np.argmax(target))
    label=np.array(label_new)
    
    return label
def correct_mismatches(label):
    label_map={0:9,1:0, 2:7, 3:6, 4:1, 5:8, 6:4, 7:3, 8:2, 9:5}
    label_new=list()
    for s in label:
        label_new.append(label_map[s])
    label_new=np.array(label_new)
    
    return label_new
    
def show_image_classes(image, label, n=10):
    label=decode_OneHotEncoding(label)
    label=correct_mismatches(label)
    fig, axarr=plt.subplots(nrows=n, ncols=n, figsize=(18, 18))
    axarr=axarr.flatten()
    plt_id=0
    start_index=0
    for sign in range(10):
        sign_indexes=np.where(label==sign)[0]
        for i in range(n):

            image_index=sign_indexes[i]
            axarr[plt_id].imshow(image[image_index], cmap='gray')
            axarr[plt_id].set_xticks([])
            axarr[plt_id].set_yticks([])
            axarr[plt_id].set_title("Sign :{}".format(sign))
            plt_id=plt_id+1
    plt.suptitle("{} Sample for Each Classes".format(n))
    plt.show()


# In[17]:


show_image_classes(image=X, label=y.copy())


# 2. Naive Model

# In[18]:


number_of_pixels = X.shape[1] * X.shape[2]
number_of_classes = y.shape[1]
print("number of pixels:", number_of_pixels)
print("number_of_classes:",number_of_classes)


# 3. Convolutional Model 1

# Conv ==> MaxPooling ==> Dense (relu) ==> Dense (softmax).

# In[19]:


def build_conv_model_1():
    model = Sequential()
    
    model.add(layers.Conv2D(64,kernel_size=(3,3),
                            padding='same',
                            activation='relu',
                            input_shape=(64,64,1)))
    
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(number_of_classes,activation='softmax'))
    
    return model


# In[20]:


model = build_conv_model_1()
evaluate_conv_model(model=model, model_name=1,X=X,y=y)


# It can be seen that model has low training accuracy rate and lower validation accuracy rate.
# Hence training acc is low and validation is overfitted. The zigzag in validation
# layers shows that it has very high variance. It would be better to add a new Convolutional layer to the model.

# 4. Convolutional Model 2 

# Conv ==> MaxPooling ==> Conv ==> MaxPooling ==> Dense (relu) ==> Dense (softmax).
# 

# In[24]:


def build_conv_model_2():
    model = Sequential()
    model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(64,64,1)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64,(3,3),activation='relu',padding='same'))
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(Flatten())
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    return model


# In[25]:


model = build_conv_model_2()
evaluate_conv_model(model=model, model_name=2, X=X, y=y)


# When the above graphs are examined, it can be seen that the model has a high training accuracy rate and a lower validation accuracy rate. This means that the model is overfitted. In addition, although the zigzags in the validation chart are reduced, they still exist. It can be assessed that the variance of validation results is still high.
# 
# In view of the above considerations, it is useful to add a new Conv layer or Dropout layer to avoid overfitted the model. First let's add a new Conv layer.

# 5. Convolutional Model 3

# Conv ==> MaxPooling ==> Conv ==> MaxPooling ==> Conv ==> MaxPooling ==> Dense (relu) ==> Dense (softmax).
# 

# In[26]:


def build_conv_model_3():
    model = Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(64,64,1)))
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    
    return model


# In[27]:


model = build_conv_model_3()
evaluate_conv_model(model=model, model_name=3, X=X, y=y)


# Although the validation accuracy rate has increased, the problem of overfitting of the model still exists. We can assume that adding a new Conv layer is not useful. In addition, although the zigzags in the validation chart are reduced, they still exist. It can be assessed that the variance of validation results is still high.
# 
# Let's try using the Dropout layer, one of the solutions to the problem of overfitting in deep networks

# 6. Convolutional Model 4

# Conv ==> MaxPooling ==> Dropout ==> Conv ==> MaxPooling ==> Dropout ==> Conv ==> MaxPooling ==> Dropout ==> Dense (relu) ==> Dense (softmax).
# 

# In[28]:


def build_conv_model_4():
    model = Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(64,64,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    
    return model


# In[29]:


model = build_conv_model_4()
evaluate_conv_model(model=model, model_name=4,X=X,y=y)


# Although the validation success accuracy has increased, the problem of overfitting of the model still exists.
# 
# Let's try adding a new Conv ==> MaxPool ==> Dropout layer.

# 7. Convolutional Model 5 

# Conv ==> MaxPooling ==> Dropout ==> Conv ==> MaxPooling ==> Dropout ==> Conv ==> MaxPooling ==> Dropout ==> Conv ==> MaxPooling ==> Dropout ==> Dense ( relu) = Dense (softmax).
# 

# In[31]:


def build_conv_model_5():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
       
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
        
    return model


# In[32]:


model=build_conv_model_5()
evaluate_conv_model(model=model, model_name=5, X=X, y=y)


# Overfitting and high variance problems were resolved, but the training and validation performance of the model was very poor. Let's remove the last Conv ==> MaxPool ==> Dropout layer added to Model 4 and try different things.
# 
# We can fine tunne another parameters to improve model performance. It is better to use Dropout layers between full connected layers and perhaps after pooling layers. We can also increase the number of nodes 128 to 256 in full connected layers.

# 8. Convolutional Model 6

# Conv ==> MaxPooling ==> Conv ==> MaxPooling ==> Conv ==> MaxPooling ==> Conv ==> MaxPooling ==> Dense (relu) ==> Dropout ==> Dense (softmax).
# 

# In[33]:


def build_conv_model_6():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
        
    return model


# In[34]:


model=build_conv_model_6()
evaluate_conv_model(model=model, model_name=6, X=X, y=y)


# Beside,overfitting and high variance problems were resolved, the training and validation performance of the model lifted up.
# 
# We can also fine tunne the number of filters in Conv layers. Filters are the feature detectors. Generally fewer filters are used at the input layer and increasingly more filters used at deeper layers.
# 
# Filter size is another parameter we can fine tunne it. The filter size should be as small as possible, but large enough to see features in the input data. It is common to use 3x3 on small images and 5x5 or 7x7 and more on larger image sizes.
# 
# BatchNormalization is another layer can be used in CNN. Although the BatchNormalization layer prolongs the training time of deep networks, it has a positive effect on the results. Let's add the BatchNormalization layer to Model 4 and see the results.

# 9. Convolutional Model 7

# Conv ==> MaxPooling ==> BatchNormalization ==> Conv ==> MaxPooling ==> BatchNormalization ==> Conv ==> MaxPooling ==> BatchNormalization ==> Conv ==> MaxPooling ==> BatchNormalization ==> Dense ( relu) ==> Dropout ==> Dense (softmax).
# 

# In[35]:


def build_conv_model_7():
    model = Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(64,64,1)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
        
    return model


# In[36]:


model=build_conv_model_7()
evaluate_conv_model(model=model, model_name=7, X=X, y=y)


# As we expect BatchNormalization increase the model performans. But there is overfitting problem in the model. To deal with that we will use Dropout layer in Conv blocks.

# 10. Convolutional Model 8 

# Conv ==> MaxPooling ==> BatchNormalization ==> Dropout ==> Conv ==> MaxPooling ==> BatchNormalization ==> Dropout ==> Conv ==> MaxPooling ==> BatchNormalization ==> Dropout ==> Conv = => MaxPooling ==> BatchNormalization ==> Dropout ==> Dense (relu) ==> Dropout ==> Dense (softmax).
# 

# In[37]:


def build_conv_model_8():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
        
    return model


# In[38]:


model=build_conv_model_8()
evaluate_conv_model(model=model, model_name=8, X=X, y=y)


# 11. Lets have a look with optimizers
# 

# In[ ]:


model = build_conv_model_8()
optimizer = optimizers.RMSprop(lr=1e-4) # default optimizer
evaluate_conv_model(model=model,model_name=8,X=X,y=y,optimizer=optimizer,epochs=200)


# In[ ]:


model=build_conv_model_8()
optimizer=optimizers.Adam(lr=0.001)
evaluate_conv_model(model=model, model_name=8, X=X, y=y, optimizer=optimizer, epochs=250)


# In[ ]:


model=build_conv_model_8()
optimizer=optimizers.Adam(lr=0.001)
evaluate_conv_model(model=model, model_name=8, X=X, y=y, optimizer=optimizer, epochs=300)


# In[ ]:


You can run these models using different optimization techniques. As I am running on 

