from utils import custom_dataset
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras import Input
from generate_LF import get_variables
from keras.utils import np_utils
import tensorflow as tf
import os
import random as rn

def create_cnn(num_classes = 3):
    #disable warning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    
    #Seeding model weights
    SEED = 10
    os.environ['PYTHONHASHSEED']=str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(42)
    rn.seed(SEED)
    
    # Model
    model = Sequential(
        [
            Input(shape=(28, 28, 3)),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model

classes,label_frac,data_path,save_path = get_variables()
data,x,y = custom_dataset(classes=[0,1,7],path=data_path ,fraction=0.01)
x_train = x
x_train = np.array(x_train).reshape(-1, 28, 28, 3)
x_train = x_train.astype("float32") / 255
y_train = [int(i) for i in y]
y_train = np_utils.to_categorical(y_train, len(classes))
batch_size = 128
epochs = 25


model = create_cnn(num_classes = 3)
# model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save('/home/akshit/Desktop/MICCAI/code/cnn/baseline.h5')