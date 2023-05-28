import numpy as np
from utils import custom_dataset
from generate_LF import get_variables
from keras.utils import np_utils
import os
from keras.models import load_model
 
#  Variables
classes,label_frac,data_path,save_path = get_variables()
num_classes = 3

# Load and Process Data
data,x,y = custom_dataset(classes=[0,1,7],path=data_path ,fraction=0)
X_test = data["test_images"]
X_test = np.array(X_test).reshape(-1, 28, 28, 3)
X_test = X_test.astype("float32") / 255

y_test = [int(i) for i in data["test_labels"]]
y_test = np_utils.to_categorical(y_test, num_classes)

# Load and evaluate model
skyline_model = load_model('/home/akshit/Desktop/MICCAI/code/cnn/skyline.h5')
baseline_model = load_model('/home/akshit/Desktop/MICCAI/code/cnn/baseline.h5')
sky_score = skyline_model.evaluate(X_test, y_test, verbose = 0 )
base_score = baseline_model.evaluate(X_test, y_test, verbose = 0 )
print("Skyline Test accuracy: ", sky_score[1]*100)
print("Baseline Test accuracy: ", base_score[1]*100)