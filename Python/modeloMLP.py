import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

def load_split(split_path):
    x_list = []
    y_list = []

    for fname in os.listdir(split_path):
        if fname.endswith(".csv"):
            fpath = os.path.join(split_path, fname)

            df = pd.read_csv(fpath)

            pixel_cols = [c for c in df.columns if c.startswith("p")]
            class_cols = [c for c in df.columns if c.startswith("class_")]

            x_list.append(df[pixel_cols].values[0])
            y_list.append(df[class_cols].values[0])

    x = np.array(x_list)
    y = np.array(y_list)

    return x, y

train_path = "/content/drive/MyDrive/Certamen_Softcomputing/dataset_csv_mfcc_tts/train"
val_path = "/content/drive/MyDrive/Certamen_Softcomputing/dataset_csv_mfcc_tts/val"
test_path = "/content/drive/MyDrive/Certamen_Softcomputing/dataset_csv_mfcc_tts/test"

x_train,y_train = load_split(train_path)
x_val,y_val = load_split(val_path)
x_test,y_test = load_split(test_path)

model = models.Sequential([
    layers.InputLayer(input_shape=(256,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.summary()

opt = optimizers.Adam(learning_rate=0.0001)

model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)