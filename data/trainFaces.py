import os
import pandas as pd
import numpy as np
import pickle
import keras

import matplotlib.pyplot as plt

from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from keras.models import Model
from keras.optimizers import Adam

from keras_vggface.vggface import VGGFace

train_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input, 
  rotation_range=20, 
  width_shift_range=0.2, 
  height_shift_range=0.2, 
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode="nearest"
)
train_generator = train_datagen.flow_from_directory("data\\datasets", target_size=(244, 244), color_mode="rgb", batch_size=32, class_mode="categorical", shuffle=True)

train_generator.class_indices.values()
NO_CLASSES = len(train_generator.class_indices.values())

base_model = VGGFace(include_top=False, model="vgg16", input_shape=(244, 244, 3))
base_model.summary()

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation="relu")(x)
x = Dense(1024, activation="relu")(x)
x = Dense(512, activation="relu")(x)

preds = Dense(NO_CLASSES, activation="softmax")(x)

model = Model(inputs = base_model.input, outputs = preds)
model.summary()

for layer in model.layers[:19]:
  layer.trainable = False

for layer in model.layers[19:]:
  layer.trainable = True

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_generator, batch_size=1, verbose=1, epochs=20)

model.save("data\\trained_face_cnn_model.h5")

class_dict = train_generator.class_indices
class_dict = {
  value: key for key, value in class_dict.items()
}

face_label_file = "data\\faceLabels.pickle"
with open(face_label_file, "wb") as f: pickle.dump(class_dict, f)