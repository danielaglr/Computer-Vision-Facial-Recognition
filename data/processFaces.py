import cv2
import os
import pickle
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

FACES_DIR = "data\\datasets"

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

face_cascade = cv2.CascadeClassifier("data\\haarcascade_frontalface_default.xml")

label_IDs = {}
current_ID = 0

for root, _, files in os.walk(FACES_DIR):
  for file in files:
    if file.endswith(".jpg"):
      path = os.path.join(root, file)

      label = os.path.basename(root).replace(" ", ".").lower()

      if not label in label_IDs:
        label_IDs[label] = current_ID
        current_ID += 1

      image_test = cv2.imread(path, cv2.IMREAD_COLOR)
      image_array = np.array(image_test, "uint8")
      
      faces = face_cascade.detectMultiScale(image_test, scaleFactor=1.1, minNeighbors=5)

      if len(faces) != 1:
        print(f'Photo Skipped')

        os.remove(path)
        continue

      for (x, y, w, h) in faces:
        face_detect = cv2.rectangle(image_test, (x, y), (x + w, y + h), (255, 0, 255), 2)
        
        plt.imshow(face_detect)

        size = (IMAGE_WIDTH, IMAGE_HEIGHT)
        roi = image_array[x: x + w, y: y + h]

        resized_image = cv2.resize(roi, size)
        image_array = np.array(resized_image, "uint8")

        os.remove(path)

        img = Image.fromarray(image_array)
        img.save(path)
