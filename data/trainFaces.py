import cv2
import os
import numpy as np

def trainFaces(dataset):
  faces, labels = [], []

  print("Training faces. This might take a while.")

  for person in os.listdir(dataset):
    person_dir = os.path.join(dataset, person)

    if os.path.isdir(person_dir):
      for filename in os.listdir(person_dir):
        if filename.endswith(".jpg"):
          image_path = os.path.join(person_dir, filename)
          image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

          label = int(ord(person[0]))

          faces.append(image)
          labels.append(label)


  faces = np.array(faces)
  labels = np.array(labels)

  recognizer = cv2.face.LBPHFaceRecognizer.create()

  recognizer.train(faces, labels)

  recognizer.save("data\\face_recognizer.yml")

trainFaces("data\\datasets")