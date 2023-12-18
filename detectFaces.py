import cv2
import pickle
import numpy as np

from keras.models import load_model

IMAGE_WIDTH = 244
IMAGE_HEIGHT = 244

face_cascade = cv2.CascadeClassifier("data\\haarcascade_frontalface_default.xml")

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

model = load_model("data\\trained_face_cnn_model.h5")

face_label_file = "data\\faceLabels.pickle"
with open(face_label_file, "rb") as f: 
  class_dict = pickle.load(f)
  labels = {key: value for key, value in class_dict.items()}
  print(labels)

live_video = cv2.VideoCapture(0)

while True:
  ret, frame = live_video.read()

  faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), scaleFactor=1.3, minNeighbors=5)


  for (x, y, w, h) in faces: 
    roi_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[y: y + h, x: x + w]
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    
    resized_image = cv2.resize(roi_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))      
    
    image_array = np.array(resized_image, "uint8")

    img = image_array.reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, 3).astype("float32") / 255.0
    
    predicted_prob = model.predict(img)

    name = labels[predicted_prob[0].argmax()]

    cv2.putText(frame, f'({name})', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

  cv2.imshow("Image", frame)
  
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    break

live_video.release()
cv2.destroyAllWindows()
