import cv2

recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("data\\face_recognizer.yml")

face_classifier = cv2.CascadeClassifier(
  cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

live_video = cv2.VideoCapture(0)

def detectFace(frame):
  gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
  return faces

def recognizeFace(frame):
  faces = detectFace(frame)

  for (x, y, w, h) in faces:
    face_roi = frame[y:y + h, x:x + w]
    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    label, confidence = recognizer.predict(face_gray)

    name = chr(label) if confidence > 80 else "Unkown Person"
    text = f'{name} - {confidence: .2f}%'

    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)


while True:
  ret, frame = live_video.read()

  recognizeFace(frame)

  cv2.imshow('Facial Recognition', frame)

  ESCAPE_KEY = 27
  if cv2.waitKey(1) == ESCAPE_KEY:
    break

live_video.release()
cv2.destroyAllWindows()