import cv2
import os

FACE_NAME = "Daniel"
FACE_PROFILE_PATH = "data\\videos\\Daniel_Face_Profile.mp4"

video_capture = cv2.VideoCapture(FACE_PROFILE_PATH)

success, image = video_capture.read()
count = 0

if not (os.path.exists(f'data\\datasets\\{FACE_NAME}')):
  os.mkdir(f'data\\datasets\\{FACE_NAME}')

while success:
  cv2.imwrite(f'data\\datasets\\{FACE_NAME}\\{FACE_NAME}_%d.jpg' % count, image)
  success, image = video_capture.read()

  print('Read a new frame: ', success)

  count += 1