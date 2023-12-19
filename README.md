# Computer-Vision-Facial-Recognition

### View demo [here](https://temp.com)

## Training Data

You can view all the training data used in the `data/datasets` and `data/videos` folders.

Utilzing several gigabytes of raw video footage, I created a script to process every individual frame with a recognizable face in it, cleaning and filtering for the model-training process. The frame collection script can be found at [`data/collectFaces.py`](https://github.com/danielaglr/Computer-Vision-Facial-Recognition/blob/main/data/collectFaces.py) and the processing script at [`data/processFaces.py`](https://github.com/danielaglr/Computer-Vision-Facial-Recognition/blob/main/data/processFaces.py).

## Usage
1. Add videos of face profiles in the `data/videos` folder and run ```python ./data/collectFaces.py``` to collect frames of visible faces for training
  - In order for the script to work you need to edit the variables `FACE_NAME` and `FACE_PROFILE_PATH` to reflect the name associated with the face in the video and the path to the video file
  - You can also add individual pictures in place of collecting frames from a video but it's much more inefficient
2. Run ```python ./data/processFaces.py``` to clean up any frames without any identifiable faces and resize to ROI around face
3. Run ```python ./data/trainFaces.py``` to train model to recognize faces in `data/datasets/`
4. Finally run ```python ./detectFaces.py``` to test the trained model in a live demo

## Dependencies
View `requirements.txt` to see a list of all packages and their used versions