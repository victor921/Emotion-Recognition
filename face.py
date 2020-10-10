import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
import json
from keras.models import load_model
import os
from stopwatch import Stopwatch

# Get a reference to webcam 
video_capture = cv2.VideoCapture(0)

# Initialize variables
face_locations = []

# Emotions
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
model = load_model("model_v6_23.hdf5")

n = 0
stopwatch = Stopwatch()
while True:
    print(str(stopwatch))
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) > 0:
        print("Found Face!")
        cv2.imwrite('Reactions/reaction%d.jpg' % n, rgb_frame)
        face_image_full = cv2.imread('Reactions/reaction%d.jpg' % n)

        top, right, bottom, left = face_locations[0]
        face_image1 = face_image_full[top:bottom, left:right]
        Image.fromarray(face_image1).save("Reactions/reaction%dCut.jpg" % n)

        face_image = cv2.imread('Reactions/reaction%dCut.jpg' % n)
        face_image = cv2.resize(face_image, (48,48))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

        prediction = np.argmax(model.predict(face_image))
        label_map = dict((v,k) for k,v in emotion_dict.items()) 
        predicted_label = label_map[prediction]
        print(predicted_label)

        found_object = {
            "frameNum": n,
            "timeFound":str(stopwatch),
            "emotion":predicted_label
        }

        json_object = json.dumps(found_object, indent = 4) 

        with open("Reactions/emotions.json", "a") as outfile: 
            outfile.write(json_object + ',\n') 

        os.remove('Reactions/reaction%d.jpg' % n)
        os.remove('Reactions/reaction%dCut.jpg' % n)
    n += 1

    # Display the results
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()