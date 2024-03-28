import cv2
import numpy as np
import os
import time

# Load object detection model
classNames = []
classFile = "/home/sirena/flaskapp/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/sirena/flaskapp/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/sirena/flaskapp/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize face recognition parameters
size = 4
classifier = '/home/sirena/flaskapp/haarcascade_frontalface_default.xml'
image_dir = 'images'
(im_width, im_height) = (120, 102)
(images, labels, names, id) = ([], [], {}, 0)

# Load training data into model
for (subdirs, dirs, files) in os.walk(image_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(image_dir, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            images.append(cv2.imread(path, 0))
            labels.append(id)
        id += 1

(images, labels) = [np.array(lis) for lis in [images, labels]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
haar_cascade = cv2.CascadeClassifier(classifier)

# GStreamer pipeline from camtest.py
gst_pipeline = (
    'libcamerasrc ! '
    'videoconvert ! '
    'videoscale ! '
    'video/x-raw,format=BGR,width=640,height=480 ! '
    'appsink drop=true sync=false'
)

# Open the camera using the GStreamer pipeline
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame from camera. Check camera index.")
        break

    # Desired classes to detect
    desired_classes = ['person', 'car', 'cat', 'dog', 'truck', 'bicycle']

    # Object detection
    classIds, confs, bbox = net.detect(frame, confThreshold=0.45, nmsThreshold=0.2)
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in desired_classes:  # Check if the class is one of the desired classes
                label = f'{className.upper()} {int(confidence * 100)}%'
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (box[0], box[1] - text_size[1] - 10),
                              (box[0] + text_size[0], box[1]), (0, 255, 255), -1)
                cv2.putText(frame, label, (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Face recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
    faces = haar_cascade.detectMultiScale(mini)
    for (x, y, w, h) in faces:
        x, y, w, h = [v * size for v in (x, y, w, h)]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        prediction = model.predict(face_resize)
        label_text = f'{names[prediction[0]]} - {int(prediction[1])}' if prediction[1] < 95 else 'Unknown'
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y - text_size[1] - 20),
                      (x + text_size[0], y), (0, 255, 255), -1)
        cv2.putText(frame, label_text, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Display the result
    cv2.imshow('Object and Face Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
