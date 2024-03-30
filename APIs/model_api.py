from flask import Flask, request
import cv2
import numpy as np
import os

app = Flask(__name__)

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

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    # Draw rectangle around the object
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    label = f'{className.upper()} {int(confidence * 100)}%'
                    # Calculate size of label and rectangle positions
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_x = box[0] + 10
                    text_y = box[1] + 30
                    # Draw filled rectangle behind text for visibility
                    cv2.rectangle(img, (box[0], box[1] - text_size[1] - 10), 
                                  (box[0] + text_size[0], box[1]), (0, 255, 255), -1)
                    # Put label text on the image
                    cv2.putText(img, label, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img, objectInfo

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

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get the image data from the request
    image_data = request.data

    # Convert the image data to a numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run object detection and face recognition on the frame
    frame, objectInfo = getObjects(frame, 0.45, 0.2, objects=['person', 'car', 'motorcycle', 'bus', 'truck', 'bicycle', 'cat', 'dog'])

    # Detect faces
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_frame)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (im_width, im_height))

        prediction = model.predict(face_resize)
        label_text = '%s - %.0f' % (names[prediction[0]], prediction[1]) if prediction[1] < 95 else 'Unknown'
        
        # Draw the rectangles and text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness=2)[0]
        text_x = x + 5
        text_y = y - 5
        cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), (0, 255, 255), -1)
        cv2.putText(frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Encode the processed frame
    ret, buffer = cv2.imencode('.jpg', frame)
    processed_frame = buffer.tobytes()

    # Return the processed frame
    return processed_frame

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)