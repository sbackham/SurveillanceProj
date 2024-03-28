from flask import Flask, Response, render_template
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

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
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

    # Set desired frame width and height
    frame_width = 640
    frame_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while True:
        success, frame = cap.read()
        if not success:
            print("Error opening camera")
            break

        # Resize frame for faster processing
        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # Detect faces on the smaller frame
        faces_small = haar_cascade.detectMultiScale(gray_small)

        for (x, y, w, h) in faces_small:
            # Scale back up face locations since the frame we detected in was scaled to 0.5 size
            x, y, w, h = [v * 2 for v in (x, y, w, h)]
            face = gray_small[y//2:(y+h)//2, x//2:(x+w)//2]
            face_resize = cv2.resize(face, (im_width, im_height))

            prediction = model.predict(face_resize)
            label_text = '%s - %.0f' % (names[prediction[0]], prediction[1]) if prediction[1] < 95 else 'Unknown'
            
            # Draw the rectangles and text as per the style in your standalone script
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness=2)[0]
            text_x = x + 5
            text_y = y - 5
            cv2.rectangle(frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), (0, 255, 255), -1)
            cv2.putText(frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Object detection - Consider doing this less frequently if performance is a concern
        _, objectInfo = getObjects(frame, 0.45, 0.2, objects=['person', 'car', 'motorcycle', 'bus', 'truck', 'bicycle', 'cat', 'dog'])

        # Encoding the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Start Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)

