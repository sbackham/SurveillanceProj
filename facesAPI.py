from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Initialize face recognition parameters
size = 4  # adjust for speed or accuracy
classifier = 'haarcascade_frontalface_default.xml'
image_dir = 'images'
(im_width, im_height) = (120, 102)
(images, labels, names, id) = ([], [], {}, 0)

# Load training data into the model
for (subdirs, dirs, files) in os.walk(image_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(image_dir, subdir)
        for filename in os.listdir(subjectpath):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.pgm')):
                path = os.path.join(subjectpath, filename)
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
        id += 1

(images, labels) = [np.array(lis) for lis in [images, labels]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
haar_cascade = cv2.CascadeClassifier(classifier)

# Define the GStreamer pipeline
gst_pipeline = (
    'libcamerasrc ! '
    'videoconvert ! '
    'videoscale ! '
    'video/x-raw,format=BGR,width=640,height=480 ! '
    'appsink drop=true sync=false'
)

def gen_frames():  
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
            faces = haar_cascade.detectMultiScale(mini)
            for (x, y, w, h) in faces:
                x, y, w, h = [v * size for v in (x, y, w, h)]
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (im_width, im_height))
                prediction = model.predict(face_resize)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                if prediction[1] < 95:
                    cv2.putText(frame, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api', methods=['GET'])
def api_root():
    return jsonify(message="Welcome to the API")

# Additional API endpoints can be added here for data interactions

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
