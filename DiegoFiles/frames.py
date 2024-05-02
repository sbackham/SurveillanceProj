#this version is for sending metadata in the captions
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore, storage
import werkzeug
import cv2
import numpy as np
import os
import datetime
import json
import uuid
from datetime import datetime, timedelta, timezone

app = Flask(__name__)

# Initialize Firebase Admin
cred = credentials.Certificate('key.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'girl-c0ded.appspot.com'})

db = firestore.client()

# Object Detection Model Setup
classNames = [line.rstrip() for line in open("/home/sirena/flaskapp/coco.names")]
configPath = "/home/sirena/flaskapp/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/sirena/flaskapp/frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Face Recognition Model Setup
size = 4
classifier = '/home/sirena/flaskapp/haarcascade_frontalface_default.xml'
image_dir = 'images'
(im_width, im_height) = (120, 102)
(images, labels, names, id) = ([], [], {}, 0)

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

def getObjects(img, thres, nms, draw=True, objects=['person', 'car', 'motorcycle', 'bus', 'truck', 'bicycle', 'cat', 'dog']):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    label = f'{className.upper()} {int(confidence * 100)}%'
                    cv2.putText(img, label, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img, objectInfo

@app.route('/process_frame', methods=['POST'])
def process_frame():
    def get_local_timestamp():
        # EST is UTC-5
        est = timezone(timedelta(hours=-5))
        return datetime.now(est).isoformat()

    def display_timestamp(iso_str):
        try:
            # Ensure the input is stripped of any unwanted characters like 'Z'
            iso_str = iso_str.rstrip('Z')
            
            # Attempt to parse the ISO format string
            dt = datetime.fromisoformat(iso_str)
            utc_offset = dt.strftime('%z')

            # Handling potential issue with offset not being present
            if not utc_offset:
                logging.error("UTC offset not found in the datetime string.")
                return dt.strftime('%B %d, %Y at %I:%M:%S %p UTC')  # Default to UTC if no offset is found

            # Processing the offset
            hours_offset = int(utc_offset[:3])  # Extracts hours with sign (±hh)
            formatted_offset = f"UTC{hours_offset}" if hours_offset < 0 else f"UTC+{hours_offset}"
            
            # Return the formatted date and time
            return dt.strftime(f'%B %d, %Y at %I:%M:%S %p {formatted_offset}')
        except Exception as e:
            logging.error(f"Failed to format timestamp: {e}")
            return None  # Return None or a default timestamp in case of failure
    if 'frame' in request.files:
        nparr = np.frombuffer(request.files['frame'].read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        _, objectInfo = getObjects(frame, 0.45, 0.2)
        print(f"Object Info: {objectInfo}")  # Debugging log

        recognized_faces = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray_resized = cv2.resize(roi_gray, (im_width, im_height))
            prediction = model.predict(roi_gray_resized)
            if prediction[1] < 98:
                recognized_face_label = names[prediction[0]]
                recognized_faces.append(recognized_face_label)
                print(f"Recognized face: {recognized_face_label}")

        detection_status = "detected" if objectInfo or recognized_faces else "not_detected"
        print(f"Detection Status: {detection_status}")  # Debugging log

        metadata = {
            'timestamp': display_timestamp(get_local_timestamp()),
            'objects': [obj[1] for obj in objectInfo],
            'faces': recognized_faces,
            'detection_status': detection_status
        }

        if detection_status == "detected":
            doc_ref = db.collection('testing').document()
            doc_ref.set(metadata)
        
        return jsonify(metadata), 200

    return jsonify({'error': 'Frame not found in request'}), 400

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        video_file = request.files.get('file')
        if not video_file:
            return jsonify({'error': 'No video file provided'}), 400

        # Extract and process metadata
        metadata = request.form.get('metadata')
        metadata_dict = json.loads(metadata) if metadata else {}
        access_token = str(uuid.uuid4())

        bucket = storage.bucket()
        blob = bucket.blob(f"videos/{werkzeug.utils.secure_filename(video_file.filename)}")

        # Setting custom metadata along with content type
        blob.metadata = {'firebaseStorageDownloadTokens': access_token,
        'customMetadata': metadata_dict}
        blob.upload_from_file(video_file, content_type='video/mp4')
        
        # Make the blob publicly accessible
        blob.make_public()

        # Construct the public URL for the uploaded video
        public_url = f"{blob.public_url}?alt=media&token={access_token}"

        return jsonify({'message': f'Successfully uploaded {video_file.filename}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5001)
