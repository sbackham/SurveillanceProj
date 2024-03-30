import cv2
import numpy as np
import os
from datetime import datetime
import threading
import firebase_admin
from firebase_admin import credentials, storage, db

# Initialize Firebase Admin SDK
cred = credentials.Certificate('path/to/serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-firebase-storage-bucket',
    'databaseURL': 'https://your-firebase-project.firebaseio.com'
})

# Create instances for Firebase Storage and Realtime Database
storage_bucket = storage.bucket()
firebase_db = db.reference()

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

def process_clip(clip, timestamp):
    clip_frames = []
    metadata = {
        'timestamp': timestamp,
        'labels': []
    }

    for frame in clip:
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

        # Collect labels for this frame
        for obj in objectInfo:
            box, label = obj
            metadata['labels'].append(label)

        clip_frames.append(frame)

    # Save the processed clip to a video file
    filename = f'motion_clip_{timestamp}.mp4'
    height, width, _ = clip_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # Assuming 30 frames per second
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for frame in clip_frames:
        video_writer.write(frame)
    video_writer.release()

    # Upload the processed clip to Firebase Storage
    upload_clip(filename)

    # Store metadata in Firebase Realtime Database
    store_metadata(filename, metadata)

def upload_clip(filename):
    blob = storage_bucket.blob(filename)
    blob.upload_from_filename(filename)
    print(f'File {filename} uploaded to Firebase Storage')

def store_metadata(filename, metadata):
    clip_ref = firebase_db.child('motion_clips').child(filename)
    clip_ref.set(metadata)
    print(f'Metadata for {filename} stored in Firebase Realtime Database')

def detect_motion():
    # Define the GStreamer pipeline
    gst_pipeline = (
        'libcamerasrc ! '
        'videoconvert ! '
        'videoscale ! '
        'video/x-raw,format=BGR,width=640,height=480 ! '
        'appsink drop=true sync=false'
    )

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    motion_detected = False
    motion_clip = []
    motion_start_time = None

    while True:
        success, frame = cap.read()
        if not success:
            print("Error reading from camera")
            break

        mask = background_subtractor.apply(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            if not motion_detected:
                motion_detected = True
                motion_start_time = datetime.now()

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            motion_clip.append(frame.copy())

            if len(motion_clip) > 30:  # Keep the last 30 frames (1 second)
                motion_clip.pop(0)

        if motion_detected and len(contours) == 0:
            motion_detected = False
            timestamp = motion_start_time.strftime("%Y%m%d_%H%M%S")
            motion_start_time = None

            # Process the motion clip in a separate thread
            clip_thread = threading.Thread(target=process_clip, args=(motion_clip, timestamp))
            clip_thread.start()

            motion_clip = []

        cv2.imshow('Motion Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_motion()