'''from flask import Flask, Response, render_template
import cv2
import requests
import numpy as np
import time
import os

app = Flask(__name__)
recording = False
video_writer = None
no_detection_counter = 0
video_folder = 'recorded_videos'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global recording, video_writer, no_detection_counter

    gst_pipeline = (
        'libcamerasrc ! '
        'videoconvert ! '
        'videoscale ! '
        'video/x-raw,format=BGR,width=1280,height=720,framerate=30/1 ! '
        'appsink drop=true sync=false'
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return

    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        response = requests.post('http://localhost:5001/process_frame', files={'frame': frame_data})

        if response.status_code == 200 and response.headers.get('Content-Type') == 'application/json':
            detection_status = response.json().get('detection_status')

            if detection_status == "detected":
                if not recording:
                    video_file_path = os.path.join(video_folder, f"{time.strftime('%Y%m%d-%H%M%S')}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(video_file_path, fourcc, 30.0, (1280, 720))
                    recording = True
                no_detection_counter = 0
            elif detection_status == "not_detected":
                no_detection_counter += 1

            if recording:
                video_writer.write(frame)
                if no_detection_counter >= 20:
                    recording = False
                    video_writer.release()
                    video_writer = None
                    print(f"Video saved: {video_file_path}")

        else:
            print(f"Error processing frame: {response.status_code}")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
    except KeyboardInterrupt:
        if recording and video_writer:
            video_writer.release()
        print("Cleanup complete.")
        
#this version sends the videos and metadata to the DB but not in the video captions
from flask import Flask, Response, render_template
import cv2
import requests
import numpy as np
import time
import os

app = Flask(__name__)
recording = False
video_writer = None
no_detection_counter = 0
video_folder = 'recorded_videos'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global recording, video_writer, no_detection_counter

    gst_pipeline = (
        'libcamerasrc ! '
        'videoconvert ! '
        'videoscale ! '
        'video/x-raw,format=BGR,width=1280,height=720,framerate=30/1 ! '
        'appsink drop=true sync=false'
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return

    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        response = requests.post('http://localhost:5001/process_frame', files={'frame': frame_data})

        if response.status_code == 200 and response.headers.get('Content-Type') == 'application/json':
            detection_status = response.json().get('detection_status')

            if detection_status == "detected":
                if not recording:
                    video_file_path = os.path.join(video_folder, f"{time.strftime('%Y%m%d-%H%M%S')}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(video_file_path, fourcc, 30.0, (1280, 720))
                    recording = True
                no_detection_counter = 0
            elif detection_status == "not_detected":
                no_detection_counter += 1

            if recording:
                video_writer.write(frame)
                if no_detection_counter >= 20:  # Changed to 20 for quicker testing
                    recording = False
                    video_writer.release()
                    video_writer = None
                    print(f"Video saved: {video_file_path}")
                    # Send the video for upload
                    with open(video_file_path, 'rb') as video_file:
                        requests.post('http://localhost:5001/upload_video', files={'file': (video_file_path, video_file, 'video/mp4')})

        else:
            print(f"Error processing frame: {response.status_code}")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
    except KeyboardInterrupt:
        if recording and video_writer:
            video_writer.release()
        print("Cleanup complete.")'''
        
from flask import Flask, Response, render_template
import cv2
import requests
import numpy as np
import time
import os
import json

app = Flask(__name__)
recording = False
video_writer = None
no_detection_counter = 0
detected_objects = set()
detected_people = set()  # Set to store unique detected people
video_folder = 'recorded_videos'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global recording, video_writer, no_detection_counter, detected_objects, detected_people

    gst_pipeline = (
        'libcamerasrc ! '
        'videoconvert ! '
        'videoscale ! '
        'video/x-raw,format=BGR,width=1280,height=720,framerate=30/1 ! '
        'appsink drop=true sync=false'
    )
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return

    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        response = requests.post('http://localhost:5001/process_frame', files={'frame': frame_data})

        if response.status_code == 200 and response.headers.get('Content-Type') == 'application/json':
            detection_status = response.json().get('detection_status')
            objects = response.json().get('objects', [])
            people = response.json().get('faces', [])  # Assuming faces are labeled and returned here

            detected_objects.update(objects)
            detected_people.update(people)  # Update detected_people set

            if detection_status == "detected":
                if not recording:
                    video_file_path = os.path.join(video_folder, f"{time.strftime('%Y%m%d-%H%M%S')}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(video_file_path, fourcc, 30.0, (1280, 720))
                    recording = True
                no_detection_counter = 0
            elif detection_status == "not_detected":
                no_detection_counter += 1

            if recording:
                video_writer.write(frame)
                if no_detection_counter >= 20:
                    recording = False
                    video_writer.release()
                    video_writer = None
                    print(f"Video saved: {video_file_path}")
                    upload_video_to_frames_py(video_file_path, list(detected_objects), list(detected_people))
                    detected_objects.clear()
                    detected_people.clear()

        else:
            print(f"Error processing frame: {response.status_code}")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def upload_video_to_frames_py(video_file_path, objects, people):
    with open(video_file_path, 'rb') as video_file:
        files = {'file': (video_file_path, video_file, 'video/mp4')}
        metadata = json.dumps({'objects': objects, 'people': people})
        data = {'metadata': metadata}
        response = requests.post('http://localhost:5001/upload_video', files=files, data=data)
        if response.status_code == 200:
            print(f"Successfully uploaded {video_file_path} to Firebase Storage with metadata.")
        else:
            print(f"Failed to upload {video_file_path}. Response: {response.content}")

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
    except KeyboardInterrupt:
        if recording and video_writer:
            video_writer.release()
        print("Cleanup complete.")



