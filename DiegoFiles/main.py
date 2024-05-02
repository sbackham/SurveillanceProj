from flask import Flask, Response, render_template, request, jsonify
import cv2
import requests
import numpy as np
import time
import os
import json
import subprocess
from datetime import datetime, timedelta, timezone

app = Flask(__name__)
recording = False
video_writer = None
detected_objects = set()
detected_people = set()
video_folder = 'recorded_videos'
original_fps = 12  # the effective FPS application is achieving


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global recording, video_writer, detected_objects, detected_people, metadata
    gst_pipeline = (
        'libcamerasrc ! '
        'videoconvert ! '
        'videoscale ! '
        'video/x-raw,format=BGR,width=1280,height=720 ! '
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

        if response.status_code == 200:
            data = response.json()
            detection_status = data.get('detection_status', "not_detected")
            objects = response.json().get('objects', [])
            detected_objects.update(data.get('objects', []))
            detected_people.update(data.get('faces', []))
            people = response.json().get('faces', [])
            timestamp = response.json().get('timestamp', [])
            
            

            if detection_status == "detected" and not recording:
                start_new_recording()

            if recording:
                video_writer.write(frame)
                if detection_status == "not_detected":
                    no_detection_counter += 1
                else:
                    no_detection_counter = 0

                if no_detection_counter >= 20:
                    stop_and_process_video()

        else:
            print(f"Error processing frame: {response.status_code}")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def start_new_recording():
    global video_writer, recording, video_file_path
    video_file_path = os.path.join(video_folder, f"{time.strftime('%Y%m%d-%H%M%S')}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_file_path, fourcc, original_fps, (1280, 720))
    recording = True

def stop_and_process_video():
    global recording, video_writer, metadata, video_file_path, detected_objects, detected_people
    recording = False
    video_writer.release()
    adjusted_video_path = video_file_path.replace('.avi', '_slow.avi')
    metadata = adjust_video_speed(video_file_path, adjusted_video_path, 5)
    upload_video_to_frames_py(adjusted_video_path, list(detected_objects), list(detected_people), metadata)
    detected_objects.clear()
    detected_people.clear()

def adjust_video_speed(input_video_path, output_video_path, slowdown_factor):
    setpts_value = f"setpts={3}*PTS"
    command = [
        'ffmpeg',
        '-i', input_video_path,
        '-filter:v', setpts_value,
        '-b:v', '5000k',
        output_video_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove(input_video_path)
    return {'objects': list(detected_objects), 'people': list(detected_people)}
    
def convert_video_format(input_video_path, output_video_path):
    command = [
        'ffmpeg',
        '-i', input_video_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        output_video_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#for 24 hour recording
'''def upload_video_to_frames_py(video_file_path, metadata):
    converted_video_path = video_file_path.replace('.avi', '.mp4')
    convert_video_format(video_file_path, converted_video_path)
    
    with open(video_file_path, 'rb') as video_file:
        files = {'file': (os.path.basename(converted_video_path), video_file, 'video/mp4')}
        data = {'metadata': json.dumps(metadata)}
        response = requests.post('http://localhost:5001/upload_video', files=files, data=data)
        if response.status_code == 200:
            print(f"Successfully uploaded {video_file_path} to Firebase Storage with metadata.")
        else:
            print(f"Failed to upload {video_file_path}. Response: {response.content}")'''
            
'''def continuous_recording():
    video_folder = '24_hour_recordings'
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    while True:  # Loop to handle continuous recording
        video_file_path = os.path.join(video_folder, f"{time.strftime('%Y%m%d-%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_file_path, fourcc, original_fps, (1280, 720))
        cap = cv2.VideoCapture(0)  # Adjust for your camera setup
        
        start_time = time.time()
        while time.time() - start_time < 60:  # Record for 1 minute
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break
        
        out.release()
        cap.release()

        # Upload and delete the video
        metadata = {'description': '24-hour continuous recording segment'}
        upload_video_to_frames_py(video_file_path, metadata)'''

def upload_video_to_frames_py(video_file_path, objects, people, metadata):
    converted_video_path = video_file_path.replace('.avi', '.mp4')
    convert_video_format(video_file_path, converted_video_path)  # Assume this function correctly converts the format
    current_timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    
    with open(converted_video_path, 'rb') as video_file:
        files = {'file': (os.path.basename(converted_video_path), video_file, 'video/mp4')}
        metadata_json = json.dumps({'objects': objects, 'people': people, 'timestamp': current_timestamp, **metadata})
        data = {'metadata': metadata_json}
        response = requests.post('http://localhost:5001/upload_video', files=files, data=data)
        if response.status_code == 200:
            print(f"Successfully uploaded {converted_video_path} to Firebase Storage with metadata.")
            print("Response Message:", response.json().get('message'))  # Print server response message
            print("Access URL:", response.json().get('url'))  # Print the URL to access the uploaded video
        else:
            print(f"Failed to upload {converted_video_path}. Response: {response.status_code}, {response.content}")



if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
    except KeyboardInterrupt:
        if recording and video_writer:
            video_writer.release()
        print("Cleanup complete.")
