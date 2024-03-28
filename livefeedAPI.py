from flask import Flask, Response
from flask_cors import CORS
import cv2

app = Flask(__name__)
CORS(app)

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
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api', methods=['GET'])
def api_root():
    return Response("API is running", mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
