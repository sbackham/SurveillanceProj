import cv2
import torch
import torchvision

print(torch.__version__)
print(torchvision.__version__)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/sirena/yolov5/my_model_weights.pt')  

# Define the GStreamer pipeline
gst_pipeline = 'libcamerasrc ! videoconvert ! videoscale ! video/x-raw,format=BGR,width=640,height=480 ! appsink'

#'libcamerasrc ! videoconvert ! autovideosink'
# 'libcamerasrc ! videoconvert ! videoscale ! video/x-raw,format=BGR,width=640,height=480 ! appsink'

# Open the camera using the GStreamer pipeline
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
try:
    if not cap.isOpened():
        print("Error opening camera")
    else:
        while True:
            # Capture a frame
            ret, frame = cap.read()

            if ret:
                # Perform inference
                results = model(frame)

                # Render the detections on the frame
                frame_with_detections = results.render()[0]

                # Display the frame with detections
                cv2.imshow("Camera Preview", frame_with_detections)

                # Break the loop when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            #else:
                #print("Failed to capture frame")

finally:
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
