import cv2

# Define a GStreamer pipeline using 'appsink' at the end
gst_pipeline = (
    'libcamerasrc ! '
    'videoconvert ! '
    'videoscale ! '
    'video/x-raw,format=BGR,width=640,height=480 ! '
    'appsink drop=true sync=false'
)

# Open the camera using the GStreamer pipeline
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error opening camera")
else:
    print("Camera opened successfully")
    while True:
        ret, frame = cap.read()
        if ret:
            # Display the captured frame
            cv2.imshow("Camera Preview", frame)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to capture frame")

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
