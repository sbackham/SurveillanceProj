import cv2

classNames = []
classFile = "/home/sirena/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/sirena/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/sirena/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo

if __name__ == "__main__":
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

    if not cap.isOpened():
        print("Error opening camera")
    else:
        while True:
            ret, img = cap.read()
            if ret:
                result, objectInfo = getObjects(img, 0.45, 0.2, objects=['person', 'car', 'motorcycle', 'bus', 'truck', 'bicycle', 'cat', 'dog'])
                cv2.imshow("Output", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Failed to capture frame")

    cap.release()
    cv2.destroyAllWindows()
