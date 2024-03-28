import cv2
import sys
import os

def create_directory_for_name(image_dir, name_class):
    path = os.path.join(image_dir, name_class)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path

def save_images_for_person(name_class, image_dir, size, classifier, im_width, im_height, count_max):
    path = create_directory_for_name(image_dir, name_class)
    pin = sorted([int(n[:n.find('.')]) for n in os.listdir(path) if n[0] != '.'] + [0])[-1] + 1
    
    haar_cascade = cv2.CascadeClassifier(classifier)

    # Define the GStreamer pipeline
    gst_pipeline = 'libcamerasrc ! videoconvert ! videoscale ! video/x-raw,format=BGR,width=640,height=480 ! appsink'
    
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

    print(f"\n\033[94mThe program will save {count_max} samples for {name_class}. Move your head around to increase diversity while it runs.\033[0m\n")

    count = 0
    pause = 0

    while count < count_max:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue

        frame = cv2.flip(frame, 1)  # Flip frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))  # Scale down for speed

        faces = haar_cascade.detectMultiScale(mini)
        faces = sorted(faces, key=lambda x: x[3])
        if faces:
            x, y, w, h = [v * size for v in faces[0]]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))

            if w * 6 >= frame.shape[1] and h * 6 >= frame.shape[0]:  # Check for face size
                if pause == 0:
                    print(f"Saving training sample {count + 1}/{count_max} for {name_class}")
                    cv2.imwrite(f'{path}/{pin}.png', face_resize)
                    pin += 1
                    count += 1
                    pause = 1

        pause = (pause + 1) % 5
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("You must provide a name.")
        sys.exit(0)

    name_class = sys.argv[1]  # Get the name from command-line argument
    image_dir = 'images'
    size = 2
    classifier = 'haarcascade_frontalface_default.xml'
    im_width, im_height = 112, 92
    count_max = 40  # Number of samples to collect

    save_images_for_person(name_class, image_dir, size, classifier, im_width, im_height, count_max)
