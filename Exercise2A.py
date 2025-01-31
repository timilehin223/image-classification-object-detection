import numpy as np
import cv2

# Paths to model, configuration, and test image (the image path is no longer needed)
prototxt_path = "C:/Users/timil/Downloads/MobileNetSSD_deploy_prototxt.txt"
model_path = "C:/Users/timil/Downloads/MobileNetSSD_deploy.caffemodel"

# Confidence threshold to filter predictions
conf_limit = 0.25

# Class labels that the MobileNetSSD model was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tv/monitor"]

# Random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the pre-trained model
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Open the camera stream (0 is the default camera)
cap = cv2.VideoCapture(0)  # Change the 0 if you have multiple cameras, e.g., 1 for the second camera

if not cap.isOpened():
    print("Error: Couldn't open the camera stream.")
    exit()

while True:
    # Capture each frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't fetch frame.")
        break

    # Get the image dimensions (height, width)
    (h, w) = frame.shape[:2]

    # Preprocess the frame into a blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                 (300, 300), 127.5)

    # Send the frame through the network
    net.setInput(blob)
    detections = net.forward()

    # Loop over all detections
    for i in np.arange(0, detections.shape[2]):
        # Extract the confidence of the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > conf_limit:
            # Extract class index and bounding box coordinates
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Prepare the label for the object
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("{}".format(label))

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Display the output frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Wait for 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
