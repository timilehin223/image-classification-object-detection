import numpy as np
import cv2

# Paths to model, configuration, and test image
prototxt_path = "C:/Users/timil/Downloads/MobileNetSSD_deploy_prototxt.txt"
model_path = "C:/Users/timil/Downloads/MobileNetSSD_deploy.caffemodel"
image_path = "C:/Users/timil/Downloads/image.jpg"

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

# Load the image from storage
image = cv2.imread(image_path)
(h, w) = image.shape[:2]

# Preprocess the image into a blob
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
                             (300, 300), 127.5)

# Pass the image through the network
print("Sending image through the network...")
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

        # Draw the bounding box and label on the image
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# Display the output image
cv2.imshow("Output", image)
cv2.waitKey(0)  # Wait for key press to close the window
cv2.destroyAllWindows()
