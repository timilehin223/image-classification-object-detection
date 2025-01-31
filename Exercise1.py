import numpy as np  
import time 
import cv2  

image_path = "C:/Users/timil/Downloads/football.jpg"
model_path = "C:/Users/timil/Downloads/googlenet.caffemodel"
label_path = "C:/Users/timil/Downloads/imagenet_labels.txt"
prototxt_path = "C:/Users/timil/Downloads/googlenet.prototxt.txt"

image = cv2.imread(image_path)

rows = open(label_path).read().strip().split("\n")
classes = [r.split(",")[0] for r in rows]

blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

print("Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

net.setInput(blob)

start = time.time()
preds = net.forward()  
end = time.time()

print("Classification time: {:.5f} seconds".format(end - start))

idxs = np.argsort(preds[0])[::-1][:5] 

for (i, idx) in enumerate(idxs):
    label = classes[idx]
    confidence = preds[0][idx] * 100

    print("{}. label: {}, probability: {:.2f}%".format(i + 1, label, confidence))

    if i == 0:
        text = "Label: {}, {:.2f}%".format(label, confidence)
        cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.imshow("Output", image)
cv2.waitKey(0) 
cv2.destroyAllWindows()
