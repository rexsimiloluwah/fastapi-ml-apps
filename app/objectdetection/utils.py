import cv2 
from PIL import Image
import os 
import time
import io
import json
import shortuuid
import cloudinary.uploader
import numpy as np
from pathlib import Path
from decouple import config
import matplotlib.image as mpimage 

cloudinary.config(
    cloud_name = config("CLOUDINARY_CLOUD_NAME"),
    api_key = config("CLOUDINARY_API_KEY"),
    api_secret = config("CLOUDINARY_API_SECRET")
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "uploads")
LABELS = os.path.join(os.path.dirname(__file__), "labels.json")

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

def read_image_file(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    if image.mode == "RGBA":
        image = image.convert("RGB")
    return np.asarray(image)

class SSDObjectDetection:
    def __init__(self, save_image = False):
        self.weights_path = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")
        self.prototxt_path = os.path.join(MODELS_DIR, "MobileNet_deploy.prototxt")
        self.labels = json.load(open(LABELS))
        self.colors = np.random.uniform(0,255, size = (len(self.labels.values()), 3))
        self.save_image = save_image
    
    def _load_model(self):
        # Using cv2 dnn module 
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.weights_path)
    
    def _detect(self, image, display=False):
        image_copy = image.copy()
        (h, w, channels) = image.shape
        print(image.shape)
        # Remove 4th channel because dnn module's Convolutional layer only supports three channels 
        if channels > 3:
            image = image[:, :, :3]
        print(image.shape)
        # Resize the image to (400, 400) because the model works with only fixed dimensions
        # Construct an input blob for the network
        image_resized = cv2.resize(image, (300,300))
        print(image_resized.shape)
        blob = cv2.dnn.blobFromImage(image_resized, 0.007843, (300,300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        # print(detections.shape)
        predictions = []
        for i in np.arange(detections.shape[2]):
            # Extract the confidence of detection (probability metric)
            confidence = detections[0, 0, i, 2]

            # Filter out detections with confidence less than the confidence threshold 
            if confidence > 0.6:
                # Class index
                idx = int(detections[0, 0 , i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
                print(list(self.labels.values())[idx])
                print(box)
                predictions.append({
                    'class' : list(self.labels.values())[idx],
                    'confidence' : float(confidence),
                    'bounding_box' : box.astype("int").tolist(),
                    'color' : tuple(map(int, self.colors[idx]))
                })

                (startX, startY, endX, endY) = box.astype("int")
                # Put a rectangle around the prediction's bounding box
                cv2.rectangle(image, (startX, startY), (endX, endY), self.colors[idx], 5)

                # Add text label 
                label = list(self.labels.values())[idx]
                y = startY - 15 if startY - 15 > 15 else startY + 15
                print(list(self.colors[idx]))
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 2, self.colors[idx], 2)
        
        if self.save_image:
            image_id = shortuuid.ShortUUID().random(length = 8)
            save_filename = "".join(["output", "-", image_id, ".jpg"])
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self.output_image_url = cloudinary.uploader.upload(os.path.join(OUTPUT_DIR, save_filename))["url"]

        if display:
            cv2.imshow("Output", image)
            cv2.waitKey(0)
        
        self.predictions = predictions

    def response(self, image):
        self._load_model()
        self._detect(image)
        response = {}
        if self.save_image:
            response["output_image_url"] = self.output_image_url 
        response["predictions"] = self.predictions
        return response


if __name__ == "__main__":
    start = time.time()
    detector = SSDObjectDetection()
    image = cv2.imread("./assets/dog-hug.jpg")
    print(detector.response(image))
    end = time.time()
    print(f"{end - start} s")