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

# COLORS = np.random.uniform(0,255, size = (len(self.labels.values()), 3))

#  For darker shades
COLORS = [
    (255,0,0),(81,14,14),(213,66,174),(87,1,64),(220,87,225),(62,18,238),
    (18,91,238),(0,255,0),(0,0,255),(18,164,238),(18,223,238),(68,87,88),
    (28,220,131),(28,220,79),(220,214,28),(48,84,0),(0,84,81),(23,41,53),
    (213,140,30),(98,39,5),(13,255,14)
]

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
        self.prototxt_path = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
        self.labels = json.load(open(LABELS))
        self.colors = COLORS
        self.save_image = save_image
    
    def _load_model(self):
        # Using cv2 dnn module 
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.weights_path)

    @staticmethod
    #SOURCE :- https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv
    def draw_text(
          img,
          text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 0, 0),
          text_color_bg=(0, 0, 0)
        ):

        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w + 20, y + text_h + 20), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    
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
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 0.007843, (300,300), 127.5)
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
                cv2.rectangle(image, (startX, startY), (endX, endY), self.colors[idx], 3)
                textboxendx = int(endX*0.3)
                # cv2.rectangle(image, (startX, startY), (startX+textboxendx, startY+60), self.colors[idx], cv2.FILLED)
                # Add text label 
                label = f"{list(self.labels.values())[idx]} {round(confidence*100,2)}%"
                y = startY - 15 if startY - 15 > 15 else startY + 15
                print(list(self.colors[idx]))
                # cv2.putText(image, label, (startX, startY + 35), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
                self.draw_text(image,label,pos=(startX,startY+35),text_color_bg=self.colors[idx])
        
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