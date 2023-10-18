import math

import cv2
from ultralytics import YOLO


def person_detection(imgPath):
    model = YOLO("yolov8n.pt")

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
                  "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                  "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                  "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                  "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
                  "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                  "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
                  "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    img = cv2.imread(imgPath)

    results = model(img, stream=True)

    count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["person"] and conf > 0.3:
                count += 1

    return count


if __name__ == "__main__":
    people = person_detection("images/13309.jpg")
    print(people)
