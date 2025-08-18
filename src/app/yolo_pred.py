import cv2 as cv
import numpy as np
from PIL import Image


def draw_bounding_boxes(image, predictions):
    image = np.array(image)
    classes = ["crazing", "inclusion", "patches","pitted_surface", "rolled-in_scale", "scratches"]

    labels=[]
    i=0


    for pred in predictions:
        clss = int(pred['class'])-1
        confidence = pred['confidence']
        x1, y1, x2, y2 = map(int, pred['bbox'])

        # get readable class name
        label = f"{i}"
        labels.append(f"{label}: {classes[clss]}  with confidence {confidence}")

        (text_w, text_h), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        center_x = x1 + (x2 - x1) // 2 - text_w // 2
        center_y = y1 + (y2 - y1) // 2 + text_h // 2


        # draw rectangle + label
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv.putText(image, label, (center_x, center_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        i=i+1

    return Image.fromarray(image),labels
