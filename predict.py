import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras_cv
from PIL import Image, ImageDraw


IMG_SIZE = 640
CLASSES = ["helmet", "head", "person"]
WEIGHTS = "helmet_yolov8.weights.h5"


CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# Minimalna pewność, żeby daną detekcję uznać za poprawna
HELMET_MIN_CONF = 0.70
HEAD_MIN_CONF = 0.40

COLORS = {"helmet": "lime", "head": "red", "person": "deepskyblue"}


backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco")

model = keras_cv.models.YOLOV8Detector(
    num_classes=len(CLASSES),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=2,
    prediction_decoder=keras_cv.layers.NonMaxSuppression(
        bounding_box_format="xyxy",
        from_logits=False,
        iou_threshold=IOU_THRESHOLD,
        confidence_threshold=CONF_THRESHOLD,
    ),
)

model.compile(
    optimizer="adam",
    classification_loss="binary_crossentropy",
    box_loss="ciou",
)

model.load_weights(WEIGHTS)


#wykrywanie

def load_image(path):
    img = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    return tf.cast(img, tf.float32)


def detect(image_path):
    #lista: (klasa, stopien pewnosci, (x1, y1, x2, y2)) w pikselach oryginalnego zdjęcia.
    img = load_image(image_path)
    orig_h, orig_w = int(img.shape[0]), int(img.shape[1])

    resized = keras_cv.layers.Resizing(
        IMG_SIZE, IMG_SIZE, pad_to_aspect_ratio=True
    )(img[None])

    pred = model.predict(resized, verbose=0)

    n = int(pred["num_detections"][0])
    boxes = pred["boxes"][0][:n]
    classes = pred["classes"][0][:n].astype(int)
    scores = pred["confidence"][0][:n]

    # Przeliczenie z obrazu 640x640 (z paddingiem) na oryginalny rozmiar
    scale = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
    pad_x = (IMG_SIZE - orig_w * scale) / 2
    pad_y = (IMG_SIZE - orig_h * scale) / 2

    detections = []
    for box, c, s in zip(boxes, classes, scores):
        if c < 0 or c >= len(CLASSES):
            continue
        x1, y1, x2, y2 = box
        x1 = float(np.clip((x1 - pad_x) / scale, 0, orig_w))
        x2 = float(np.clip((x2 - pad_x) / scale, 0, orig_w))
        y1 = float(np.clip((y1 - pad_y) / scale, 0, orig_h))
        y2 = float(np.clip((y2 - pad_y) / scale, 0, orig_h))
        detections.append((CLASSES[c], float(s), (x1, y1, x2, y2)))

    detections.sort(key=lambda d: d[1], reverse=True)
    return detections


def evaluate(detections):

    helmets = [d for d in detections if d[0] == "helmet" and d[1] >= HELMET_MIN_CONF]
    heads = [d for d in detections if d[0] == "head" and d[1] >= HEAD_MIN_CONF]
    persons = [d for d in detections if d[0] == "person"]

    if heads:
        return "UWAGA: wykryto głowę bez kasku - wymaga sprawdzenia."
    if helmets:
        return "OK: wykryto kask."
    if persons or any(d[0] == "helmet" for d in detections):
        return "Brak pewnej detekcji kasku - wymaga sprawdzenia."
    return "Nie wykryto osób ani kasków."


def draw_results(image_path, detections, save_path):
    pil = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for name, score, box in detections:
        draw.rectangle(box, outline=COLORS[name], width=3)
        draw.text((box[0] + 3, box[1] + 3), f"{name} {score:.2f}", fill=COLORS[name])
    pil.save(save_path)


def check_helmet(image_path, save_path="result.jpg"):
    detections = detect(image_path)

    print("\nWykryto:")
    if not detections:
        print("Brak detekcji.")
    for name, score, (x1, y1, x2, y2) in detections:
        print(f"{name:7s} {score:.2f}  [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    print(f"\n{evaluate(detections)}")

    draw_results(image_path, detections, save_path)
    print(f"Zapisano w: {save_path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    check_helmet(path)