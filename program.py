import tensorflow as tf
import keras_cv

import random
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_SIZE = 640
CLASSES = ["helmet", "head"]
RANDOM_SEED = 123


def parse_xml(xml_path, img_dir):
    root = ET.parse(xml_path).getroot()
    boxes, classes = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASSES:
            continue
        b = obj.find("bndbox")
        boxes.append([float(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")])
        classes.append(CLASSES.index(name))
    img_path = str(Path(img_dir) / root.find("filename").text)
    return img_path, np.array(boxes, "float32"), np.array(classes, "float32")

samples = [parse_xml(x, "dataset/images") for x in Path("dataset/annotations").glob("*.xml")]
# delete empty samples
samples = [s for s in samples if len(s[1]) > 0 and Path(s[0]).exists()]
# randomize
random.Random(RANDOM_SEED).shuffle(samples)


def make_dataset(samples, batch=16, train=True):
    paths = [s[0] for s in samples]
    boxes = tf.ragged.constant([s[1] for s in samples], ragged_rank=1)
    classes = tf.ragged.constant([s[2] for s in samples])

    ds = tf.data.Dataset.from_tensor_slices((paths, classes, boxes))

    def load(path, cls, bx):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32)
        return {"images": img,
                "bounding_boxes": {"classes": cls, "boxes": bx}}

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.shuffle(500)

    resize = keras_cv.layers.JitteredResize(
        target_size=(IMG_SIZE, IMG_SIZE), scale_factor=(0.75, 1.3),
        bounding_box_format="xyxy") if train else keras_cv.layers.Resizing(
        IMG_SIZE, IMG_SIZE, pad_to_aspect_ratio=True, bounding_box_format="xyxy")

    ds = ds.ragged_batch(batch, drop_remainder=True).map(resize, num_parallel_calls=tf.data.AUTOTUNE)

    if train:
        aug = keras_cv.layers.Augmenter([
            keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
            keras_cv.layers.RandomHue(0.1, [0, 255]),
        ])
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.map(lambda d: (d["images"], d["bounding_boxes"]),
                  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

split = int(0.8 * len(samples))
train_ds = make_dataset(samples[:split])
val_ds = make_dataset(samples[split:], train=False)

backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco")

model = keras_cv.models.YOLOV8Detector(
    num_classes=len(CLASSES),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=2,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3, global_clipnorm=10.0),
    classification_loss="binary_crossentropy",
    box_loss="ciou",
)


model.fit(train_ds, validation_data=val_ds, epochs=60)
model.save_weights("helmet_yolov8.weights.h5")


img = tf.io.decode_image(tf.io.read_file("test.jpg"), channels=3)
img = keras_cv.layers.Resizing(IMG_SIZE, IMG_SIZE, pad_to_aspect_ratio=True)(img[None])

model.prediction_decoder = keras_cv.layers.NonMaxSuppression(
    bounding_box_format="xyxy", from_logits=False,
    iou_threshold=0.5, confidence_threshold=0.5)

pred = model.predict(img)
labels = [CLASSES[int(c)] for c in pred["classes"][0] if c >= 0]  # -1 = padding

if "head" in labels:
    print("DENIED: person without helmet")
elif "helmet" in labels:
    print("OK")


