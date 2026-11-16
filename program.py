
import tensorflow as tf
import keras_cv

import random
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 1e-3

CLASSES = ["helmet", "head", "person"]
RANDOM_SEED = 123

IMAGE_DIR = Path("dataset/images")
ANNOTATION_DIR = Path("dataset/annotations")

WEIGHTS = "helmet_yolov8.weights.h5"

tf.keras.utils.set_random_seed(RANDOM_SEED)


def parse_xml(xml_path):

    root = ET.parse(xml_path).getroot()

    filename = root.findtext("filename")

    if not filename:
        return None

    img_path = IMAGE_DIR / filename

    if not img_path.exists():
        return None

    size = root.find("size")

    if size is None:
        return None

    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    if width <= 0 or height <= 0:
        return None

    boxes = []
    classes = []

    for obj in root.findall("object"):

        name = obj.findtext("name", "").strip().lower()

        if name not in CLASSES:
            continue

        b = obj.find("bndbox")

        if b is None:
            continue

        xmin = float(b.findtext("xmin"))
        ymin = float(b.findtext("ymin"))
        xmax = float(b.findtext("xmax"))
        ymax = float(b.findtext("ymax"))

        xmin = np.clip(xmin, 0, width)
        xmax = np.clip(xmax, 0, width)
        ymin = np.clip(ymin, 0, height)
        ymax = np.clip(ymax, 0, height)

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        classes.append(CLASSES.index(name))

    return (
        str(img_path),
        np.array(boxes, dtype="float32"),
        np.array(classes, dtype="float32")
    )


#zaciegniecie zbioru
samples = []

for xml in sorted(ANNOTATION_DIR.glob("*.xml")):

    try:
        sample = parse_xml(xml)

        if sample is not None and len(sample[1]) > 0:
            samples.append(sample)

    except Exception as e:
        print(f"Błąd XML {xml.name}: {e}")


if len(samples) < 2:
    raise RuntimeError("Za mało poprawnych zdjęć w zbiorze.")

random.Random(RANDOM_SEED).shuffle(samples)

print(f"Liczba zdjęć: {len(samples)}")


#80 : 20 podzielenie

split = int(0.8 * len(samples))

train_samples = samples[:split]
val_samples = samples[split:]

print(f"Train: {len(train_samples)}")
print(f"Validation: {len(val_samples)}")


#data set

def make_dataset(samples, train=True):

    paths = [s[0] for s in samples]

    boxes = tf.ragged.constant(
        [s[1] for s in samples],
        ragged_rank=1,
        dtype=tf.float32
    )

    classes = tf.ragged.constant(
        [s[2] for s in samples],
        ragged_rank=1,
        dtype=tf.float32
    )

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, classes, boxes)
    )

    def load(path, cls, bx):

        img = tf.io.decode_image(
            tf.io.read_file(path),
            channels=3,
            expand_animations=False
        )

        img.set_shape([None, None, 3])
        img = tf.cast(img, tf.float32)

        return {
            "images": img,
            "bounding_boxes": {
                "classes": cls,
                "boxes": bx
            }
        }

    ds = ds.map(load, num_parallel_calls=2)

    if train:
        ds = ds.shuffle(
            min(len(samples), 500),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.ragged_batch(
        BATCH_SIZE,
        drop_remainder=False
    )

    if train:

        resize = keras_cv.layers.JitteredResize(
            target_size=(IMG_SIZE, IMG_SIZE),
            scale_factor=(0.75, 1.3),
            bounding_box_format="xyxy"
        )

    else:

        resize = keras_cv.layers.Resizing(
            IMG_SIZE,
            IMG_SIZE,
            pad_to_aspect_ratio=True,
            bounding_box_format="xyxy"
        )

    ds = ds.map(resize, num_parallel_calls=2)

    if train:

        aug = keras_cv.layers.Augmenter([
            keras_cv.layers.RandomFlip(
                mode="horizontal",
                bounding_box_format="xyxy"
            ),
            keras_cv.layers.RandomHue(
                factor=0.05,
                value_range=(0, 255)
            )
        ])

        ds = ds.map(aug, num_parallel_calls=2)

    return ds.map(
        lambda d: (d["images"], d["bounding_boxes"]),
        num_parallel_calls=2
    ).prefetch(1)


train_ds = make_dataset(train_samples, train=True)
val_ds = make_dataset(val_samples, train=False)

#yolo

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_s_backbone_coco"
)

model = keras_cv.models.YOLOV8Detector(
    num_classes=len(CLASSES),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=2
)



steps_per_epoch = (
    len(train_samples) + BATCH_SIZE - 1
) // BATCH_SIZE

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=EPOCHS * steps_per_epoch,
    alpha=0.05
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        global_clipnorm=10.0
    ),
    classification_loss="binary_crossentropy",
    box_loss="ciou"
)



callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        WEIGHTS,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
]




print("\nStart treningu...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save_weights(WEIGHTS)

print("\nZapisano model:", WEIGHTS)


# =====================================================
# TEST PREDYKCJI
# =====================================================

model.prediction_decoder = keras_cv.layers.NonMaxSuppression(
    bounding_box_format="xyxy",
    from_logits=False,
    iou_threshold=0.5,
    confidence_threshold=0.05
)

def check_helmet(image_path):

    img = tf.io.decode_image(
        tf.io.read_file(image_path),
        channels=3,
        expand_animations=False
    )

    img = tf.cast(img, tf.float32)

    img = keras_cv.layers.Resizing(
        IMG_SIZE,
        IMG_SIZE,
        pad_to_aspect_ratio=True
    )(img[None])

    pred = model.predict(img, verbose=0)

    labels = [
        CLASSES[int(c)]
        for c in pred["classes"][0]
        if c >= 0
    ]

    print("\nWykryte obiekty:", labels)

    if "head" in labels:
        print("Wykryto głowę - wymaga sprawdzenia.")
    elif "helmet" in labels:
        print("Wykryto kask.")
    else:
        print("Nie wykryto obiektów.")


if Path("test.jpg").exists():
    check_helmet("test.jpg")
