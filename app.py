import os
import threading

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras_cv
import gradio as gr
from PIL import Image, ImageDraw, ImageFont


IMG_SIZE = 640
CLASSES = ["helmet", "head", "person"]
WEIGHTS = "helmet_yolov8.weights.h5"

# Dekoder NMS dostaje niski próg; właściwy próg ustawia suwak w interfejsie
NMS_CONF = 0.15
IOU_THRESHOLD = 0.5

DEFAULT_DET_CONF = 0.35      # minimalna pewność
DEFAULT_HELMET_CONF = 0.60   # tyle trzeba, żeby uznać "kask OK"
DEFAULT_HEAD_CONF = 0.40     # od tego progu "głowa" oznacza brak kasku

COLORS = {"helmet": (0, 200, 0), "head": (230, 30, 30), "person": (30, 144, 255)}
LABELS_PL = {"helmet": "kask", "head": "głowa bez kasku", "person": "osoba"}

# =====================================================
# MODEL
# =====================================================

print("Ładowanie modelu...")

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
        confidence_threshold=NMS_CONF,
    ),
)
model.compile(
    optimizer="adam",
    classification_loss="binary_crossentropy",
    box_loss="ciou",
)
model.load_weights(WEIGHTS)

resizer = keras_cv.layers.Resizing(IMG_SIZE, IMG_SIZE, pad_to_aspect_ratio=True)
MODEL_LOCK = threading.Lock()

print("Model gotowy.")

#detect

def detect(img_np, det_conf):

    h, w = img_np.shape[:2]
    x = tf.cast(img_np[None], tf.float32)
    x = resizer(x)

    with MODEL_LOCK:
        pred = model.predict(x, verbose=0)

    n = int(pred["num_detections"][0])
    boxes = pred["boxes"][0][:n]
    classes = pred["classes"][0][:n].astype(int)
    scores = pred["confidence"][0][:n]


    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    pad_x = (IMG_SIZE - w * scale) / 2
    pad_y = (IMG_SIZE - h * scale) / 2

    result = []
    for box, c, s in zip(boxes, classes, scores):
        if c < 0 or c >= len(CLASSES) or s < det_conf:
            continue
        x1, y1, x2, y2 = box
        x1 = float(np.clip((x1 - pad_x) / scale, 0, w))
        x2 = float(np.clip((x2 - pad_x) / scale, 0, w))
        y1 = float(np.clip((y1 - pad_y) / scale, 0, h))
        y2 = float(np.clip((y2 - pad_y) / scale, 0, h))
        result.append((CLASSES[c], float(s), (x1, y1, x2, y2)))

    result.sort(key=lambda d: d[1], reverse=True)
    return result


def evaluate(dets, helmet_conf, head_conf):
    helmets = [d for d in dets if d[0] == "helmet" and d[1] >= helmet_conf]
    heads = [d for d in dets if d[0] == "head" and d[1] >= head_conf]
    persons = [d for d in dets if d[0] == "person"]

    if heads:
        return "alert", " UWAGA: wykryto głowę bez kasku"
    if helmets:
        return "ok", " OK: wykryto kask"
    if persons or any(d[0] == "helmet" for d in dets):
        return "check", " Brak pewnego kasku - zweryfikj"
    return "none", " Nie wykryto osób ani kasków"


def draw(img_np, dets, helmet_conf, head_conf):
    pil = Image.fromarray(img_np).convert("RGB")
    d = ImageDraw.Draw(pil)
    thickness = max(2, int(max(pil.size) / 300))
    try:
        font = ImageFont.load_default(size=max(12, int(max(pil.size) / 45)))
    except TypeError:  # starsze Pillow bez parametru size
        font = ImageFont.load_default()

    for name, score, box in dets:
        color = COLORS[name]
        d.rectangle(box, outline=color, width=thickness)
        text = f"{LABELS_PL[name]} {score:.2f}"
        tx, ty = box[0] + 3, max(0, box[1] - 18)
        d.text((tx, ty), text, fill=color, font=font)
    return pil


def analyze(img_np, det_conf, helmet_conf, head_conf):
    dets = detect(img_np, det_conf)
    level, text = evaluate(dets, helmet_conf, head_conf)
    return draw(img_np, dets, helmet_conf, head_conf), dets, level, text


# =====================================================
# ZAKŁADKA 1: WIELE ZDJĘĆ
# =====================================================


def process_files(files, det_conf, helmet_conf, head_conf, progress=gr.Progress()):
    if not files:
        return [], [], "Dodaj przynajmniej jedno zdjęcie."

    gallery, rows = [], []
    counts = {"ok": 0, "alert": 0, "check": 0, "none": 0}

    for f in progress.tqdm(files, desc="Analiza zdjęć"):
        path = f if isinstance(f, str) else getattr(f, "name", str(f))
        name = os.path.basename(path)
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except Exception as e:
            rows.append([name, "-", "-", "-", f"Błąd odczytu: {e}"])
            continue

        annotated, dets, level, text = analyze(img, det_conf, helmet_conf, head_conf)
        counts[level] += 1

        n_helmets = sum(1 for d in dets if d[0] == "helmet" and d[1] >= helmet_conf)
        n_heads = sum(1 for d in dets if d[0] == "head" and d[1] >= head_conf)
        n_persons = sum(1 for d in dets if d[0] == "person")

        gallery.append((annotated, f"{name} - {text}"))
        rows.append([name, n_helmets, n_heads, n_persons, text])

    summary = (
        f"**Przeanalizowano {len(files)} zdjęć:** "
        f"OK {counts['ok']} OK, "
        f"Brak kasku️ {counts['alert']} bez kasku, "
        f"Zweryfikuj {counts['check']} do sprawdzenia, "
        f"Nie wykryto nikogo {counts['none']} bez detekcji."
    )
    return gallery, rows, summary


#livecam

def process_frame(frame, det_conf, helmet_conf, head_conf):
    if frame is None:
        return None, ""
    annotated, _, _, text = analyze(frame, det_conf, helmet_conf, head_conf)
    return annotated, f"### {text}"

#interfejs

with gr.Blocks(title="Detekcja kasków", theme=gr.themes.Soft()) as demo:
    gr.Markdown("Detekcja kasków")

    with gr.Accordion("Ustawienia progów", open=False):
        det_conf = gr.Slider(
            0.05, 0.95, value=DEFAULT_DET_CONF, step=0.05,
            label="Minimalna pewność pokazywanych ramek",
        )
        helmet_conf = gr.Slider(
            0.05, 0.95, value=DEFAULT_HELMET_CONF, step=0.05,
            label="Pewność kasku wymagana do oceny „OK”",
        )
        head_conf = gr.Slider(
            0.05, 0.95, value=DEFAULT_HEAD_CONF, step=0.05,
            label="Pewność głowy uznawana za brak kasku",
        )

    with gr.Tabs():
        with gr.Tab("📁 Zdjęcia"):
            files = gr.File(
                label="Wgraj jedno lub wiele zdjęć",
                file_count="multiple",
                file_types=["image"],
            )
            run_btn = gr.Button("Analizuj", variant="primary")
            summary = gr.Markdown()
            gallery = gr.Gallery(label="Wyniki", columns=3, height="auto", object_fit="contain")
            table = gr.Dataframe(
                headers=["Plik", "Kaski", "Głowy bez kasku", "Osoby", "Ocena"],
                interactive=False,
                wrap=True,
            )
            run_btn.click(
                process_files,
                inputs=[files, det_conf, helmet_conf, head_conf],
                outputs=[gallery, table, summary],
            )

        with gr.Tab("📷 Kamera na żywo"):
            gr.Markdown("Kliknij ikonę kamery i zezwól przeglądarce na dostęp. Analiza działa w pętli.")
            with gr.Row():
                cam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Kamera",
                )
                live_out = gr.Image(label="Wynik", type="pil")
            live_status = gr.Markdown()
            cam.stream(
                process_frame,
                inputs=[cam, det_conf, helmet_conf, head_conf],
                outputs=[live_out, live_status],
                stream_every=0.5,
                time_limit=None,
                concurrency_limit=1,
            )

if __name__ == "__main__":
    demo.queue().launch()
