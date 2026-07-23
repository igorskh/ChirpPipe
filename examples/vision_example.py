import logging

from chirps.utils import init_default_logger
from chirps.image.crop_image import CropImage
from chirps.ml.libre_yolo_detection import LibreYoloDetection
from chirps.ml.bioclip_inference import BioClipInference

from PIL import Image, ImageDraw, ImageFont

import argparse

if __name__ == "__main__":
    init_default_logger()

    parser = argparse.ArgumentParser(
        description="Run LibreYOLO detection and BioClip inference on an image.")
    parser.add_argument("--image_path", type=str, default="demo/P7180276.jpeg",
                        help="Path to the input image file.")
    args = parser.parse_args()

    image_path = args.image_path

    yolo_detection = LibreYoloDetection()
    yolo_detection.configure({"model_path": "models/LibreYOLO9c.pt"})
    detection_res = yolo_detection.process(
        input=image_path)
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    class_names = yolo_detection.model.names
    cropped_images = {}
    n_birds = 0
    for i, b in enumerate(detection_res["detections"].boxes):
        if int(b.cls) == 14:
            xyxy = b.xyxy.tolist()[0]
            crop_image = CropImage()
            crop_image.configure({"out_postfix": f"cropped_{i}"})
            crop_res = crop_image.process(path=image_path, xyxy=xyxy)
            cropped_images[i] = {
                "output_path": crop_res["output_path"],
                "xyxy": xyxy
            }
            n_birds += 1
        else:
            cropped_images[i] = {
                "output_path": None,
                "xyxy": b.xyxy.tolist()[0]
            }

    if n_birds == 0:
        logging.info("No birds detected in the image.")
        exit(0)
    logging.info(
        f"Detected {n_birds} birds in the image. Running BioClip inference on cropped images.")

    for i, cropped in cropped_images.items():
        if cropped["output_path"] is None:
            continue
        bio_clip = BioClipInference()
        bio_clip_res = bio_clip.process(image_path=cropped["output_path"])
        cropped_images[i]["bio_clip_res"] = bio_clip_res["predictions"][0] if bio_clip_res["predictions"] else None
        print(bio_clip_res)


    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except:
        font = ImageFont.load_default()

    for i, b in enumerate(detection_res["detections"].boxes):
        cls = int(b.cls)
        has_bird_pred = "bio_clip_res" in cropped_images[
            i] and cropped_images[i]["bio_clip_res"] is not None
        color = "green" if has_bird_pred else "red"
        class_name = cropped_images[i]["bio_clip_res"]["common_name"] if has_bird_pred else class_names[cls]
        conf = float(b.conf)
        xyxy = b.xyxy.tolist()[0]
        label = f"{class_name}: {conf:.2f}"
        draw.rectangle(xyxy, outline=color, width=6)
        text_size = draw.textbbox((0, 0), label, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        draw.rectangle((xyxy[0], xyxy[1], xyxy[0] + text_width +
                       4, xyxy[1] + text_height + 4), fill=color)
        draw.text((xyxy[0] + 2, xyxy[1] + 2), label, fill="white", font=font)

    output_path = image_path.replace(
        ".jpeg", "_annotated.jpeg").replace(".jpg", "_annotated.jpg")
    img.save(output_path)
