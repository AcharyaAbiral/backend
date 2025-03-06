from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil

from django.apps import apps
from deep_translator import GoogleTranslator


map = {
    1: 'व्यक्ति',         # person
    2: 'गाडी',          # car
    3: 'मोटरसाइकल',    # motorcycle
    4: 'बिरालो',        # cat
    5: 'कुकुर',        # dog
    6: 'झोला',        # backpack
    7: 'खेलकुद बल',   # sports ball
    8: 'बोतल',        # bottle
    9: 'कप',          # cup
    10: 'कुर्सी',       # chair
    11: 'सोफा',        # couch
    12: 'ओछ्यान',      # bed
    13: 'भान्साको टेबल',  # dining table
    14: 'शौचालय',      # toilet
    15: 'टिभी',       # tv
    16: 'ल्यापटप',     # laptop
    17: 'माउस',       # mouse
    18: 'किबोर्ड',     # keyboard
    19: 'मोबाइल फोन',  # cell phone
    20: 'रेफ्रिजिरेटर',  # refrigerator
}


def perform_object_detection(image_path):

    image_to_text = apps.get_app_config('image_processing').image_to_text
    caption = image_to_text(image_path)
    caption = caption[0]['generated_text']
    # translator = apps.get_app_config('image_processing').translator
    # caption = await translator.translate(caption, src='en', dest='ne')
    # caption = caption.text
    # print(caption)
    depth_model = apps.get_app_config('image_processing').depth_model
    raw_img = cv2.imread(image_path)
    depth = depth_model.infer_image(raw_img)
    print("depth model predicted")

    # yolo_model = YOLO("yolo_wts.pt")
    yolo_model = apps.get_app_config('image_processing').yolo_model
    with torch.no_grad():
        predictions = yolo_model.predict(source=image_path, save=False)
    print("yolo model predicted")
    output = []

    image = Image.open(image_path)
    img_array = np.array(image)

    sorted_predictions = []
    for result in predictions:
        boxes = result.boxes

    # Calculate area for each box
        areas = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)

    # Create a list of (index, area) pairs
        box_indices_with_areas = list(enumerate(areas))

    # Sort by area (ascending order)
        sorted_indices = [idx for idx, _ in sorted(
            box_indices_with_areas, key=lambda x: x[1])]

    # Create a new result with sorted boxes
        sorted_boxes = [boxes[i] for i in sorted_indices]

    # Store the sorted result
        sorted_result = result
        sorted_result.boxes = sorted_boxes
        sorted_predictions.append(sorted_result)

# Replace original predictions with sorted ones
    predictions = sorted_predictions
    print("done")
    for index, item in enumerate(predictions):
        boxes = item.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            confidence = box.conf[0].item()
            # class_id = int(box.cls[0].item())

            class_label = int(box.cls[0].item())
            if (class_label == 0):
                temp_img = img_array[y1:y2, x1:x2, :]
                temp_img = Image.fromarray(temp_img)
                temp_img_path = "media/temp_img.png"
                temp_img.save(temp_img_path)

                img = DeepFace.extract_faces(
                    temp_img_path, detector_backend="opencv", enforce_detection=False)

                for i in img:
                    # print(i)
                    person_name = "व्यक्ति"
                    if (i['confidence'] > 0.8 and (i['facial_area']['left_eye'] != None or i['facial_area']['right_eye'] != None)):

                        db_path = "db/"
                        num_files = len([f for f in os.listdir(
                            db_path) if os.path.isfile(os.path.join(db_path, f))])

                        if (num_files > 1):
                            recognition = DeepFace.find(
                                img_path=temp_img_path, db_path=db_path, detector_backend="opencv")
                            recognition = recognition[0]
                        # print(recognition)
                            if (len(recognition) != 0):
                                person_name = recognition['identity'][0][3:-4]
                                print(f"found {person_name}")
                            else:
                                pass
                        # do age and gender here only

                        # analysis = DeepFace.analyze(img_path=temp_img_path, actions=[
                        #     "age", "gender", "emotion"])
                        print("done")
                        # analysis = analysis[0]
                        class_label = person_name
                        # class_label = f"{person_name} age {analysis['age']} gender {analysis['dominant_gender']} emotion {analysis['dominant_emotion']}"
                        break
                    else:
                        class_label = "व्यक्ति"

                # os.remove(temp_img_path)

            else:
                # class_label = str(yolo_model.names[class_label])
                class_label = map[class_label+1]
            temp = {

                'box': box.xyxy[0].tolist(),
                'class': class_label,
                'distance': str(round(np.median(depth[y1:y2, x1:x2]), 2))
                # cdistance': str(np.median(depth[y1:y2, x1:x2]))

            }
            # print(str(round(np.median(depth[y1:y2, x1:x2])), 2))
            output.append(temp)
    print(caption)
    print(type(caption))
    caption = GoogleTranslator(
        source='en', target='ne').translate(caption)
    print(caption)
    # caption = caption.encode('utf-8').decode()
    # print(caption)
    return caption, output


# async def translate(input):
#     translator = apps.get_app_config('image_processing').translator
#     print(input)
#     output = await translator.translate(input, src='en', dest='ne')
#     print(output.text)
#     return output.text


def perform_face_registration(image_path, name):
    print(f"image_path={image_path}")
    img = DeepFace.extract_faces(
        image_path, detector_backend="opencv", enforce_detection=False)
    # print(img)
    for i in img:

        if (i['confidence'] > 0.8 and (i['facial_area']['left_eye'] != None or i['facial_area']['right_eye'] != None)):

            save_img = i['face']*255.0
            save_img = save_img.astype(np.uint8)

            # Image.fromarray(save_img).convert("RGB").save(f"db/{name}.png")
            # os.cop
            shutil.copy(image_path, f"db/{name}{image_path[-4:]}")

            # image.save(f"db/{name}.png")

            break
