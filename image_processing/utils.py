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


def perform_object_detection(image_path):

    depth_model_config = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vits'
    dataset = 'hypersim'
    max_depth = 10
    depth_model = DepthAnythingV2(
        **{**depth_model_config[encoder], 'max_depth': max_depth})
    depth_model.load_state_dict(torch.load(
        f'depth_anything_v2_metric_hypersim_vits.pth', map_location='cpu'))
    depth_model.eval()
    raw_img = cv2.imread(image_path)
    depth = depth_model.infer_image(raw_img)
    print("depth model predicted")

    yolo_model = YOLO("yolo_wts.pt")
    with torch.no_grad():
        predictions = yolo_model.predict(source=image_path, save=False)
    print("yolo model predicted")
    output = []

    image = Image.open(image_path)
    img_array = np.array(image)

    for index, item in enumerate(predictions):
        boxes = item.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # plt.subplot(1, len(predictions), index+1)

            # plt.imshow(img_array[y1:y2, x1:x2, :])

            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())

            class_label = int(box.cls[0].item())
            if (class_label == 0):
                temp_img = img_array[y1:y2, x1:x2, :]
                temp_img = Image.fromarray(temp_img)
                temp_img_path = "media/temp_img.png"
                temp_img.save(temp_img_path)

                img = DeepFace.extract_faces(
                    temp_img_path, detector_backend="opencv", enforce_detection=False)

                for i in img:
                    print(i)
                    person_name = "person"
                    if (i['confidence'] > 0.8 and (i['facial_area']['left_eye'] != None or i['facial_area']['right_eye'] != None)):

                        # if (img[0]['confidence'] > 0.8):
                        recognition = DeepFace.find(
                            img_path=temp_img_path, db_path="db/", detector_backend="opencv")
                        recognition = recognition[0]
                        print(recognition)
                        if (len(recognition) != 0):
                            person_name = recognition['identity'][0][3:-4]
                            # print(f"found {person_name}")
                        else:
                            pass
                        # do age and gender here only

                        analysis = DeepFace.analyze(img_path=temp_img_path, actions=[
                            "age", "gender", "emotion"])
                        print("done")
                        analysis = analysis[0]
                        class_label = f"{person_name} age {analysis['age']} gender {analysis['dominant_gender']} emotion {analysis['dominant_emotion']}"
                        break
                    else:
                        class_label = "person"
                # os.remove(temp_img_path)

            else:
                class_label = str(yolo_model.names[class_label])
            temp = {

                'box': box.xyxy[0].tolist(),
                'class': class_label,
                'distance': str(round(np.median(depth[y1:y2, x1:x2]), 2))
                # cdistance': str(np.median(depth[y1:y2, x1:x2]))

            }
            # print(str(round(np.median(depth[y1:y2, x1:x2])), 2))
            output.append(temp)
    plt.show()
    return output


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
