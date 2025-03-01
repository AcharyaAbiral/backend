from django.apps import AppConfig
from ultralytics import YOLO
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
import torch
import transformers
from transformers import pipeline
# from deep_translator import GoogleTranslator


class ImageProcessingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'image_processing'

    def ready(self):
        # global yolo_model

        self.yolo_model = YOLO("yolo_wts.pt")
        print("yolo_model loaded")

        depth_model_config = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        encoder = 'vits'
        dataset = 'hypersim'
        max_depth = 14
        self.depth_model = DepthAnythingV2(
            **{**depth_model_config[encoder], 'max_depth': max_depth})
        self.depth_model.load_state_dict(torch.load(
            f'depth_anything_v2_metric_hypersim_vits.pth', map_location='cpu'))
        self.depth_model.eval()

        print("depth model loaded")

        self.image_to_text = pipeline(
            "image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

        print("capioning_model_ready")

        # self.translator = GoogleTranslator(source='en', target='ne')
        # print("translator ready")
