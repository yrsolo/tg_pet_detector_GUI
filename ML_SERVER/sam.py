from functools import lru_cache

import torch
import numpy as np
import cv2
from PIL import Image
from markdown_it.rules_inline import image
from seaborn import histplot
from torchvision.transforms.v2.functional import crop_mask
from transformers import SamProcessor, SamModel, BitsAndBytesConfig
from matplotlib import pyplot as plt
import seaborn as sns

from IPython.display import display
from matplotlib import pyplot as plt

import numpy as np
import requests
from PIL import Image
import io

from utils import pic2float, pic2int, pic2pil, sigmoid, swimg

from utils import memo

# MODEL_NAME = "facebook/sam-vit-large"
MODEL_NAME = 'facebook/sam-vit-base'
DTYPE = torch.float16

from constant import device

def advanced_mask(logits, threshold=0.5, sigma=5, alpha=10):
    """
    Создает сложную маску с чёткими внутренними объектами и мягкими границами.

    :param logits: Логиты (numpy массив)
    :param threshold: Порог для бинаризации
    :param sigma: Параметр размытия для сглаживания границ
    :param alpha: Коэффициент крутизны для сигмоиды
    :return: Маска с плавными границами (numpy массив)
    """
    # 1. Бинаризация логитов
    sigmoid = 1 / (1 + np.exp(-logits))  # Преобразуем логиты в вероятности
    binary_mask = (sigmoid >= threshold).astype(np.float32)  # Бинарная маска

    # 2. Размытие бинарной маски для выделения границ
    blurred_binary = cv2.GaussianBlur(binary_mask, (0, 0), sigma)

    # Нормализация размытой маски (для диапазона 0-1)
    blurred_binary = 0.25 - (blurred_binary - 0.5) ** 2
    blurred_binary = cv2.GaussianBlur(blurred_binary, (0, 0), sigma)

    blurred_binary = blurred_binary / np.max(blurred_binary)
    blurred_binary = np.clip(blurred_binary, 0, 1)
    # print(blurred_binary.max(), blurred_binary.min())

    # return blurred_binary
    # 3. Применение размытой маски к сигмоиде
    alpha = 4
    soft_mask = 1 / (1 + np.exp(-alpha * (sigmoid - threshold)))  # Сигмоидная маска
    soft_mask = 1 * (sigmoid - 0.5) + 0.5

    # binary_mask = cv2.GaussianBlur(binary_mask, (0, 0), 1)

    final_mask = soft_mask * blurred_binary + binary_mask * (1 - blurred_binary)
    final_mask = np.clip(final_mask, 0, 1)

    return final_mask


def mask_crop(image, mask):
    coords = np.where(mask)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    return image[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]


def center(image, mask, shape=(512, 512), boudary=0.2):
    h, w = shape
    bh = int(h * (1 - boudary))
    bw = int(w * (1 - boudary))
    obj_h, obj_w, _ = image.shape
    scale = min(bh / obj_h, bw / obj_w)
    new_h, new_w = int(obj_h * scale), int(obj_w * scale)
    # display(pic2pil(image))
    if scale < 1:
        algo = cv2.INTER_AREA
    else:
        algo = cv2.INTER_AREA
        algo = cv2.INTER_LINEAR
        # algo = cv2.INTER_CUBIC
        # algo = cv2.INTER_LANCZOS4

    image = cv2.resize(image, (new_w, new_h), interpolation=algo)

    mask = cv2.resize(mask, (new_w, new_h))

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    new_mask = np.zeros((h, w, 3), dtype=np.float32)
    new_mask[top:top + new_h, left:left + new_w] = mask
    #
    new_image = np.zeros((h, w, 3), dtype=np.float32)
    new_image[top:top + new_h, left:left + new_w] = image

    return new_image, new_mask


class Predictor():
    def __init__(self, model=None, processor=None, device=None, model_name=MODEL_NAME, type=DTYPE):
        self.dtype = type

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,  # включаем 4-битную квантизацию
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4"  # можно попробовать и другие варианты, например 'fp4'
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if model is None:
            model = SamModel.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                quantization_config=quant_config
            ).to(self.device)
        if processor is None:
            processor = SamProcessor.from_pretrained(MODEL_NAME)

        self.model = model.to(self.device)
        self.processor = processor

    def predict(self, image, input_points=None, bbox=None):
        image = pic2float(image)

        if bbox:
            # bbox = [[int(coord) for coord in bbox]]
            bbox = [[bbox]]
            print(bbox)
            pass
        elif input_points is None:
            bbox = None
            input_points = [[[image.shape[1] // 2, image.shape[0] // 2]]]

        inputs = self.processor(
            image,
            input_points=input_points,
            input_boxes=bbox,
            return_tensors="pt", do_rescale=False).to(self.dtype).to("cuda")

        with torch.inference_mode():
            outputs = self.model(**inputs)

        scores = outputs.iou_scores[0][0].cpu().detach().numpy().astype('float')
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
            binarize=False
        )[0][0].cpu().numpy().astype('float')

        return scores, masks

    @staticmethod
    def best_masks(scores, masks, n=4):

        best_masks = []
        best_masks_indexex = np.argsort(scores)[::-1][:n]

        for idx in best_masks_indexex:
            mask, score = masks[idx], scores[idx]
            mask = advanced_mask(mask)
            mask = np.stack([mask, mask, mask], axis=-1)
            best_masks.append(mask)
        return best_masks


sam_predictor = Predictor()

@memo
def sam_process(image, text=None, bbox=None):
    image = pic2float(image)

    scores, masks = sam_predictor.predict(image, bbox=bbox)
    masks = sam_predictor.best_masks(scores, masks, 4)

    composes = []
    crop_masks = []

    for mask in masks:
        temp_image = image.copy()
        temp_image, mask = mask_crop(temp_image, mask)
        temp_image, mask = center(temp_image, mask)
        bg = np.ones_like(temp_image)
        # print(temp_image.shape, mask.shape, bg.shape)
        compose = temp_image * mask + (1 - mask) * bg

        composes.append(compose)
        crop_masks.append(mask)
        
    return composes, crop_masks, text

def test_sam():
    test_image = "../image.jpg"
    test_image = Image.open(test_image).convert("RGB")
    test_bbox = [ 5.8209e+01,  2.9224e+01,  2.9198e+02,  1.8345e+02]
    print(sam_process(test_image, bbox=test_bbox))

if __name__ == "__main__":
    test_sam()