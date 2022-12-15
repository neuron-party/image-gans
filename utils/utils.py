from PIL import Image
import numpy as np
import cv2


def postprocess_img(original_img: Image.Image, colored_img: Image.Image) -> Image.Image:
    original_np, colored_np = np.array(original_img), np.array(colored_img)
    original_yuv = cv2.cvtColor(original_np, cv2.COLOR_BGR2YUV)
    predicted_yuv = cv2.cvtColor(colored_np, cv2.COLOR_BGR2YUV)

    processed_img = original_yuv.copy()
    processed_img[:, :, 1:] = predicted_yuv[:, :, 1:]
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_YUV2BGR)
    processed_img = Image.fromarray(processed_img)
    return processed_img