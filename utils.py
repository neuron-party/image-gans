from PIL import Image
import numpy as np
import cv2


# collection of random utility functions (probably need to reformat this in the future)
def postprocess_img(original_img: Image.Image, colored_img: Image.Image) -> Image.Image:
    original_np, colored_np = np.array(original_img), np.array(colored_img)
    original_yuv = cv2.cvtColor(original_np, cv2.COLOR_BGR2YUV)
    predicted_yuv = cv2.cvtColor(colored_np, cv2.COLOR_BGR2YUV)

    processed_img = original_yuv.copy()
    processed_img[:, :, 1:] = predicted_yuv[:, :, 1:]
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_YUV2BGR)
    processed_img = Image.fromarray(processed_img)
    return processed_img


def create_gram_matrix(x, normalize=False):
    b, c, h, w = x.shape
    x, x_T = x.flatten(start_dim=2), x.flatten(start_dim=2).permute(0, 2, 1)
    gram_matrix = x @ x_T if not normalize else (x @ x_T) / (c * h * w) # normalization from perceptual loss paper
    return gram_matrix