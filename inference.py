import cv2
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


# inference script for huggingface space (assumes model is already loaded)
def inference(url, postprocess=False):
    response = requests.get(url)
    original_img = Image.open(BytesIO(response.content))
    img = np.array(original_img)
    
    img = cv2.resize(img, (512, 512))
    img = img / 255
    assert np.min(img) >= 0
    assert np.max(img) <= 1
    
    if len(img.shape) < 3: # grayscale coloring
        x = torch.Tensor(img)
        x = torch.stack([x, x, x], dim=0)
        x = normalize(x)
        x = x.unsqueeze(0)

    else: # RGB reconstruction
        x = torch.Tensor(img).permute(2, 0, 1)
        x = normalize(x)
        x_gs = cv2.cvtColor(x.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2GRAY)
        x_gs = np.dstack([x_gs, x_gs, x_gs])
        x = torch.Tensor(x_gs).permute(2, 0, 1).unsqueeze(0)
        
    pred = model(x)
    res = unnormalize(pred.squeeze(0))
    res = res.clamp(0, 1)
    res = res.permute(1, 2, 0).detach().cpu().numpy()
    
    colored_img = cv2.resize(res, original_img.size)
    colored_img = Image.fromarray((colored_img * 255).astype(np.uint8))
    
    if postprocess and len(img.shape) >= 3:
        colored_img = postprocess_img(original_img, colored_img)
        
    return colored_img, original_img, original_img.convert('L')


# torchvision transforms
normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)

unnormalize = transforms.Compose([
    transforms.Normalize(
        mean = [0., 0., 0.],
        std = [1/0.229, 1/0.224, 1/0.225]
    ),
    transforms.Normalize(
        mean = [-0.485, -0.456, -0.406],
        std = [1., 1., 1.]
    )
])