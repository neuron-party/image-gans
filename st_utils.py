# utils for style transfers
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import tensorflow as tf


def compute_layer_masks(mask, layers, mask_net):
    '''
    Masks: read-in (using PIL or something) numpy array of shape [h, w, c]
    '''
    masks_tf = mask.transpose(2, 0, 1).transpose([1, 2, 0]) # [h, w, masks]
    masks_tf = masks_tf[np.newaxis, :, :, :]
    
    mask_dict = {}
    with tf.Session() as sess:
        for layer in layers:
            out = sess.run(mask_net[layer], feed_dict={input: masks_tf})
            mask_dict[layer] = out[0].transpose([2, 0, 1])
    
    return mask_dict


def masked_gram_matrix(feature_map, masked_feature_map):
    '''
    Compute the masked Gram Matrix (torch)
    Feature_map: Tensor of shape [b, c, h, w]
    Masked_feature_map: Tensor of shape [c, h, w]
    '''
    c, h, w = masked_feature_map.shape
    b, N, _, _ = feature_map.shape
    masked_feature_map = masked_feature_map.reshape(c, h * w)
    feature_map = feature_map.view(N, h * w)
    masked_gram = []
    for i in range(c):
        mask = masked_feature_map[i]
        masked_x = feature_map * mask
        K = 1. / torch.sum(mask ** 2)
        gram = K * (masked_x @ masked_x.T)
        masked_gram.append(gram)
        
    return torch.stack(masked_gram)


def gram_matrix_torch(x):
    b, c, h, w = x.shape
    x = x.view(h * w, -1)
    gram = (x.T @ x) / (h * w)
    return gram