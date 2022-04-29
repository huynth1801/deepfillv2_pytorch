import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from options.test_options import TestOptions
from utils.utils import create_generator
from PIL import Image

# Options
opt = TestOptions().parse()

def load_model(generator, epoch, opt):
    pre_dict = torch.load(opt.load_name, map_location='cpu')
    generator.load_state_dict(pre_dict)

    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)

# Load  moel generator
model = create_generator(opt).eval()
load_model(model, opt.epochs, opt)
print("Load pretrained model generator")


def predict(img, mask, opt=None):
    # img = img.convert('RGB')
    # img_raw = np.array(img)
    img = cv2.imread(img)
    mask = cv2.imread(mask, 0)
    img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, (128,128), interpolation = cv2.INTER_AREA)
    # print("Mask: ", mask.shape)
    # print(mask.shape)
    h_raw, w_raw, _ = img.shape
    h_t, w_t = h_raw // 8*8, w_raw // 8*8

    # img = img.resize((w_t, h_t))
    # img = np.array(img)
    # img = np.vstack(img).astype(np.float64)
    # print(img.shape)
    img = np.transpose(img, axes=[2, 0, 1])

    # mask_raw = np.array(mask)[..., None] > 0
    # mask = mask.resize((128, 128))
    # print(mask.shape)
    # mask = np.array(mask, dtype=np.float32)
    mask = torch.from_numpy(mask)
    # mask = mask.permute(2, 0, 1)
    img = torch.from_numpy(img)

    img = (img / 255 - 0.5) / 0.5
    img = img[None]
    mask = mask[None, None]


    with torch.no_grad():
        generated, second_out = model(img , mask)
    
    generated = torch.clamp(generated, -1., 1.)
    generated = (generated + 1) / 2 * 255.0
    generated = generated.cpu().nupy().astype(np.uint8)
    generated = generated[0].permute((1, 2, 0))
    result = generated * mask + img * (1 - mask)
    result = result.astype(np.unit8)
    print("DONE INPAINTING")
