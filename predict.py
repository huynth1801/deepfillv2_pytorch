import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from options.test_options import TestOptions
from utils.utils import create_generator
from PIL import Image
from utils.utils import save_sample_png
import matplotlib.pyplot as plt


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
    mask = np.expand_dims(mask, axis=-1)
    # print("Mask: ", mask.shape)
    img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, (128,128), interpolation = cv2.INTER_AREA)
    # print(mask.shape)
    # h_raw, w_raw, _ = img.shape
    # h_t, w_t = h_raw // 8*8, w_raw // 8*8

    # img = img.resize((w_t, h_t))
    # img = np.array(img)
    # img = np.vstack(img).astype(np.float64)
    # print(img.shape)
    img = np.transpose(img, axes=[2, 0, 1])

    # mask_raw = np.array(mask)[..., None] > 0
    # mask = mask.resize((128, 128))
    # print(mask.shape)
    # mask = np.array(mask, dtype=np.float32)
    mask = torch.from_numpy(mask.astype(np.float32) / 255.0).contiguous()
    # mask = mask.permute(2, 0, 1)
    img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).contiguous()

    # img = (img / 255 - 0.5) / 0.5
    # img = img.unsqueeze(0)
    mask = mask[None, None]


    with torch.no_grad():
        first_out, second_out = model(img , mask)
    
    # second_out = torch.clamp(second_out, -1., 1.)
    # second_out = (second_out + 1) / 2 * 255.0
    # generated = generated.cpu().numpy().astype(np.uint8)
    print(second_out.shape)
    print(mask.shape)
    # generated = generated.transpose((1, 2, 0))
    result = img * (1 - mask) + second_out * mask
    print(img.shape)
    # result = result.permute(1, 2, 0).cpu().numpy()
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) 
    # cv2.imwrite('result.png', result)
    # img_list = [result]
    # name_list = ["second_out"]
    # save_sample_png(sample_folder = r"C:\Users\acer\deepfillv2_pytorch\outputs", sample_name = '%d' % (0 + 1), 
    #             img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

    result = result * 255
    # Process img_copy and do not destroy the data of img
    img_copy = result.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
    img_copy = np.clip(img_copy, 0, 255.0)
    img_copy = img_copy.astype(np.uint8)
    # Save to certain path
    save_img_name = '1' + '_' + 'second_out' + '.png'
    save_img_path = os.path.join(r"C:\Users\acer\deepfillv2_pytorch\outputs", save_img_name)
    print(img_copy.shape)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imsave("predict.png", img_copy)
    # cv2.imwrite(save_img_path, img_copy)
    print("DONE INPAINTING")
