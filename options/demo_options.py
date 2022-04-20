import argparse

from yaml import parse

parser = argparse.ArgumentParser(description="Image Inpainting")

#
parser.add_argument("--test_dir", type=str, default='./test_folder/img', help="image foldere for testing with application")
parser.add_argument("--pretrained", type=str, default='./save_models/free_form/deepfillv2_G_epoch40_batchsize16.pth', help="pretrained model of dataset celeb or places2")
# Folder to save outputs
parser.add_argument('--outputs', type=str, default='./outputs', help="path to save ouputs of demo")
# Options for paint
parser.add_argument('--thick', type=int, default=15, help='thick of pen for image inpainting task')
parser.add_argument('--painter', default='freeform', choices=('freeform', 'bbox'), help="painters")
# Network parameters
parser.add_argument('--in_channels', type=int, default=4, help='The input of RGB image + 1 channel of mask')
parser.add_argument('--out_channels', type=int, default=3, help='Output RGB image')
parser.add_argument('--latent_channels', type=int, default=48, help='Latent channels')
parser.add_argument('--pad_type', type=str, default='zero', help='Padding types: zero, reflect, replicate')
parser.add_argument('--activation', type=str, default='ELU', help='Activation types: ReLU, LeakyReLU, ELU, SELU, PReLU, Tanh, Sigmoid, none')
parser.add_argument('--norm', type=str, default = 'in', help = 'normalization type')
parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal|xavier|kaiming|orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='The initialized gain')

opt = parser.parse_args()
