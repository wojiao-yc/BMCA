import os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import open_clip

os.environ['http_proxy'] = 'http://127.0.0.1:7896'
os.environ['https_proxy'] = 'http://127.0.0.1:7896'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7897'

# Initialize device (use CUDA if available)
device = "cuda:7" if torch.cuda.is_available() else "cpu"

# _, _, preprocess = open_clip.create_model_and_transforms(
#     model_name="ViT-H-14",
#     pretrained="laion2b_s32b_b79k",
#     precision='fp32'
# )

# Load configuration from YAML file
cfg = OmegaConf.load(os.path.join("/mnt/dataset1/ldy/Workspace/FLORA/configs/config.yaml"))
cfg = OmegaConf.structured(cfg)

# Get image directories from config
img_directory_training = cfg.eegdataset.img_directory_training
img_directory_test = cfg.eegdataset.img_directory_test

img_directory = img_directory_training 
# img_directory = img_directory_test
        
# Get all image folders and sort them
all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
all_folders.sort()

images = []  # Initialize images list
for folder in all_folders:
    folder_path = os.path.join(img_directory, folder)
    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_images.sort()  
    images.extend(os.path.join(folder_path, img) for img in all_images)
print(images[:10])  # Print first 10 image paths