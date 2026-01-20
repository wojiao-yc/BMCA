import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
SRC_DIRS = {
    'training_images': '/mnt/dataset1/ldy/4090_Workspace/4090_THINGS/images_set/training_images',
    'test_images': '/mnt/dataset1/ldy/4090_Workspace/4090_THINGS/images_set/test_images',
}
SAVE_ROOT = '/home/wenxiao/workspace/qhy/BMCA/data/Image_set_Resize'

t1 = transforms.Resize((224,224))

for split_name, data_dir in SRC_DIRS.items():
    save_dir = os.path.join(SAVE_ROOT, split_name)
    os.makedirs(save_dir, exist_ok=True)
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))

    for path in tqdm(image_paths, desc=split_name, unit="img"):
        img = Image.open(path)
        img = t1(img)
        rel_path = os.path.relpath(path, data_dir)
        save_path = os.path.join(save_dir, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
