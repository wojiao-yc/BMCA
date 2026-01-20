import os
os.environ['http_proxy'] = 'http://127.0.0.1:7896'
os.environ['https_proxy'] = 'http://127.0.0.1:7896'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7897'

import argparse
import os
from pathlib import Path
import sys
from typing import List, Sequence

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from inpating_data import FoveaBlur
import open_clip


def build_text_descriptions(directory: str) -> List[str]:
    """Mimic dataset text extraction based on folder naming convention."""
    dirnames = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]
    dirnames.sort()

    texts: List[str] = []
    for dirname in dirnames:
        try:
            idx = dirname.index("_")
            description = dirname[idx + 1 :]
        except ValueError:
            print(f"Skipped text for folder '{dirname}' (missing '_').")
            continue
        texts.append(f"This picture is {description}")
    return texts


def build_image_list(directory: str) -> List[str]:
    """Replicate the image collection logic from the dataset."""
    all_folders = [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]
    all_folders.sort()

    images: List[str] = []
    for folder in all_folders:
        folder_path = os.path.join(directory, folder)
        all_images = [
            img
            for img in os.listdir(folder_path)
            if img.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        all_images.sort()
        images.extend(os.path.join(folder_path, img) for img in all_images)

    return images


def encode_texts(
    texts: Sequence[str],
    model,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Encode and normalize text prompts."""
    features: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        text_inputs = open_clip.tokenize(batch).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        features.append(F.normalize(text_features, dim=-1).detach())
    return torch.cat(features, dim=0) if features else torch.empty(0)


def encode_images(
    images: Sequence[str],
    model,
    preprocess,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    """Encode and normalize images in batches."""
    image_features: List[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch_paths = images[start : start + batch_size]
        batch_tensors = []
        for path in batch_paths:
            with Image.open(path).convert("RGB") as img:
                batch_tensors.append(preprocess(img))
        image_inputs = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(image_inputs)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        image_features.append(batch_features.detach())
    return torch.cat(image_features, dim=0) if image_features else torch.empty(0)

def infer_preprocess_size(preprocess, fallback: int) -> tuple[int, int]:
    if isinstance(preprocess, transforms.Compose):
        for t in preprocess.transforms:
            size = getattr(t, "size", None)
            if size is None:
                continue
            if isinstance(size, int):
                return (size, size)
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                return (int(size[0]), int(size[1]))
    return (fallback, fallback)


def build_preprocess_with_blur(preprocess, blur_transform):
    return lambda img: preprocess(blur_transform(img))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and save CLIP text/image features."
    )
    parser.add_argument(
        "--config",
        default="/mnt/dataset1/ldy/Workspace/FLORA/configs/config.yaml",
        help="Path to the OmegaConf config file.",
    )
    parser.add_argument(
        "--model-type",
        default="ViT-H-14",
        help="open_clip model identifier.",
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s32b_b79k",
        help="open_clip pretrained weights to load.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="test",
        help="Select training or test image directory from the config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to store the feature tensor (defaults to <model>_features_<mode>.pt).",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=256,
        help="Batch size for text encoding.",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=20,
        help="Batch size for image encoding (fits GPU memory).",
    )
    parser.add_argument(
        "--blur-radius",
        type=int,
        default=0,
        help="Apply FoveaBlur when > 0 (kernel size, should be odd).",
    )
    parser.add_argument(
        "--blur-size",
        type=int,
        default=224,
        help="Size (H=W) used to build the FoveaBlur mask if not inferred.",
    )
    parser.add_argument(
        "--blur-curve",
        default="exp",
        help="FoveaBlur curve type (e.g., exp, linear, quadratic, log).",
    )
    parser.add_argument(
        "--blur-system-g",
        type=float,
        default=3.0,
        help="system_g parameter for the exp curve.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--gpu",
        default="cuda:7",
        help="Force CPU even if CUDA is available.",
    )
    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available() and not args.cpu:
        device = args.gpu

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.structured(cfg)

    # img_directory = (
    #     cfg.eegdataset.img_directory_training
    #     if args.mode == "train"
    #     else cfg.eegdataset.img_directory_test
    # )

    img_directory = (
        "/home/wenxiao/workspace/qhy/BMCA/data/Image_set_Resize/train_images"
        if args.mode == "train"
        else "/home/wenxiao/workspace/qhy/BMCA/data/Image_set_Resize/test_images"
    )

    print(f"Collecting text and image metadata from: {img_directory}")
    texts = build_text_descriptions(img_directory)
    images = build_image_list(img_directory)
    print(f"Found {len(texts)} text prompts and {len(images)} image files.")

    model, _, _ = open_clip.create_model_and_transforms(
        args.model_type,
        pretrained=args.pretrained,
        precision="fp32",
        device=device,
    )
    process_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    preprocess = process_transform

    if args.blur_radius > 0:
        if args.blur_radius % 2 == 0:
            args.blur_radius += 1
            print(f"Adjusted blur radius to odd value: {args.blur_radius}")

        height, width = infer_preprocess_size(preprocess, args.blur_size)
        blur_transform = FoveaBlur(
            h=height,
            w=width,
            blur_kernel_size=args.blur_radius,
            curve_type=args.blur_curve,
            system_g=args.blur_system_g,
        )
        preprocess = build_preprocess_with_blur(preprocess, blur_transform)
        print(
            f"Enabled FoveaBlur: size=({height},{width}), radius={args.blur_radius}, curve={args.blur_curve}"
        )

    print("Encoding text features...")
    text_features = encode_texts(
        texts, model, device=device, batch_size=args.text_batch_size
    )
    print("Encoding image features...")
    image_features = encode_images(
        images,
        model,
        preprocess,
        device=device,
        batch_size=args.image_batch_size,
    )

    output_path = args.output or f"{args.model_type}_features_{args.blur_radius}_{args.mode}.pt"
    torch.save(
        {
            "text": texts,
            "images": images,
            "text_features": text_features.cpu(),
            "img_features": image_features.cpu(),
            "model_type": args.model_type,
            "pretrained": args.pretrained,
            "mode": args.mode,
        },
        output_path,
    )
    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    main()
