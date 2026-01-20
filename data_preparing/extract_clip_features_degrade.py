import argparse
import io
import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

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


def build_preprocess_with_transform(preprocess, image_transform):
    if isinstance(preprocess, transforms.Compose):
        new_transforms = []
        inserted = False
        for t in preprocess.transforms:
            if not inserted and isinstance(t, transforms.ToTensor):
                new_transforms.append(image_transform)
                inserted = True
            new_transforms.append(t)
        if not inserted:
            new_transforms.insert(0, image_transform)
        return transforms.Compose(new_transforms)
    return lambda img: preprocess(image_transform(img))


def _ensure_odd(value: int, minimum: int = 1) -> int:
    if value < minimum:
        value = minimum
    if value % 2 == 0:
        value += 1
    return value


def apply_lowres(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    down_w = max(1, int(round(w * scale)))
    down_h = max(1, int(round(h * scale)))
    img_small = img.resize((down_w, down_h), Image.BICUBIC)
    return img_small.resize((w, h), Image.BICUBIC)


def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _center_rect(w: int, h: int, scale: float) -> Tuple[int, int, int, int]:
    rect_w = max(1, int(round(w * scale)))
    rect_h = max(1, int(round(h * scale)))
    x = max(0, (w - rect_w) // 2)
    y = max(0, (h - rect_h) // 2)
    return x, y, rect_w, rect_h


def _soft_mask(mask: np.ndarray, blur_ksize: int) -> np.ndarray:
    if blur_ksize <= 1:
        return mask.astype(np.float32)
    blur_ksize = _ensure_odd(blur_ksize, minimum=3)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (blur_ksize, blur_ksize), 0)
    if blurred.max() > 0:
        blurred = blurred / blurred.max()
    return blurred


def apply_subject_blur(
    img: Image.Image,
    blur_kernel_size: int,
    rect_scale: float,
    mask_blur: int,
    iterations: int,
) -> Image.Image:
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    blur_kernel_size = _ensure_odd(blur_kernel_size, minimum=3)

    mask = np.zeros((h, w), np.uint8)
    rect = _center_rect(w, h, rect_scale)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            bgr,
            mask,
            rect,
            bgd_model,
            fgd_model,
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )
        fg_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
        ).astype(np.uint8)
    except cv2.error:
        fg_mask = np.zeros((h, w), np.uint8)
        center = (w // 2, h // 2)
        axes = (max(1, int(w * rect_scale / 2)), max(1, int(h * rect_scale / 2)))
        cv2.ellipse(fg_mask, center, axes, 0, 0, 360, 1, -1)

    soft_mask = _soft_mask(fg_mask, mask_blur)
    soft_mask_3 = np.repeat(soft_mask[:, :, None], 3, axis=2)

    blurred = cv2.GaussianBlur(bgr, (blur_kernel_size, blur_kernel_size), 0)
    blended = bgr * soft_mask_3 + blurred * (1 - soft_mask_3)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and save CLIP text/image features with degradations."
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
        help="Where to store the feature tensor (defaults to <model>_<method>_<mode>.pt).",
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
        "--method",
        choices=["lowres", "jpeg", "subject"],
        default="lowres",
        help="Degradation method: downsample+upsample, JPEG compression, or subject-aware blur.",
    )
    parser.add_argument(
        "--lowres-scale",
        type=float,
        default=0.5,
        help="Downsample scale for low-res (0-1, then upsample back).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=50,
        help="JPEG quality for compression (1-95).",
    )
    parser.add_argument(
        "--subject-blur-radius",
        type=int,
        default=21,
        help="Gaussian blur kernel size for background (odd).",
    )
    parser.add_argument(
        "--subject-rect-scale",
        type=float,
        default=0.6,
        help="Center rectangle scale for GrabCut (0-1).",
    )
    parser.add_argument(
        "--subject-mask-blur",
        type=int,
        default=11,
        help="Gaussian blur kernel size for soft mask (odd).",
    )
    parser.add_argument(
        "--subject-iter",
        type=int,
        default=5,
        help="GrabCut iterations for subject extraction.",
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

    img_directory = (
        cfg.eegdataset.img_directory_training
        if args.mode == "train"
        else cfg.eegdataset.img_directory_test
    )

    print(f"Collecting text and image metadata from: {img_directory}")
    texts = build_text_descriptions(img_directory)
    images = build_image_list(img_directory)
    print(f"Found {len(texts)} text prompts and {len(images)} image files.")

    model, preprocess, _ = open_clip.create_model_and_transforms(
        args.model_type,
        pretrained=args.pretrained,
        precision="fp32",
        device=device,
    )

    if args.method == "lowres":
        def lowres_transform(img: Image.Image) -> Image.Image:
            return apply_lowres(img, args.lowres_scale)

        preprocess = build_preprocess_with_transform(preprocess, lowres_transform)
        print(f"Enabled low-res: scale={args.lowres_scale}")
    elif args.method == "jpeg":
        def jpeg_transform(img: Image.Image) -> Image.Image:
            return apply_jpeg(img, args.jpeg_quality)

        preprocess = build_preprocess_with_transform(preprocess, jpeg_transform)
        print(f"Enabled JPEG compression: quality={args.jpeg_quality}")
    elif args.method == "subject":
        def subject_transform(img: Image.Image) -> Image.Image:
            return apply_subject_blur(
                img,
                blur_kernel_size=args.subject_blur_radius,
                rect_scale=args.subject_rect_scale,
                mask_blur=args.subject_mask_blur,
                iterations=args.subject_iter,
            )

        preprocess = build_preprocess_with_transform(preprocess, subject_transform)
        print(
            "Enabled subject-aware blur: "
            f"radius={args.subject_blur_radius}, rect_scale={args.subject_rect_scale}"
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

    output_path = args.output or f"{args.model_type}_{args.method}_{args.mode}.pt"
    torch.save(
        {
            "text": texts,
            "images": images,
            "text_features": text_features.cpu(),
            "img_features": image_features.cpu(),
            "model_type": args.model_type,
            "pretrained": args.pretrained,
            "mode": args.mode,
            "method": args.method,
        },
        output_path,
    )
    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    main()
