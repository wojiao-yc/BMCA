import argparse
import os
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import open_clip
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from utils import instantiate_from_config, get_device

os.environ['http_proxy'] = 'http://127.0.0.1:7896'
os.environ['https_proxy'] = 'http://127.0.0.1:7896'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7897'

PRETRAIN_MAP = {
    "RN50": {"pretrained": "openai", "resize": (224, 224)},
    "RN101": {"pretrained": "openai", "resize": (224, 224)},
    "ViT-B-16": {"pretrained": "laion2b_s34b_b88k", "resize": (224, 224)},
    "ViT-B-32": {"pretrained": "laion2b_s34b_b79k", "resize": (224, 224)},
    "ViT-L-14": {"pretrained": "laion2b_s32b_b82k", "resize": (224, 224)},
    "ViT-H-14": {"pretrained": "laion2b_s32b_b79k", "resize": (224, 224)},
    "ViT-g-14": {"pretrained": "laion2b_s34b_b88k", "resize": (224, 224)},
    "ViT-bigG-14": {"pretrained": "laion2b_s39b_b160k", "resize": (224, 224)},
}


def load_img_text_from_pt(data_path: str, avg: bool) -> Tuple[List[str], List[str]]:
    loaded_data = torch.load(data_path, weights_only=False)
    images = np.array(loaded_data["img"])
    texts = np.array(loaded_data["text"])
    if avg:
        images = images[:, 0]
        texts = texts[:, 0]
    else:
        images = images.reshape(-1)
        texts = texts.reshape(-1)
    return images.tolist(), texts.tolist()


def build_process_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


@torch.no_grad()
def encode_images(
    images: Sequence[str],
    model,
    process_transform,
    blur_transform,
    image_root: str,
    device: str,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    set_images = list(set(images))
    set_images.sort()
    image_features_list: List[torch.Tensor] = []
    for start in tqdm(range(0, len(set_images), batch_size)):
        batch_images = set_images[start : start + batch_size]
        batch_tensors = []
        for rel_path in batch_images:
            img_path = os.path.join(image_root, rel_path)
            with Image.open(img_path).convert("RGB") as img:
                batch_tensors.append(process_transform(blur_transform(img)))
        image_inputs = torch.stack(batch_tensors).to(device)
        batch_features = model.encode_image(image_inputs)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        image_features_list.append(batch_features.detach())
    if not image_features_list:
        return {}
    image_features = torch.cat(image_features_list, dim=0)
    return {
        set_images[i]: image_features[i].float().cpu()
        for i in range(len(set_images))
    }


@torch.no_grad()
def encode_texts(
    texts: Sequence[str],
    model,
    device: str,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    set_texts = list(set(texts))
    set_texts.sort()
    text_features_list: List[torch.Tensor] = []
    for start in range(0, len(set_texts), batch_size):
        batch_texts = set_texts[start : start + batch_size]
        prompts = [f"This is a {t}." for t in batch_texts]
        text_inputs = open_clip.tokenize(prompts).to(device)
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_list.append(text_features.detach())
    if not text_features_list:
        return {}
    text_features = torch.cat(text_features_list, dim=0)
    return {set_texts[i]: text_features[i].float().cpu() for i in range(len(set_texts))}


def build_blur_transforms(config) -> object:
    blur_param = deepcopy(config["data"]["blur_type"])
    return instantiate_from_config(blur_param)


def apply_blur_overrides(config, args) -> None:
    overrides = (
        args.blur_kernel_size is not None
        or args.blur_h is not None
        or args.blur_w is not None
        or args.blur_curve is not None
        or args.blur_system_g is not None
        or args.blur_target is not None
    )
    if not overrides:
        return
    if "data" not in config or "blur_type" not in config["data"]:
        raise KeyError("Missing data.blur_type in config; cannot apply blur overrides.")
    blur_cfg = deepcopy(config["data"]["blur_type"])
    params = dict(blur_cfg.get("params") or {})
    if args.blur_kernel_size is not None:
        params["blur_kernel_size"] = args.blur_kernel_size
        if "blur_kernel_size" in config:
            config["blur_kernel_size"] = args.blur_kernel_size
    if args.blur_h is not None:
        params["h"] = args.blur_h
    if args.blur_w is not None:
        params["w"] = args.blur_w
    if args.blur_curve is not None:
        params["curve_type"] = args.blur_curve
    if args.blur_system_g is not None:
        params["system_g"] = args.blur_system_g
        if "system_g" in config:
            config["system_g"] = args.blur_system_g
    if args.blur_target is not None:
        blur_cfg["target"] = args.blur_target
    blur_cfg["params"] = params
    config["data"]["blur_type"] = blur_cfg


def resolve_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "auto":
        if torch.cuda.is_available():
            device_index = get_device("auto")
            return f"cuda:{device_index}"
        return "cpu"
    return device_arg


def infer_mode_from_path(data_path: str) -> str:
    name = os.path.basename(data_path).lower()
    if "train" in name:
        return "train"
    if "test" in name:
        return "test"
    return "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CLIP image/text features for EEG data."
    )
    parser.add_argument("--config", required=True, help="Path to config file.")
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default=None,
        help="Optional split name for default output naming.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to a .pt file that contains img/text arrays.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Override Image_set_Resize root.",
    )
    parser.add_argument(
        "--model-type",
        default="RN50",
        help="CLIP vision backbone (model_type).",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Override CLIP pretrained tag (defaults to model_type mapping).",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=128,
        help="Batch size for image encoding.",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=256,
        help="Batch size for text encoding.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda:0, etc.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pt file path (defaults to Image_feature/<blur>/<model_type>_<mode>.pt).",
    )
    parser.add_argument(
        "--blur-kernel-size",
        type=int,
        default=None,
        help="Override blur kernel size in config.",
    )
    parser.add_argument(
        "--blur-h",
        type=int,
        default=None,
        help="Override blur height in config.",
    )
    parser.add_argument(
        "--blur-w",
        type=int,
        default=None,
        help="Override blur width in config.",
    )
    parser.add_argument(
        "--blur-curve",
        default=None,
        help="Override blur curve_type in config (e.g., exp, linear).",
    )
    parser.add_argument(
        "--blur-system-g",
        type=float,
        default=None,
        help="Override blur system_g in config.",
    )
    parser.add_argument(
        "--blur-target",
        default=None,
        help="Override blur_type target in config.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config["data"]["model_type"] = args.model_type
    apply_blur_overrides(config, args)

    data_path = os.path.abspath(args.data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    mode = args.mode or infer_mode_from_path(data_path)
    avg_key = f"{mode}_avg"
    if avg_key not in config["data"]:
        raise KeyError(f"Missing {avg_key} in config data; pass --mode train/test.")
    avg = bool(config["data"][avg_key])
    images, texts = load_img_text_from_pt(data_path, avg)

    data_dir = os.path.dirname(os.path.dirname(data_path))
    image_root = args.image_root or os.path.join(data_dir, "..", "Image_set")
    image_root = os.path.abspath(image_root)

    if args.output:
        output_path = args.output
    else:
        blur_target = config["data"]["blur_type"]["target"].rsplit(".", 1)[-1]
        feature_root = os.path.join(data_dir, "..", "Image_feature", blur_target)
        os.makedirs(feature_root, exist_ok=True)
        output_path = os.path.join(feature_root, f"{args.model_type}_{mode}.pt")

    device = resolve_device(args.device)
    model_type = config["data"]["model_type"]
    if model_type not in PRETRAIN_MAP:
        raise ValueError(f"Unsupported model_type: {model_type}")

    pretrained = args.pretrained or PRETRAIN_MAP[model_type]["pretrained"]
    model, _, _ = open_clip.create_model_and_transforms(
        model_type,
        pretrained=pretrained,
        precision="fp32",
        device=device,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    process_transform = build_process_transform()
    blur_transform = build_blur_transforms(config)
    if config.get("data", {}).get("uncertainty_aware"):
        print("Note: uncertainty_aware is enabled in config, but using single blur transform.")
    img_features = encode_images(
        images,
        model,
        process_transform,
        blur_transform,
        image_root,
        device,
        args.image_batch_size,
    )

    text_features = encode_texts(
        texts,
        model,
        device,
        args.text_batch_size,
    )

    torch.save(
        {
            "text_features": text_features,
            "img_features": img_features,
        },
        output_path,
    )
    print(f"Saved features to: {output_path}")


if __name__ == "__main__":
    main()
