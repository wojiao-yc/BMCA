import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_preparing.blureegdatasets_selected_quality_avg import EEGDataset
from model.EEG_MedformerTS import eeg_encoder


DEFAULT_RELIABILITY_K = [1, 2, 4, 8, 16, 40]
DEFAULT_RETRIEVAL_K = [1, 2, 4, 8, 16, 32, 80]


def parse_k_list(value, default):
    if value is None:
        return default
    items = value.replace(",", " ").split()
    return [int(x) for x in items if x]


def parse_subjects(value):
    if value is None or value.strip() == "":
        return [f"sub-{i:02d}" for i in range(1, 11)]
    return [v.strip() for v in value.split(",") if v.strip()]


def extract_done_k(curve):
    if not curve:
        return set()
    done = set()
    for item in curve:
        if isinstance(item, dict) and "k" in item:
            try:
                done.add(int(item["k"]))
            except (TypeError, ValueError):
                continue
    return done


def warn_resume_mismatch(existing, current, label):
    if not isinstance(existing, dict):
        return
    diffs = [k for k, v in current.items() if existing.get(k) != v]
    if diffs:
        joined = ", ".join(diffs)
        print(f"Warning: resume {label} differs for keys: {joined}")


def resolve_device(device_str):
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def load_encoder(model_path, device):
    if model_path is None:
        raise ValueError("model_path is required for embedding/retrieval experiments.")
    model = eeg_encoder().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")
    if any(k.startswith("brain.") for k in state.keys()):
        state = {k.replace("brain.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Warning: load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def collect_features(dataset, model, device, space, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    features = []
    with torch.no_grad():
        for eeg_data, _, _, _, _, _ in loader:
            if space == "embedding":
                eeg_data = eeg_data.to(device)
                out = model(eeg_data)
                features.append(out.detach().cpu())
            else:
                features.append(eeg_data.detach().cpu())
    return torch.cat(features, dim=0)


def batch_cosine_similarity(a, b):
    a = a.flatten(1)
    b = b.flatten(1)
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return (a * b).sum(dim=1)


def batch_pearson_corr(a, b, eps=1e-8):
    a = a.flatten(1)
    b = b.flatten(1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    denom = (a.norm(dim=1) * b.norm(dim=1)).clamp_min(eps)
    return (a * b).sum(dim=1) / denom


def summarize_ci(values, ci=0.95):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(arr, 100 * alpha)),
        "ci_high": float(np.percentile(arr, 100 * (1.0 - alpha))),
    }


def atomic_write_json(path, payload):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def save_results(output_path, results):
    if output_path:
        atomic_write_json(output_path, results)


def load_results(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_plot_dir(plot_dir, output_path):
    if plot_dir:
        return plot_dir
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            return output_dir
    return "."


def maybe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Matplotlib not available ({exc}); skipping plots.")
        return None
    return plt


def plot_reliability_curve(curve, output_path, metric, space, dpi):
    plt = maybe_import_matplotlib()
    if plt is None:
        return
    ks = [item["k"] for item in curve]
    means = [item["mean"] for item in curve]
    lows = [item["ci_low"] for item in curve]
    highs = [item["ci_high"] for item in curve]
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(ks, means, marker="o", color="#1f77b4", label="mean")
    plt.fill_between(ks, lows, highs, color="#1f77b4", alpha=0.2, label="CI")
    metric_label = "corr" if metric == "corr" else metric
    plt.xlabel("k (trials averaged)")
    plt.ylabel(f"{metric_label} similarity")
    plt.title(f"Reliability vs k ({space})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_retrieval_curve(curve, output_path, k_way, dpi):
    plt = maybe_import_matplotlib()
    if plt is None:
        return
    ks = [item["k"] for item in curve]
    means = [item["mean_acc"] for item in curve]
    stds = [item["std_acc"] for item in curve]
    plt.figure(figsize=(6.0, 4.0))
    plt.errorbar(ks, means, yerr=stds, fmt="-o", color="#ff7f0e", capsize=3)
    plt.xlabel("k (trials averaged)")
    plt.ylabel("Top-1 accuracy")
    plt.title(f"Retrieval accuracy vs k (k-way={k_way})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_from_results_file(results_path, plot_dir=None, dpi=150):
    results = load_results(results_path)
    plot_dir = resolve_plot_dir(plot_dir, results_path)
    os.makedirs(plot_dir, exist_ok=True)
    reliability = results.get("reliability")
    if reliability:
        curve = reliability.get("curve", [])
        if curve:
            output_path = os.path.join(plot_dir, "reliability_curve.png")
            plot_reliability_curve(
                curve,
                output_path,
                reliability.get("metric", "cosine"),
                reliability.get("space", "embedding"),
                dpi,
            )
    retrieval = results.get("retrieval")
    if retrieval:
        curve = retrieval.get("curve", [])
        if curve:
            output_path = os.path.join(plot_dir, "retrieval_curve.png")
            plot_retrieval_curve(curve, output_path, retrieval.get("k_way"), dpi)


def evaluate_retrieval(img_features_all, dataloader, model, device, k_way, rng):
    model.eval()
    img_features_all = img_features_all.to(device).float()
    num_classes = img_features_all.size(0)
    all_labels = list(range(num_classes))
    correct = 0
    total = 0
    with torch.no_grad():
        for eeg_data, labels, _, _, _, _ in dataloader:
            eeg_data = eeg_data.to(device)
            labels = labels.to(device)
            eeg_features = model(eeg_data)
            logit_scale = model.logit_scale
            for idx, label in enumerate(labels):
                label_int = int(label.item())
                if k_way is None or k_way >= num_classes:
                    selected_classes = all_labels
                else:
                    candidates = [c for c in all_labels if c != label_int]
                    selected_classes = rng.sample(candidates, k_way - 1) + [label_int]
                    rng.shuffle(selected_classes)
                selected_img_features = img_features_all[selected_classes]
                logits = logit_scale * (eeg_features[idx] @ selected_img_features.T)
                pred = selected_classes[int(torch.argmax(logits).item())]
                if pred == label_int:
                    correct += 1
                total += 1
    return correct / total if total else 0.0


def run_reliability(
    subjects,
    data_path,
    k_list,
    n_boot,
    base_seed,
    space,
    metric,
    model,
    device,
    batch_size,
    num_workers,
    dataset_kwargs,
    ci=0.95,
    curve=None,
    save_cb=None,
    log_prefix="[reliability]",
):
    metric_fn = batch_cosine_similarity if metric == "cosine" else batch_pearson_corr
    if curve is None:
        curve = []
    total_k = len(k_list)
    for k_idx, k in enumerate(k_list, 1):
        print(f"{log_prefix} k={k} ({k_idx}/{total_k}) start", flush=True)
        boot_scores = []
        for b in range(n_boot):
            print(f"{log_prefix} k={k} bootstrap {b + 1}/{n_boot}", flush=True)
            sub_scores = []
            for s_idx, sub in enumerate(subjects):
                seed_a = base_seed + b * 2 + s_idx * 1000
                seed_b = base_seed + b * 2 + s_idx * 1000 + 1
                ds_a = EEGDataset(
                    data_path,
                    subjects=[sub],
                    train=False,
                    test_avg_k=k,
                    test_avg_seed=seed_a,
                    **dataset_kwargs,
                )
                ds_b = EEGDataset(
                    data_path,
                    subjects=[sub],
                    train=False,
                    test_avg_k=k,
                    test_avg_seed=seed_b,
                    **dataset_kwargs,
                )
                feats_a = collect_features(ds_a, model, device, space, batch_size, num_workers)
                feats_b = collect_features(ds_b, model, device, space, batch_size, num_workers)
                sims = metric_fn(feats_a, feats_b)
                sub_scores.append(float(sims.mean().item()))
            boot_scores.append(float(np.mean(sub_scores)))
        stats = summarize_ci(boot_scores, ci=ci)
        stats["k"] = k
        curve.append(stats)
        print(
            f"{log_prefix} k={stats['k']}: mean={stats['mean']:.4f}, "
            f"ci=[{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]",
            flush=True,
        )
        if save_cb:
            save_cb()
    return curve


def run_retrieval(
    subjects,
    data_path,
    k_list,
    n_boot,
    base_seed,
    model,
    device,
    k_way,
    batch_size,
    num_workers,
    dataset_kwargs,
    curve=None,
    save_cb=None,
    log_prefix="[retrieval]",
):
    if curve is None:
        curve = []
    total_k = len(k_list)
    for k_idx, k_trials in enumerate(k_list, 1):
        print(f"{log_prefix} k_trials={k_trials} ({k_idx}/{total_k}) start", flush=True)
        boot_scores = []
        for b in range(n_boot):
            print(f"{log_prefix} k_trials={k_trials} bootstrap {b + 1}/{n_boot}", flush=True)
            sub_scores = []
            for s_idx, sub in enumerate(subjects):
                seed = base_seed + b + s_idx * 1000
                ds = EEGDataset(
                    data_path,
                    subjects=[sub],
                    train=False,
                    test_avg_k=k_trials,
                    test_avg_seed=seed,
                    **dataset_kwargs,
                )
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                rng = random.Random(base_seed + b * 997 + s_idx * 10007)
                acc = evaluate_retrieval(ds.img_features, loader, model, device, k_way, rng)
                sub_scores.append(float(acc))
            boot_scores.append(float(np.mean(sub_scores)))
        arr = np.asarray(boot_scores, dtype=np.float64)
        curve.append(
            {
                "k": k_trials,
                "mean_acc": float(arr.mean()),
                "std_acc": float(arr.std()),
                "var_acc": float(arr.var()),
            }
        )
        stats = curve[-1]
        print(
            f"{log_prefix} k_trials={stats['k']}: mean_acc={stats['mean_acc']:.4f}, "
            f"std={stats['std_acc']:.4f}",
            flush=True,
        )
        if save_cb:
            save_cb()
    return curve


def main():
    parser = argparse.ArgumentParser(description="Trial-averaging reliability and retrieval scaling.")
    parser.add_argument("--data-path", default="/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz")
    parser.add_argument("--model-path", default="/home/wenxiao/workspace/qhy/BMCA/data/blur+avgs+med/joint-subject/checkpoints/epoch=136-val_top1_acc=0.4950.ckpt")
    parser.add_argument("--subjects", default=None, help="Comma-separated subject list, e.g. sub-01,sub-02.")
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--space", choices=["erp", "embedding"], default="embedding")
    parser.add_argument("--metric", choices=["cosine", "corr", "correlation"], default="cosine")
    parser.add_argument("--reliability-k-list", default=None)
    parser.add_argument("--retrieval-k-list", default=None)
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--k-way", type=int, default=200)
    parser.add_argument("--run", choices=["both", "reliability", "retrieval"], default="both")
    parser.add_argument("--time-window-start", type=float, default=0.0)
    parser.add_argument("--time-window-end", type=float, default=1.0)
    parser.add_argument("--output", default="/home/wenxiao/workspace/qhy/BMCA/evaluate/trail_scaling.json", help="Optional path to save JSON results.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSON and skip completed k values.")
    parser.add_argument("--plot", action="store_true", help="Plot curves from saved results.")
    parser.add_argument("--plot-dir", default=None, help="Directory for plots (defaults to output dir).")
    parser.add_argument("--plot-dpi", type=int, default=150)
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    reliability_k_list = parse_k_list(args.reliability_k_list, DEFAULT_RELIABILITY_K)
    retrieval_k_list = parse_k_list(args.retrieval_k_list, DEFAULT_RETRIEVAL_K)
    device = resolve_device(args.device)

    dataset_kwargs = {
        "avg": True,
        "time_window": [args.time_window_start, args.time_window_end],
    }

    model = None
    if args.run in ("both", "retrieval") or (args.run in ("both", "reliability") and args.space == "embedding"):
        model = load_encoder(args.model_path, device)

    current_config = {
        "subjects": subjects,
        "data_path": args.data_path,
        "model_path": args.model_path,
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "time_window": [args.time_window_start, args.time_window_end],
    }
    results = None
    if args.resume and args.output and os.path.exists(args.output):
        try:
            results = load_results(args.output)
        except Exception as exc:
            print(f"Warning: failed to load {args.output} ({exc}); starting fresh.")
            results = None
    if not isinstance(results, dict):
        results = {"config": current_config}
    else:
        warn_resume_mismatch(results.get("config", {}), current_config, "config")
        results["config"] = current_config
    save_cb = lambda: save_results(args.output, results)
    save_cb()

    metric = "corr" if args.metric == "correlation" else args.metric
    if args.run in ("both", "reliability"):
        print(f"Running reliability curve (space={args.space}, metric={metric})")
        existing_curve = []
        if args.resume:
            existing_curve = results.get("reliability", {}).get("curve", [])
        results["reliability"] = {
            "space": args.space,
            "metric": metric,
            "k_list": reliability_k_list,
            "bootstrap": args.bootstrap,
            "curve": existing_curve,
        }
        save_cb()
        pending_k = reliability_k_list
        if args.resume:
            done_k = extract_done_k(existing_curve)
            pending_k = [k for k in reliability_k_list if k not in done_k]
        if not pending_k:
            print("[reliability] all requested k values already computed; skipping.")
        else:
            run_reliability(
                subjects=subjects,
                data_path=args.data_path,
                k_list=pending_k,
                n_boot=args.bootstrap,
                base_seed=args.seed,
                space=args.space,
                metric=metric,
                model=model,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dataset_kwargs=dataset_kwargs,
                curve=results["reliability"]["curve"],
                save_cb=save_cb,
            )

    if args.run in ("both", "retrieval"):
        print(f"Running retrieval scaling (k_way={args.k_way})")
        existing_curve = []
        if args.resume:
            existing_curve = results.get("retrieval", {}).get("curve", [])
        results["retrieval"] = {
            "k_list": retrieval_k_list,
            "bootstrap": args.bootstrap,
            "k_way": args.k_way,
            "curve": existing_curve,
        }
        save_cb()
        pending_k = retrieval_k_list
        if args.resume:
            done_k = extract_done_k(existing_curve)
            pending_k = [k for k in retrieval_k_list if k not in done_k]
        if not pending_k:
            print("[retrieval] all requested k values already computed; skipping.")
        else:
            run_retrieval(
                subjects=subjects,
                data_path=args.data_path,
                k_list=pending_k,
                n_boot=args.bootstrap,
                base_seed=args.seed + 10000,
                model=model,
                device=device,
                k_way=args.k_way,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                dataset_kwargs=dataset_kwargs,
                curve=results["retrieval"]["curve"],
                save_cb=save_cb,
            )
    save_cb()
    if args.output:
        print(f"Saved results to {args.output}")
    if args.plot:
        if not args.output:
            raise ValueError("--plot requires --output to be set.")
        plot_from_results_file(args.output, plot_dir=args.plot_dir, dpi=args.plot_dpi)
        plot_dir = resolve_plot_dir(args.plot_dir, args.output)
        print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
