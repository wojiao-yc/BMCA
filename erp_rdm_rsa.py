import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_preparing.blureegdatasets_selected_avg import EEGDataset

# ========= 1) 统计/相关工具 =========

def _rank_ordinal(x: np.ndarray) -> np.ndarray:
    """
    Spearman 的 rank：用 ordinal rank（不处理 ties 的平均排名；RDM 向量 ties 很少，够用）
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks

def _pearsonr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y) + eps)
    return float(np.dot(x, y) / denom)

def spearmanr_fast(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rank_ordinal(a)
    rb = _rank_ordinal(b)
    return _pearsonr(ra, rb)

def partial_spearman_fast(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    partial Spearman: 先 rank 再做 partial Pearson
    r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
    """
    rx = _rank_ordinal(x)
    ry = _rank_ordinal(y)
    rz = _rank_ordinal(z)

    r_xy = _pearsonr(rx, ry)
    r_xz = _pearsonr(rx, rz)
    r_yz = _pearsonr(ry, rz)

    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2) + 1e-12)
    return float((r_xy - r_xz * r_yz) / denom)

# ========= 2) RDM 构建 =========

def rdm_vec_from_eeg(X: torch.Tensor, eps: float = 1e-6) -> np.ndarray:
    """
    EEG RDM：用 correlation distance = 1 - corr
    X: [N, F] (N=图片数, F=展开后的特征维)
    """
    # 行内 z-score
    X = X - X.mean(dim=1, keepdim=True)
    X = X / (X.std(dim=1, keepdim=True, unbiased=True) + eps)
    # corr = (X @ X.T) / (F - 1)
    Fdim = X.shape[1]
    corr = (X @ X.t()) / max(Fdim - 1, 1)
    rdm = 1.0 - corr
    N = rdm.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    return rdm[iu[0], iu[1]].cpu().numpy()

def rdm_vec_from_img(E: torch.Tensor) -> np.ndarray:
    """
    图像 RDM：用 cosine distance = 1 - cos
    E: [N, D]
    """
    E = F.normalize(E, dim=1)
    sim = E @ E.t()
    rdm = 1.0 - sim
    N = rdm.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    return rdm[iu[0], iu[1]].cpu().numpy()

# ========= 3) 读取 image pt（兼容 dict / tensor + 可选按重复平均） =========

def _sorted_keys(keys):
    # 尽可能按数字排序（如果 key 是 "0","1"... 或 0,1...）
    def key_fn(k):
        try:
            return int(k)
        except Exception:
            return str(k)
    return sorted(keys, key=key_fn)

def load_img_features_pt(pt_path: str, n_images: int = 200, repeats_hint: int = 80) -> torch.Tensor:
    """
    返回 per-image 的 embedding: [n_images, D]
    兼容两种存法：
      - saved['img_features'] 是 dict（key->tensor）
      - saved['img_features'] 是 tensor
    若第一维是 n_images * repeats（如 16000），则 reshape 后对 repeats 取均值。
    """
    saved = torch.load(pt_path, weights_only=False)
    img_feat = saved["img_features"]

    if isinstance(img_feat, dict):
        keys = _sorted_keys(list(img_feat.keys()))
        feat = torch.stack([img_feat[k] for k in keys], dim=0)
    elif torch.is_tensor(img_feat):
        feat = img_feat
    else:
        raise TypeError(f"Unsupported img_features type: {type(img_feat)}")

    feat = feat.float().cpu()
    M, D = feat.shape

    if M == n_images:
        return feat

    # 如果是 200*80 这种形式：按 repeats 平均成 per-image
    if M % n_images == 0:
        rep = M // n_images
        # 有时 rep=80；也可能是 1（已是 per-image）
        feat = feat.view(n_images, rep, D).mean(dim=1)
        return feat

    raise ValueError(f"Cannot reshape img features: first dim={M}, expected {n_images} or multiple of it.")

# ========= 4) 主流程：RSA + Partial RSA =========

def run_rsa(
    data_path: str,
    img_pt_orig: str,
    img_pt_blur: str,
    subjects=None,
    time_window=(0.0, 1.0),
    win_ms=50.0,
    step_ms=10.0,
    out_npz="rsa_raw_eeg_results.npz",
    plot_png=None,
):
    # --- 读取 EEG（test） ---
    ds = EEGDataset(data_path=data_path, subjects=subjects, train=False, avg=True, time_window=list(time_window))
    eeg = ds.data.float().cpu()       # [n_sub*200, C, T_sel]
    labels = ds.labels.cpu().numpy()  # [n_sub*200]
    n_sub = ds.n_sub
    n_images = ds.n_cls  # 200

    # 注意：ds.times 没被 extract_eeg 同步裁剪，这里按 time_window 重建 times_sel
    times_full = ds.times.cpu().numpy() if torch.is_tensor(ds.times) else np.array(ds.times)
    t0, t1 = time_window
    sel = (times_full >= t0) & (times_full <= t1)
    times_sel = times_full[sel]
    assert eeg.shape[-1] == len(times_sel), "Time axis mismatch: eeg last dim != len(times_sel)"

    # --- 读取图像 embedding（orig / blur）并做 per-image 对齐 ---
    img_orig = load_img_features_pt(img_pt_orig, n_images=n_images)
    img_blur = load_img_features_pt(img_pt_blur, n_images=n_images)

    # 图像 RDM 向量（固定，不随 subject / time 变）
    rdm_img_orig = rdm_vec_from_img(img_orig)
    rdm_img_blur = rdm_vec_from_img(img_blur)

    # --- 构造滑窗 ---
    win_s = win_ms / 1000.0
    step_s = step_ms / 1000.0
    starts = np.arange(times_sel[0], times_sel[-1] - win_s + 1e-9, step_s)
    centers = starts + win_s / 2.0

    # 保存：每个 subject × 每个时间窗
    rho_orig = np.zeros((n_sub, len(starts)), dtype=np.float64)
    rho_blur = np.zeros((n_sub, len(starts)), dtype=np.float64)
    prho_blur_given_orig = np.zeros((n_sub, len(starts)), dtype=np.float64)
    prho_orig_given_blur = np.zeros((n_sub, len(starts)), dtype=np.float64)

    # --- 按 subject 分块（你的 ds 是按 subject 依次 append 的，所以这里直接切片） ---
    for si in range(n_sub):
        s0 = si * n_images
        s1 = (si + 1) * n_images
        eeg_s = eeg[s0:s1]  # [200, C, T]
        lab_s = labels[s0:s1]
        # 确保标签是 0..199 的一一对应（如果不是，就按 label 重排）
        if not np.array_equal(lab_s, np.arange(n_images)):
            order = np.argsort(lab_s)
            eeg_s = eeg_s[order]

        # 每个时间窗计算 EEG-RDM，然后 RSA / partial RSA
        for wi, st in enumerate(starts):
            ed = st + win_s
            i0 = int(np.searchsorted(times_sel, st, side="left"))
            i1 = int(np.searchsorted(times_sel, ed, side="right"))
            if i1 <= i0 + 1:
                rho_orig[si, wi] = np.nan
                rho_blur[si, wi] = np.nan
                prho_blur_given_orig[si, wi] = np.nan
                prho_orig_given_blur[si, wi] = np.nan
                continue

            seg = eeg_s[:, :, i0:i1]             # [200, C, W]
            X = seg.reshape(n_images, -1)        # [200, F]
            rdm_eeg = rdm_vec_from_eeg(X)

            rho_orig[si, wi] = spearmanr_fast(rdm_eeg, rdm_img_orig)
            rho_blur[si, wi] = spearmanr_fast(rdm_eeg, rdm_img_blur)

            prho_blur_given_orig[si, wi] = partial_spearman_fast(rdm_eeg, rdm_img_blur, rdm_img_orig)
            prho_orig_given_blur[si, wi] = partial_spearman_fast(rdm_eeg, rdm_img_orig, rdm_img_blur)

        print(f"[Subject {si+1}/{n_sub}] done.")

    # --- 保存 ---
    np.savez(
        out_npz,
        times_centers=centers,
        rho_orig=rho_orig,
        rho_blur=rho_blur,
        prho_blur_given_orig=prho_blur_given_orig,
        prho_orig_given_blur=prho_orig_given_blur,
    )
    print(f"Saved: {out_npz}")

    # --- 可选画图（均值±bootstrap CI/或简单 mean±sem） ---
    if plot_png is not None:
        mean_orig = np.nanmean(rho_orig, axis=0)
        mean_blur = np.nanmean(rho_blur, axis=0)
        mean_p_blur = np.nanmean(prho_blur_given_orig, axis=0)
        mean_p_orig = np.nanmean(prho_orig_given_blur, axis=0)

        # 简单 SEM（够用；你也可以替换为 bootstrap CI）
        sem_orig = np.nanstd(rho_orig, axis=0) / np.sqrt(n_sub)
        sem_blur = np.nanstd(rho_blur, axis=0) / np.sqrt(n_sub)
        sem_p_blur = np.nanstd(prho_blur_given_orig, axis=0) / np.sqrt(n_sub)
        sem_p_orig = np.nanstd(prho_orig_given_blur, axis=0) / np.sqrt(n_sub)

        plt.figure(figsize=(10, 4))
        plt.plot(centers, mean_orig, label="RSA: EEG RDM vs Orig Img RDM")
        plt.fill_between(centers, mean_orig - sem_orig, mean_orig + sem_orig, alpha=0.2)

        plt.plot(centers, mean_blur, label="RSA: EEG RDM vs Blur Img RDM")
        plt.fill_between(centers, mean_blur - sem_blur, mean_blur + sem_blur, alpha=0.2)

        plt.xlabel("Time (s, window center)")
        plt.ylabel("Spearman corr")
        plt.title("Raw-EEG RSA (mean ± SEM across subjects)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_png, dpi=200)
        print(f"Saved plot: {plot_png}")

        plt.figure(figsize=(10, 4))
        plt.plot(centers, mean_p_blur, label="Partial RSA: EEG vs Blur | Orig")
        plt.fill_between(centers, mean_p_blur - sem_p_blur, mean_p_blur + sem_p_blur, alpha=0.2)

        plt.plot(centers, mean_p_orig, label="Partial RSA: EEG vs Orig | Blur")
        plt.fill_between(centers, mean_p_orig - sem_p_orig, mean_p_orig + sem_p_orig, alpha=0.2)

        plt.xlabel("Time (s, window center)")
        plt.ylabel("Partial Spearman corr")
        plt.title("Partial RSA (mean ± SEM across subjects)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_png.replace(".png", "_partial.png"), dpi=200)
        print(f"Saved plot: {plot_png.replace('.png', '_partial.png')}")

# ========= 5) CLI =========

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="THINGS-EEG preprocessed root path")
    parser.add_argument("--img_pt_orig", type=str, required=True, help="PT file with orig image CLIP features (test)")
    parser.add_argument("--img_pt_blur", type=str, required=True, help="PT file with blur image CLIP features (test)")
    parser.add_argument("--subjects", type=str, nargs="*", default=["sub-01"], help="Subject folder names (default: all)")
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--win_ms", type=float, default=50.0)
    parser.add_argument("--step_ms", type=float, default=10.0)
    parser.add_argument("--out_npz", type=str, default="rsa_raw_eeg_results.npz")
    parser.add_argument("--plot_png", type=str, default=None)

    args = parser.parse_args()

    run_rsa(
        data_path=args.data_path,
        img_pt_orig=args.img_pt_orig,
        img_pt_blur=args.img_pt_blur,
        subjects=args.subjects,
        time_window=(args.t0, args.t1),
        win_ms=args.win_ms,
        step_ms=args.step_ms,
        out_npz=args.out_npz,
        plot_png=args.plot_png,
    )
