import numpy as np
import matplotlib.pyplot as plt

priors = ["Fovea Blur", "Low-Res", "JPEG"]
encoders = ["RN50", "ViT-H-14", "ViT-g-14"]

in_top1 = np.array([
    [52.5, 40.4, 39.7],
    [47.5, 36.1, 33.8],
    [48.8, 34.8, 34.7],
], dtype=float)

in_top5 = np.array([
    [79.9, 71.7, 71.6],
    [78.5, 66.7, 64.6],
    [81.2, 34.7, 66.5],
], dtype=float)

joint_top1 = np.array([
    [46.3, 36.9, 36.1],
    [44.0, 29.6, 27.1],
    [46.2, 28.3, 30.3],
], dtype=float)

joint_top5 = np.array([
    [79.2, 71.4, 68.9],
    [76.0, 62.3, 60.4],
    [81.2, 64.2, 63.6],
], dtype=float)

# ====== Font sizes (tweak here) ======
FS_TITLE = 14       # subplot title ("Top-1 (%)")
FS_AXIS = 12        # axis labels ("Vision Encoders", "Priors")
FS_TICK = 9        # tick labels (encoders / priors)
FS_CBAR_LABEL = 9  # colorbar label
FS_CBAR_TICK = 9   # colorbar tick labels
FS_CELL = 11        # numbers inside each cell (smaller)

fig, axes = plt.subplots(
    1, 2, figsize=(9, 3.6),
    constrained_layout=True, sharey=True
)

for idx, (ax, mat, title) in enumerate(zip(axes, [in_top1, joint_top1], ["Intra-subject Top-1 accuracy(%)", "Joint-shubject Top-1 accuracy(%)"])):

    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_title(title, fontsize=FS_TITLE)

    ax.set_xticks(np.arange(len(encoders)))
    ax.set_yticks(np.arange(len(priors)))
    ax.set_xticklabels(encoders, rotation=30, ha="right", fontsize=FS_TICK)
    ax.set_xlabel("Vision Encoders", fontsize=FS_AXIS)

    # make ticks a bit more readable
    ax.tick_params(axis="both", which="major", labelsize=FS_TICK)

    if idx == 0:
        ax.set_yticklabels(priors, rotation=90, fontsize=FS_TICK)
        ax.set_ylabel("Priors", fontsize=FS_AXIS)
    else:
        ax.tick_params(axis="y", labelleft=False, left=False)

    # text in cells (smaller)
    thresh = (mat.min() + mat.max()) / 2.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] >= thresh else "black"
            ax.text(
                j, i, f"{mat[i, j]:.1f}",
                ha="center", va="center",
                color=color, fontsize=FS_CELL
            )

    # colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label("Performance (%)", fontsize=FS_CBAR_LABEL)
    cbar.ax.tick_params(labelsize=FS_CBAR_TICK)

fig.savefig("heatmap_top1_top5.png", dpi=600, bbox_inches="tight")
plt.show()
