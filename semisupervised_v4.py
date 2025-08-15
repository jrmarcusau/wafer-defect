#!/usr/bin/env python3
"""
semisup_pipeline_v4.py
──────────────────────
End-to-end wafer-clustering pipeline in **three clean stages** (v4 changes):

• No weak buckets (30–33 removed) — every kept image is assigned to **one of 30 clusters (0–29)**.
• Noise filter is stricter (lower NOISE_CONF) so **more unlabeled are routed to noise** and excluded.
• We still save noise rejects into <out_dir>/type_noise/ for auditability.

Stages
------

1.  **Duplicate removal**
    • Drops files in *Unlabeled/** whose basename already exists in any *Label_*/ folder (in-memory: originals stay on disk).

2.  **Noise filter**
    • Trains a light ResNet-18 binary head
         – Positive = all Label_* images
         – Negative = all Noise/ images
      (4-channel input: gray, Sobel-x, Coord-x, Coord-y) – 5 epochs.
    • Classifies the (deduplicated) *Unlabeled/** set; wafers with **p(noise) ≥ 0.80** are discarded from clustering and copied to type_noise/.

3.  **Semi-supervised dual-stream DEC**
    • Polar + Cartesian streams; input channels = gray + Sobel + Coord.
    • **Exactly 30 clusters** (no Weak buckets).
    • Loss = reconstruction + 0.1·KL(DEC) + 0.3·CE(seeds) + 0.05·prototype.
    • Early DEC (warm-up 2 epochs) and centroid refresh every 5 epochs.

Outputs (`<out_dir>/`)
    checkpoints/ figures/ summaries/{top9,random9}/ type_<id>/ type_noise/
    (id = 0–29)

Run
----
python semisup_pipeline_v4.py \
       --data_dir  data/Op3176_DefectMap_Labeled \
       --out_dir   outputs/semisup_v4
"""

# ───────────────────────── Imports ──────────────────────────
from __future__ import annotations

import argparse
import os
import random
import shutil
from glob import glob
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm

# ───────────────────── Global hyper-parameters ─────────────────────
IMG_SIZE = 256

# noise-head
CLS_BATCH = 64
CLS_EPOCHS = 5
NOISE_CONF = 0.90            # v4: lower threshold → more routed to noise

# main model
MAIN_BATCH = 32
WARMUP_EPOCHS = 2
TOTAL_EPOCHS = 100
UPDATE_INT = 5
TSNE_INT = 10

Z_STREAM = 256
Z_FUSED = 256
K_CLUSTERS = 30

GAMMA_DEC = 0.10
LAMBDA_CE = 0.30
LAMBDA_PROTO = 0.05

LR = 1e-3
SEED = 42
TOP_K, RAND_K = 9, 9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"

# new AMP API (PyTorch ≥ 2.1)
from torch.amp import GradScaler, autocast

SCALER = GradScaler(enabled=AMP)

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def to_four_channel(arr_u8: np.ndarray) -> torch.Tensor:
    """
    Convert an H×W uint8 image to a 4×H×W torch.Tensor:
      [gray, |sobel_x|, coord_x, coord_y], all normalized [0,1].
    """
    # 1) resize to IMG_SIZE
    gray = cv2.resize(arr_u8, (IMG_SIZE, IMG_SIZE), cv2.INTER_LINEAR)

    # 2) sobel-x
    sob = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sob = cv2.convertScaleAbs(np.abs(sob))

    # 3) coordinate grids
    coords = np.linspace(-1.0, 1.0, IMG_SIZE, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(coords, coords)  # both shape (IMG_SIZE, IMG_SIZE)

    # 4) stack and convert
    stacked = np.stack([
        gray.astype(np.float32) / 255.0,
        sob.astype(np.float32)  / 255.0,
        grid_x,
        grid_y
    ], axis=0)  # shape (4, IMG_SIZE, IMG_SIZE)

    return torch.from_numpy(stacked)


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# -----------------------------------------------------------------------------
# Stage 0 – build initial path lists (deduplicate by filename)
# -----------------------------------------------------------------------------
def collect_paths(root: str) -> Tuple[List[str], List[int], List[str], List[str], int]:
    """Return (seed_paths, seed_labels, unlabeled_paths, noise_paths, n_seed_classes)"""
    label_dirs = sorted(d for d in os.listdir(root) if d.startswith("Label_"))
    seed_paths, seed_labels, name_set = [], [], set()

    for idx, d in enumerate(label_dirs):
        for p in glob(os.path.join(root, d, "*.png")) + glob(
            os.path.join(root, d, "*.PNG")
        ):
            seed_paths.append(p)
            seed_labels.append(idx)
            name_set.add(os.path.basename(p))

    unlabeled = [
        p
        for p in glob(os.path.join(root, "Unlabeled", "*.png"))
        + glob(os.path.join(root, "Unlabeled", "*.PNG"))
        if os.path.basename(p) not in name_set
    ]

    noise = glob(os.path.join(root, "Noise", "*.png")) + glob(
        os.path.join(root, "Noise", "*.PNG")
    )

    return seed_paths, seed_labels, unlabeled, noise, len(label_dirs)


# -----------------------------------------------------------------------------
# Stage 1 – tiny ResNet 18 noise classifier
# -----------------------------------------------------------------------------
class NoiseDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int]) -> None:
        self.paths = paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        arr = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        return to_four_channel(arr), torch.tensor(self.labels[idx], dtype=torch.long)


def build_noise_head() -> nn.Module:
    net = resnet18(num_classes=2)
    net.conv1 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
    return net.to(DEVICE)


def train_noise_head(positive: List[str], negative: List[str]) -> nn.Module:
    dataset = NoiseDataset(positive + negative, [1] * len(positive) + [0] * len(negative))
    loader = DataLoader(dataset, CLS_BATCH, shuffle=True, num_workers=4, pin_memory=True)

    net = build_noise_head()
    optimizer = torch.optim.Adam(net.parameters(), 3e-4)
    criterion = nn.CrossEntropyLoss()

    net.train()
    for epoch in range(1, CLS_EPOCHS + 1):
        running = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type, enabled=AMP):
                loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"[NoiseHead] epoch {epoch}/{CLS_EPOCHS}  loss = {running / len(loader):.3f}")

    net.eval()
    return net


def filter_unlabeled(paths: List[str], net: nn.Module) -> List[str]:
    """Return paths classified as **non-noise** (prob_noise < NOISE_CONF)."""
    clean = []
    with torch.no_grad():
        for p in paths:
            arr = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            logits = net(to_four_channel(arr).unsqueeze(0).to(DEVICE))
            prob_noise = torch.softmax(logits, 1)[0, 1].item()
            if prob_noise < NOISE_CONF:
                clean.append(p)
    print(f"Noise filter kept {len(clean)} / {len(paths)} unlabeled wafers.")
    return clean


# -----------------------------------------------------------------------------
# Stage 2 – semi-supervised dual-stream dataset & model
# -----------------------------------------------------------------------------
class SemiDataset(Dataset):
    def __init__(
        self, seed_paths: List[str], seed_labels: List[int], unlabeled_paths: List[str]
    ) -> None:
        self.paths = seed_paths + unlabeled_paths
        self.is_seed = np.array([1] * len(seed_paths) + [0] * len(unlabeled_paths), bool)
        self.seed_labels = np.array(seed_labels + [-1] * len(unlabeled_paths))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        arr = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        x = to_four_channel(arr)
        return x, self.is_seed[idx], self.seed_labels[idx], idx, self.paths[idx]


class StreamEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256→128
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128→64
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64→32
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, Z_STREAM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StreamDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_STREAM, 128 * 32 * 32),
            nn.Unflatten(1, (128, 32, 32)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SemiDualNet(nn.Module):
    """Dual-stream AE + supervised head"""

    def __init__(self, n_seed_classes: int) -> None:
        super().__init__()
        self.enc_pol = StreamEncoder()
        self.enc_cart = StreamEncoder()
        self.dec_pol = StreamDecoder()
        self.dec_cart = StreamDecoder()

        self.fuse = nn.Sequential(
            nn.Linear(2 * Z_STREAM, 512),
            nn.ReLU(),
            nn.Linear(512, Z_FUSED),
        )
        self.head = nn.Linear(Z_FUSED, n_seed_classes)

    def forward(
        self, x_polar: torch.Tensor, x_cart: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_p = self.enc_pol(x_polar)
        z_c = self.enc_cart(x_cart)
        fused = self.fuse(torch.cat([z_c, z_p], 1))

        rec_p = self.dec_pol(z_p)
        rec_c = self.dec_cart(z_c)
        logits = self.head(fused)

        return fused, rec_p, rec_c, logits


# -----------------------------------------------------------------------------
# DEC helper functions
# -----------------------------------------------------------------------------
def soft_assign(z: torch.Tensor, centers: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Student-t similarity used in DEC."""
    dist2 = torch.cdist(z, centers).pow(2)
    q = (1.0 + dist2 / alpha).pow(-(alpha + 1) / 2)
    return q / q.sum(dim=1, keepdim=True)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    w = (q**2) / q.sum(0)
    return (w.t() / w.sum(1)).t()


# -----------------------------------------------------------------------------
# Training & diagnostics helpers
# -----------------------------------------------------------------------------
def save_loss_plot(
    out_dir: str, epochs, rec_p, rec_c, kl, ce
) -> None:
    plt.figure()
    plt.plot(epochs, rec_p, label="rec_polar")
    plt.plot(epochs, rec_c, label="rec_cart")
    plt.plot(epochs, kl, label="KL")
    plt.plot(epochs, ce, label="CE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "loss_curve.png"))
    plt.close()


def save_cluster_hist(out_dir: str, epoch: int, labels: np.ndarray) -> None:
    plt.figure()
    plt.bar(range(K_CLUSTERS), np.bincount(labels, minlength=K_CLUSTERS))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", f"hist_{epoch:03d}.png"))
    plt.close()


def save_tsne(
    out_dir: str, epoch: int, feats: np.ndarray, labels: np.ndarray
) -> None:
    try:
        tsne = TSNE(
            2,
            init="pca",
            learning_rate="auto",
            perplexity=min(30, max(5, len(labels) // 20)),
            random_state=SEED,
        ).fit_transform(feats)
        plt.figure(figsize=(6, 5))
        s = plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, s=6, cmap="tab20")
        plt.colorbar(s)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "figures", f"tsne_{epoch:03d}.png"))
        plt.close()
    except Exception as exc:
        print("[t-SNE warn]", exc)


def save_montage(img_paths: List[str], out_file: str, k: int) -> None:
    tf_resize = Resize((IMG_SIZE, IMG_SIZE))
    imgs = [ToTensor()(tf_resize(Image.open(p).convert("RGB"))) for p in img_paths]
    while len(imgs) < k:
        imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
    grid = vutils.make_grid(imgs, nrow=int(np.ceil(np.sqrt(k))), padding=2)
    vutils.save_image(grid, out_file)


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------
def train_dataset(root: str, out_dir: str) -> None:  # noqa: C901 (long but clear)
    # ------------------------------------------------------------------
    # 0.  Build lists & deduplicate
    seed_paths, seed_lbls, unl_paths, noise_paths, n_seed = collect_paths(root)

    # ------------------------------------------------------------------
    # 1.  Train noise head & filter unlabeled
    noise_net = train_noise_head(seed_paths, noise_paths)
    clean_unlabeled = filter_unlabeled(unl_paths, noise_net)

    #     Save noise rejects to type_noise/
    noise_rejects = list(set(unl_paths) - set(clean_unlabeled))
    if len(noise_rejects) > 0:
        noise_dir = os.path.join(out_dir, "type_noise")
        ensure_dirs(noise_dir)
        for p in noise_rejects:
            shutil.copy2(p, os.path.join(noise_dir, os.path.basename(p)))

    # ------------------------------------------------------------------
    # 2.  Semi-supervised DEC dataset & loader
    dataset = SemiDataset(seed_paths, seed_lbls, clean_unlabeled)
    loader = DataLoader(
        dataset,
        MAIN_BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    # model & optim
    model = SemiDualNet(n_seed).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), LR)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    # running prototypes for prototype loss
    prototypes = torch.zeros(n_seed, Z_FUSED, device=DEVICE)
    proto_count = torch.zeros(n_seed, device=DEVICE)

    # place-holders for DEC
    centers, target_q = None, None

    # logs
    epochs, rec_p_hist, rec_c_hist, kl_hist, ce_hist = [], [], [], [], []

    # make dirs
    ensure_dirs(
        out_dir,
        os.path.join(out_dir, "figures"),
        os.path.join(out_dir, "checkpoints"),
        os.path.join(out_dir, "summaries", "top9"),
        os.path.join(out_dir, "summaries", "random9"),
    )

    # ------------------------------------------------------------------
    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        rec_p_run = rec_c_run = kl_run = ce_run = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}", leave=False):
            x_all, is_seed, lbl_seed, idx_batch, _ = batch
            x_all = x_all.to(DEVICE)
            is_seed = is_seed.to(DEVICE)
            lbl_seed = lbl_seed.to(DEVICE)

            # Here both streams receive the same tensor (polar/cart = gray)
            x_polar = x_cart = x_all

            with autocast(device_type=DEVICE.type, enabled=AMP):
                z, rec_polar, rec_cart, logits = model(x_polar, x_cart)

                # reconstruction
                l_rec_p = mse_loss(rec_polar, x_polar[:, :1])
                l_rec_c = mse_loss(rec_cart, x_cart[:, :1])

                # supervised CE (seed only)
                mask = is_seed.bool()
                l_ce = (
                    ce_loss(logits[mask], lbl_seed[mask])
                    if mask.any()
                    else torch.zeros((), device=DEVICE)
                )

                # prototype loss (seed only)
                with torch.no_grad():
                    for cls in lbl_seed[mask]:
                        mask_cls = lbl_seed[mask] == cls
                        prototypes[cls] += z[mask][mask_cls].sum(0)
                        proto_count[cls] += mask_cls.sum()
                proto_norm = prototypes.clone()
                valid = proto_count > 0
                proto_norm[valid] = proto_norm[valid] / proto_count[valid].unsqueeze(1)
                l_proto = (
                    (z[mask] - proto_norm[lbl_seed[mask]]).pow(2).sum(1).mean()
                    if mask.any()
                    else torch.zeros((), device=DEVICE)
                )

                # DEC KL
                l_kl = torch.zeros((), device=DEVICE)
                if epoch > WARMUP_EPOCHS and centers is not None:
                    q_batch = soft_assign(z, centers)
                    p_batch = target_q[idx_batch].to(DEVICE)
                    l_kl = (p_batch * torch.log(p_batch / (q_batch + 1e-8))).sum(1).mean()

                loss = (
                    l_rec_p
                    + l_rec_c
                    + GAMMA_DEC * l_kl
                    + LAMBDA_CE * l_ce
                    + LAMBDA_PROTO * l_proto
                )

            optimizer.zero_grad(set_to_none=True)
            SCALER.scale(loss).backward()
            SCALER.step(optimizer)
            SCALER.update()

            rec_p_run += l_rec_p.item()
            rec_c_run += l_rec_c.item()
            kl_run += l_kl.item()
            ce_run += l_ce.item()
            n_batches += 1

        # ── epoch log
        rec_p_avg = rec_p_run / n_batches
        rec_c_avg = rec_c_run / n_batches
        kl_avg = kl_run / n_batches
        ce_avg = ce_run / n_batches

        print(
            f"Ep {epoch:03d}  recP {rec_p_avg:.4f}  recC {rec_c_avg:.4f}  "
            f"KL {kl_avg:.4f}  CE {ce_avg:.4f}"
        )

        epochs.append(epoch)
        rec_p_hist.append(rec_p_avg)
        rec_c_hist.append(rec_c_avg)
        kl_hist.append(kl_avg)
        ce_hist.append(ce_avg)

        # ─────────────────── DEC K-means refresh ────────────────────
        if epoch % UPDATE_INT == 0 or epoch == TOTAL_EPOCHS:
            model.eval()
            feats_list, path_list = [], []
            with torch.no_grad():
                for batch in DataLoader(dataset, MAIN_BATCH, False, num_workers=4):
                    x_all, _, _, _, pth = batch
                    x_all = x_all.to(DEVICE)
                    z, *_ = model(x_all, x_all)
                    feats_list.append(z.cpu())
                    path_list.extend(pth)
            feats = torch.cat(feats_list, dim=0).float()

            # K-means
            kmeans = KMeans(K_CLUSTERS, n_init=20, random_state=SEED).fit(feats)
            centers = torch.tensor(
                kmeans.cluster_centers_, device=DEVICE, dtype=torch.float
            )
            q_all = soft_assign(feats.to(DEVICE), centers).cpu()
            target_q = target_distribution(q_all)
            cluster_id = kmeans.labels_

            # ─ diagnostics ─
            save_loss_plot(out_dir, epochs, rec_p_hist, rec_c_hist, kl_hist, ce_hist)
            save_cluster_hist(out_dir, epoch, cluster_id)

            if epoch % TSNE_INT == 0:
                save_tsne(out_dir, epoch, feats.numpy(), cluster_id)

            # top9 / rand9 montages
            top_dir = os.path.join(out_dir, "summaries", "top9")
            rnd_dir = os.path.join(out_dir, "summaries", "random9")
            for k in range(K_CLUSTERS):
                idx = np.where(cluster_id == k)[0]
                if len(idx) == 0:
                    continue
                dist = np.linalg.norm(feats[idx].numpy() - centers[k].cpu().numpy(), axis=1)
                save_montage(
                    [path_list[i] for i in idx[np.argsort(dist)][:TOP_K]],
                    os.path.join(top_dir, f"cluster_{k}.png"),
                    TOP_K,
                )
                rnd_sel = np.random.choice(idx, min(len(idx), RAND_K), replace=False)
                save_montage(
                    [path_list[i] for i in rnd_sel],
                    os.path.join(rnd_dir, f"cluster_{k}.png"),
                    RAND_K,
                )

            # copy buckets (force every kept image into one of 0–29)
            for pth, lbl in zip(path_list, cluster_id):
                dst_dir = os.path.join(out_dir, f"type_{lbl}")
                ensure_dirs(dst_dir)
                shutil.copy2(pth, os.path.join(dst_dir, os.path.basename(pth)))

            # checkpoint
            ckpt_name = f"ckpt_{epoch:03d}.pth"
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "centers": centers.cpu()},
                os.path.join(out_dir, "checkpoints", ckpt_name),
            )

    print("✔ Training complete")


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/Op3176_DefectMap_Labeled')
    parser.add_argument("--out_dir",  default='outputs/sup_v4')
    args = parser.parse_args()

    train_dataset(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
