#!/usr/bin/env python3
"""
infer_v8_256_memsafe_diag.py
———————————
Inference + diagnostics for the Conv-AE + DEC model trained with
main_v8_256_memsafe.py.

Outputs
-------
<output_dir>/
    type_<k>/               # copies of original wafer PNGs by cluster
    summaries/
        top9/cluster_<k>.png
        random9/cluster_<k>.png
    diagnostics/
        cluster_sizes.png
        confidence_hist.png
        tsne_feats.png      (needs scikit-learn)

Usage
-----
python infer_v8_256_memsafe_diag.py \
       --input_dir  data/Op3176_DefectMap \
       --checkpoint outputs/v8_256_memsafe/checkpoints/ckpt_099.pth \
       --output_dir outputs/inference_run \
       --batch 64
"""

import os, argparse, shutil, random, cv2, numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from skimage.feature import hog
from sklearn.manifold import TSNE     # t-SNE

# ────────────────── constants (match training) ────────────────────
IMG_SIZE = 256
Z_DIM    = 256
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED     = 42

# ───────────────────── dataset (same preprocessing) ───────────────
class WaferDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted(glob(os.path.join(root, "*.png")) +
                            glob(os.path.join(root, "*.PNG")))
        if not self.paths:
            raise RuntimeError(f"No images in {root}")

        dummy = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
        self.hog_dim = hog(dummy, orientations=9,
                           pixels_per_cell=(16,16),
                           cells_per_block=(1,1),
                           feature_vector=True).shape[0]

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        arr  = np.array(Image.open(path).convert("L"))

        polar = cv2.warpPolar(arr, (IMG_SIZE,IMG_SIZE),
                              (arr.shape[1]//2, arr.shape[0]//2),
                              maxRadius=arr.shape[0]//2,
                              flags=cv2.WARP_POLAR_LINEAR)
        polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
        edge  = cv2.Canny(polar, 50, 150)

        hog_vec = hog(polar, orientations=9,
                      pixels_per_cell=(16,16),
                      cells_per_block=(1,1),
                      feature_vector=True).astype(np.float32)

        p = torch.from_numpy(polar/255.).unsqueeze(0).float()
        e = torch.from_numpy(edge /255.).unsqueeze(0).float()
        xs = torch.linspace(-1,1,IMG_SIZE).view(1,1,-1).expand(1,IMG_SIZE,IMG_SIZE)
        ys = torch.linspace(-1,1,IMG_SIZE).view(1,-1,1).expand(1,IMG_SIZE,IMG_SIZE)
        coord = torch.cat([xs,ys],0)
        x = torch.cat([p,e,coord],0)                 # 4×256×256
        return x, torch.from_numpy(hog_vec), path

# ─────────────────────── model (encoder only) ─────────────────────
class ConvAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                        # 256→128
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                        # 128→64
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                        #  64→32
            nn.Flatten(),
            nn.Linear(128*32*32, z_dim),
        )
    def forward(self,x): return self.encoder(x)

# ────────────────────── helper functions ──────────────────────────
def soft_assign(feats, centers, alpha=1.0):
    centers = centers.to(feats.dtype)
    d2 = torch.cdist(feats, centers)**2
    q  = (1 + d2/alpha).pow(-(alpha+1)/2)
    return q / q.sum(1, keepdim=True)

# ─────────────────────────── main() ───────────────────────────────
@torch.no_grad()
def main(args):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    # 1) load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    centers = ckpt.get("centers")          # training saved this key
    if centers is None:
        centers = ckpt.get("cluster_centers")
    if centers is None:
        raise RuntimeError("No 'centers' key in checkpoint")
    centers = centers.to(DEVICE)
    K = centers.size(0)

    # 2) data
    ds = WaferDataset(args.input_dir)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=4, pin_memory=torch.cuda.is_available())

    # 3) model
    model = ConvAE(Z_DIM).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)   # ignore decoder weights
    model.eval()

    # 4) inference
    feats, paths = [], []
    for x, hog_vec, pth in tqdm(loader, desc="Encoding"):
        x, hog_vec = x.to(DEVICE, non_blocking=True), hog_vec.to(DEVICE, non_blocking=True)
        z = model(x)
        feats.append(torch.cat([z, hog_vec],1).cpu())
        paths.extend(pth)
    feats = torch.cat(feats)
    q = soft_assign(feats.to(DEVICE), centers).cpu().numpy()
    cluster = np.argmax(q,1)
    conf    = np.max(q,1)

    # after computing `cluster` array
    if args.save_assignments:
        np.save(os.path.join(args.output_dir, "assignments.npy"), cluster)
        print("✓ Saved assignments.npy")

    # 5) make dirs
    for k in range(K):
        os.makedirs(os.path.join(args.output_dir, f"type_{k}"), exist_ok=True)
    sum_top = os.path.join(args.output_dir, "summaries", "top9");   os.makedirs(sum_top,  exist_ok=True)
    sum_rnd = os.path.join(args.output_dir, "summaries", "random9");os.makedirs(sum_rnd,  exist_ok=True)
    diag_dir = os.path.join(args.output_dir, "diagnostics");        os.makedirs(diag_dir, exist_ok=True)

    # 6) copy wafers
    for src,cid in zip(paths, cluster):
        shutil.copy2(src, os.path.join(args.output_dir, f"type_{cid}", os.path.basename(src)))

    # 7) montage helpers
    tf_resize = T.Resize((IMG_SIZE,IMG_SIZE))
    def montage(sel, k, outdir):
        imgs=[T.ToTensor()(tf_resize(Image.open(paths[i]).convert("RGB"))) for i in sel]
        while len(imgs)<9: imgs.append(torch.zeros(3,IMG_SIZE,IMG_SIZE))
        grid=vutils.make_grid(imgs,nrow=3,padding=2)
        vutils.save_image(grid, os.path.join(outdir, f"cluster_{k}.png"))

    # 8) summaries
    for k in range(K):
        idx=np.where(cluster==k)[0]
        if len(idx)==0: continue
        top = idx[np.argsort(-conf[idx])][:9]
        montage(top, k, sum_top)
        rnd = np.random.choice(idx, min(len(idx),9), replace=False)
        montage(rnd, k, sum_rnd)

    # 9) diagnostics ────────────────────────────────────────────────
    # 9a) cluster sizes
    counts = np.bincount(cluster, minlength=K)
    plt.figure(); plt.bar(range(K), counts)
    plt.xlabel("cluster"); plt.ylabel("count"); plt.title("Cluster sizes")
    plt.tight_layout(); plt.savefig(os.path.join(diag_dir, "cluster_sizes.png")); plt.close()

    # 9b) confidence histogram
    plt.figure(); plt.hist(conf, bins=30, range=(0,1))
    plt.xlabel("max soft-assignment"); plt.ylabel("freq"); plt.title("Confidence histogram")
    plt.tight_layout(); plt.savefig(os.path.join(diag_dir, "confidence_hist.png")); plt.close()

    # 9c) t-SNE
    try:
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                    perplexity=min(30, max(5, len(paths)//20)),
                    random_state=SEED)
        emb2 = tsne.fit_transform(feats.numpy())
        plt.figure(figsize=(6,5))
        sc=plt.scatter(emb2[:,0], emb2[:,1], c=cluster, s=8, cmap="tab20")
        plt.colorbar(sc,label="cluster"); plt.title("t-SNE of DEC features")
        plt.tight_layout(); plt.savefig(os.path.join(diag_dir,"tsne_feats.png")); plt.close()
    except Exception as e:
        print("[warn] t-SNE failed:", e)

    print("✓ Buckets   :", args.output_dir)
    print("✓ Summaries :", os.path.join(args.output_dir,"summaries"))
    print("✓ Diagnostics:", diag_dir)

# ──────────────────────────── CLI ─────────────────────────────────
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input_dir",  required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch",      type=int, default=64)
    ap.add_argument("--save_assignments", action="store_true",
                help="Save original cluster IDs as .npy")
    args=ap.parse_args()
    main(args)
