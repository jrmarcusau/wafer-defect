#!/usr/bin/env python3
"""
postcluster_v8.py
————————
Re-cluster embeddings produced by main_v8_256_memsafe.py.

Two modes
---------
1.  HDBSCAN on the **entire** embedding space
    $ python postcluster_v8.py --method hdbscan …

2.  Hierarchical K-means *inside* one or more existing DEC clusters
    $ python postcluster_v8.py --method subkmeans --targets 0 7 13 --subK 3 …

Outputs
-------
<output_dir>/
    reassign.csv          (# image_path, new_label)
    tsne_post.png         (t-SNE with new labels)
    buckets/label_<n>/    (copied PNGs)
"""

import os, argparse, shutil, random, cv2, numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from skimage.feature import hog
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import csv

# Optional density-based clustering
try:
    import hdbscan
except ImportError:
    hdbscan = None

# —— constants (same as training) ————————————————
IMG_SIZE = 256
Z_DIM    = 256
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED     = 42

# —— dataset (identical to training pre-proc) ——————————
class WaferDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted(glob(os.path.join(root, "*.png")) +
                            glob(os.path.join(root, "*.PNG")))
        if not self.paths:
            raise RuntimeError(f"No PNGs in {root}")

        dummy = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
        self.hog_dim = hog(dummy, orientations=9,
                           pixels_per_cell=(16,16),
                           cells_per_block=(1,1),
                           feature_vector=True).shape[0]

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        g = np.array(Image.open(p).convert("L"))
        polar = cv2.warpPolar(g, (IMG_SIZE,IMG_SIZE),
                              (g.shape[1]//2, g.shape[0]//2),
                              maxRadius=g.shape[0]//2,
                              flags=cv2.WARP_POLAR_LINEAR)
        polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
        edge  = cv2.Canny(polar, 50,150)

        hog_vec = hog(polar, orientations=9,
                      pixels_per_cell=(16,16),
                      cells_per_block=(1,1),
                      feature_vector=True).astype(np.float32)

        p_t = torch.from_numpy(polar/255.).unsqueeze(0).float()
        e_t = torch.from_numpy(edge /255.).unsqueeze(0).float()
        xs  = torch.linspace(-1,1,IMG_SIZE).view(1,1,-1).expand(1,IMG_SIZE,IMG_SIZE)
        ys  = torch.linspace(-1,1,IMG_SIZE).view(1,-1,1).expand(1,IMG_SIZE,IMG_SIZE)
        x   = torch.cat([p_t, e_t, torch.cat([xs,ys],0)], 0)
        return x, torch.from_numpy(hog_vec), p

# —— encoder (same as training) ————————————————
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,32,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*32*32, Z_DIM)
        )
    def forward(self,x): return self.net(x)

# —— helper ——————————————————————————————
def l2(t): return t / (t.norm(dim=1, keepdim=True) + 1e-8)

@torch.no_grad()
def encode_all(loader, model):
    feats, paths = [], []
    for x, hog_vec, p in tqdm(loader, desc="Encoding"):
        x, hog_vec = x.to(DEVICE), hog_vec.to(DEVICE)
        z = model(x)
        feats.append(torch.cat([z, hog_vec],1).cpu())
        paths.extend(p)
    return torch.cat(feats).numpy(), paths

# —— main ————————————————————————————————
def main(a):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    

    # load encoder weights
    ckpt = torch.load(a.checkpoint, map_location=DEVICE)
    enc  = Encoder().to(DEVICE)
    enc.load_state_dict(ckpt["model_state"], strict=False)  # ignore decoder
    enc.eval()

    # encode dataset
    ds = WaferDataset(a.input_dir)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False,
                        num_workers=4, pin_memory=torch.cuda.is_available())
    feats, paths = encode_all(loader, enc)

    # —— choose re-clustering method ——————————
    if a.method == "hdbscan":
        if hdbscan is None:
            raise RuntimeError("pip install hdbscan first")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=a.min_size,
                                    min_samples=max(a.min_size//2, 5))
        new_lbl = clusterer.fit_predict(feats)
        print(f"HDBSCAN produced {new_lbl.max()+1} clusters "
              f"and {np.sum(new_lbl==-1)} noise samples")
    else:  # sub-kmeans
        orig = np.load(a.assign_file)
        new_lbl = orig.copy()
        for tgt in a.targets:
            idx = np.where(orig==tgt)[0]
            if len(idx)==0: continue
            sub = feats[idx]
            km  = KMeans(a.subK, n_init=20, random_state=SEED).fit(sub)
            new_lbl[idx] = km.labels_ + new_lbl.max() + 1   # relabel uniquely
        print(f"Sub-Kmeans produced {new_lbl.max()+1} labels total")

    # —— outputs ————————————————————————————
    os.makedirs(a.output_dir, exist_ok=True)
    # CSV
    with open(os.path.join(a.output_dir,"reassign.csv"),"w",newline="") as f:
        csv.writer(f).writerows([("image_path","label"), *zip(paths,new_lbl)])
    # buckets
    for l in np.unique(new_lbl):
        os.makedirs(os.path.join(a.output_dir,f"label_{l}"), exist_ok=True)
    for src,lbl in zip(paths,new_lbl):
        shutil.copy2(src, os.path.join(a.output_dir,f"label_{lbl}",os.path.basename(src)))
    # t-SNE plot
    emb2 = TSNE(2, init="pca", random_state=SEED,
                perplexity=min(30,max(5,len(paths)//20))).fit_transform(feats)
    plt.figure(figsize=(6,5))
    sc=plt.scatter(emb2[:,0], emb2[:,1], c=new_lbl, s=8, cmap="tab20")
    plt.colorbar(sc,label="label")
    plt.title("t-SNE after re-clustering")
    plt.tight_layout(); plt.savefig(os.path.join(a.output_dir,"tsne_post.png")); plt.close()
    print("✓ Results in", a.output_dir)

# —— CLI ————————————————————————————————
if __name__=="__main__":
    import csv
    ap=argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--method", choices=["hdbscan","subkmeans"], default="hdbscan")
    # HDBSCAN
    ap.add_argument("--min_size", type=int, default=30,
                    help="min_cluster_size for HDBSCAN")
    # sub-kmeans
    ap.add_argument("--targets", type=int, nargs="+", default=[],
                    help="original cluster IDs to split")
    ap.add_argument("--subK", type=int, default=3,
                    help="K inside each target blob")
    ap.add_argument("--assign_file", default=None,
                help="NumPy .npy file with original DEC assignments")
    a=ap.parse_args()
    main(a)
