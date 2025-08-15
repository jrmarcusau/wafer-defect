#!/usr/bin/env python3
"""
cluster_wafer.py  –  wafer-level SimCLR + k-means (≤10 clusters)

Outputs
-------
outputs/v1/
    type_1/ … type_N/      full wafer PNGs
    summary.csv            cluster , num_images , image_names
"""

from pathlib import Path
import csv, math, random, shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ---- paths & knobs ----------------------------------------------------------
DATA_DIR   = Path("data/Op3176_DefectMap2")
OUT_DIR    = Path("outputs/v2")
IMG_SIDE   = 256               # wafer resized to 256×256
BATCH      = 64
EPOCHS     = 30
MAX_TYPES  = 10
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42
random.seed(SEED); torch.manual_seed(SEED)

# ---- dataset (whole images, two SimCLR views) -------------------------------
class WaferDS(Dataset):
    def __init__(self, root: Path):
        self.files = sorted([p for p in root.rglob("*") if p.suffix.lower() in {".png",".jpg",".jpeg"}])
        if not self.files:
            raise RuntimeError("No wafer images found.")

        self.aug = T.Compose([
            T.Resize(IMG_SIDE, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(IMG_SIDE),
            T.RandomResizedCrop(IMG_SIDE, scale=(0.8,1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor()
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.aug(img), self.aug(img), self.files[idx].as_posix()

# ---- simple ResNet18 encoder -------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        r18=torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        r18.fc=nn.Identity()
        self.backbone=r18
        self.proj=nn.Sequential(nn.Linear(512,256),nn.ReLU(True),
                                nn.Linear(256,out_dim))
    def forward(self,x):
        return nn.functional.normalize(self.proj(self.backbone(x)),dim=1)

def nt_xent(z1,z2,tau=0.2):
    B=z1.size(0); z=torch.cat([z1,z2],0)
    sim=(z @ z.t())/tau; sim.fill_diagonal_(-9e15)
    lbl=torch.arange(B,device=z.device)
    logits=torch.cat([sim[:B,B:],sim[B:,:B]],0)
    return nn.CrossEntropyLoss()(logits,lbl.repeat(2))

# ---- train SimCLR ------------------------------------------------------------
def train_ssl(ds):
    net=Encoder().to(DEVICE)
    opt=torch.optim.Adam(net.parameters(),lr=3e-4)
    loader=DataLoader(ds,batch_size=BATCH,shuffle=True,drop_last=True,num_workers=4)
    for ep in range(1,EPOCHS+1):
        tot=0; net.train()
        for v1,v2,_ in loader:
            v1,v2=v1.to(DEVICE),v2.to(DEVICE)
            loss=nt_xent(net(v1),net(v2))
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()
        print(f"[{ep:02}/{EPOCHS}] SSL loss {tot/len(loader):.4f}")
    return net

# ---- embed + k-means ---------------------------------------------------------
def cluster(net, ds):
    loader=DataLoader(ds,batch_size=128,shuffle=False,num_workers=4)
    embs, names = [], []
    net.eval()
    with torch.no_grad():
        for v,_, fns in loader:
            embs.append(net(v.to(DEVICE)).cpu().numpy())
            names.extend(fns)
    embs=StandardScaler().fit_transform(np.vstack(embs))
    k=min(MAX_TYPES,len(set(names)))
    kmeans=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit(embs)
    return kmeans.labels_, names, k

# ---- save results ------------------------------------------------------------
def save(labels, names, k):
    if OUT_DIR.exists(): shutil.rmtree(OUT_DIR)
    for t in range(k): (OUT_DIR/f"type_{t+1}").mkdir(parents=True,exist_ok=True)

    by_cluster={t:[] for t in range(k)}
    for fn,lbl in zip(names,labels):
        by_cluster[lbl].append(fn)
        shutil.copy(fn, OUT_DIR/f"type_{lbl+1}"/Path(fn).name)

    with open(OUT_DIR/"summary.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["cluster","num_images","images"])
        for t in range(k):
            imgs=[Path(p).name for p in by_cluster[t]]
            w.writerow([f"type_{t+1}", len(imgs), ";".join(imgs)])

# ---- main --------------------------------------------------------------------
def main():
    ds=WaferDS(DATA_DIR)
    print(f"Found {len(ds)} wafer images.")
    enc=train_ssl(ds)
    labels,names,k=cluster(enc,ds)
    save(labels,names,k)
    print(f"✅ Done – {k} clusters in {OUT_DIR}")

if __name__=="__main__":
    from pathlib import Path; main()
