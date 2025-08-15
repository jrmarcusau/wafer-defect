#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_v1.py  –  Unsupervised wafer-defect clustering
└─ outputs/v1/
   ├─ type_1/                 (full wafer images)
   │   └─ type_1.csv          (one line per blob: file,y0,x0,y1,x1)
   ├─ type_2/
   │   └─ type_2.csv
   ├─ …
   └─ summary_counts.csv      (cluster , blob_count)
"""
# ------------------------------------------------- imports & constants
from pathlib import Path
import csv, math, random, shutil, warnings
import numpy as np
from PIL import Image
from scipy.ndimage import label
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

DATA_DIR   = Path("data/Op3176_DefectMap2")
OUT_ROOT   = Path("outputs/v1")
PATCH_SIDE = 128
BATCH      = 128
EPOCHS     = 20
MAX_TYPES  = 10
MIN_PIXELS = 20
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42
random.seed(SEED); torch.manual_seed(SEED)

# ------------------------------------------------- purple-blob finder
def find_blobs(rgb: np.ndarray, min_pixels=MIN_PIXELS):
    if rgb.shape[2] == 4: rgb = rgb[...,:3]
    r,g,b = rgb[...,0].astype(int), rgb[...,1].astype(int), rgb[...,2].astype(int)
    mask  = (r>100) & (b>100) & (g < 0.6*((r+b)/2))
    lbl,n = label(mask, structure=np.ones((3,3)))
    for i in range(1,n+1):
        ys,xs = np.where(lbl==i)
        if len(ys)>=min_pixels:
            yield ys.min(), xs.min(), ys.max()+1, xs.max()+1

# ------------------------------------------------- dataset (SimCLR pair)
class PatchDS(Dataset):
    def __init__(self, root:Path):
        self.samples=[]
        paths=[p for p in root.rglob("*") if p.suffix.lower() in {".png",".jpg",".jpeg"}]
        for p in tqdm(paths,desc="🔍 Scanning wafer maps"):
            rgb=np.array(Image.open(p).convert("RGB"))
            for bb in find_blobs(rgb): self.samples.append((p,bb))
        self.aug=T.Compose([
            T.RandomResizedCrop(PATCH_SIDE,scale=(0.8,1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor()
        ])
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        path,bb=self.samples[idx]
        rgb=np.array(Image.open(path).convert("RGB"))
        y0,x0,y1,x1=bb
        patch=Image.fromarray(rgb[y0:y1,x0:x1])
        return self.aug(patch),self.aug(patch),path.as_posix(),bb

# ------------------------------------------------- SimCLR encoder
class Encoder(nn.Module):
    def __init__(self,out_dim=128):
        super().__init__()
        res=torch.hub.load('pytorch/vision','resnet18',pretrained=False)
        res.fc=nn.Identity()
        self.backbone=res
        self.proj=nn.Sequential(nn.Linear(512,256),nn.ReLU(True),
                                nn.Linear(256,out_dim))
    def forward(self,x):
        return nn.functional.normalize(self.proj(self.backbone(x)),dim=1)

def nt_xent(z1,z2,tau=0.2):
    B=z1.size(0); z=torch.cat([z1,z2],0)
    sim=(z@z.t())/tau; sim.fill_diagonal_(-9e15)
    lbl=torch.arange(B,device=z.device)
    logits=torch.cat([sim[:B,B:],sim[B:,:B]],0)
    return nn.CrossEntropyLoss()(logits,lbl.repeat(2))

def train_ssl(ds):
    net=Encoder().to(DEVICE); opt=torch.optim.Adam(net.parameters(),lr=3e-4)
    loader=DataLoader(ds,batch_size=BATCH,shuffle=True,drop_last=True,
                      num_workers=4,pin_memory=True)
    for ep in range(1,EPOCHS+1):
        net.train(); tot=0
        for v1,v2,*_ in loader:
            v1,v2=v1.to(DEVICE),v2.to(DEVICE)
            loss=nt_xent(net(v1),net(v2))
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()
        print(f"[{ep:02}/{EPOCHS}] SSL loss {tot/len(loader):.4f}")
    return net

def cluster_embeddings(net,ds):
    loader=DataLoader(ds,batch_size=256,shuffle=False,
                      num_workers=4,pin_memory=True)
    embs,metas=[],[]
    net.eval()
    with torch.no_grad():
        for v1,_,paths,bbs in loader:
            embs.append(net(v1.to(DEVICE)).cpu().numpy())
            metas.extend(zip(paths,bbs))
    embs=StandardScaler().fit_transform(np.vstack(embs))
    k=min(MAX_TYPES,max(2,int(math.sqrt(len(embs)//5))))
    km=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit(embs)
    return km.labels_,metas,k

# ------------------------------------------------- save outputs
def save_outputs(labels, metas, n_types):
    if OUT_ROOT.exists(): shutil.rmtree(OUT_ROOT)
    for n in range(n_types):
        (OUT_ROOT/f"type_{n+1}").mkdir(parents=True,exist_ok=True)

    # prepare per-cluster csv rows and copy full images
    rows_by_cluster={n:[] for n in range(n_types)}
    copied=set()
    for (path,bb),lbl in zip(metas,labels):
        if lbl<0: continue
        y0,x0,y1,x1=[int(v) for v in bb[:4]]
        rows_by_cluster[lbl].append([Path(path).name,y0,x0,y1,x1])
        dst=OUT_ROOT/f"type_{lbl+1}"/Path(path).name
        if path not in copied:
            shutil.copy(path,dst)
            copied.add(path)

    # write per-cluster csv
    for lbl,rows in rows_by_cluster.items():
        csv_path=OUT_ROOT/f"type_{lbl+1}"/f"type_{lbl+1}.csv"
        with open(csv_path,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["file","y0","x0","y1","x1"]); w.writerows(rows)

    # write summary_counts.csv
    with open(OUT_ROOT/"summary_counts.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["cluster","blob_count"])
        for lbl in range(n_types):
            w.writerow([f"type_{lbl+1}", len(rows_by_cluster[lbl])])

# ------------------------------------------------- main
def main():
    ds=PatchDS(DATA_DIR)
    if len(ds)==0:
        warnings.warn("No blobs found – check data path or MIN_PIXELS."); return
    print(f"Found {len(ds)} blob patches.")
    enc=train_ssl(ds)
    labels,metas,n_types=cluster_embeddings(enc,ds)
    save_outputs(labels,metas,n_types)
    print(f"✅ Done – {n_types} clusters written to {OUT_ROOT}")

if __name__=="__main__":
    main()
