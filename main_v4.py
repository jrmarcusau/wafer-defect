#!/usr/bin/env python3
# v5_fast.py  –  polar-edge SimCLR + k-means (≤10 clusters)  *fast preset*

from pathlib import Path
import csv, math, random, shutil, os, time
import numpy as np, cv2, torch
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small

# ----------------------------- hyper-params you may tune --------------------
DATA_DIR    = Path("data/Op3176_DefectMap2")
CACHE_DIR   = Path("cache")           # holds *.npy 2-channel tensors
OUT_DIR     = Path("outputs/v4")
POLAR_SIZE  = 256                     # ↓ from 512  (4× faster)
BATCH       = 16
EPOCHS      = 100                      # checkpoints every 5
MAX_TYPES   = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP         = DEVICE.type == "cuda"   # mixed precision only on GPU
SEED        = 42
random.seed(SEED); torch.manual_seed(SEED)

# ----------------------------- helper: purple → mask ------------------------
def purple_mask(rgb):
    if rgb.shape[2] == 4: rgb = rgb[..., :3]
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    return ((r>100)&(b>100)&(g<0.6*((r+b)//2))).astype(np.uint8)

def to_polar_edge(mask):
    H,W = mask.shape; center=(W//2,H//2); maxR=W//2
    polar = cv2.warpPolar(mask*255,(POLAR_SIZE,POLAR_SIZE),center,maxR,
                          cv2.WARP_POLAR_LINEAR)
    polar = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    polar = (polar>127).astype(np.float32)
    edge  = cv2.Canny((polar*255).astype(np.uint8),50,150)/255.0
    return np.stack([polar,edge]).astype(np.float16)  # C×H×W

# ----------------------------- dataset --------------------------------------
class PolarDS(Dataset):
    def __init__(self, root:Path):
        self.files = sorted(p for p in root.rglob("*")
                            if p.suffix.lower() in {".png",".jpg",".jpeg"})
        if not self.files: raise RuntimeError("No images found.")
        CACHE_DIR.mkdir(exist_ok=True)
        self._prepare_cache()
        self.aug = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])
    def _prepare_cache(self):
        for p in tqdm(self.files, desc="Caching polar tensors"):
            npy = CACHE_DIR / (p.stem + ".npy")
            if npy.exists(): continue
            rgb  = np.array(Image.open(p).convert("RGB"))
            mask = purple_mask(rgb)
            tensor = to_polar_edge(mask)
            np.save(npy, tensor, allow_pickle=False)
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        tensor = np.load(CACHE_DIR / (Path(self.files[idx]).stem + ".npy"))
        t = torch.from_numpy(tensor)         # C×H×W float16
        t1 = self.aug(t); t2 = self.aug(t.clone())
        return t1.float(), t2.float(), self.files[idx].as_posix()

# ----------------------------- 2-ch MobileNet-v3-small -----------------------
class Encoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        mb = mobilenet_v3_small(weights=None)
        # patch first conv (in_ch=2 → 16)
        mb.features[0][0] = nn.Conv2d(2,16,kernel_size=3,stride=2,padding=1,bias=False)
        mb.classifier = nn.Identity()
        self.backbone = mb
        self.proj = nn.Sequential(nn.Linear(576,256), nn.ReLU(True),
                                  nn.Linear(256,out_dim))
    def forward(self,x):
        h = self.backbone(x)
        if h.ndim==4: h = h.mean([-2,-1])   # global pool (B×C×1×1 → B×C)
        return nn.functional.normalize(self.proj(h), dim=1)

def nt_xent(z1,z2,tau=0.2):
    B=z1.size(0); z=torch.cat([z1,z2],0)
    sim=(z@z.t())/tau; sim.fill_diagonal_(-9e15)
    lbl=torch.arange(B,device=z.device)
    logits=torch.cat([sim[:B,B:],sim[B:,:B]],0)
    return nn.CrossEntropyLoss()(logits,lbl.repeat(2))

# ----------------------------- training + checkpoints ------------------------
def train_ssl(ds):
    net,scaler = Encoder().to(DEVICE), torch.cuda.amp.GradScaler(enabled=AMP)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=AMP)

    for ep in range(1,EPOCHS+1):
        net.train(); running=0.0
        for v1,v2,_ in tqdm(loader, desc=f"Epoch {ep}/{EPOCHS}", leave=False):
            v1,v2 = v1.to(DEVICE), v2.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=AMP):
                loss = nt_xent(net(v1), net(v2))
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item()
        print(f"[{ep:02}/{EPOCHS}] loss {running/len(loader):.4f}")

        if ep % 5 == 0:
            ckpt = OUT_DIR / f"ckpt_ep{ep}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), ckpt)

    return net

# ----------------------------- cluster & save (unchanged) --------------------
def cluster(net,ds):
    loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0)
    embs,names=[],[]
    net.eval()
    with torch.no_grad():
        for v,_,fns in tqdm(loader,desc="Embedding",unit="batch"):
            embs.append(net(v.to(DEVICE)).cpu().numpy()); names.extend(fns)
    embs=StandardScaler().fit_transform(np.vstack(embs))
    k=min(MAX_TYPES,len(set(names)))
    km=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit(embs)
    return km.labels_,names,k

def save(labels,names,k):
    if OUT_DIR.exists(): shutil.rmtree(OUT_DIR)
    for t in range(k): (OUT_DIR/f"type_{t+1}").mkdir(parents=True,exist_ok=True)
    clusters={t:[] for t in range(k)}
    for fn,lbl in zip(names,labels):
        clusters[lbl].append(fn)
        shutil.copy(fn, OUT_DIR/f"type_{lbl+1}"/Path(fn).name)
    with open(OUT_DIR/"summary.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["cluster","num_images","images"])
        for t in range(k):
            w.writerow([f"type_{t+1}",len(clusters[t]),
                        ";".join(Path(p).name for p in clusters[t])])

# ----------------------------- main ------------------------------------------
def main():
    start=time.time()
    ds=PolarDS(DATA_DIR)
    print(f"Found {len(ds)} wafers. Pre-processing done in {time.time()-start:.1f}s")
    enc=train_ssl(ds)
    labels,names,k=cluster(enc,ds)
    save(labels,names,k)
    print(f"✅ Completed in {(time.time()-start)/60:.1f} min – {k} clusters in {OUT_DIR}")

if __name__=="__main__":
    main()
