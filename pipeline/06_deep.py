"""Model C: per-date CNN -> temporal Transformer (S2 chips + optional S1 field-mean sequence).

Inputs per field (one field-season = one sample):
  S2 tokens: one per CLEAR acquisition (>= 60% of interior clear). Chip = 24 x 24 px at 10 m (240 m x 240 m)
     centred on the parcel. Channels: 8 reflectance bands (B02 B03 B04 B05 B08 B8A B11 B12; the 20 m bands were
     read onto the 10 m grid by nearest neighbour - no added detail) + pixel-valid mask (clear & inside the
     buffered parcel) + parcel mask. Pixels outside the parcel / cloudy are zeroed after normalisation, and the
     CNN output is pooled only over parcel-interior valid pixels, limiting neighbouring-field influence.
  S1 tokens (fusion variant): one per acquisition of the block's dominant relative orbit: field-mean
     VV dB, VH dB, VH-VV dB (= linear cross-ratio), via a separate linear branch (sensor-specific).
  Time: sinusoidal encoding of the actual acquisition day-of-year + learned sensor embedding. No resampling
  to a regular grid, no interpolated tokens. Padding is masked in attention.
Training: normalisation stats from TRAIN only; class-weighted BCE; augmentation = random flips/rot90, random
  S2 token dropout, and for 30% of samples a random 30-45 day optical blackout (Aug 15 - Nov 15) - needed
  before claiming any resilience to optical gaps. Early stopping on val AUC. Threshold + Platt calibration on val.
Outputs: data/results/deep_<variant>.json, checkpoints in data/models/.
"""
import os, sys, json, time, math, glob, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from common import ROOT, YEAR, log
import importlib.util
spec = importlib.util.spec_from_file_location("m5", __file__.replace("06_deep.py", "05_models.py"))

SEED = int(os.environ.get("HTH_SEED", 20251114))
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from multiyear import RESULTS as RES
MOD = RES / "models"; MOD.mkdir(exist_ok=True)
MAX_S2, MAX_S1 = 64, 64
DOY0 = pd.Timestamp(f"{YEAR}-01-01")


def metrics(y, pred, score=None):
    from sklearn.metrics import confusion_matrix, average_precision_score
    y, pred = np.asarray(y), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sd = lambda a, b: a / b if b else float("nan")
    m = dict(n=int(len(y)), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), coverage=1.0,
             precision_silage=sd(tp, tp + fp), recall_silage=sd(tp, tp + fn), precision_grain=sd(tn, tn + fn),
             recall_grain=sd(tn, tn + fp), fpr_silage=sd(fp, fp + tn), accuracy=sd(tp + tn, len(y)))
    m["balanced_accuracy"] = (m["recall_silage"] + m["recall_grain"]) / 2
    m["f1_grain"] = sd(2 * m["precision_grain"] * m["recall_grain"], m["precision_grain"] + m["recall_grain"])
    m["f1_silage"] = sd(2 * m["precision_silage"] * m["recall_silage"], m["precision_silage"] + m["recall_silage"])
    if score is not None:
        m["roc_auc"] = roc_auc_score(y, score); m["pr_auc_grain"] = average_precision_score(1 - y, -np.asarray(score))
    return m


# ---------------- data ----------------
def load():
    """All field-seasons from every configured year; keys are '<year>:<BRP id>'; role = train/val/test/unused."""
    from multiyear import roots, TRAIN_YEARS, TEST_YEAR
    parts = []
    for yr, R in roots().items():
        lab = pd.read_parquet(R / "features" / "labels.parquet")[["field_id", "label"]]
        sp = pd.read_csv(R / "features" / "splits.csv").merge(lab, on="field_id")
        sp["year"] = yr; sp["uid"] = [f"{yr}:{f}" for f in sp.field_id]
        parts.append(sp)
    splits = pd.concat(parts, ignore_index=True)
    splits["role"] = "unused"
    splits.loc[splits.year.isin(TRAIN_YEARS) & (splits.split == "train"), "role"] = "train"
    splits.loc[splits.year.isin(TRAIN_YEARS) & (splits.split == "val"), "role"] = "val"
    splits.loc[(splits.year == TEST_YEAR) & (splits.split == "test"), "role"] = "test"
    fid2i = {f: i for i, f in enumerate(splits.uid)}
    N = len(splits)
    s2_x, s2_t, s2_f = [], [], []
    fmask = np.zeros((N, 24, 24), bool); imask = np.zeros((N, 24, 24), bool)
    clear_l, s1 = [], {}
    for yr, R in roots().items():
        d0 = pd.Timestamp(f"{yr}-01-01")
        for p in sorted(glob.glob(str(R / "s2" / "chips_*.npz"))):
            z = np.load(p)
            fids = np.array([f"{yr}:{f}" for f in z["field_id"]])
            keep = np.array([f in fid2i for f in fids])
            s2_x.append(z["x"][keep]); clear_l.append(z["clear"][keep])
            s2_t.append(np.array([(pd.Timestamp(str(d)) - d0).days for d in z["date"][keep]]))
            s2_f.append(np.array([fid2i[f] for f in fids[keep]]))
            for f, fm, im in zip(z["chip_field_id"], z["field_mask"], z["interior_mask"]):
                k = fid2i.get(f"{yr}:{f}")
                if k is not None: fmask[k] = fm; imask[k] = im
        dom = json.load(open(R / "features" / "s1_dominant_orbit.json"))
        for p in sorted(glob.glob(str(R / "s1" / "s1_*.parquet"))):
            b = os.path.basename(p)[3:-8]
            d = pd.read_parquet(p)
            if b in dom: d = d[d.rel_orbit == dom[b]]
            for f, g in d.groupby("field_id"):
                k = fid2i.get(f"{yr}:{f}")
                if k is not None:
                    g = g.sort_values("date")
                    s1[k] = (np.array([(pd.Timestamp(str(x)) - d0).days for x in g.date]), g[["vv_db", "vh_db", "cr_db"]].values.astype(np.float32))
    X = np.concatenate(s2_x); C = np.concatenate(clear_l); T = np.concatenate(s2_t); Fi = np.concatenate(s2_f)
    order = np.lexsort((T, Fi)); X, C, T, Fi = X[order], C[order], T[order], Fi[order]
    starts = np.searchsorted(Fi, np.arange(N + 1))
    return splits, X, C, T, starts, fmask, imask, s1


class DS(torch.utils.data.Dataset):
    def __init__(self, idx, y, X, C, T, starts, fmask, imask, s1, mu, sd, s1mu, s1sd, train, use_s1, blackout=None):
        self.idx, self.y = idx, y
        self.X, self.C, self.T, self.st, self.fm, self.im, self.s1 = X, C, T, starts, fmask, imask, s1
        self.mu, self.sd, self.s1mu, self.s1sd = mu, sd, s1mu, s1sd
        self.train, self.use_s1, self.blackout = train, use_s1, blackout

    def __len__(self): return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        a, b = self.st[i], self.st[i + 1]
        t = self.T[a:b]; sel = np.arange(a, b)
        if self.blackout is not None:  # fixed simulated gap (evaluation)
            keep = (t < self.blackout[0]) | (t > self.blackout[1]); sel, t = sel[keep], t[keep]
        if self.train and len(sel) > 4:
            keep = np.random.rand(len(sel)) > np.random.uniform(0, 0.3)
            if np.random.rand() < 0.3:
                s = np.random.randint(226, 320 - 30); L = np.random.randint(30, 46)
                keep &= (t < s) | (t > s + L)
            if keep.sum() >= 3: sel, t = sel[keep], t[keep]
        if len(sel) > MAX_S2:
            pick = np.sort(np.random.choice(len(sel), MAX_S2, replace=False)) if self.train else np.linspace(0, len(sel) - 1, MAX_S2).astype(int)
            sel, t = sel[pick], t[pick]
        x = (self.X[sel].astype(np.float32) - self.mu[None, :, None, None]) / self.sd[None, :, None, None]
        valid = (self.C[sel] & self.im[i][None]).astype(np.float32)
        x = x * valid[:, None]
        x = np.concatenate([x, valid[:, None], np.repeat(self.fm[i][None, None].astype(np.float32), len(sel), 0)], 1)
        if self.train:
            if np.random.rand() < 0.5: x = x[..., ::-1]
            if np.random.rand() < 0.5: x = x[..., ::-1, :]
            x = np.rot90(x, np.random.randint(4), axes=(2, 3))
        n2 = len(sel)
        xs = np.zeros((MAX_S2, 10, 24, 24), np.float32); xs[:n2] = x
        ts = np.zeros(MAX_S2, np.float32); ts[:n2] = t
        m2 = np.zeros(MAX_S2, bool); m2[:n2] = True
        s1v = np.zeros((MAX_S1, 3), np.float32); s1t = np.zeros(MAX_S1, np.float32); m1 = np.zeros(MAX_S1, bool)
        if self.use_s1 and i in self.s1:
            tt, vv = self.s1[i]
            if len(tt) > MAX_S1:
                p = np.linspace(0, len(tt) - 1, MAX_S1).astype(int); tt, vv = tt[p], vv[p]
            if self.train and len(tt) > 4:
                keep = np.random.rand(len(tt)) > 0.15; tt, vv = tt[keep], vv[keep]
            s1v[:len(tt)] = (vv - self.s1mu) / self.s1sd; s1t[:len(tt)] = tt; m1[:len(tt)] = True
        return (torch.from_numpy(np.ascontiguousarray(xs)), torch.from_numpy(ts), torch.from_numpy(m2),
                torch.from_numpy(s1v), torch.from_numpy(s1t), torch.from_numpy(m1), torch.tensor(float(self.y[k])))


# ---------------- model ----------------
def time_enc(t, d):
    # periods geometric from 366 d down to 4 d, so both season position and ~weekly spacing are encoded
    i = torch.arange(d // 2, device=t.device, dtype=torch.float32)
    period = 366.0 * (4.0 / 366.0) ** (i / (d // 2 - 1))
    ang = t[..., None] * 2 * math.pi / period
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class Net(nn.Module):
    def __init__(self, d=64, use_s1=True, use_cnn=True):
        super().__init__()
        self.use_s1, self.use_cnn, self.d = use_s1, use_cnn, d
        self.cnn = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.BatchNorm2d(64), nn.GELU())  # 24 -> 12
        self.cnn_proj = nn.Linear(64 + 8, d) if use_cnn else nn.Linear(8, d)
        self.s1_proj = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.sensor = nn.Embedding(3, d)  # 0 = cls, 1 = S2, 2 = S1
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, xs, ts, m2, s1v, s1t, m1):
        B, T = xs.shape[:2]
        flat = xs.flatten(0, 1)
        valid = flat[:, 8:9]
        vsum = valid.sum((2, 3)).clamp(min=1)
        bandmean = (flat[:, :8] * valid).sum((2, 3)) / vsum  # masked field mean of normalised bands
        if self.use_cnn:
            h = self.cnn(flat)
            vm = F.avg_pool2d(valid, 2)
            h = (h * vm).sum((2, 3)) / vm.sum((2, 3)).clamp(min=1e-3)
            h = torch.cat([h, bandmean], 1)
        else:
            h = bandmean
        tok2 = self.cnn_proj(h).view(B, T, -1) + time_enc(ts, self.d) + self.sensor.weight[1]
        toks, mask = [self.cls.expand(B, 1, -1) + self.sensor.weight[0], tok2], [torch.ones(B, 1, dtype=torch.bool, device=xs.device), m2]
        if self.use_s1:
            toks.append(self.s1_proj(s1v) + time_enc(s1t, self.d) + self.sensor.weight[2]); mask.append(m1)
        z = self.tr(torch.cat(toks, 1), src_key_padding_mask=~torch.cat(mask, 1))
        return self.head(z[:, 0]).squeeze(-1)


def run(variant):
    use_s1 = "s1" in variant; use_cnn = "nocnn" not in variant
    t_load = time.time()
    splits, X, C, T, starts, fmask, imask, s1 = load()
    y = (splits.label == "silage").astype(int).values
    tr, va, te = (np.flatnonzero(splits.role.values == s) for s in ("train", "val", "test"))
    has = np.diff(starts) >= 3
    tr, va, te_all = tr[has[tr]], va[has[va]], te
    te = te[has[te]]
    # normalisation from TRAIN chips only (interior clear pixels)
    trmask = np.isin(np.repeat(np.arange(len(splits)), np.diff(starts)), tr)
    sub = np.flatnonzero(trmask); sub = np.random.choice(sub, min(20000, len(sub)), replace=False)
    xs = X[sub].astype(np.float32); vm = C[sub] & imask[np.repeat(np.arange(len(splits)), np.diff(starts))[sub]]
    mu = np.array([xs[:, c][vm].mean() for c in range(8)], np.float32); sd = np.array([xs[:, c][vm].std() for c in range(8)], np.float32)
    s1all = np.concatenate([s1[i][1] for i in tr if i in s1]); s1mu, s1sd = s1all.mean(0), s1all.std(0)
    log(f"deep {variant}: loaded in {time.time()-t_load:.0f}s; tokens {len(X)}; train {len(tr)} val {len(va)} test {len(te)} (test without >=3 clear looks: {len(te_all)-len(te)})", "deep.log")
    mk = lambda idx, train, bo=None: DS(idx, y[idx], X, C, T, starts, fmask, imask, s1, mu, sd, s1mu, s1sd, train, use_s1, bo)
    w = np.where(y[tr] == 1, 1.0, (y[tr] == 1).sum() / (y[tr] == 0).sum())   # balanced sampling weights
    sampler = torch.utils.data.WeightedRandomSampler(w, len(tr), replacement=True)
    dl_tr = torch.utils.data.DataLoader(mk(tr, True), batch_size=64, sampler=sampler, num_workers=12, persistent_workers=True)
    dl = lambda idx, bo=None: torch.utils.data.DataLoader(mk(idx, False, bo), batch_size=128, num_workers=12)
    net = Net(use_s1=use_s1, use_cnn=use_cnn).to(DEV)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    EPOCHS, PATIENCE = int(os.environ.get("HTH_EPOCHS", 40)), 8
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EPOCHS * len(dl_tr))

    def predict(loader):
        net.eval(); out = []
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
            for bt in loader:
                out.append(net(*[t.to(DEV, non_blocking=True) for t in bt[:6]]).float().cpu())
        return torch.cat(out).numpy()

    best, bad, t0 = (-1, None), 0, time.time()
    for ep in range(EPOCHS):
        net.train(); tot = 0
        for bt in dl_tr:
            bt = [t.to(DEV, non_blocking=True) for t in bt]
            with torch.autocast(DEV, dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(net(*bt[:6]).float(), bt[6])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item()
        auc = roc_auc_score(y[va], predict(dl(va)))
        log(f"deep {variant} ep{ep}: loss {tot/len(dl_tr):.4f} val_auc {auc:.4f} ({time.time()-t0:.0f}s)", "deep.log")
        if auc > best[0]:
            best, bad = (auc, ep), 0; torch.save(net.state_dict(), MOD / f"deep_{variant}_seed{SEED}.pt")
        else:
            bad += 1
            if bad >= PATIENCE: break
    t_train = time.time() - t0
    net.load_state_dict(torch.load(MOD / f"deep_{variant}_seed{SEED}.pt"))
    lv = predict(dl(va))
    cal = LogisticRegression(C=1e6).fit(lv[:, None], y[va]); calib = lambda l: cal.predict_proba(l[:, None])[:, 1]
    pv = calib(lv); ths = np.linspace(0.05, 0.95, 91)
    thr = float(ths[int(np.argmax([metrics(y[va], (pv >= t).astype(int))["balanced_accuracy"] for t in ths]))])
    t1 = time.time(); lt = predict(dl(te)); t_inf = time.time() - t1
    pt = calib(lt)
    res = dict(variant=variant, seed=SEED, params=int(nparam), best_epoch=best[1], val_auc=best[0], threshold=thr,
               train_seconds=t_train, test_inference_seconds=t_inf, n_test=len(te), n_test_excluded_few_looks=int(len(te_all) - len(te)),
               hardware=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
               test=metrics(y[te], (pt >= thr).astype(int), pt))
    lg = predict(dl(te, (244, 288)))   # SIMULATED: no optical looks Sep 1 - Oct 15
    pg = calib(lg)
    res["simulated_gap_sep1_oct15"] = metrics(y[te], (pg >= thr).astype(int), pg)
    pd.DataFrame({"uid": splits.uid.values[te], "p_silage": pt, "p_silage_gap": pg}).to_parquet(RES / f"deep_{variant}_test_preds.parquet")
    json.dump(res, open(RES / f"deep_{variant}_seed{SEED}.json", "w"), indent=1, default=float)
    log(f"deep {variant}: test bal_acc {res['test']['balanced_accuracy']:.3f} auc {res['test']['roc_auc']:.3f} | gap bal_acc {res['simulated_gap_sep1_oct15']['balanced_accuracy']:.3f} | params {nparam} train {t_train:.0f}s", "deep.log")


if __name__ == "__main__":
    for v in sys.argv[1:]:
        run(v)
