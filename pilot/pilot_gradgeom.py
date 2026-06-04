"""
Pilot: gradient-geometry diagnostics for KD data selection (CPU-only, minutes).

Tests four hypotheses on a small teacher->student MLP knowledge distillation
(sklearn 'digits', 1797 x 64, 10 classes):

  A (Idea 1): per-sample KD-loss gradients are LOW-RANK across samples,
              and lower rank than per-sample CE gradients.
  B (Idea 2): parameter-gradient space is lower (normalized) rank than
              input-saliency space.
  C (Idea 7): a sample's STRUCTURAL energy (projection onto the top-k
              gradient principal subspace) is NOT strongly correlated with
              its KD loss  ->  "structural != hard".
  D (Idea 3): a D-optimal coreset spanning the principal gradient subspace
              beats a random / loss-based coreset of equal size when the
              student is retrained on it.

This is a SMALL-SCALE diagnostic pilot, not a full experiment. It exists to
produce empirical signal cheaply; conclusions must be re-tested at LM scale.
"""
import json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

t0 = time.time()
torch.manual_seed(0)
np.random.seed(0)

# ---------------- data ----------------
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
Xte_t, yte_t = torch.tensor(Xte), torch.tensor(yte)
D_in, N_cls = X.shape[1], 10
T = 4.0  # KD temperature

def make_mlp(hidden):
    layers, prev = [], D_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]; prev = h
    layers += [nn.Linear(prev, N_cls)]
    return nn.Sequential(*layers)

def acc(model, Xv, yv):
    with torch.no_grad():
        return (model(Xv).argmax(1) == yv).float().mean().item()

# ---------------- teacher ----------------
teacher = make_mlp([128, 128])
opt = torch.optim.Adam(teacher.parameters(), lr=0.02, weight_decay=1e-4)
for _ in range(250):
    opt.zero_grad(); F.cross_entropy(teacher(Xtr_t), ytr_t).backward(); opt.step()
teacher.eval()
with torch.no_grad():
    teach_logits = teacher(Xtr_t)
teacher_acc = acc(teacher, Xte_t, yte_t)

def fresh_student(seed=0):
    torch.manual_seed(seed)
    return make_mlp([32])

# ---------------- per-sample gradients (fresh student) ----------------
student = fresh_student(0)
params = [p for p in student.parameters()]
P = sum(p.numel() for p in params)

N = min(1200, Xtr_t.shape[0])
G_kd  = np.zeros((N, P), dtype=np.float32)     # grad of KD loss wrt student params
G_ce  = np.zeros((N, P), dtype=np.float32)     # grad of CE loss wrt student params
G_sal = np.zeros((N, D_in), dtype=np.float32)  # grad of KD loss wrt input (saliency)
kd_loss = np.zeros(N, dtype=np.float32)

for i in range(N):
    x = Xtr_t[i:i+1].clone().requires_grad_(True)
    s_logits = student(x)
    t_soft = F.softmax(teach_logits[i:i+1] / T, dim=1).detach()
    kd = (T * T) * F.kl_div(F.log_softmax(s_logits / T, dim=1), t_soft, reduction='batchmean')
    g_par = torch.autograd.grad(kd, params, retain_graph=True)
    G_kd[i] = torch.cat([g.reshape(-1) for g in g_par]).numpy()
    g_in = torch.autograd.grad(kd, x)[0]
    G_sal[i] = g_in.reshape(-1).detach().numpy()
    kd_loss[i] = float(kd.detach())
    ce = F.cross_entropy(student(Xtr_t[i:i+1]), ytr_t[i:i+1])
    g_ce = torch.autograd.grad(ce, params)
    G_ce[i] = torch.cat([g.reshape(-1) for g in g_ce]).numpy()

# ---------------- spectrum helpers ----------------
def spectrum(G):
    s = np.linalg.svd(G, compute_uv=False)
    s2 = s ** 2
    e = s2 / s2.sum()
    eff = float(np.exp(-(e * np.log(e + 1e-12)).sum()))   # entropy effective rank
    mn = min(G.shape)
    return dict(eff_rank=round(eff, 2),
                norm_eff_rank=round(eff / mn, 4),
                stable_rank=round(float(s2.sum() / s2.max()), 2),
                ambient_dim=mn,
                top1_energy=round(float(e[0]), 4),
                top5_energy=round(float(e[:5].sum()), 4),
                top10_energy=round(float(e[:10].sum()), 4))

specA_kd = spectrum(G_kd)
specA_ce = spectrum(G_ce)
specB_sal = spectrum(G_sal)

# ---------------- Pilot C: structural energy vs sample loss ----------------
k = 10
# uncentered SVD: dominant directions include the (highly structural) mean gradient
_, S, Vt = np.linalg.svd(G_kd, full_matrices=False)
coords_full = G_kd @ Vt.T
struct = (coords_full[:, :k] ** 2).sum(1)
total = (G_kd ** 2).sum(1)
struct_frac = struct / (total + 1e-12)
corr_struct_loss = float(np.corrcoef(struct_frac, kd_loss)[0, 1])
corr_gnorm_loss  = float(np.corrcoef(np.sqrt(total), kd_loss)[0, 1])
hi = kd_loss > np.median(kd_loss)
struct_frac_hi = float(struct_frac[hi].mean())
struct_frac_lo = float(struct_frac[~hi].mean())

# ---------------- Pilot D: coreset selection + retraining ----------------
r = 20
coords = G_kd @ Vt[:r].T            # principal coordinates for D-optimal
sal_coords = G_sal @ np.linalg.svd(G_sal, full_matrices=False)[2][:min(r, D_in)].T

def d_optimal(C, m, lam=1e-3):
    Npts, rr = C.shape
    A_inv = np.eye(rr) / lam
    chosen, avail = [], np.ones(Npts, bool)
    for _ in range(m):
        proj = C @ A_inv
        quad = np.einsum('ij,ij->i', proj, C)
        quad[~avail] = -np.inf
        p = int(np.argmax(quad)); chosen.append(p); avail[p] = False
        g = C[p]; Ag = A_inv @ g
        A_inv = A_inv - np.outer(Ag, Ag) / (1.0 + g @ Ag)
    return np.array(chosen)

m = N // 2
sel_dopt = d_optimal(coords, m)
sel_sal  = d_optimal(sal_coords, m)
sel_hi   = np.argsort(-kd_loss)[:m]      # hard samples
sel_lo   = np.argsort(kd_loss)[:m]       # easy samples

def kd_train(train_idx, seed, epochs=200):
    st = fresh_student(seed)
    op = torch.optim.Adam(st.parameters(), lr=0.03, weight_decay=1e-4)
    Xs = Xtr_t[:N][train_idx]; ts = teach_logits[:N][train_idx]
    t_soft = F.softmax(ts / T, dim=1).detach()
    for _ in range(epochs):
        op.zero_grad()
        loss = (T * T) * F.kl_div(F.log_softmax(st(Xs) / T, dim=1), t_soft, reduction='batchmean')
        loss.backward(); op.step()
    return acc(st, Xte_t, yte_t)

seeds = [1, 2, 3, 4, 5]
results = {}
for name, sel in [('full', np.arange(N)), ('d_optimal_grad', sel_dopt),
                   ('d_optimal_saliency', sel_sal), ('hard_highloss', sel_hi),
                   ('easy_lowloss', sel_lo)]:
    accs = [kd_train(sel, s) for s in seeds]
    results[name] = dict(mean=round(float(np.mean(accs)), 4),
                         std=round(float(np.std(accs)), 4), n=len(sel))
rand_runs = []
for s in seeds:
    ridx = np.random.RandomState(s).choice(N, m, replace=False)
    rand_runs.append(kd_train(ridx, s))
results['random'] = dict(mean=round(float(np.mean(rand_runs)), 4),
                         std=round(float(np.std(rand_runs)), 4), n=m)

# ---------------- report ----------------
out = {
    "setup": dict(dataset="sklearn-digits", n_grad_samples=N, student_params=P,
                  teacher_acc=round(teacher_acc, 4), temperature=T),
    "pilot_A_spectrum": dict(KD_grad=specA_kd, CE_grad=specA_ce,
        verdict=("KD lower-rank than CE" if specA_kd['norm_eff_rank'] < specA_ce['norm_eff_rank']
                 else "KD NOT lower-rank than CE")),
    "pilot_B_space": dict(param_grad_KD=specA_kd, input_saliency_KD=specB_sal,
        verdict=("param-grad lower norm-rank than saliency"
                 if specA_kd['norm_eff_rank'] < specB_sal['norm_eff_rank']
                 else "saliency NOT higher norm-rank than param-grad")),
    "pilot_C_structural": dict(
        corr_structuralfrac_vs_loss=round(corr_struct_loss, 4),
        corr_gradnorm_vs_loss=round(corr_gnorm_loss, 4),
        struct_frac_highloss=round(struct_frac_hi, 4),
        struct_frac_lowloss=round(struct_frac_lo, 4),
        verdict=("structural != hard (weak corr)" if abs(corr_struct_loss) < 0.3
                 else "structural correlates with loss")),
    "pilot_D_coreset": results,
    "runtime_sec": round(time.time() - t0, 1),
}
with open("pilot/PILOT_RESULTS.json", "w") as f:
    json.dump(out, f, indent=2)

print("=" * 64)
print("PILOT: gradient-geometry diagnostics for KD data selection")
print("=" * 64)
print(f"teacher test acc = {teacher_acc:.3f} | student params = {P} | N = {N}")
print()
print("[A] KD vs CE per-sample gradient spectrum (across samples)")
print(f"  KD grad : norm_eff_rank={specA_kd['norm_eff_rank']:.4f}  "
      f"top5_energy={specA_kd['top5_energy']:.3f}  stable_rank={specA_kd['stable_rank']}")
print(f"  CE grad : norm_eff_rank={specA_ce['norm_eff_rank']:.4f}  "
      f"top5_energy={specA_ce['top5_energy']:.3f}  stable_rank={specA_ce['stable_rank']}")
print(f"  -> {out['pilot_A_spectrum']['verdict']}")
print()
print("[B] parameter-gradient space vs input-saliency space")
print(f"  param-grad : norm_eff_rank={specA_kd['norm_eff_rank']:.4f} (ambient {specA_kd['ambient_dim']})")
print(f"  saliency   : norm_eff_rank={specB_sal['norm_eff_rank']:.4f} (ambient {specB_sal['ambient_dim']})")
print(f"  -> {out['pilot_B_space']['verdict']}")
print()
print("[C] structural energy vs sample loss")
print(f"  corr(structural_frac, kd_loss) = {corr_struct_loss:+.3f}")
print(f"  struct_frac  high-loss={struct_frac_hi:.3f}  low-loss={struct_frac_lo:.3f}")
print(f"  -> {out['pilot_C_structural']['verdict']}")
print()
print("[D] coreset retraining (50% budget, mean test acc over 5 seeds)")
for k_ in ['full', 'd_optimal_grad', 'random', 'd_optimal_saliency', 'easy_lowloss', 'hard_highloss']:
    rd = results[k_]
    print(f"  {k_:20s} n={rd['n']:4d}  acc={rd['mean']:.4f} +/- {rd['std']:.4f}")
print()
print(f"runtime = {out['runtime_sec']}s  ->  pilot/PILOT_RESULTS.json")
