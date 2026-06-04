"""
Pilot v2: budget sweep for the D-optimal gradient-subspace coreset.

Pilot v1 tested only a 50% budget, where random selection already spans the
(rank ~23) gradient subspace, so D-optimal showed no advantage. Theory predicts
the D-optimal advantage appears when BUDGET is close to the intrinsic gradient
rank. This sweep checks budgets from ~1x to ~25x the intrinsic rank.

Methods: random / d_optimal (gradient-PCA) / easy_lowloss.
CPU-only, sklearn 'digits'. Runtime: ~1-2 min.
"""
import json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

t0 = time.time()
torch.manual_seed(0); np.random.seed(0)

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
Xte_t, yte_t = torch.tensor(Xte), torch.tensor(yte)
D_in, N_cls = X.shape[1], 10
T = 4.0

def make_mlp(hidden):
    layers, prev = [], D_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]; prev = h
    layers += [nn.Linear(prev, N_cls)]
    return nn.Sequential(*layers)

def acc(m, Xv, yv):
    with torch.no_grad():
        return (m(Xv).argmax(1) == yv).float().mean().item()

teacher = make_mlp([128, 128])
opt = torch.optim.Adam(teacher.parameters(), lr=0.02, weight_decay=1e-4)
for _ in range(250):
    opt.zero_grad(); F.cross_entropy(teacher(Xtr_t), ytr_t).backward(); opt.step()
teacher.eval()
with torch.no_grad():
    teach_logits = teacher(Xtr_t)

def fresh_student(seed=0):
    torch.manual_seed(seed); return make_mlp([32])

student = fresh_student(0)
params = [p for p in student.parameters()]
P = sum(p.numel() for p in params)
N = min(1200, Xtr_t.shape[0])
G_kd = np.zeros((N, P), dtype=np.float32)
kd_loss = np.zeros(N, dtype=np.float32)
for i in range(N):
    t_soft = F.softmax(teach_logits[i:i+1] / T, dim=1).detach()
    kd = (T * T) * F.kl_div(F.log_softmax(student(Xtr_t[i:i+1]) / T, dim=1),
                            t_soft, reduction='batchmean')
    g = torch.autograd.grad(kd, params)
    G_kd[i] = torch.cat([gi.reshape(-1) for gi in g]).numpy()
    kd_loss[i] = float(kd.detach())

_, S, Vt = np.linalg.svd(G_kd, full_matrices=False)
e = S**2 / (S**2).sum()
eff_rank = float(np.exp(-(e * np.log(e + 1e-12)).sum()))

def d_optimal(C, m, lam=1e-3):
    Npts, rr = C.shape
    A_inv = np.eye(rr) / lam
    chosen, avail = [], np.ones(Npts, bool)
    for _ in range(m):
        quad = np.einsum('ij,ij->i', C @ A_inv, C)
        quad[~avail] = -np.inf
        p = int(np.argmax(quad)); chosen.append(p); avail[p] = False
        g = C[p]; Ag = A_inv @ g
        A_inv = A_inv - np.outer(Ag, Ag) / (1.0 + g @ Ag)
    return np.array(chosen)

def kd_train(idx, seed, epochs=400):
    st = fresh_student(seed)
    op = torch.optim.Adam(st.parameters(), lr=0.03, weight_decay=1e-4)
    Xs = Xtr_t[:N][idx]; t_soft = F.softmax(teach_logits[:N][idx] / T, dim=1).detach()
    for _ in range(epochs):
        op.zero_grad()
        F.kl_div(F.log_softmax(st(Xs) / T, dim=1), t_soft, reduction='batchmean').mul(T*T).backward()
        op.step()
    return acc(st, Xte_t, yte_t)

seeds = [1, 2, 3, 4, 5]
budgets = [24, 48, 96, 192, 384, 600]
r = 20
coords = G_kd @ Vt[:r].T
table = {}
for m in budgets:
    sel_d = d_optimal(coords, m)
    sel_e = np.argsort(kd_loss)[:m]
    a_d = [kd_train(sel_d, s) for s in seeds]
    a_e = [kd_train(sel_e, s) for s in seeds]
    a_r = [kd_train(np.random.RandomState(s).choice(N, m, replace=False), s) for s in seeds]
    table[m] = dict(
        d_optimal=[round(float(np.mean(a_d)), 4), round(float(np.std(a_d)), 4)],
        random=[round(float(np.mean(a_r)), 4), round(float(np.std(a_r)), 4)],
        easy_lowloss=[round(float(np.mean(a_e)), 4), round(float(np.std(a_e)), 4)],
        lift_dopt_vs_random=round(float(np.mean(a_d) - np.mean(a_r)), 4))

out = dict(intrinsic_eff_rank=round(eff_rank, 2), n_pool=N, budgets=budgets,
           sweep=table, runtime_sec=round(time.time() - t0, 1))
with open("pilot/PILOT_BUDGET_SWEEP.json", "w") as f:
    json.dump(out, f, indent=2)

print("=" * 70)
print(f"BUDGET SWEEP  (intrinsic effective rank of gradient matrix = {eff_rank:.1f})")
print("=" * 70)
print(f"{'budget':>7} {'d_optimal':>16} {'random':>16} {'easy_lowloss':>16} {'lift':>8}")
for m in budgets:
    rd = table[m]
    print(f"{m:>7} {rd['d_optimal'][0]:>10.4f}+-{rd['d_optimal'][1]:.3f} "
          f"{rd['random'][0]:>10.4f}+-{rd['random'][1]:.3f} "
          f"{rd['easy_lowloss'][0]:>10.4f}+-{rd['easy_lowloss'][1]:.3f} "
          f"{rd['lift_dopt_vs_random']:>+8.4f}")
print(f"\nruntime = {out['runtime_sec']}s  ->  pilot/PILOT_BUDGET_SWEEP.json")
