# -*- coding: utf-8 -*-
"""
截断边界消融实验 —— 收紧边界 vs v4 原边界
===========================================
目的：验证 v4 的截断边界是否设得太宽（没截到东西）。

  v4 原边界（安全缓冲）: x∈[-3.5,3.6], y∈[-2.0,2.0]
  本次紧边界（真实数据极值）: x∈[-2.8,3.4], y∈[-1.8,1.96]

紧边界来自 _probe 输出（train+val+test 的落点标准化坐标极值）：
  x∈[-3.480, 3.524], y∈[-1.962, 1.953]
  保守取 x∈[-2.8,3.4], y∈[-1.8,1.96]（比真实极值各收 0.1-0.2σ）

其他一切与 tgmm_v4.py 相同（同 encoder、同 optimizer、同训练），只改边界常数。
产物写入 results/v4b_tight.json / .pt / .npz
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tgmm_v4 import (to_real, full_tgmm, _TGMMWrapper, trunc_log_prob_2d,
                    mixture_metrics_tgmm, collect_and_evaluate, S_COV, K_EXP)
from data_pipeline import ShuttleGenDataset, collate_fn, OUT_DIR, TYPE_NAMES
from baseline_v1 import BATCH, LR, WEIGHT_DECAY, EPOCHS, PATIENCE, SEED
from moe_v3 import TwoStageMoE

# === 边界常数 ===
X_LOW, X_HIGH = -2.8, 3.4     # 紧：比真实极值各收 0.2σ
Y_LOW, Y_HIGH = -1.8, 1.96    # 紧：y 方向真实极值 [-1.96, 1.95]

# 关键：patch tgmm_v4 模块里的边界常数，让 trunc_log_prob_2d 等函数用到紧边界
import tgmm_v4 as v4mod
v4mod.X_LOW, v4mod.X_HIGH = X_LOW, X_HIGH
v4mod.Y_LOW, v4mod.Y_HIGH = Y_LOW, Y_HIGH

# 导入 patch 后才能正确工作的函数
# 注意：trunc_log_prob_2d 已经在导入时用了 v4mod 的边界（运行时读全局变量，OK）

RESULTS = Path(__file__).resolve().parent / "results"

print(f"=== v4b 截断边界消融（紧边界）===")
print(f"边界: x∈[{X_LOW},{X_HIGH}] y∈[{Y_LOW},{Y_HIGH}]")

torch.manual_seed(SEED); np.random.seed(SEED)
dl_train = DataLoader(ShuttleGenDataset(OUT_DIR/"train.npz"), batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
dl_val = DataLoader(ShuttleGenDataset(OUT_DIR/"val.npz"), batch_size=BATCH, collate_fn=collate_fn)
dl_test = DataLoader(ShuttleGenDataset(OUT_DIR/"test.npz"), batch_size=BATCH, collate_fn=collate_fn)

model = TwoStageMoE(n_players=35)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
ce = nn.CrossEntropyLoss()

best_nll, best_epoch, best_state = float("inf"), -1, None
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    for bi, (bx, by) in enumerate(dl_train, 1):
        g, e_pi, e_mu, e_sigma = model(bx, torch.device("cpu"))
        y_type = by["type_id"]; y_land = by["landing"]
        loss_gate = ce(g, y_type)
        b_idx = torch.arange(len(y_type))
        t_pi = e_pi[b_idx, y_type]; t_mu = e_mu[b_idx, y_type]; t_sg = e_sigma[b_idx, y_type]
        exp_tgmm = _TGMMWrapper(t_pi, t_mu, t_sg, n_comp=t_pi.shape[1])
        loss_land = -exp_tgmm.log_prob(y_land).mean()
        loss = loss_gate + loss_land
        opt.zero_grad(); loss.backward(); opt.step()

    vm, _ = collect_and_evaluate(model, dl_val, torch.device("cpu"))
    f = vm["full"]
    flag = ""
    if f["nll"] < best_nll:
        best_nll, best_epoch = f["nll"], epoch
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        flag = " ← 最好"
    print(f"epoch {epoch:>2} | val: NLL={f['nll']:.3f} MAE={f['mae_mean']:.2f} "
          f"minADE5={f['minade5']:.2f} Cov={f['cov90']:.3f}{flag}")
    if epoch - best_epoch >= PATIENCE: break

print(f"\n训练结束（{(time.time()-t0)/60:.1f} min），加载 epoch {best_epoch} 最优")
model.load_state_dict(best_state)
vm, _ = collect_and_evaluate(model, dl_val, torch.device("cpu"))
tm, _ = collect_and_evaluate(model, dl_test, torch.device("cpu"))

RESULTS.mkdir(exist_ok=True)
torch.save(best_state, RESULTS/"v4b_tight.pt")
results = {
    "model": "v4b_tight_boundary_ablation",
    "k_expert": K_EXP,
    "boundaries_norm": {"x": [X_LOW, X_HIGH], "y": [Y_LOW, Y_HIGH]},
    "selection": "val NLL (truncated soft routing)", "seed": SEED,
    "protocol": "同 v4，但边界更紧",
    "val": vm, "test": tm,
    "v4_wide_target": {"mae": 76.91, "minade5": 38.64, "nll": 1.211, "cov90": 0.895, "oracle_mae": 50.10},
}
(RESULTS/"v4b_tight.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

f, o = tm["full"], tm["oracle"]
print(f"\n===== v4b 紧边界结果 =====")
print(f"{'':16} {'MAE':>7} {'minADE5':>8} {'NLL':>7} {'Cov90':>7}")
print(f"{'软路由':<16} {f['mae_mean']:>7.2f} {f['minade5']:>8.2f} {f['nll']:>7.3f} {f['cov90']:>7.3f}")
print(f"{'oracle':<16} {o['mae_mean']:>7.2f} {o['minade5']:>8.2f} {o['nll']:>7.3f} {o['cov90']:>7.3f}")
print(f"\n与 v4 宽边界对比：")
print(f"v4  宽边界: MAE=76.91 minADE5=38.64 NLL=1.211 Cov=0.895")
print(f"v4b 紧边界: MAE={f['mae_mean']:.2f} minADE5={f['minade5']:.2f} NLL={f['nll']:.3f} Cov={f['cov90']:.3f}")

