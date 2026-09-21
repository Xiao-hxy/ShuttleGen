# -*- coding: utf-8 -*-
"""
MC Dropout baseline —— 用 v1 点回归的 dropout 做不确定性估计
===============================================================
做法（Gal & Ghahramani 2016 的经典方法）：
  1. 加载 v1 PointBaseline 权重（不需要重训）
  2. 推理时 model.train() 保持 dropout 打开，前向跑 S=20 次
  3. 得到 S 个预测点 μ_i (B, 2)，每个预测被解释为一个高斯分量的中心
  4. 共享 v1 残差估计的 σ，构造均匀混合的 20 分量高斯分布
  5. 同协议算四个指标：MAE / minADE-5 / Cov@90% / NLL

为什么做这个？Deep Ensembles 是"多模型方差"，MC Dropout 是"单模型 dropout 方差"——
两者是不确定性估计的两大流派，审稿人大概率会问。而且 MC Dropout 零训练成本。

运行：python mc_dropout.py
产物：results/mc_dropout.json
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pipeline import (ShuttleGenDataset, collate_fn, OUT_DIR,
                           MEAN_X, STD_X, MEAN_Y, STD_Y)
from baseline_v1 import PointBaseline, mae_report, BATCH, SEED

RESULTS = Path(__file__).resolve().parent / "results"
S_FORWARD = 20   # MC Dropout 前向次数
S_COV = 200

V1_MAE = 76.78
V4 = {"mae": 76.91, "minade5": 38.64, "nll": 1.211, "cov90": 0.895}
DE = {"mae": 76.47, "minade5": 64.62, "nll": 2.148, "cov90": 0.943}


def to_real(t):
    return torch.stack([t[..., 0] * STD_X + MEAN_X,
                        t[..., 1] * STD_Y + MEAN_Y], dim=-1)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    print("=== MC Dropout Baseline ===")
    print(f"加载 v1 权重，推理时跑 {S_FORWARD} 次前向，保持 dropout 打开")

    model = PointBaseline(n_players=35).to(device)
    model.load_state_dict(torch.load(str(RESULTS / "v1_point_baseline.pt"),
                                      weights_only=True))
    # 关键：不调用 model.eval() —— 保持 dropout 层激活
    print(f"参数量: {sum(p.numel() for p in model.parameters())/1e4:.1f} 万")

    dl_val = DataLoader(ShuttleGenDataset(OUT_DIR/"val.npz"),
                        batch_size=BATCH, collate_fn=collate_fn)
    dl_test = DataLoader(ShuttleGenDataset(OUT_DIR/"test.npz"),
                         batch_size=BATCH, collate_fn=collate_fn)

    # 1) 用 val 集估计 residual σ（标准化空间）
    print("\n用 val 集估计 residual σ...")
    val_preds, val_truths = [], []
    with torch.no_grad():
        for bx, by in dl_val:
            val_preds.append(model(bx, device).cpu())
            val_truths.append(by["landing"])
    vp = torch.cat(val_preds); vt = torch.cat(val_truths)
    sigma_x = (vp[:, 0] - vt[:, 0]).std().item()
    sigma_y = (vp[:, 1] - vt[:, 1]).std().item()
    print(f"σ（标准化空间）: x={sigma_x:.4f}, y={sigma_y:.4f}")

    # 2) MC Dropout 推理（保持 dropout，跑 S 次）
    print(f"\nMC Dropout 推理（S={S_FORWARD} 次前向）...")
    all_preds = []                       # list of (N_test, 2)，S 个
    ys = []
    with torch.no_grad():
        for bx, by in dl_test:
            # 每次前向有不同 dropout mask → 不同预测
            batch_preds = []
            for _ in range(S_FORWARD):
                batch_preds.append(model(bx, device).cpu())
            all_preds.append(torch.stack(batch_preds))   # (S, B, 2)
            ys.append(by["landing"])

    preds = torch.cat(all_preds, dim=1)    # (S, N, 2) 标准化空间
    y = torch.cat(ys)                       # (N, 2)

    N = y.shape[0]
    sigma = torch.tensor([sigma_x, sigma_y])   # (2,)
    pi_logits = torch.zeros(N, S_FORWARD)      # 均匀混合

    # 3) MAE：S 个预测的均值
    mean_norm = preds.mean(0)                 # (N, 2)
    mae = mae_report(to_real(mean_norm), to_real(y))

    # 4) NLL：均匀混合 S 个高斯，共享残差 σ
    # 每个分量 N(μ_i, σ²)，σ 从 v1 残差估计
    lp_per_comp = torch.zeros(N, S_FORWARD)
    for c in range(S_FORWARD):
        lp_x = D.Normal(preds[c, :, 0], sigma[0]).log_prob(y[:, 0])
        lp_y = D.Normal(preds[c, :, 1], sigma[1]).log_prob(y[:, 1])
        lp_per_comp[:, c] = lp_x + lp_y
    nll = -(F.log_softmax(pi_logits, -1) + lp_per_comp).logsumexp(-1).mean().item()

    # 5) minADE-5：从 S 个预测中随机抽 5 个当假设
    torch.manual_seed(SEED)
    idx = torch.randperm(S_FORWARD)[:5]
    hyp = to_real(preds[idx])                  # (5, N, 2)
    truth = to_real(y).unsqueeze(0)
    d = (hyp - truth).abs().sum(-1) / 2
    minade5 = d.min(0).values.mean().item()

    # 6) Coverage@90%
    s_lp_all = []
    for c in range(S_FORWARD):
        lp_x = D.Normal(preds[c, :, 0], sigma[0]).log_prob(y[:, 0])
        lp_y = D.Normal(preds[c, :, 1], sigma[1]).log_prob(y[:, 1])
        s_lp_all.append(lp_x + lp_y)
    lp_true_all = torch.stack(s_lp_all, dim=1)   # (N, S)
    # 先从每个分量采 S_COV/S_FORWARD 个样本，统一算 logpdf
    per_cov = S_COV // S_FORWARD
    all_logps = []
    for c in range(S_FORWARD):
        sx = D.Normal(preds[c, :, 0], sigma[0]).sample((per_cov,))  # (per_cov, N)
        sy = D.Normal(preds[c, :, 1], sigma[1]).sample((per_cov,))
        lp_s = D.Normal(preds[c, :, 0], sigma[0]).log_prob(sx) + \
               D.Normal(preds[c, :, 1], sigma[1]).log_prob(sy)     # (per_cov, N)
        all_logps.append(lp_s)
    s_lp = torch.cat(all_logps, dim=0)          # (S_COV, N)
    thr = torch.quantile(s_lp, 0.10, dim=0)
    # 每个样本的 log pdf = Σ π_c · p(y|c) → logsumexp
    s_lp_weighted = F.log_softmax(pi_logits, -1) + lp_per_comp   # (N, S)
    lp_true_mix = s_lp_weighted.logsumexp(-1)                     # (N,)
    cov = (lp_true_mix >= thr).float().mean().item()

    print("\n===== MC Dropout 测试结果 =====")
    print(f"MAE(均值)  = {mae['mae_mean']:.2f}   (v1={V1_MAE}, v4={V4['mae']})")
    print(f"minADE-5  = {minade5:.2f}   (v4={V4['minade5']})")
    print(f"NLL       = {nll:.3f}   (DE={DE['nll']}, v4={V4['nll']})")
    print(f"Cov@90%   = {cov:.3f}   (理想 0.90, DE={DE['cov90']}, v4={V4['cov90']})")
    print(f"\nΔx={mae['mae_x']:.2f}, Δy={mae['mae_y']:.2f}")

    RESULTS.mkdir(exist_ok=True)
    results = {
        "model": "MC_Dropout",
        "s_forward": S_FORWARD,
        "sigma_norm": {"x": sigma_x, "y": sigma_y},
        "protocol": "MC Dropout on v1 PointBaseline; sigma from v1 val residual; mixture of S_FORWARD Gaussians",
        "test": {
            "mae_x": round(mae["mae_x"], 3),
            "mae_y": round(mae["mae_y"], 3),
            "mae_mean": round(mae["mae_mean"], 3),
            "nll": round(nll, 3),
            "minade5": round(minade5, 3),
            "cov90": round(cov, 3),
        },
        "targets": {"v1": V1_MAE, "deep_ensemble": DE, "v4": V4},
    }
    (RESULTS / "mc_dropout.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n指标 → {RESULTS/'mc_dropout.json'}")

    print("\n===== 主结果表 =====")
    print(f"{'模型':<25} {'MAE':>7} {'minADE5':>8} {'NLL':>7} {'Cov90':>7}")
    print(f"{'v1 点回归':<25} {V1_MAE:>7.2f} {'—':>8} {'—':>7} {'—':>7}")
    print(f"{'Deep Ensembles':<25} {DE['mae']:>7.2f} {DE['minade5']:>8.2f} {DE['nll']:>7.3f} {DE['cov90']:>7.3f}")
    print(f"{'MC Dropout':<25} {mae['mae_mean']:>7.2f} {minade5:>8.2f} {nll:>7.3f} {cov:>7.3f}")
    print(f"{'v2 MDN':<25} {77.92:>7.2f} {39.56:>8.2f} {1.377:>7.3f} {0.863:>7.3f}")
    print(f"{'v4 TGMM (ours)':<25} {V4['mae']:>7.2f} {V4['minade5']:>8.2f} {V4['nll']:>7.3f} {V4['cov90']:>7.3f}")


if __name__ == "__main__":
    main()
