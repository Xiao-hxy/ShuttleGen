# -*- coding: utf-8 -*-
"""
Deep Ensembles —— 5 × v1 独立训练的 GRU 点回归集成
======================================================
对比目的：这是"最便宜的分布 baseline"——审稿人最可能问的就是它。

方法：
  训练 5 个独立的 PointBaseline（v1 架构、不同随机种子），每个输出单点 μ_i。
  集成分布 = (1/5) Σ_i N(μ_i, σ²)，其中 σ 从各模型的 val 集残差估计。

指标计算：
  - MAE = 集成期望点 (Σ μ_i / 5) 与真值的 MAE（与 v1 同协议）
  - NLL = -log(平均高斯密度)
  - minADE-5 = 采 5 个分量中心，取最优的 MAE
  - Coverage@90% = 用采样法估 HPD，与 v2/v3/v4 同协议

运行：
    python deep_ensemble.py smoke   # 1 个 seed × 2 epoch
    python deep_ensemble.py         # 5 seed 完整训练（CPU 约 20 分钟）
产物：results/deep_ensemble.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_pipeline import (ShuttleGenDataset, collate_fn, OUT_DIR,
                           MEAN_X, STD_X, MEAN_Y, STD_Y)
from baseline_v1 import (PointBaseline, mae_report,
                         BATCH, LR, WEIGHT_DECAY, EPOCHS, PATIENCE)

RESULTS = Path(__file__).resolve().parent / "results"
N_MEMBERS = 5
SEEDS = [1, 42, 123, 2024, 9999]
S_COV = 200


def to_real(t):
    return torch.stack([t[..., 0] * STD_X + MEAN_X,
                        t[..., 1] * STD_Y + MEAN_Y], dim=-1)


# ============ 单模型训练（复制 baseline_v1 的逻辑，加 seed 参数）============

def train_one(seed, epochs, max_batches=None):
    """训练一个 PointBaseline，返回 (best_state, val_mae, val_residual_std, val_preds, test_preds)。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    dl_train = DataLoader(ShuttleGenDataset(OUT_DIR / "train.npz"),
                          batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ShuttleGenDataset(OUT_DIR / "val.npz"),
                        batch_size=BATCH, collate_fn=collate_fn)
    dl_test = DataLoader(ShuttleGenDataset(OUT_DIR / "test.npz"),
                         batch_size=BATCH, collate_fn=collate_fn)

    model = PointBaseline(n_players=35).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_mae, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(1, epochs + 1):
        model.train()
        for bi, (bx, by) in enumerate(dl_train, 1):
            if max_batches and bi > max_batches:
                break
            pred = model(bx, device)
            loss = F.mse_loss(pred, by["landing"].to(device))
            opt.zero_grad(); loss.backward(); opt.step()

        # val
        model.eval()
        val_preds, val_truths = [], []
        with torch.no_grad():
            for bx, by in dl_val:
                val_preds.append(model(bx, device).cpu())
                val_truths.append(by["landing"])
        vp = torch.cat(val_preds); vt = torch.cat(val_truths)
        val_mae = mae_report(to_real(vp), to_real(vt))["mae_mean"]

        if val_mae < best_val_mae:
            best_val_mae, best_epoch = val_mae, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if max_batches:
            continue
        if epoch - best_epoch >= PATIENCE:
            break

    # 加载最优权重，跑 test
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_truths = [], []
    val_preds2, val_truths2 = [], []
    with torch.no_grad():
        for bx, by in dl_val:
            val_preds2.append(model(bx, device).cpu())
            val_truths2.append(by["landing"])
        for bx, by in dl_test:
            test_preds.append(model(bx, device).cpu())
            test_truths.append(by["landing"])

    vp = torch.cat(val_preds2); vt = torch.cat(val_truths2)
    # 估计该模型的 per-dim residual std（标准化空间）
    res_x = (vp[:, 0] - vt[:, 0]).std().item()
    res_y = (vp[:, 1] - vt[:, 1]).std().item()

    tp = torch.cat(test_preds); tt = torch.cat(test_truths)
    return best_state, best_val_mae, (res_x, res_y), tp, tt


# ============ 主流程 ============

def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    epochs = 2 if smoke else EPOCHS
    max_batches = 15 if smoke else None

    print(f"=== Deep Ensembles ({N_MEMBERS} members, seeds={SEEDS}) ===")
    print(f"模式: {'冒烟' if smoke else '完整训练'}")
    t0 = time.time()

    all_test_mus, all_val_sigmas, all_val_maes = [], [], []
    member_states = []

    for i, seed in enumerate(SEEDS):
        print(f"\n--- 成员 {i+1}/{N_MEMBERS} (seed={seed}) ---")
        state, val_mae, (rx, ry), test_mu, test_truth = train_one(
            seed, epochs, max_batches)
        print(f"  best val MAE={val_mae:.2f} epoch={len(member_states)+1} "
              f"residual σ: x={rx:.4f} y={ry:.4f}")
        all_test_mus.append(test_mu)          # (N_test, 2) 标准化
        all_val_sigmas.append((rx, ry))
        all_val_maes.append(val_mae)
        member_states.append(state)

    # 用第一个模型的 test_truth（所有模型 test set 相同）
    y = test_truth
    mus = torch.stack(all_test_mus)          # (5, N, 2)

    # 平均 residual sigma 作为集成分布的分量标准差
    sigmas = torch.tensor([[rx, ry] for rx, ry in all_val_sigmas]).mean(0)  # (2,)
    sigma = sigmas.unsqueeze(0).expand(N_MEMBERS, -1)                        # (5, 2)

    N = len(y)
    pi_logits = torch.zeros(N, N_MEMBERS)                                    # 均匀混合

    print(f"\n=== 集成评估 ===")
    print(f"平均 residual σ（标准化空间）: x={sigmas[0].item():.4f} y={sigmas[1].item():.4f}")

    # —— MAE：集成期望点 ——
    mean_norm = mus.mean(0)                                                  # (N, 2)
    mae = mae_report(to_real(mean_norm), to_real(y))
    print(f"MAE(集成期望) = {mae['mae_mean']:.2f}   (v1 单模型 = 76.78)")

    # —— NLL：混合高斯 log pdf ——
    pi = F.log_softmax(pi_logits, dim=-1)          # (N, 5) log(0.2) each
    lp_per_comp = torch.zeros(N, N_MEMBERS)
    for c in range(N_MEMBERS):
        lp_x = Normal(mus[c, :, 0], sigma[c, 0]).log_prob(y[:, 0])
        lp_y = Normal(mus[c, :, 1], sigma[c, 1]).log_prob(y[:, 1])
        lp_per_comp[:, c] = lp_x + lp_y
    nll = -(pi + lp_per_comp).logsumexp(-1).mean().item()
    print(f"NLL = {nll:.3f}   (v2=1.377 v3=1.316 v4=1.211)")

    # —— minADE-5：用 5 个分量中心当假设 ——
    hyp = to_real(mus)                          # (5, N, 2) 真实坐标
    truth = to_real(y).unsqueeze(0)             # (1, N, 2)
    d = (hyp - truth).abs().sum(-1) / 2         # (5, N)
    minade5 = d.min(0).values.mean().item()
    print(f"minADE-5 = {minade5:.2f}")

    # —— Coverage@90% —— 采样法
    torch.manual_seed(42)
    # 采 S_COV × N_MEMBERS 个样本（每个分量采 S_COV/5 个）
    n_per_comp = S_COV // N_MEMBERS
    samples = []
    for c in range(N_MEMBERS):
        sx = Normal(mus[c, :, 0], sigma[c, 0]).sample((n_per_comp,))  # (n_pc, N)
        sy = Normal(mus[c, :, 1], sigma[c, 1]).sample((n_per_comp,))
        samples.append(torch.stack([sx, sy], dim=-1))                  # (n_pc, N, 2)
    samples = torch.cat(samples, dim=0)                                # (S_COV, N, 2)
    # 计算每个样本的 log pdf（混合）
    lp_samples = torch.zeros(S_COV, N, N_MEMBERS)
    for c in range(N_MEMBERS):
        lp_x = Normal(mus[c, :, 0], sigma[c, 0]).log_prob(samples[:, :, 0])
        lp_y = Normal(mus[c, :, 1], sigma[c, 1]).log_prob(samples[:, :, 1])
        lp_samples[:, :, c] = lp_x + lp_y
    s_lp = (pi.unsqueeze(0) + lp_samples).logsumexp(-1)                # (S_COV, N)
    thr = torch.quantile(s_lp, 0.10, dim=0)                             # (N,)
    # 真值的 log pdf
    lp_true = (pi + lp_per_comp).logsumexp(-1)                         # (N,)
    cov = (lp_true >= thr).float().mean().item()
    print(f"Cov@90% = {cov:.3f}   (理想 0.90)")

    elapsed = (time.time() - t0) / 60
    print(f"\n总耗时 {elapsed:.1f} 分钟")

    # —— 保存 ——
    RESULTS.mkdir(exist_ok=True)
    for i, (seed, state) in enumerate(zip(SEEDS, member_states)):
        torch.save(state, RESULTS / f"de_member_{i}_seed{seed}.pt")

    results = {
        "model": "Deep_Ensembles",
        "n_members": N_MEMBERS,
        "seeds": SEEDS,
        "member_val_mae": [round(v, 3) for v in all_val_maes],
        "sigma_norm": {"x": sigmas[0].item(), "y": sigmas[1].item()},
        "protocol": "MAE: ensemble mean; distribution: uniform-weighted 5-component Gaussian mixture",
        "test": {
            "mae_x": round(mae["mae_x"], 3),
            "mae_y": round(mae["mae_y"], 3),
            "mae_mean": round(mae["mae_mean"], 3),
            "nll": round(nll, 3),
            "minade5": round(minade5, 3),
            "cov90": round(cov, 3),
        },
    }

    (RESULTS / "deep_ensemble.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n指标 → {RESULTS/'deep_ensemble.json'}")
    print("\n===== 主结果表更新 =====")
    print(f"{'模型':<25} {'MAE':>7} {'minADE5':>8} {'NLL':>7} {'Cov90':>7}")
    print(f"{'v1 点回归':<25} {76.78:>7.2f} {'—':>8} {'—':>7} {'—':>7}")
    print(f"{'Deep Ensembles (5×v1)':<25} {mae['mae_mean']:>7.2f} {minade5:>8.2f} "
          f"{nll:>7.3f} {cov:>7.3f}")
    print(f"{'v2 MDN':<25} {77.92:>7.2f} {39.56:>8.2f} {1.377:>7.3f} {0.863:>7.3f}")
    print(f"{'v4 TGMM':<25} {76.91:>7.2f} {38.64:>8.2f} {1.211:>7.3f} {0.895:>7.3f}")


if __name__ == "__main__":
    main()
