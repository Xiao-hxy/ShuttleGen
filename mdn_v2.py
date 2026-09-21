# -*- coding: utf-8 -*-
"""
ShuttleGen v2 —— MDN（混合密度网络）：分布估计的最小对照实验
================================================================
与 v1 的关系（控制变量法）：
  编码器一字不改（直接继承 v1 的类），只把"输出一个点"的 MLP 头
  换成"输出 K 个高斯分量"的混合头。于是 v1 vs v2 的全部差异
  = 任务定义的差异（点回归 vs 条件分布估计）。

模型输出（K=5 个分量，共 5K 维）：
  π_k  混合权重（softmax 归一）
  μ_k  每个分量的中心 (x, y)
  σ_k  每个分量的对角标准差（softplus 保证为正）

训练损失：负对数似然 NLL = -log Σ_k π_k · N(y | μ_k, σ_k)
  与 v1 的 MSE 是本质不同的训练信号：它奖励"把概率质量铺在真相周围"，
  而不是"猜一个平均值"。

本版新增的分布指标（论文主结果表的 v2 行）：
  MAE(K=1)     用混合期望 E[X] 当单点预测（与 v1 同协议， apples-to-apples）
  minADE-5     从分布采 5 个假设取最近——"防住任意一个"的误差
  NLL          测试集平均负对数似然（标准化空间）
  Coverage@90% 真值是否落在 90% 最高密度区内（采样近似 HPD，理想值 0.90）

运行：
    python mdn_v2.py smoke   # 冒烟
    python mdn_v2.py         # 完整训练（CPU 约 4 分钟）
产物（results/）：
    v2_mdn.json / v2_mdn.pt / v2_test_predictions.npz / v2_mdn_density.png
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from data_pipeline import (ShuttleGenDataset, collate_fn, OUT_DIR,
                           MEAN_X, STD_X, MEAN_Y, STD_Y)
from baseline_v1 import (PointBaseline, real_coords, mae_report,
                         BATCH, LR, WEIGHT_DECAY, EPOCHS, PATIENCE, SEED)

RESULTS = Path(__file__).resolve().parent / "results"
K = 5          # 高斯分量数
S_COV = 200    # Coverage 的采样近似数
V1_TEST_MAE = 76.78   # v1 立的靶子（对照打印用）


# ============ 模型：v1 编码器 + MDN 头 ============

class MDNBaseline(PointBaseline):
    """继承 v1 → 编码器（嵌入/GRU/球员条件）全部复用，只替换 head。"""

    def __init__(self, n_types=10, n_players=35, type_dim=16,
                 player_dim=8, hidden=128, k=K):
        super().__init__(n_types, n_players, type_dim, player_dim, hidden)
        self.k = k
        self.head = nn.Sequential(                       # 5K 维输出
            nn.Linear(hidden + player_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 5 * k),
        )

    def forward(self, bx, device):
        # —— 编码部分与 v1 完全一致（"只换头"是字面意思）——
        num = torch.cat([
            bx["landing"], bx["player_loc"], bx["opp_loc"],
            bx["scores"].float() / 21.0,
            bx["ball_round"].float().unsqueeze(-1) / 20.0,
            bx["backhand"].float().unsqueeze(-1),
            bx["aroundhead"].float().unsqueeze(-1),
        ], dim=-1).to(device)
        seq = torch.cat([
            self.type_emb(bx["type_id"].to(device)),
            self.player_emb(bx["player_id"].to(device)),
            num,
        ], dim=-1)
        lengths = bx["mask"].sum(1).cpu()
        packed = pack_padded_sequence(seq, lengths,
                                      batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h = h[-1]
        nxt = self.player_emb(bx["next_player_id"].to(device))
        raw = self.head(torch.cat([h, nxt], dim=-1))     # (B, 5K)

        # —— 拆参数：π 走 log_softmax（数值稳）；σ 过 softplus（恒正）——
        pi = F.log_softmax(raw[:, :self.k], dim=-1)                # (B,K) log π
        mu = raw[:, self.k:3 * self.k].view(-1, self.k, 2)        # (B,K,2)
        sigma = F.softplus(raw[:, 3 * self.k:]).view(-1, self.k, 2) + 1e-3
        return pi, mu, sigma


def build_mixture(pi_logits, mu, sigma):
    """把三组参数装进 torch 的混合分布（log_prob / sample 全都用它）"""
    comp = D.Independent(D.Normal(mu, sigma), 1)          # 对角二维高斯
    return D.MixtureSameFamily(D.Categorical(logits=pi_logits), comp)


def to_real(t):
    """标准化 → 真实坐标，支持任意前导维度 (..., 2)。
    注意：v1 的 real_coords 只认 (N, 2)，对 (5,B,2) 这种采样张量会把
    维度搞混（本文件第一版 minADE=326 的 bug 就是这么来的）。"""
    return torch.stack([t[..., 0] * STD_X + MEAN_X,
                        t[..., 1] * STD_Y + MEAN_Y], dim=-1)


# ============ 评估：四个指标一次算齐 ============

@torch.no_grad()
def collect_and_evaluate(model, loader, device):
    """收齐测试集分布参数后统一算：MAE / minADE-5 / NLL / Coverage@90%"""
    model.eval()
    pis, mus, sigmas, ys = [], [], [], []
    for bx, by in loader:
        pi, mu, sigma = model(bx, device)
        pis.append(pi.cpu()); mus.append(mu.cpu()); sigmas.append(sigma.cpu())
        ys.append(by["landing"])                           # 标准化坐标 (B,2)
    pi, mu, sigma, y = (torch.cat(t) for t in (pis, mus, sigmas, ys))
    mix = build_mixture(pi, mu, sigma)

    # 1) MAE(K=1)：混合期望做单点预测，换算真实坐标后与 v1 同协议
    mean_norm = (pi.exp().unsqueeze(-1) * mu).sum(1)       # E[X] (B,2)
    mae = mae_report(real_coords(mean_norm), real_coords(y))

    # 2) NLL（标准化空间；与真实空间的差只是常数 log-det，不影响模型间比较）
    nll = -mix.log_prob(y).mean().item()

    # 3) minADE-5：采 5 个假设，取"最好的那个"的误差
    torch.manual_seed(SEED)                                # 指标采样可复现
    hyp = to_real(mix.sample((5,)))                        # (5,B,2)
    truth = to_real(y).unsqueeze(0)                        # (1,B,2)
    d = ((hyp - truth).abs().sum(-1)) / 2                  # (5,B) 每假设的 MAE
    minade5 = d.min(0).values.mean().item()

    # 4) Coverage@90%：真值的 logpdf 若不低于采样 logpdf 的 10% 分位 → 被覆盖
    s_lp = mix.log_prob(mix.sample((S_COV,)))              # (S,B)
    thr = torch.quantile(s_lp, 0.10, dim=0)                # (B,)
    cov = (mix.log_prob(y) >= thr).float().mean().item()

    return {"mae_x": mae["mae_x"], "mae_y": mae["mae_y"], "mae_mean": mae["mae_mean"],
            "nll": round(nll, 3), "minade5": round(minade5, 3),
            "coverage90": round(cov, 3)}, (pi, mu, sigma, y, mean_norm)


# ============ 主流程 ============

def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}｜模式: {'冒烟' if smoke else '完整训练'}｜K={K}")

    dl_train = DataLoader(ShuttleGenDataset(OUT_DIR / "train.npz"),
                          batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ShuttleGenDataset(OUT_DIR / "val.npz"),
                        batch_size=BATCH, collate_fn=collate_fn)
    dl_test = DataLoader(ShuttleGenDataset(OUT_DIR / "test.npz"),
                         batch_size=BATCH, collate_fn=collate_fn)

    model = MDNBaseline(n_players=35).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e4:.1f} 万")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_nll, best_epoch, best_state = float("inf"), -1, None
    t0 = time.time()
    max_batches = 15 if smoke else None

    for epoch in range(1, (2 if smoke else EPOCHS) + 1):
        model.train()
        for bi, (bx, by) in enumerate(dl_train, 1):
            if max_batches and bi > max_batches:
                break
            pi, mu, sigma = model(bx, device)
            loss = -build_mixture(pi, mu, sigma).log_prob(
                by["landing"].to(device)).mean()           # NLL
            opt.zero_grad(); loss.backward(); opt.step()
            if bi % 100 == 0:
                print(f"  epoch {epoch} batch {bi}/{len(dl_train)} nll={loss.item():.4f}")

        # 验证集：模型选择用 NLL（分布模型按自己的目标选，与 v1 用 MAE 对等）
        vm, _ = collect_and_evaluate(model, dl_val, device)
        flag = ""
        if vm["nll"] < best_nll:
            best_nll, best_epoch = vm["nll"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = " ← 目前最好"
        print(f"epoch {epoch:>2} | val: MAE={vm['mae_mean']:.2f} NLL={vm['nll']:.3f} "
              f"minADE5={vm['minade5']:.2f} Cov90={vm['coverage90']:.3f}{flag}")

        if smoke:
            print("冒烟通过：分布头 / NLL / 四指标链路无 bug")
            return
        if epoch - best_epoch >= PATIENCE:
            print(f"早停：val NLL 已 {PATIENCE} 个 epoch 无改善")
            break

    print(f"\n训练结束（{(time.time()-t0)/60:.1f} 分钟），加载 epoch {best_epoch} 最优权重")
    model.load_state_dict(best_state)

    vm, _ = collect_and_evaluate(model, dl_val, device)
    tm, (pi, mu, sigma, y, mean_norm) = collect_and_evaluate(model, dl_test, device)

    RESULTS.mkdir(exist_ok=True)
    torch.save(best_state, RESULTS / "v2_mdn.pt")
    np.savez_compressed(RESULTS / "v2_test_predictions.npz",
                        pi=pi.numpy(), mu=mu.numpy(), sigma=sigma.numpy(),
                        truth=y.numpy(), mean_pred=mean_norm.numpy())
    results = {"model": "v2_mdn", "k": K, "selection": "val NLL", "seed": SEED,
               "protocol": "MAE/minADE 真实坐标单位；NLL 标准化空间；Coverage@90% 理想值 0.90",
               "val": vm, "test": tm, "v1_test_mae_mean": V1_TEST_MAE}
    (RESULTS / "v2_mdn.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # —— 密度可视化：挑 4 个测试样本，画出"多峰终于被表达出来"——
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, i in zip(axes.flat, [0, len(y)//3, len(y)//2, len(y)-1]):
        xs = torch.linspace(-3, 3, 120); ys_ = torch.linspace(-3, 3, 120)
        gx, gy = torch.meshgrid(xs, ys_, indexing="xy")
        pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)       # (G,2)
        mix_i = build_mixture(pi[i:i+1].expand(len(pts), -1),
                              mu[i:i+1].expand(len(pts), -1, -1),
                              sigma[i:i+1].expand(len(pts), -1, -1))
        lp = mix_i.log_prob(pts).reshape(len(ys_), len(xs))
        ax.contourf(xs, ys_, lp, levels=20, cmap="viridis")
        ax.scatter([y[i, 0]], [y[i, 1]], c="red", s=60, marker="*",
                   label="truth", zorder=3)
        ax.scatter(mu[i, :, 0], mu[i, :, 1], c="white", s=40, marker="x",
                   label="component μ", zorder=3)
        ax.set_title(f"test sample #{i}  Cov-check")
        ax.legend(fontsize=7)
    fig.suptitle("v2 MDN: predicted landing density (normalized coords)", y=0.995)
    fig.tight_layout()
    fig.savefig(RESULTS / "v2_mdn_density.png", dpi=150)

    print("\n===== v2 MDN 结果（测试集）=====")
    print(f"  MAE(K=1) = {tm['mae_mean']:.2f}   （v1 点回归 = {V1_TEST_MAE}，需不输）")
    print(f"  minADE-5 = {tm['minade5']:.2f}   （点回归无此指标）")
    print(f"  NLL      = {tm['nll']:.3f}")
    print(f"  Cov@90%  = {tm['coverage90']:.3f} （理想 0.90，越近越校准）")
    print(f"\nΔx={tm['mae_x']:.2f}（v1: 72.27） Δy={tm['mae_y']:.2f}（v1: 81.29）")
    print(f"指标 → {RESULTS/'v2_mdn.json'}")


if __name__ == "__main__":
    main()
