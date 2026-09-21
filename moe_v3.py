# -*- coding: utf-8 -*-
"""
ShuttleGen v3 —— 两阶段混合专家（论文核心模型）
================================================================
架构（相对 v2 只动输出侧，编码器依旧一字不改）：

  编码器 z（与 v1/v2 完全一致，GRU + 球员条件）
      ├── 阶段一 · 类型门控：p(击球类型 t | s)   —— 10 类分类器，CE 损失
      └── 阶段二 · 落点专家：p(落点 | t, s)      —— 每类一个小混合分布 (K=3)

  推理时软路由（论文的"概率咬合"）：
      p(落点 | s) = Σ_t p(t|s) · p(落点 | t, s)     共 10×3 = 30 个分量

训练策略（写进论文方法节的两句话）：
  * 门控只吃分类损失，不回传落点梯度——与落点生成解耦，
    这正是 5.5 噪声实验里"识别器软标签直接替换门控"的接口前提；
  * 专家用 teacher forcing：训练时按真值类型路由，保证每个专家
    只见自己类型的落点（专门化）；推理时换预测的门控概率（软路由）。

对照目标（同协议测试集）：
  v1 点回归: MAE=76.78
  v2 MDN   : MAE=77.92  minADE-5=39.56  NLL=1.377  Cov90=0.863
  v3 要打的两个缺口：MAE(K=1) 反超 v1；Coverage 逼近 0.90。

另产出"oracle 路由"诊断（按真值类型路由的上限）——
门控不完美损失了多少，一眼可见（论文分析节的素材）。

运行：
    python moe_v3.py smoke   # 冒烟
    python moe_v3.py         # 完整训练（CPU 约 5-6 分钟）
产物（results/）：v3_moe.json / v3_moe.pt / v3_test_predictions.npz / v3_moe_density.png
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
                           MEAN_X, STD_X, MEAN_Y, STD_Y, TYPE_NAMES)
from baseline_v1 import (PointBaseline, mae_report,
                         BATCH, LR, WEIGHT_DECAY, EPOCHS, PATIENCE, SEED)

RESULTS = Path(__file__).resolve().parent / "results"
K_EXP = 3      # 每个专家内部的高斯分量数
S_COV = 200    # Coverage 的采样近似数
N_TYPES = 10
V1_MAE, V2 = 76.78, {"mae": 77.92, "minade5": 39.56, "nll": 1.377, "cov90": 0.863}


def to_real(t):
    """标准化 → 真实坐标，支持任意前导维度 (..., 2)。"""
    return torch.stack([t[..., 0] * STD_X + MEAN_X,
                        t[..., 1] * STD_Y + MEAN_Y], dim=-1)


# ============ 模型 ============

class TwoStageMoE(PointBaseline):
    """v1 的编码器 + 门控 + 10 专家。self.head 置空（不再要单点输出）。"""

    def __init__(self, n_types=N_TYPES, n_players=35, type_dim=16,
                 player_dim=8, hidden=128, k_exp=K_EXP):
        super().__init__(n_types, n_players, type_dim, player_dim, hidden)
        self.n_types, self.k_exp = n_types, k_exp
        self.head = None                                   # 注销 v1 的单点头
        self.gate = nn.Sequential(                         # 阶段一：类型门控
            nn.Linear(hidden + player_dim, 128), nn.ReLU(),
            nn.Linear(128, n_types))
        self.experts = nn.ModuleList([                     # 阶段二：每类一个专家
            nn.Sequential(
                nn.Linear(hidden + player_dim, 128), nn.ReLU(),
                nn.Linear(128, 5 * k_exp))                 # π(K)+μ(2K)+σ(2K)
            for _ in range(n_types)])

    def encode(self, bx, device):
        """与 v1/v2 完全一致的编码器（提出来给三个头共用）。"""
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
        nxt = self.player_emb(bx["next_player_id"].to(device))
        return torch.cat([h[-1], nxt], dim=-1)             # z (B, hidden+player_dim)

    def forward(self, bx, device):
        z = self.encode(bx, device)
        gate_logits = self.gate(z)                         # (B, 10)
        raw = torch.stack([exp(z) for exp in self.experts], dim=1)   # (B, 10, 5K)
        k = self.k_exp
        e_pi = F.log_softmax(raw[..., :k], dim=-1)         # (B,10,K) 各专家内部 log π
        e_mu = raw[..., k:3 * k].reshape(-1, self.n_types, k, 2)
        e_sigma = F.softplus(raw[..., 3 * k:]).reshape(-1, self.n_types, k, 2) + 1e-3
        return gate_logits, e_pi, e_mu, e_sigma


# ============ 分布装配 ============

def gather_expert(e_pi, e_mu, e_sigma, y_type):
    """按类型索引挑出对应专家的参数（teacher forcing / oracle 共用）。"""
    b = torch.arange(len(y_type))
    return e_pi[b, y_type], e_mu[b, y_type], e_sigma[b, y_type]


def make_mixture(pi_logits, mu, sigma):
    comp = D.Independent(D.Normal(mu, sigma), 1)
    return D.MixtureSameFamily(D.Categorical(logits=pi_logits), comp)


def full_mixture(gate_logits, e_pi, e_mu, e_sigma):
    """软路由：log p(t|s) + log π_{t,k} → 30 分量联合混合。"""
    log_gate = F.log_softmax(gate_logits, dim=-1)          # (B,10)
    log_joint = log_gate.unsqueeze(-1) + e_pi              # (B,10,K)
    B = log_joint.shape[0]
    return make_mixture(log_joint.reshape(B, -1),
                        e_mu.reshape(B, -1, 2), e_sigma.reshape(B, -1, 2))


# ============ 评估 ============

def mixture_metrics(mix, y_norm):
    """MAE(期望点) / NLL / minADE-5 / Coverage@90 / 最优假设分轴误差"""
    probs = mix.mixture_distribution.probs.unsqueeze(-1)      # (B,C,1)
    mean = (probs * mix.component_distribution.mean).sum(1)
    mae = mae_report(to_real(mean), to_real(y_norm))
    nll = -mix.log_prob(y_norm).mean().item()
    torch.manual_seed(SEED)
    hyp, truth = to_real(mix.sample((5,))), to_real(y_norm).unsqueeze(0)
    d = (hyp - truth).abs().sum(-1) / 2
    win = d.argmin(0)
    b = torch.arange(len(y_norm))
    wht, wtr = hyp[win, b], truth[0]
    s_lp = mix.log_prob(mix.sample((S_COV,)))                     # Coverage@90%
    thr = torch.quantile(s_lp, 0.10, dim=0)
    cov = (mix.log_prob(y_norm) >= thr).float().mean().item()
    return {"mae_x": mae["mae_x"], "mae_y": mae["mae_y"], "mae_mean": mae["mae_mean"],
            "nll": round(nll, 3), "minade5": round(d.min(0).values.mean().item(), 3),
            "cov90": round(cov, 3),
            "win_dx": round((wht[:, 0] - wtr[:, 0]).abs().mean().item(), 2),
            "win_dy": round((wht[:, 1] - wtr[:, 1]).abs().mean().item(), 2)}


@torch.no_grad()
def collect_and_evaluate(model, loader, device):
    model.eval()
    gates, EPs, EMs, ESs, types, ys = [], [], [], [], [], []
    for bx, by in loader:
        g, e_pi, e_mu, e_sigma = model(bx, device)
        gates.append(g.cpu()); EPs.append(e_pi.cpu())
        EMs.append(e_mu.cpu()); ESs.append(e_sigma.cpu())
        types.append(by["type_id"]); ys.append(by["landing"])
    g = torch.cat(gates); e_pi = torch.cat(EPs)
    e_mu = torch.cat(EMs); e_sigma = torch.cat(ESs)
    y_type = torch.cat(types); y = torch.cat(ys)

    gate_top1 = (g.argmax(-1) == y_type).float().mean().item()
    gate_top3 = (g.topk(3, -1).indices == y_type.unsqueeze(-1)).any(-1).float().mean().item()
    full = mixture_metrics(full_mixture(g, e_pi, e_mu, e_sigma), y)
    o_pi, o_mu, o_sg = gather_expert(e_pi, e_mu, e_sigma, y_type)
    oracle = mixture_metrics(make_mixture(o_pi, o_mu, o_sg), y)
    return ({"gate_top1": round(gate_top1, 3), "gate_top3": round(gate_top3, 3),
             "full": full, "oracle": oracle},
            (g, e_pi, e_mu, e_sigma, y_type, y))


# ============ 主流程 ============

def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}｜模式: {'冒烟' if smoke else '完整训练'}｜专家数 10 × K={K_EXP}")

    dl_train = DataLoader(ShuttleGenDataset(OUT_DIR / "train.npz"),
                          batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ShuttleGenDataset(OUT_DIR / "val.npz"),
                        batch_size=BATCH, collate_fn=collate_fn)
    dl_test = DataLoader(ShuttleGenDataset(OUT_DIR / "test.npz"),
                         batch_size=BATCH, collate_fn=collate_fn)

    model = TwoStageMoE(n_players=35).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e4:.1f} 万")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    best_nll, best_epoch, best_state = float("inf"), -1, None
    t0 = time.time()
    max_batches = 15 if smoke else None

    for epoch in range(1, (2 if smoke else EPOCHS) + 1):
        model.train()
        for bi, (bx, by) in enumerate(dl_train, 1):
            if max_batches and bi > max_batches:
                break
            g, e_pi, e_mu, e_sigma = model(bx, device)
            y_type = by["type_id"].to(device)
            y_land = by["landing"].to(device)
            # 损失 1：门控分类（不吃落点梯度——解耦是特性不是偷懒）
            loss_gate = ce(g, y_type)
            # 损失 2：teacher forcing 的落点 NLL（只用真值类型的专家）
            t_pi, t_mu, t_sg = gather_expert(e_pi, e_mu, e_sigma, y_type)
            loss_land = -make_mixture(t_pi, t_mu, t_sg).log_prob(y_land).mean()
            loss = loss_gate + loss_land
            opt.zero_grad(); loss.backward(); opt.step()
            if bi % 100 == 0:
                print(f"  epoch {epoch} batch {bi}/{len(dl_train)} "
                      f"gate={loss_gate.item():.3f} land={loss_land.item():.3f}")

        vm, _ = collect_and_evaluate(model, dl_val, device)
        f = vm["full"]
        flag = ""
        if f["nll"] < best_nll:
            best_nll, best_epoch = f["nll"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = " ← 目前最好"
        print(f"epoch {epoch:>2} | val: NLL={f['nll']:.3f} MAE={f['mae_mean']:.2f} "
              f"minADE5={f['minade5']:.2f} Cov={f['cov90']:.3f} "
              f"gate@1={vm['gate_top1']:.3f}{flag}")

        if smoke:
            print("冒烟通过：门控/专家/软路由/oracle 链路无 bug")
            return
        if epoch - best_epoch >= PATIENCE:
            print(f"早停：val NLL 已 {PATIENCE} 个 epoch 无改善")
            break

    print(f"\n训练结束（{(time.time()-t0)/60:.1f} 分钟），加载 epoch {best_epoch} 最优权重")
    model.load_state_dict(best_state)

    vm, _ = collect_and_evaluate(model, dl_val, device)
    tm, (g, e_pi, e_mu, e_sigma, y_type, y) = collect_and_evaluate(model, dl_test, device)

    RESULTS.mkdir(exist_ok=True)
    torch.save(best_state, RESULTS / "v3_moe.pt")
    np.savez_compressed(RESULTS / "v3_test_predictions.npz",
                        gate=g.numpy(), e_pi=e_pi.numpy(), e_mu=e_mu.numpy(),
                        e_sigma=e_sigma.numpy(), y_type=y_type.numpy(), truth=y.numpy())
    results = {"model": "v3_two_stage_moe", "k_expert": K_EXP,
               "selection": "val NLL (soft routing)", "seed": SEED,
               "protocol": "同 v2：MAE/minADE 真实坐标；NLL 标准化空间",
               "val": vm, "test": tm,
               "targets": {"v1_mae": V1_MAE, "v2": V2}}
    (RESULTS / "v3_moe.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # —— 密度可视化：真值类型 vs 门控 top-2，看"专家各管一段"——
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, i in zip(axes.flat, [0, len(y)//3, len(y)//2, len(y)-1]):
        mix_i = full_mixture(g[i:i+1], e_pi[i:i+1], e_mu[i:i+1], e_sigma[i:i+1])
        xs = torch.linspace(-3, 3, 120); ys_ = torch.linspace(-3, 3, 120)
        gx, gy = torch.meshgrid(xs, ys_, indexing="xy")
        pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)
        mix_g = make_mixture(
            mix_i.mixture_distribution.logits.expand(len(pts), -1),
            mix_i.component_distribution.mean.expand(len(pts), -1, -1),
            mix_i.component_distribution.stddev.expand(len(pts), -1, -1))
        lp = mix_g.log_prob(pts).reshape(len(ys_), len(xs))
        ax.contourf(xs, ys_, lp, levels=20, cmap="viridis")
        ax.scatter([y[i, 0]], [y[i, 1]], c="red", s=80, marker="*", zorder=3)
        w = mix_i.mixture_distribution.probs[0]
        top = w.topk(6)
        mus = mix_i.component_distribution.mean[0]
        ax.scatter(mus[top.indices, 0], mus[top.indices, 1], c="white", s=60,
                   marker="o", alpha=.8, zorder=3)
        p = F.softmax(g[i], -1)
        t2 = p.topk(2)
        ax.set_title(f"#{i} truth={TYPE_NAMES[y_type[i]][:12]} | "
                     f"gate: {TYPE_NAMES[t2.indices[0]][:10]}({p[t2.indices[0]]:.2f}) "
                     f"{TYPE_NAMES[t2.indices[1]][:10]}({p[t2.indices[1]]:.2f})", fontsize=9)
    fig.suptitle("v3 two-stage MoE: soft-routed landing density (normalized coords)", y=0.995)
    fig.tight_layout()
    fig.savefig(RESULTS / "v3_moe_density.png", dpi=150)

    f, o = tm["full"], tm["oracle"]
    print("\n===== v3 两阶段 MoE 结果（测试集）=====")
    print(f"{'':16} {'MAE':>7} {'minADE5':>8} {'NLL':>7} {'winΔx':>7} {'winΔy':>7}")
    print(f"{'软路由(完整)':<16} {f['mae_mean']:>7.2f} {f['minade5']:>8.2f} "
          f"{f['nll']:>7.3f} {f['win_dx']:>7.2f} {f['win_dy']:>7.2f}")
    print(f"{'oracle(真值路由)':<16} {o['mae_mean']:>7.2f} {o['minade5']:>8.2f} "
          f"{o['nll']:>7.3f} {o['win_dx']:>7.2f} {o['win_dy']:>7.2f}")
    print(f"门控: top1={tm['gate_top1']:.3f} top3={tm['gate_top3']:.3f}")
    print(f"\n对照: v1 MAE={V1_MAE} | v2 MAE={V2['mae']} minADE5={V2['minade5']} "
          f"NLL={V2['nll']}")
    print(f"指标 → {RESULTS/'v3_moe.json'}")


if __name__ == "__main__":
    main()
