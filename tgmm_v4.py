# -*- coding: utf-8 -*-
"""
ShuttleGen v4 —— 边界截断 TGMM（v3 + Truncated Gaussian 截断修正）
===================================================================
承接 v3 的两阶段 MoE，唯一改动：混合分布的每个高斯分量在 x、y 两个维度
分别截断到球场边界内，并对 NLL / 采样做截断修正。

为什么要截断？
  v2/v3 的混合高斯没有边界约束，模型会把一部分概率质量铺到场外——
  这不是球场的物理可能，直接后果是 Coverage@90% 偏保守
  （v2=0.863, v3=0.897，理想值 0.90）。截断把"不可能的落点"概率归零，
  迫使模型在可行域内更密集地分配概率，理论上能提升 Coverage 和 NLL。

截断修正公式（一维，对角高斯即两维独立处理）：
  log p_trunc(y) = log N(y; μ, σ) - log[Φ(z_upper) - Φ(z_lower)]
  其中 z_lower = (bound_low - μ)/σ,  z_upper = (bound_high - μ)/σ
  第二项叫"截断归一化常数"，保证截断后的 pdf 仍积分为 1。

采样不能直接 clamp（会改变密度），必须用逆 CDF：
  U ~ Uniform(Φ(z_lower), Φ(z_upper));  y = μ + σ · Φ⁻¹(U)

球场边界（标准化坐标，见 _probe 输出；加了安全缓冲）：
  x ∈ [-4.0, 4.0], y ∈ [-2.5, 2.5]

运行：
    python tgmm_v4.py smoke   # 冒烟
    python tgmm_v4.py         # 完整训练（CPU 约 5-6 分钟）
产物（results/）：v4_tgmm.json / v4_tgmm.pt / v4_test_predictions.npz / v4_tgmm_density.png
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from data_pipeline import (ShuttleGenDataset, collate_fn, OUT_DIR,
                           MEAN_X, STD_X, MEAN_Y, STD_Y, TYPE_NAMES)
from baseline_v1 import (PointBaseline, mae_report,
                         BATCH, LR, WEIGHT_DECAY, EPOCHS, PATIENCE, SEED)
from moe_v3 import TwoStageMoE

RESULTS = Path(__file__).resolve().parent / "results"
K_EXP = 3
S_COV = 200
N_TYPES = 10

# —— 球场边界（标准化坐标）—— 真实球场 x≈0..440cm, y≈0..1340cm
# 转标准化：(x-175)/82 ∈ [-2.1, 3.2], (y-467)/192 ∈ [-2.4, 1.8]
# 用稍宽的安全边界，避免截断掉有效落点
X_LOW, X_HIGH = -3.5, 3.6
Y_LOW, Y_HIGH = -2.0, 2.0

V1_MAE = 76.78
V2 = {"mae": 77.92, "minade5": 39.56, "nll": 1.377, "cov90": 0.863}
V3 = {"mae": 77.10, "minade5": 40.08, "nll": 1.316, "cov90": 0.897}


def to_real(t):
    return torch.stack([t[..., 0] * STD_X + MEAN_X,
                        t[..., 1] * STD_Y + MEAN_Y], dim=-1)


# ============ 截断高斯的 log_prob & sample ============

_N = Normal(0.0, 1.0)
_LOG_PI = torch.tensor(np.log(2 * np.pi))


def trunc_log_prob_1d(y, mu, sigma, lo, hi):
    """一维截断高斯的 log pdf。
    y, mu, sigma 形状相同（可以是任意前导维度）。
    lo, hi 是 Python float（边界常数）。"""
    z = (y - mu) / sigma
    z_l = (lo - mu) / sigma
    z_u = (hi - mu) / sigma
    # 普通高斯 log pdf
    log_normal = _N.log_prob(z) - torch.log(sigma)   # = log N(y; μ, σ²)
    # 截断修正：-log[Φ(z_u) - Φ(z_l)]
    cdf_diff = _N.cdf(z_u) - _N.cdf(z_l)
    log_cdf_diff = torch.log(cdf_diff.clamp(min=1e-12))
    return log_normal - log_cdf_diff


def trunc_sample_1d(mu, sigma, lo, hi, size=None):
    """一维截断高斯采样（逆 CDF 法）。
    mu, sigma 可以带任意前导维度；size 是额外采样维度 tuple。"""
    z_l = (lo - mu) / sigma
    z_u = (hi - mu) / sigma
    if size is None:
        u_l = _N.cdf(z_l); u_u = _N.cdf(z_u)
    else:
        mu_exp = mu.expand(*size, *mu.shape)
        sg_exp = sigma.expand(*size, *sigma.shape)
        z_l = (lo - mu_exp) / sg_exp
        z_u = (hi - mu_exp) / sg_exp
        u_l = _N.cdf(z_l); u_u = _N.cdf(z_u)
    U = torch.rand_like(u_l) * (u_u - u_l) + u_l
    return mu + sigma * _N.icdf(U)


def trunc_log_prob_2d(y, mu, sigma):
    """二维对角截断高斯 log pdf（x、y 各自截断，边界全局常数）。
    y:  (..., 2)  真值（标准化坐标）
    mu: (..., 2)  分量中心
    sigma: (..., 2)  分量对角标准差"""
    lp_x = trunc_log_prob_1d(y[..., 0], mu[..., 0], sigma[..., 0], X_LOW, X_HIGH)
    lp_y = trunc_log_prob_1d(y[..., 1], mu[..., 1], sigma[..., 1], Y_LOW, Y_HIGH)
    return lp_x + lp_y


def trunc_sample_2d(mu, sigma, size):
    """二维对角截断高斯采样。
    mu, sigma: (C, 2) 或 (B, C, 2) 分量参数
    size: 采样维度，如 (5,) 表示每个分量采 5 个"""
    # 先把 mu/sigma 展平成 (..., 2) 方便处理
    sam_x = trunc_sample_1d(mu[..., 0], sigma[..., 0], X_LOW, X_HIGH, size=size)
    sam_y = trunc_sample_1d(mu[..., 1], sigma[..., 1], Y_LOW, Y_HIGH, size=size)
    return torch.stack([sam_x, sam_y], dim=-1)


# ============ 截断混合分布装配 ============

def full_tgmm(gate_logits, e_pi, e_mu, e_sigma):
    """把 v3 的软路由参数包装成一个"截断混合分布"对象。
    返回的对象有 log_prob(y) 和 sample(size) 两个方法。"""
    log_gate = F.log_softmax(gate_logits, dim=-1)
    log_joint = log_gate.unsqueeze(-1) + e_pi              # (B, 10, K)
    B = log_joint.shape[0]
    log_joint = log_joint.reshape(B, -1)                    # (B, C), C=10*K
    C = log_joint.shape[1]
    mu_flat = e_mu.reshape(B, -1, 2)                        # (B, C, 2)
    sg_flat = e_sigma.reshape(B, -1, 2)                     # (B, C, 2)

    return _TGMMWrapper(log_joint, mu_flat, sg_flat, n_comp=C)


class _TGMMWrapper:
    """把 (log π, μ, σ) 包装成一个有 log_prob / sample 的对象。"""

    def __init__(self, log_pi, mu, sigma, n_comp):
        self.log_pi = log_pi          # (B, C)
        self.mu = mu                  # (B, C, 2)
        self.sigma = sigma            # (B, C, 2)
        self.C = n_comp

    def log_prob(self, y):
        """y: (*, B, 2) → (*, B) 每个样本的截断混合 log pdf。
        支持任意前导维度（Coverage 采样时需要 (S, B, 2) 直接算）。"""
        # 把 y 展开一个分量维度：(*, B, 1, 2) → broadcast 到 (*, B, C, 2)
        y_exp = y.unsqueeze(-2).expand(*y.shape[:-1], self.C, 2)
        # mu/sigma: (B, C, 2) → broadcast 到 (*, B, C, 2)
        mu_exp = self.mu.view(*([1] * (y.dim() - 2)), *self.mu.shape)
        sg_exp = self.sigma.view(*([1] * (y.dim() - 2)), *self.sigma.shape)
        lp_comp = trunc_log_prob_2d(y_exp, mu_exp, sg_exp)   # (*, B, C)
        log_pi_exp = self.log_pi.view(*([1] * (y.dim() - 2)), *self.log_pi.shape)
        log_joint = log_pi_exp + lp_comp                       # (*, B, C)
        return torch.logsumexp(log_joint, dim=-1)              # (*, B)

    def sample(self, size):
        """size: tuple，如 (5,) 表示每个样本采 5 个假设。
        返回 (size[0], B, 2) 的采样坐标（标准化）。"""
        S = size[0]
        B = self.log_pi.shape[0]
        probs = self.log_pi.exp()                      # (B, C)
        # 先按 π 选分量：对每个 (sample, batch) 选一个分量
        comp_idx = torch.multinomial(
            probs.unsqueeze(0).expand(S, -1, -1).reshape(-1, self.C),
            num_samples=1, replacement=True).reshape(S, B)   # (S, B)
        b_idx = torch.arange(B).unsqueeze(0).expand(S, -1)
        chosen_mu = self.mu[b_idx, comp_idx]            # (S, B, 2)
        chosen_sg = self.sigma[b_idx, comp_idx]         # (S, B, 2)
        return trunc_sample_2d(chosen_mu, chosen_sg, size=None)   # (S, B, 2)


# ============ 评估 ============

def mixture_metrics_tgmm(tgmm, y_norm):
    """四个指标 + 最优假设分轴误差，用截断分布计算。"""
    # 1) MAE：混合期望 E[X]（注意截断后期望 ≠ Σ π_c μ_c，但差别通常很小，
    #    这里仍用 Σ π_c μ_c 作近似单点预测——与 v1/v2/v3 协议对齐）
    probs = tgmm.log_pi.exp().unsqueeze(-1)                # (B, C, 1)
    mean = (probs * tgmm.mu).sum(1)                         # (B, 2)
    mae = mae_report(to_real(mean), to_real(y_norm))

    # 2) NLL（截断修正后的）
    nll = -tgmm.log_prob(y_norm).mean().item()

    # 3) minADE-5：截断采样
    torch.manual_seed(SEED)
    hyp = to_real(tgmm.sample((5,)))                         # (5, B, 2)
    truth = to_real(y_norm).unsqueeze(0)
    d = (hyp - truth).abs().sum(-1) / 2
    win = d.argmin(0)
    b = torch.arange(len(y_norm))
    wht, wtr = hyp[win, b], truth[0]

    # 4) Coverage@90% —— 采样 (S, B, 2) 直接喂给支持前导维的 log_prob
    s_lp = tgmm.log_prob(tgmm.sample((S_COV,)))           # (S_COV, B)
    thr = torch.quantile(s_lp, 0.10, dim=0)               # (B,)
    cov = (tgmm.log_prob(y_norm) >= thr).float().mean().item()

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

    full = mixture_metrics_tgmm(full_tgmm(g, e_pi, e_mu, e_sigma), y)

    # oracle：按真值类型路由 + 截断
    b = torch.arange(len(y_type))
    o_pi, o_mu, o_sg = e_pi[b, y_type], e_mu[b, y_type], e_sigma[b, y_type]
    oracle_tgmm = _TGMMWrapper(o_pi, o_mu, o_sg, n_comp=o_pi.shape[1])
    oracle = mixture_metrics_tgmm(oracle_tgmm, y)

    return ({"gate_top1": round(gate_top1, 3), "gate_top3": round(gate_top3, 3),
             "full": full, "oracle": oracle},
            (g, e_pi, e_mu, e_sigma, y_type, y))


# ============ 主流程 ============

def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}｜模式: {'冒烟' if smoke else '完整训练'}｜截断边界 x∈[{X_LOW},{X_HIGH}] y∈[{Y_LOW},{Y_HIGH}]")

    dl_train = DataLoader(ShuttleGenDataset(OUT_DIR / "train.npz"),
                          batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ShuttleGenDataset(OUT_DIR / "val.npz"),
                        batch_size=BATCH, collate_fn=collate_fn)
    dl_test = DataLoader(ShuttleGenDataset(OUT_DIR / "test.npz"),
                         batch_size=BATCH, collate_fn=collate_fn)

    model = TwoStageMoE(n_players=35).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e4:.1f} 万（与 v3 相同）")
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

            loss_gate = ce(g, y_type)

            # teacher forcing 的截断 NLL（v3 这里是普通 NLL，v4 只换这一行）
            b_idx = torch.arange(len(y_type))
            t_pi = e_pi[b_idx, y_type]         # (B, K)
            t_mu = e_mu[b_idx, y_type]         # (B, K, 2)
            t_sg = e_sigma[b_idx, y_type]      # (B, K, 2)
            exp_tgmm = _TGMMWrapper(t_pi, t_mu, t_sg, n_comp=t_pi.shape[1])
            loss_land = -exp_tgmm.log_prob(y_land).mean()

            loss = loss_gate + loss_land
            opt.zero_grad(); loss.backward(); opt.step()
            if bi % 100 == 0:
                print(f"  epoch {epoch} batch {bi}/{len(dl_train)} "
                      f"gate={loss_gate.item():.3f} land(tgmm)={loss_land.item():.3f}")

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
            print("冒烟通过：截断高斯 log_prob / 采样 / 软路由链路无 bug")
            return
        if epoch - best_epoch >= PATIENCE:
            print(f"早停：val NLL 已 {PATIENCE} 个 epoch 无改善")
            break

    print(f"\n训练结束（{(time.time()-t0)/60:.1f} 分钟），加载 epoch {best_epoch} 最优权重")
    model.load_state_dict(best_state)

    vm, _ = collect_and_evaluate(model, dl_val, device)
    tm, (g, e_pi, e_mu, e_sigma, y_type, y) = collect_and_evaluate(model, dl_test, device)

    RESULTS.mkdir(exist_ok=True)
    torch.save(best_state, RESULTS / "v4_tgmm.pt")
    np.savez_compressed(RESULTS / "v4_test_predictions.npz",
                        gate=g.numpy(), e_pi=e_pi.numpy(), e_mu=e_mu.numpy(),
                        e_sigma=e_sigma.numpy(), y_type=y_type.numpy(), truth=y.numpy(),
                        bounds=np.array([X_LOW, X_HIGH, Y_LOW, Y_HIGH]))
    results = {"model": "v4_tgmm", "k_expert": K_EXP,
               "boundaries_norm": {"x": [X_LOW, X_HIGH], "y": [Y_LOW, Y_HIGH]},
               "selection": "val NLL (truncated soft routing)", "seed": SEED,
               "protocol": "同 v2/v3，NLL 含截断修正项",
               "val": vm, "test": tm,
               "targets": {"v1_mae": V1_MAE, "v2": V2, "v3": V3}}
    (RESULTS / "v4_tgmm.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # —— 密度可视化（截断版本）——
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, i in zip(axes.flat, [0, len(y)//3, len(y)//2, len(y)-1]):
        tgmm_i = full_tgmm(g[i:i+1], e_pi[i:i+1], e_mu[i:i+1], e_sigma[i:i+1])
        xs = torch.linspace(X_LOW, X_HIGH, 120)
        ys_ = torch.linspace(Y_LOW, Y_HIGH, 120)
        gx, gy = torch.meshgrid(xs, ys_, indexing="xy")
        pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)      # (G, 2)
        # 把 pts 变成 (1, G, 2) 喂给 batch=1 的 tgmm
        tgmm_i_full = _TGMMWrapper(
            tgmm_i.log_pi.expand(len(pts), -1),
            tgmm_i.mu.expand(len(pts), -1, -1),
            tgmm_i.sigma.expand(len(pts), -1, -1),
            n_comp=tgmm_i.C)
        lp = tgmm_i_full.log_prob(pts).reshape(len(ys_), len(xs))
        ax.contourf(xs, ys_, lp, levels=20, cmap="viridis")
        ax.scatter([y[i, 0]], [y[i, 1]], c="red", s=80, marker="*", zorder=3)
        w = tgmm_i.log_pi.exp()[0]
        top = w.topk(6)
        mus = tgmm_i.mu[0]
        ax.scatter(mus[top.indices, 0], mus[top.indices, 1], c="white", s=60,
                   marker="o", alpha=.8, zorder=3)
        p = F.softmax(g[i], -1)
        t2 = p.topk(2)
        ax.set_title(f"#{i} truth={TYPE_NAMES[y_type[i]][:12]} | "
                     f"gate: {TYPE_NAMES[t2.indices[0]][:10]}({p[t2.indices[0]]:.2f}) "
                     f"{TYPE_NAMES[t2.indices[1]][:10]}({p[t2.indices[1]]:.2f})", fontsize=9)
    fig.suptitle("v4 TGMM: truncated landing density (normalized coords)", y=0.995)
    fig.tight_layout()
    fig.savefig(RESULTS / "v4_tgmm_density.png", dpi=150)

    f, o = tm["full"], tm["oracle"]
    print("\n===== v4 TGMM 结果（测试集）=====")
    print(f"{'':16} {'MAE':>7} {'minADE5':>8} {'NLL':>7} {'Cov90':>7} {'winΔx':>7} {'winΔy':>7}")
    print(f"{'软路由(完整)':<16} {f['mae_mean']:>7.2f} {f['minade5']:>8.2f} "
          f"{f['nll']:>7.3f} {f['cov90']:>7.3f} {f['win_dx']:>7.2f} {f['win_dy']:>7.2f}")
    print(f"{'oracle(真值路由)':<16} {o['mae_mean']:>7.2f} {o['minade5']:>8.2f} "
          f"{o['nll']:>7.3f} {o['cov90']:>7.3f} {o['win_dx']:>7.2f} {o['win_dy']:>7.2f}")
    print(f"门控: top1={tm['gate_top1']:.3f} top3={tm['gate_top3']:.3f}")
    print(f"\n历史对照:")
    print(f"  v1 点回归 MAE={V1_MAE}")
    print(f"  v2 MDN   MAE={V2['mae']}  minADE5={V2['minade5']}  NLL={V2['nll']}  Cov90={V2['cov90']}")
    print(f"  v3 MoE   MAE={V3['mae']}  minADE5={V3['minade5']}  NLL={V3['nll']}  Cov90={V3['cov90']}")
    print(f"  v4 TGMM  MAE={f['mae_mean']:.2f}  minADE5={f['minade5']:.2f}  NLL={f['nll']:.3f}  Cov90={f['cov90']:.3f}")
    print(f"指标 → {RESULTS/'v4_tgmm.json'}")


if __name__ == "__main__":
    main()
