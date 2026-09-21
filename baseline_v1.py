# -*- coding: utf-8 -*-
"""
ShuttleGen v1 —— 点回归基线模型
================================
任务：输入羽毛球比赛回合中前 t 拍的全部可见信息，预测第 t+1 拍的落点坐标 (x, y)。

模型结构：
  GRU 序列编码器  ──编码整个回合的时序信息──▶  隐藏状态向量
                ──拼接"下一拍击球者"的嵌入向量──▶  MLP 回归头  ──▶  (x, y) 坐标
  总参数量约 15 万。

本脚本的角色：
  1. 实现"确定性点回归"范式——模型只输出一个确定的坐标值，不输出概率分布；
  2. 作为后续分布模型（v2 MDN、v3 两阶段 MoE）的对照基线，后者必须在 MAE 上超过它；
  3. 同时实现"常数预测"基线：永远猜训练集平均落点，作为最基础的参照。

评估指标（所有版本 v1~v4 统一使用，保证结果可比）：
  MAE = mean( (|Δx| + |Δy|) / 2 )，单位为真实球场坐标单位（厘米 cm）
  同时报告 |Δx|、|Δy| 两个分轴的平均绝对误差

运行方式：
    conda activate badminton_cv
    python baseline_v1.py smoke   # 冒烟测试：只跑几十个 batch，用来快速检验代码是否能跑通
    python baseline_v1.py         # 完整训练：CPU 上约 5~10 分钟

输出文件（保存在 results/ 目录下）：
    v1_point_baseline.pt        训练好的模型权重（PyTorch 格式）
    v1_point_baseline.json      全部评估指标（JSON 格式，用于论文表格）
    v1_test_predictions.npz     测试集上逐样本的预测值与真值（用于画图或复算）
    v1_pred_scatter.png         预测落点 vs 真实落点的散点图（用于直观观察预测分布）
"""

# ==================== 导入区 ====================

import json          # 读写 JSON 文件：把训练结果字典写入 v1_point_baseline.json
import sys           # 读取命令行参数：用来判断用户是否传入了 "smoke" 参数（冒烟测试模式）
import time          # 计时：记录训练过程耗时
from pathlib import Path   # Python 标准路径工具，用对象表示文件路径，跨平台

import numpy as np   # NumPy：科学计算库，用来处理数组和设置随机种子
import torch         # PyTorch 核心库：提供张量运算、自动求导、GPU/CPU 切换等功能
import torch.nn as nn      # PyTorch 神经网络模块：Linear、GRU、Embedding、Dropout 等层都在这里
from torch.nn.utils.rnn import pack_padded_sequence   # RNN 工具：把变长序列打包，让 GRU 忽略 padding（补零）部分
from torch.utils.data import DataLoader               # 数据加载器：自动按 batch 切分数据、打乱顺序

# 从同目录下的数据处理模块导入：
#   ShuttleGenDataset  —— 数据集类：从 npz 文件读取数据，按羽毛球回合为单位返回样本
#   collate_fn         —— 拼 batch 的函数：处理变长序列，将多个样本组合成一个 batch
#   OUT_DIR            —— 数据输出目录：v0 脚本预处理后的数据存放在这里
#   MEAN_X / STD_X / MEAN_Y / STD_Y  —— 训练集落点的均值和标准差，用于 z-score 标准化
from data_pipeline import (ShuttleGenDataset, collate_fn, OUT_DIR,
                           MEAN_X, STD_X, MEAN_Y, STD_Y)

# 结果输出目录：本脚本所在目录下的 results/
RESULTS = Path(__file__).resolve().parent / "results"

# ==================== 超参数 ====================
# 超参数在训练前设定好，控制模型训练过程。全部写在文件顶部并保存到结果 JSON 中，
# 保证实验可以被精确复现。

SEED = 42          # 随机种子：固定后，每次运行的权重初始化和数据打乱都完全一样
BATCH = 64         # 批大小：每次迭代同时处理 64 个回合，使梯度估计更稳定、训练更快
LR = 1e-3          # 学习率（Learning Rate）：每次参数更新的幅度。值太大震荡不收敛，太小收敛极慢
WEIGHT_DECAY = 1e-4   # 权重衰减（L2 正则化）：在损失函数中加入参数幅度的惩罚项，防止过拟合
EPOCHS = 30        # 最大训练轮数：整个训练集被完整遍历 30 次（配合早停机制可能用不满）
PATIENCE = 6       # 早停耐心值：如果验证集 MAE 连续 6 轮没有改善，就提前停止训练

# ------ 自动生成的运行元信息（用于留痕，别手动改）------
# 每次脚本启动时生成唯一标识，结果 JSON 里会带上它
import datetime, subprocess, os
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# 记录训练集 npz 的修改时间戳作为数据版本标识；文件不存在则退化为 None
try:
    _DATA_MTIME = (RESULTS.parent / "processed" / "train.npz").stat().st_mtime
    _DATA_VERSION = datetime.datetime.fromtimestamp(_DATA_MTIME).strftime("%Y%m%d_%H%M%S")
except FileNotFoundError:
    _DATA_VERSION = None


def _get_git_commit_hash() -> str:
    """尝试取当前代码的 git commit 哈希。如果项目没 git 就返回 None（不报错）。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


_GIT_HASH = _get_git_commit_hash()


# ==================== 模型定义 ====================

class PointBaseline(nn.Module):
    """GRU 点回归基线模型。

    每一拍的输入特征由三部分拼接而成：
      1. 球种嵌入 (16 维) —— 把球种类别 ID 映射为可学习的向量
      2. 击球者嵌入 (8 维) —— 把球员 ID 映射为可学习的向量，编码该球员的击球风格
      3. 11 维数值特征 —— 包含：落点 (2) + 击球者位置 (2) + 对手位置 (2) + 比分 (2)
         + 回合深度 (1) + 是否反手 (1) + 是否绕头 (1)

    完整前向流程：
      输入序列 (B, T, 35) → GRU 编码 → 最终隐藏状态 (B, 128)
                           → 拼接"下一拍击球者"嵌入 (B, 8)
                           → MLP 回归头 → 输出 (B, 2)，即标准化后的 (x, y)

    这是一个确定性模型：对同一输入，它只输出一个坐标值，不输出概率分布。
    """

    def __init__(self, n_types=10, n_players=35,
                 type_dim=16, player_dim=8, hidden=128):
        """
        参数说明：
          n_types     —— 球种类别数，数据集中有杀球、吊球等共 10 种
          n_players   —— 球员总数，数据集中共有 35 名不同球员
          type_dim    —— 球种嵌入向量的维度
          player_dim  —— 球员嵌入向量的维度
          hidden      —— GRU 隐藏状态的维度（也是整个"回合编码"的特征维度上限）
        """
        super().__init__()          # PyTorch 要求：继承 nn.Module 的类必须先调用父类的 __init__
        # 球种嵌入层：查表，把整数 ID (0~9) 转换为 16 维浮点向量，向量值随训练学习
        self.type_emb = nn.Embedding(n_types, type_dim)
        # 球员嵌入层：查表，把整数 ID (0~34) 转换为 8 维浮点向量
        self.player_emb = nn.Embedding(n_players, player_dim)
        # 每一拍的输入总维度 = 球种嵌入(16) + 球员嵌入(8) + 数值特征(11) = 35
        in_dim = type_dim + player_dim + 11
        # GRU 序列编码器：输入维度 35，隐藏状态维度 128
        # batch_first=True 表示输入张量形状约定为 (B, T, 特征)
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        # 回归头（多层感知机 MLP）：把 GRU 编码结果映射为坐标
        self.head = nn.Sequential(
            # 全连接层 136→256，接 ReLU 激活函数，训练时随机丢弃 20% 的神经元以减轻过拟合
            nn.Linear(hidden + player_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            # 全连接层 256→128，接 ReLU 激活函数
            nn.Linear(256, 128), nn.ReLU(),
            # 输出层 128→2：两个值分别是标准化后的 x 和 y 坐标
            nn.Linear(128, 2),
        )

    def forward(self, bx, device):
        """
        前向传播。

        参数：
          bx     —— 一个 batch 的输入字典，由 collate_fn 函数组装好，包含多种张量
          device —— 计算设备（torch.device 类型），值为 "cpu" 或 "cuda"。
                    所有张量和模型参数必须在同一设备上才能运算。

        返回：
          形状为 (B, 2) 的张量，B = batch size，每行是一个样本的标准化坐标 (x, y)。
        """
        # ---------- 第一步：组装 11 维数值特征 ----------
        # 从 bx 字典中取出 6 组张量，按最后一维拼接成 (B, T, 11)
        num = torch.cat([
            bx["landing"],                                      # 历史落点 (B, T, 2)，已做 z-score 标准化
            bx["player_loc"], bx["opp_loc"],                    # 击球者站位 (B, T, 2) + 对手站位 (B, T, 2)，已标准化
            bx["scores"].float() / 21.0,                        # 当前比分 (B, T, 2)，除以 21 压到 ~0~1 范围
            bx["ball_round"].float().unsqueeze(-1) / 20.0,      # 回合内第几拍 (B, T) → unsqueeze 变 (B, T, 1) → 再÷20 缩量纲
            bx["backhand"].float().unsqueeze(-1),               # 是否反手 0/1，形状 (B, T) → (B, T, 1)
            bx["aroundhead"].float().unsqueeze(-1),             # 是否绕头 0/1，形状 (B, T) → (B, T, 1)
        ], dim=-1).to(device)

        # ---------- 第二步：每拍最终特征 = 嵌入 + 数值特征 ----------
        # 三路拼接 → 形状 (B, T, 35)
        seq = torch.cat([
            self.type_emb(bx["type_id"].to(device)),           # 球种 ID → 嵌入向量 (B, T, 16)
            self.player_emb(bx["player_id"].to(device)),       # 击球者 ID → 嵌入向量 (B, T, 8)
            num,                                                # 数值特征 (B, T, 11)
        ], dim=-1)

        # ---------- 第三步：GRU 编码（处理变长序列）----------
        # 一个 batch 里的羽毛球回合长度不同，短回合会用零填充到相同长度 T。
        # 如果不处理，GRU 会把填充的零当作真实输入，污染编码结果。
        # 解决方案：用 pack_padded_sequence 把变长序列打包，告诉 GRU 哪些位置是 padding。

        # 从 0/1 掩码中算出每个回合的真实长度。.cpu() 是 pack_padded_sequence 要求的。
        lengths = bx["mask"].sum(1).cpu()
        # 打包：pack_padded_sequence 不需要预先按长度排序（enforce_sorted=False）
        packed = pack_padded_sequence(seq, lengths,
                                      batch_first=True, enforce_sorted=False)
        # 跑 GRU。返回的 h 是最终隐藏状态，形状 (num_layers, B, 128)
        # 因为这里用的是单层 GRU，所以 h[0] 就是整个回合的编码结果
        _, h = self.gru(packed)
        h = h[-1]                                  # 去掉层数维 → (B, 128)

        # ---------- 第四步：拼接下一拍击球者条件 → 回归头 ----------
        # 查出"下一拍由谁打"的嵌入向量 (B, 8)。这个向量作为条件告诉模型：
        # "即将击球的人有自己的风格，预测落点时要考虑这一点"
        nxt = self.player_emb(bx["next_player_id"].to(device))
        # 拼接 (B, 128) + (B, 8) = (B, 136)，送入回归头 → 输出 (B, 2)
        return self.head(torch.cat([h, nxt], dim=-1))


# ==================== 训练与评估 ====================

def real_coords(t: torch.Tensor) -> torch.Tensor:
    """把 z-score 标准化的坐标还原为真实球场坐标。

    训练时模型输出的坐标是标准化值（均值 0、标准差 1），
    评估和展示时需要还原为真实的毫米单位坐标。

    参数：
      t —— 形状 (N, 2) 的张量，每行是一个样本的标准化坐标 [z_x, z_y]

    返回：
      形状 (N, 2) 的张量，每行是真实坐标 [x_cm, y_cm]
    """
    out = t.clone()                       # 先复制一份，避免原地修改输入参数（好习惯）
    # z-score 标准化公式：z = (x - μ) / σ；反变换：x = z * σ + μ
    out[:, 0] = t[:, 0] * STD_X + MEAN_X  # 还原 x 坐标
    out[:, 1] = t[:, 1] * STD_Y + MEAN_Y  # 还原 y 坐标
    return out
    # 为什么训练用标准化、评估用真实坐标？
    #   —— 训练时，两个坐标轴的量纲统一在 ~0 附近，梯度下降更容易收敛；
    #   —— 报告时，"平均差 76.78 厘米"比"标准化空间差 0.83"对人更直观。


def evaluate(model, loader, device):
    """在给定数据集上评估模型。

    参数：
      model  —— 要评估的 PyTorch 模型
      loader —— DataLoader，遍历验证集或测试集
      device —— 计算设备（cpu 或 cuda）

    返回：
      (平均 MSE 损失, 所有样本的预测坐标 (N,2), 所有样本的真实坐标 (N,2))
      坐标已经还原为真实球场单位。
    """
    model.eval()                   # 切换到评估模式：Dropout 层关闭，不再随机丢弃神经元
    loss_fn = nn.MSELoss()         # 损失函数用 MSE（和训练一致）
    tot, n = 0.0, 0                # 累计损失 / 累计样本数
    preds, truths = [], []         # 收集每个 batch 的预测和真值
    with torch.no_grad():          # 关闭梯度追踪：评估不需要反向传播，节省显存并加速
        for bx, by in loader:      # 逐 batch 遍历数据
            pred = model(bx, device)                    # 前向传播，得到 (B, 2) 标准化预测
            truth = by["landing"].to(device)            # 标签：(B, 2) 标准化真值
            tot += loss_fn(pred, truth).item() * len(pred)  # batch 平均损失 × 样本数 = 批次总损失
            n += len(pred)                              # 累计样本数
            preds.append(real_coords(pred).cpu())       # 预测还原为真实坐标，搬回 CPU
            truths.append(real_coords(truth).cpu())     # 真值同样处理
    # 平均损失 + 把所有 batch 拼接成 (N, 2) 全量张量
    return tot / n, torch.cat(preds), torch.cat(truths)


def mae_report(pred: torch.Tensor, truth: torch.Tensor) -> dict:
    """计算 MAE 指标，返回分轴和平均的绝对误差。

    评估协议：
      MAE_x = mean(|预测x - 真实x|)
      MAE_y = mean(|预测y - 真实y|)
      MAE   = (MAE_x + MAE_y) / 2

    参数：
      pred  —— 形状 (N, 2) 的预测坐标（真实单位）
      truth —— 形状 (N, 2) 的真实坐标（真实单位）

    返回：
      字典 {"mae_x": ..., "mae_y": ..., "mae_mean": ...}，保留 3 位小数
    """
    dx = (pred[:, 0] - truth[:, 0]).abs().mean().item()   # x 轴所有样本绝对误差的平均值
    dy = (pred[:, 1] - truth[:, 1]).abs().mean().item()   # y 轴所有样本绝对误差的平均值
    return {"mae_x": round(dx, 3), "mae_y": round(dy, 3),
            "mae_mean": round((dx + dy) / 2, 3)}


def constant_baseline(train_ds, eval_loaders):
    """最简单的基线：永远预测训练集所有落点的均值。

    用途：作为参照下限——如果你的模型连"永远猜同一个点"都打不过，说明训练有问题。

    参数：
      train_ds    —— 训练集 ShuttleGenDataset 实例
      eval_loaders —— 字典，键是数据集名称（如 "val"、"test"），值是对应的 DataLoader

    返回：
      字典，和 eval_loaders 键对应，每个值是 mae_report 的结果字典
    """
    # Step 1：从训练集中提取所有"真实拍"的落点，排除 padding 部分
    mask = np.zeros(train_ds.landing.shape[:2], dtype=bool)  # 创建与 landing 同形状的布尔掩码，全为 False
    for i, L in enumerate(train_ds.length):   # 遍历每个回合的真实长度
        mask[i, :L] = True                     # 标记前 L 拍为有效，后面 padding 保持 False
    pts = train_ds.landing[mask]               # 布尔索引 → (总有效拍数, 2) 的落点数组
    mean_pt = torch.from_numpy(pts.mean(0))    # 沿第 0 维求均值 → (2,) 的均值坐标张量

    # Step 2：在每个评估集上计算 MAE
    out = {}
    for name, loader in eval_loaders.items():
        truths, preds = [], []
        for _, by in loader:                   # 遍历评估集（只用标签，模型输出恒定所以输入无所谓）
            truths.append(real_coords(by["landing"]))           # 真值还原为真实坐标
            # 均值 (2,) → 扩展为 (B, 2)，每个样本的预测都一样
            preds.append(real_coords(mean_pt.unsqueeze(0).expand(len(by["landing"]), 2)))
        # 用和其他模型完全相同的评估方法，确保结果可比
        out[name] = mae_report(torch.cat(preds), torch.cat(truths))
    return out


# ==================== 主流程 ====================

def main():
    # ------ 1. 初始化 ------
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"  # 如果命令行传入了 "smoke"，则进入冒烟测试模式
    torch.manual_seed(SEED)       # 固定 PyTorch 随机种子（影响权重初始化、Dropout 等所有随机过程）
    np.random.seed(SEED)          # 固定 NumPy 随机种子
    # 自动选择计算设备：有 CUDA GPU 就用 GPU，没有退回 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}｜模式: {'冒烟测试' if smoke else '完整训练'}")

    # ------ 2. 加载数据集 ------
    # 训练集：用于学习模型参数
    ds_train = ShuttleGenDataset(OUT_DIR / "train.npz")
    # 验证集：用于选择最优模型权重和早停判断
    ds_val = ShuttleGenDataset(OUT_DIR / "val.npz")
    # 测试集：最终评估，仅用于论文报告，不参与模型选择
    ds_test = ShuttleGenDataset(OUT_DIR / "test.npz")

    # DataLoader 负责按 batch 切分数据
    # shuffle=True：训练集每个 epoch 打乱顺序，防止模型记住数据排列顺序
    # collate_fn：自定义函数，处理变长序列，把多个回合拼成一个 batch
    dl_train = DataLoader(ds_train, batch_size=BATCH, shuffle=True,
                          collate_fn=collate_fn)
    # 验证集和测试集不需要打乱顺序
    dl_val = DataLoader(ds_val, batch_size=BATCH, collate_fn=collate_fn)
    dl_test = DataLoader(ds_test, batch_size=BATCH, collate_fn=collate_fn)

    # ------ 3. 构建模型 ------
    model = PointBaseline(n_players=35)   # 35 是数据集中确认的球员总数
    model.to(device)                      # 把模型的所有参数搬到计算设备上
    # 统计可学习参数的总数量（p.numel() 返回每个参数张量的元素总数）
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e4:.1f} 万")

    # ------ 4. 优化器和损失函数 ------
    # Adam：自适应学习率的梯度下降优化器。对学习率初始值不太敏感，训练更稳定。
    # weight_decay 会在损失函数里加入参数幅度的惩罚，减轻过拟合。
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # MSE 损失：点回归任务的标准损失，计算预测坐标与真实坐标之间的均方误差。
    # 注意：MSE 本身有一个特性——当多个不同的落点都是合理答案时，
    # 它会"惩罚"任何偏离均值的预测，导致模型倾向于输出多个可能落点的平均值。
    # 这不是模型能力不足，而是 MSE 损失函数的特性决定的。这正是本文要解决的问题之一。
    loss_fn = nn.MSELoss()

    # ------ 5. 训练循环 ------
    # 追踪历史最优验证 MAE：初始值设为正无穷大，保证第一轮一定会刷新记录
    best_val, best_epoch, best_state = float("inf"), -1, None
    t0 = time.time()                # 记录训练开始时间
    max_batches = 15 if smoke else None   # 冒烟模式每轮只跑 15 个 batch，够验证链路即可

    # 冒烟模式只跑 2 轮；完整训练最多跑 EPOCHS（30）轮
    for epoch in range(1, (2 if smoke else EPOCHS) + 1):
        model.train()               # 切换训练模式：Dropout 层启用，随机丢弃部分神经元减轻过拟合
        for bi, (bx, by) in enumerate(dl_train, 1):   # 遍历训练 batch，bi 从 1 开始编号
            if max_batches and bi > max_batches:      # 冒烟模式：跑满 15 个 batch 就跳过剩余的
                break

            # --- 训练五步曲 ---
            pred = model(bx, device)                          # ① 前向传播：算出预测 (B, 2)
            loss = loss_fn(pred, by["landing"].to(device))    # ② 计算损失：预测 vs 标准化真值
            opt.zero_grad()          # ③ 清空梯度：PyTorch 梯度默认会累加，不清空就会累积上一轮的值
            loss.backward()          # ④ 反向传播：自动计算每个参数的梯度
            opt.step()               # ⑤ 更新参数：优化器按梯度和学习率调整所有参数
            # -------------------

            if bi % 100 == 0:        # 每 100 个 batch 打印一次当前损失，观察是否整体下降
                print(f"  epoch {epoch} batch {bi}/{len(dl_train)} loss={loss.item():.4f}")

        # ------ 每轮结束后，在验证集上评估 ------
        _, vp, vt = evaluate(model, dl_val, device)   # 验证集全量预测和真值，还原为真实坐标
        val_mae = mae_report(vp, vt)                  # 用统一协议计算 MAE

        flag = ""
        if val_mae["mae_mean"] < best_val:            # 是否刷新了历史最佳验证 MAE？
            best_val, best_epoch = val_mae["mae_mean"], epoch   # 记录新的最佳值和 epoch 编号
            # 保存当前模型的完整参数快照
            # state_dict() 返回一个字典：{参数名: 参数张量}
            # detach() 从计算图中分离、clone() 深拷贝——确保后续训练不会修改这份快照
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = " ← 目前最好"
        print(f"epoch {epoch:>2} | 验证集 MAE = {val_mae['mae_mean']:.3f} "
              f"(Δx={val_mae['mae_x']:.3f}, Δy={val_mae['mae_y']:.3f}){flag}")

        if smoke:                    # 冒烟模式到此为止，链路跑通即表示代码没问题
            print("冒烟测试通过：前向/反向/评估链路无 bug，可以放心完整训练")
            return
        # 早停判断：如果距离最后一次刷新最佳值已经过了 PATIENCE 轮，就提前停止
        if epoch - best_epoch >= PATIENCE:
            print(f"早停：验证集 MAE 已连续 {PATIENCE} 个 epoch 无改善")
            break

    # ------ 6. 加载最优权重，做最终评估 ------
    print(f"\n训练结束（用时 {(time.time()-t0)/60:.1f} 分钟），加载第 {best_epoch} 轮的最优权重")
    # 关键说明：加载的不是最后一轮的权重，而是验证集 MAE 最低那轮的权重。
    # 最后一轮可能已经过拟合了。
    model.load_state_dict(best_state)

    # 测试集：仅评估一次，用于论文报告
    _, tp, tt = evaluate(model, dl_test, device)
    # 验证集也再评估一次（论文附录对照用）
    _, vp, vt = evaluate(model, dl_val, device)

    # 汇总结果为字典，稍后写入 JSON
    results = {
        "model": "v1_point_baseline_gru",
        "protocol": "MAE = mean((|Δx|+|Δy|)/2)，真实球场坐标单位",
        "seed": SEED, "params": n_params, "best_epoch": best_epoch,   # 记录种子、参数量、最优轮数，保证可复现
        # ===== 留痕元信息（自动生成，勿手动改） =====
        "run_id": RUN_ID,
        "run_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "model_version": "v1_point_baseline",
            "seed": SEED,
            "lr": LR,
            "batch_size": BATCH,
            "epochs_max": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "device": str(device),
            "data_path": str(OUT_DIR),
            "data_version_mtime": _DATA_VERSION,
            "git_commit": _GIT_HASH,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        # ============================================
        "val": mae_report(vp, vt),
        "test": mae_report(tp, tt),
    }

    # 计算常数基线的 MAE（用于对照）
    results["constant_baseline"] = constant_baseline(ds_train,
                                                      {"val": dl_val, "test": dl_test})

    # ------ 7. 保存全部结果 ------
    RESULTS.mkdir(exist_ok=True)      # 结果目录不存在则创建（已存在不报错）
    torch.save(best_state, RESULTS / "v1_point_baseline.pt")   # 保存模型权重
    np.savez_compressed(RESULTS / "v1_test_predictions.npz",   # 保存测试集逐样本预测与真值（压缩格式）
                        pred=tp.numpy(), truth=tt.numpy())
    # 保存指标字典为 JSON 文件。ensure_ascii=False 允许中文直接写入，indent=2 缩进两格便于阅读
    (RESULTS / "v1_point_baseline.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------ 8. 画预测 vs 真值散点图 ------
    # matplotlib 放在 main 最后导入：冒烟测试走不到这里，可以省掉这个依赖
    import matplotlib
    matplotlib.use("Agg")            # 无界面后端：不弹窗，直接存文件（服务器/脚本环境必须）
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))            # 6×6 正方形画布，保证 x/y 轴比例一致
    ax.scatter(tt[:, 0], tt[:, 1], c="0.75", s=5, label="truth")        # 真值：灰色小点
    ax.scatter(tp[:, 0], tp[:, 1], c="#02743B", s=5, alpha=.5, label="prediction")  # 预测：绿色半透明小点
    ax.set_title(f"v1 point baseline (test MAE={results['test']['mae_mean']:.2f})")  # 标题嵌入测试 MAE
    ax.legend()
    fig.tight_layout()               # 自动收紧边距
    fig.savefig(RESULTS / "v1_pred_scatter.png", dpi=150)   # 存为 150dpi PNG

    # ------ 9. 控制台打印最终结果 ------
    print("\n===== v1 点回归基线结果（真实坐标单位）=====")
    print(f"{'':10} {'Δx':>8} {'Δy':>8} {'MAE':>8}")
    for name, r in [("GRU 点回归", results["test"]),
                    ("常数预测", results["constant_baseline"]["test"])]:
        print(f"{name:<10} {r['mae_x']:>8.2f} {r['mae_y']:>8.2f} {r['mae_mean']:>8.2f}")
    print(f"\n全部指标 → {RESULTS/'v1_point_baseline.json'}")
    print("以上数字是后续分布模型（v2/v3）必须超过的基线。")


if __name__ == "__main__":   # 直接运行本文件才执行 main()；被别的文件 import 时只暴露类和函数
    main()
