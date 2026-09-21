# -*- coding: utf-8 -*-
"""
ShuttleGen v0 —— 数据管线
=========================
把 ShuttleSet22 原始标注变成训练样本：**给定回合前 t 拍 → 预测第 t+1 拍的 (击球类型, 落点)**。

运行方式（只需跑一次，产物存到 processed/ 供 v1-v4 反复读取）：
    conda activate badminton_cv
    python data_pipeline.py

处理流程（对齐官方 preprocess_data.py 的规则，并修正其脚本里的两个问题：
官方脚本第 19 行有一句 1/0 会直接崩溃；getpoint_player 替换有笔误——我们都不需要）：

  原始 CSV（像素坐标）
    → 清洗：丢标注瑕疵/未知球种/场外击球/坐标缺失的回合
    → 球种 18 类合并为 10 类（官方规则）
    → 单应矩阵投影：像素坐标 → 真实球场坐标
    → z-score 标准化（常数沿用官方）
    → 按官方划分 train / val / test（按整场比赛划分，防止同场数据泄漏）
    → 存成"每个回合一条变长序列"的填充矩阵

样本定义（这就是论文的任务本身）：
    输入 = 前 t 拍（球种链 / 历史落点 / 球员 / 位置 / 比分 …），t = 1..L-1
    目标 = 第 t+1 拍的 type_id 和 landing (x, y)
    一个长度 L 的回合产生 L-1 个样本
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ============ 路径与官方常量 ============
# 原始数据目录：请把 ShuttleSet22 从 CoachAI-Projects 仓库下载后，
# 将 set/ 文件夹放到仓库根目录的 datasets/ShuttleSet22/ 下。
# 也可以改成你自己的绝对路径（仅本地使用，不上传到 GitHub）。
RAW_DIR = Path(__file__).resolve().parent / "datasets" / "ShuttleSet22" / "set"
OUT_DIR = Path(__file__).resolve().parent / "processed"

# 官方划分：这 5 场做验证、这 9 场做测试，其余全部训练
VAL_ID = [38, 46, 47, 48, 49]
TEST_ID = [39, 40, 50, 52, 53, 54, 55, 56, 57]

# ---- 球种 18 → 10（先中文内部合并，再翻译成英文 canonical 名）----
COMBINED_TYPES = {
    '小平球': '平球', '後場抽平球': '平球',          # 小平球是旧版命名，并入平球
    '過度切球': '切球',                              # 被动切球并入切球
    '點扣': '殺球',                                  # 点扣（手腕杀）并入杀球
    '擋小球': '接殺防守', '防守回挑': '接殺防守', '防守回抽': '接殺防守',
    '放小球': '網前球', '勾球': '網前球',
    '推球': '推撲球', '撲球': '推撲球',
}
ZH_TO_EN = {
    '發短球': 'short service', '發長球': 'long service', '長球': 'clear',
    '挑球': 'lob', '切球': 'drop', '殺球': 'smash', '平球': 'drive',
    '推撲球': 'push/rush', '網前球': 'net shot', '接殺防守': 'defensive shot',
}
# type_id 的固定编号顺序（0-9）
TYPE_NAMES = ['short service', 'long service', 'clear', 'lob', 'drop',
              'smash', 'drive', 'push/rush', 'net shot', 'defensive shot']
TYPE_TO_ID = {t: i for i, t in enumerate(TYPE_NAMES)}

# 坐标标准化常数（真实球场坐标系；官方 preprocess_data.py 给定的均值/方差）
MEAN_X, STD_X = 175., 82.
MEAN_Y, STD_Y = 467., 192.

OUTSIDE_AREAS = [10, 11, 12, 13, 14, 15, 16]   # 场外区域编号
AREA_COLS = ["landing_area", "player_location_area", "opponent_location_area"]


# ============ 第一步：读原始 CSV ============

def load_raw():
    """读 match.csv + homography.csv + 每场的 set{1,2,3}.csv。

    返回：(每拍一张表的明细 DataFrame, {match_id: 3x3 单应矩阵}, 球员名单)
    """
    match = pd.read_csv(RAW_DIR / "match.csv")
    hom = pd.read_csv(RAW_DIR / "homography.csv")
    hom_dict = {int(r["id"]): np.array(json.loads(r["homography_matrix"]))
                for _, r in hom.iterrows()}

    # 球员姓名 → 全局编号（"球员风格编码器"的输入；官方数据共 35 名顶尖单打球手）
    players = sorted(set(match["winner"]) | set(match["loser"]))
    player_to_id = {p: i for i, p in enumerate(players)}

    frames = []
    for _, m in match.iterrows():
        folder = RAW_DIR / m["video"]
        if not folder.exists():
            print(f"[警告] 缺少比赛文件夹：{m['video']}")
            continue
        for csv in sorted(folder.glob("set*.csv")):
            df = pd.read_csv(csv)
            df["set"] = int(re.findall(r"\d+", csv.name)[0])   # 第几局
            df["match_id"] = int(m["id"])
            # 官方约定：A = 该场胜方，B = 负方 → 替换成全局球员编号
            df["player_id"] = df["player"].map(
                {"A": player_to_id[m["winner"]], "B": player_to_id[m["loser"]]})
            frames.append(df)

    strokes = pd.concat(frames, ignore_index=True, sort=False)
    print(f"读入 {match.shape[0]} 场比赛 / {len(frames)} 局 / {len(strokes)} 拍 / "
          f"{len(players)} 名球员")
    return strokes, hom_dict, players


# ============ 第二步：清洗（规则与官方一致） ============

def clean(strokes: pd.DataFrame) -> pd.DataFrame:
    """逐条规则丢"整个回合"（一拍坏则全回合丢弃，避免脏上下文）。"""

    def drop_rallies_with(df, mask, reason):
        bad = df.loc[mask, "rally_id"].unique()
        df = df[~df["rally_id"].isin(bad)]
        print(f"  丢 {reason:<10} 回合 {len(bad):>4} 个 → 剩 "
              f"{df['rally_id'].nunique()} 回合 / {len(df)} 拍")
        return df

    strokes = strokes.copy()
    # 全局回合编号（比赛×局×回合 唯一）
    strokes["rally_id"] = strokes.groupby(["match_id", "set", "rally"]).ngroup()
    print("清洗：")

    strokes = drop_rallies_with(strokes, strokes["flaw"].notna(), "标注瑕疵")
    strokes = drop_rallies_with(strokes, strokes["type"] == "未知球種", "未知球种")

    # 发球拍的击球区域官方视为中场 7 号区，再剔掉击球点在界外的回合
    strokes.loc[strokes["server"] == 1, "hit_area"] = 7
    strokes = drop_rallies_with(strokes, strokes["hit_area"].isin(OUTSIDE_AREAS), "场外击球")
    strokes = drop_rallies_with(strokes, strokes["hit_area"].isna(), "击球区缺失")
    strokes = drop_rallies_with(strokes, strokes["landing_area"].isna(), "落点区缺失")

    # 防御性检查：模型要用到的坐标/区域都不能缺（比官方更严一点）
    need = ["landing_x", "landing_y",
            "player_location_x", "player_location_y",
            "opponent_location_x", "opponent_location_y"] + AREA_COLS
    strokes = drop_rallies_with(strokes, strokes[need].isna().any(axis=1), "坐标缺失")

    # 球种 18 → 10 → 英文名 → 编号；仍映射不上的回合整段丢弃
    strokes["type"] = strokes["type"].replace(COMBINED_TYPES).map(ZH_TO_EN)
    strokes = drop_rallies_with(strokes, strokes["type"].isna(), "球种漏网")
    strokes["type_id"] = strokes["type"].map(TYPE_TO_ID).astype(np.int64)

    # 位置区域：场外一律记 10；二值标志补零
    for col in AREA_COLS:
        strokes.loc[strokes[col].isin(OUTSIDE_AREAS), col] = 10
        strokes[col] = strokes[col].astype(np.int64)
    strokes["backhand"] = strokes["backhand"].fillna(0).astype(np.int64)
    strokes["aroundhead"] = strokes["aroundhead"].fillna(0).astype(np.int64)
    strokes["landing_height"] = pd.to_numeric(
        strokes["landing_height"], errors="coerce").fillna(0).astype(np.int64)
    strokes["ball_round"] = strokes["ball_round"].astype(np.int64)

    return strokes.reset_index(drop=True)


# ============ 第三步：坐标投影 + 标准化 ============

def project_and_normalize(strokes: pd.DataFrame, hom_dict: dict) -> pd.DataFrame:
    """像素坐标 --单应矩阵 H--> 真实球场坐标 --z-score--> 标准化坐标。

    单应投影：p_real = H @ [x, y, 1]^T，再除以齐次分量（官方 preprocess 的向量化版）。
    对落点、击球者位置、接球者位置三组坐标都做同样的处理。
    """
    strokes = strokes.copy()
    for prefix in ["landing", "player_location", "opponent_location"]:
        col_x, col_y = f"{prefix}_x", f"{prefix}_y"
        new_x = pd.Series(np.nan, index=strokes.index)
        new_y = pd.Series(np.nan, index=strokes.index)
        for mid, g in strokes.groupby("match_id", sort=False):
            H = hom_dict[mid]
            pts = np.stack([g[col_x].to_numpy(dtype=float),
                            g[col_y].to_numpy(dtype=float),
                            np.ones(len(g))], axis=1)        # (n, 3)
            real = pts @ H.T                                   # (n, 3)
            real = real[:, :2] / real[:, 2:3]                  # 齐次除法
            new_x.loc[g.index] = real[:, 0]
            new_y.loc[g.index] = real[:, 1]
        strokes[col_x] = new_x.to_numpy()
        strokes[col_y] = new_y.to_numpy()

    # z-score 标准化（六列共用两组常数；真实球场 x/y 的量纲一致）
    for col, m, s in [("landing_x", MEAN_X, STD_X), ("landing_y", MEAN_Y, STD_Y),
                      ("player_location_x", MEAN_X, STD_X),
                      ("player_location_y", MEAN_Y, STD_Y),
                      ("opponent_location_x", MEAN_X, STD_X),
                      ("opponent_location_y", MEAN_Y, STD_Y)]:
        strokes[col] = (strokes[col] - m) / s
    return strokes


# ============ 第四步：转成"每回合一条序列"的填充矩阵 ============

def build_arrays(strokes: pd.DataFrame) -> dict:
    """把一个 split 的所有回合堆成 (N, L) / (N, L, 2) 的填充数组。

    N = 回合数，L = 最长回合拍数；不足 L 的回合尾部补 0，
    真实长度存在 length 里（使用时必须配合 length 做 mask）。
    """
    grouped = strokes.sort_values(["rally_id", "ball_round"]).groupby("rally_id", sort=True)
    rallies = [g for _, g in grouped]
    N, L = len(rallies), max(len(g) for g in rallies)

    def pad1(col, dtype=np.int64):
        out = np.zeros((N, L), dtype=dtype)
        for i, g in enumerate(rallies):
            out[i, :len(g)] = g[col].to_numpy()
        return out

    def pad_pair(cx, cy, dtype=np.float32):
        out = np.zeros((N, L, 2), dtype=dtype)
        for i, g in enumerate(rallies):
            out[i, :len(g), 0] = g[cx].to_numpy(dtype=float)
            out[i, :len(g), 1] = g[cy].to_numpy(dtype=float)
        return out

    return {
        "type_id": pad1("type_id"),                                    # 击球类型 0-9
        "landing": pad_pair("landing_x", "landing_y"),                 # 落点 (x,y) 已标准化
        "player_id": pad1("player_id"),                                # 击球者全局编号
        "player_loc": pad_pair("player_location_x", "player_location_y"),
        "opp_loc": pad_pair("opponent_location_x", "opponent_location_y"),
        "scores": np.stack([pad1("roundscore_A"), pad1("roundscore_B")], axis=-1),
        "ball_round": pad1("ball_round"),                              # 回合内第几拍
        "backhand": pad1("backhand"),                                  # 反手 0/1
        "aroundhead": pad1("aroundhead"),                              # 绕头 0/1
        "landing_height": pad1("landing_height"),                      # 落点过网高度档
        "landing_area": pad1("landing_area"),                          # 落点区域 1-10
        "length": np.array([len(g) for g in rallies], dtype=np.int64),
        "match_id": np.array([int(g["match_id"].iloc[0]) for g in rallies], dtype=np.int64),
    }


def run_pipeline():
    strokes, hom_dict, players = load_raw()
    strokes = clean(strokes)
    strokes = project_and_normalize(strokes, hom_dict)

    OUT_DIR.mkdir(exist_ok=True)
    is_val = strokes["match_id"].isin(VAL_ID)
    is_test = strokes["match_id"].isin(TEST_ID)
    splits = {"train": strokes[~(is_val | is_test)],
              "val": strokes[is_val],
              "test": strokes[is_test]}

    meta = {"type_names": TYPE_NAMES, "player_names": players,
            "norm": {"mean_x": MEAN_X, "std_x": STD_X,
                     "mean_y": MEAN_Y, "std_y": STD_Y},
            "val_match_id": VAL_ID, "test_match_id": TEST_ID, "splits": {}}

    for name, df in splits.items():
        arrs = build_arrays(df)
        n_strokes = int(arrs["length"].sum())
        n_samples = int(np.maximum(arrs["length"] - 1, 0).sum())  # 每回合 L 拍 → L-1 个样本
        np.savez_compressed(OUT_DIR / f"{name}.npz", **arrs)
        meta["splits"][name] = {"matches": int(df["match_id"].nunique()),
                                "rallies": len(arrs["length"]),
                                "strokes": n_strokes, "samples": n_samples}
        print(f"[{name:>5}] {df['match_id'].nunique():>2} 场 / "
              f"{len(arrs['length']):>5} 回合 / {n_strokes:>6} 拍 / {n_samples:>6} 预测样本")

    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"\n产物已写入 {OUT_DIR}")


# ============ PyTorch Dataset（v1 起直接 import 使用） ============

class ShuttleGenDataset(Dataset):
    """给定前 t 拍 → 预测第 t+1 拍的 (type_id, landing)。

    用法：
        ds = ShuttleGenDataset("processed/train.npz")
        dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    """

    def __init__(self, npz_path):
        d = np.load(npz_path)
        for k in d.files:
            setattr(self, k, d[k])
        # 每个回合贡献 length-1 个样本；offsets 做全局样本号 → (回合, 时刻) 的映射
        self.n_per_rally = np.maximum(self.length - 1, 0)
        self.offsets = np.concatenate([[0], np.cumsum(self.n_per_rally)])

    def __len__(self):
        return int(self.offsets[-1])

    def rally_and_time(self, idx):
        """全局样本号 → (回合下标 r, 历史长度 t)。目标即为第 t+1 拍。"""
        r = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        t = idx - int(self.offsets[r]) + 1
        return r, t

    def __getitem__(self, idx):
        r, t = self.rally_and_time(idx)
        x = {  # —— 输入：前 t 拍的全部可见信息 ——
            "type_id": torch.from_numpy(self.type_id[r, :t]),
            "landing": torch.from_numpy(self.landing[r, :t]),
            "player_id": torch.from_numpy(self.player_id[r, :t]),
            "player_loc": torch.from_numpy(self.player_loc[r, :t]),
            "opp_loc": torch.from_numpy(self.opp_loc[r, :t]),
            "scores": torch.from_numpy(self.scores[r, :t]),
            "ball_round": torch.from_numpy(self.ball_round[r, :t]),
            "backhand": torch.from_numpy(self.backhand[r, :t]),
            "aroundhead": torch.from_numpy(self.aroundhead[r, :t]),
            # 下一拍的击球者 = 目标拍的执行者（球员风格编码器条件）
            "next_player_id": int(self.player_id[r, t]),
        }
        y = {  # —— 目标：第 t+1 拍 ——
            "type_id": int(self.type_id[r, t]),
            "landing": torch.from_numpy(self.landing[r, t].copy()),   # (2,)
        }
        return x, y


def collate_fn(batch):
    """把变长样本 pad 成规整 batch（训练时 DataLoader 必须用它）。"""
    xs, ys = zip(*batch)
    B = len(batch)
    T = max(x["type_id"].numel() for x in xs)

    def pad1(key, dtype):
        out = torch.zeros(B, T, dtype=dtype)
        for i, x in enumerate(xs):
            out[i, :x[key].numel()] = x[key]
        return out

    def pad2(key, dtype):
        out = torch.zeros(B, T, 2, dtype=dtype)
        for i, x in enumerate(xs):
            out[i, :x[key].shape[0]] = x[key]
        return out

    mask = torch.zeros(B, T, dtype=torch.bool)     # True = 有效位置
    for i, x in enumerate(xs):
        mask[i, :x["type_id"].numel()] = True

    batched_x = {
        "type_id": pad1("type_id", torch.long),
        "landing": pad2("landing", torch.float32),
        "player_id": pad1("player_id", torch.long),
        "player_loc": pad2("player_loc", torch.float32),
        "opp_loc": pad2("opp_loc", torch.float32),
        "scores": pad2("scores", torch.long),   # (t, 2) 二维，用 pad2
        "ball_round": pad1("ball_round", torch.long),
        "backhand": pad1("backhand", torch.long),
        "aroundhead": pad1("aroundhead", torch.long),
        "next_player_id": torch.tensor([x["next_player_id"] for x in xs], dtype=torch.long),
        "mask": mask,
    }
    batched_y = {
        "type_id": torch.tensor([y["type_id"] for y in ys], dtype=torch.long),
        "landing": torch.stack([y["landing"] for y in ys]),            # (B, 2)
    }
    return batched_x, batched_y


# ============ 验证：跑完管线后自检 + 输出统计图 ============

def verify():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.loads((OUT_DIR / "meta.json").read_text(encoding="utf-8"))
    ds = ShuttleGenDataset(OUT_DIR / "train.npz")
    print(f"\n自检：训练集 {len(ds.length)} 回合 → {len(ds)} 样本")

    # 1) 解码一个样本，肉眼核对
    idx = len(ds) // 2
    x, y = ds[idx]
    r, t = ds.rally_and_time(idx)
    hist = " -> ".join(TYPE_NAMES[i] for i in x["type_id"].tolist())
    lx = y["landing"][0].item() * STD_X + MEAN_X
    ly = y["landing"][1].item() * STD_Y + MEAN_Y
    print(f"  样本#{idx}（回合{r}，历史{t}拍）")
    print(f"    球种链: {hist}")
    print(f"    目标: 第{t + 1}拍 = {TYPE_NAMES[y['type_id']]}，"
          f"落点=({lx:.0f}, {ly:.0f}) 真实坐标")

    # 2) DataLoader 冒烟测试（v1 直接可用）
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    bx, by = next(iter(dl))
    print(f"  DataLoader batch: type_id{tuple(bx['type_id'].shape)} "
          f"landing{tuple(bx['landing'].shape)} mask{tuple(bx['mask'].shape)} "
          f"目标landing{tuple(by['landing'].shape)}")

    # 3) 球种分布 + 落点范围（把有效拍抠出来，排除 padding）
    tid, land, length = ds.type_id, ds.landing, ds.length
    valid_t, valid_l = [], []
    for i in range(len(length)):
        valid_t.extend(tid[i, :length[i]])
        valid_l.append(land[i, :length[i]])
    valid_t = np.array(valid_t)
    valid_l = np.concatenate(valid_l)
    real_x = valid_l[:, 0] * STD_X + MEAN_X
    real_y = valid_l[:, 1] * STD_Y + MEAN_Y
    print("  训练集球种分布：")
    for i, name in enumerate(TYPE_NAMES):
        print(f"    {name:<15} {(valid_t == i).sum():>6} 拍")
    print(f"  落点范围: x∈[{real_x.min():.0f}, {real_x.max():.0f}], "
          f"y∈[{real_y.min():.0f}, {real_y.max():.0f}]（真实坐标）")

    # 4) 统计图：球种分布 + 落点散点（这张散点图就是论文 Figure 1 的素材）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order = np.argsort(-np.bincount(valid_t, minlength=10))
    axes[0].barh([TYPE_NAMES[i] for i in order],
                 [(valid_t == i).sum() for i in order], color="#02743B")
    axes[0].set_title("Training stroke type distribution")
    axes[0].invert_yaxis()
    sc = axes[1].scatter(real_x, real_y, c=valid_t, cmap="tab10", s=4, alpha=.4)
    axes[1].set_title("Training landing points (real-court coords)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].legend(*sc.legend_elements(num=10), loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stats.png", dpi=150)
    print(f"  统计图已存 {OUT_DIR / 'stats.png'}")


if __name__ == "__main__":
    run_pipeline()
    verify()
