# -*- coding: utf-8 -*-
"""
2026 MCM Problem C - 问题一（基于“预处理后面板数据”的更新版可运行代码）
====================================================

你已提供的数据预处理逻辑（关键点）：
- weekX_judgeY_score 已统一转数值，N/A -> NaN，不填补
- 结构性缺失已区分并得到每季真实周数与每周评委人数
- “淘汰后记为 0”为状态占位，已构造 last_active_week，仅保留实际参赛周
- 宽表已转为长表/面板：选手–赛季–周（panel）
- 已生成与赛制同构的核心变量：
    total_judge_score, judge_rank, judge_percent
- 静态属性保守清洗，为后续问题准备

本代码只解决“问题一”：
- 反演每位选手每周粉丝投票份额 vote_share
- 并给出不确定性区间（min/max）与宽度（width）
- 同时给出每周“可行性”（是否存在投票份额使淘汰一致）

方法严格对应前述解题思路：
- Percent 合成（赛季 3-27）：基于 judge_percent + vote_share 的淘汰一致性约束
- Rank 合成（赛季 1-2, 28-34）：基于 judge_rank + fan_rank 的淘汰一致性约束
- 输出 vote_share：以“份额”为主（绝对票数不可识别，不强行假设总票数）

性能说明：
- 默认使用“可行采样法”估计区间与最大熵代表点（速度快，适合全赛季）
- 可选对 Percent 周次用线性规划求精确区间（慢）：MODE="exact_percent"

输出（保存到当前目录，并在运行过程中显示关键表与图）：
- q1_vote_share_estimates.csv
- q1_feasibility_summary.csv
- q1_season_example_heatmap.png
- q1_season_example_uncertainty.png
- q1_feasibility_overview.png
"""

import os
import math
import time
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize, linprog

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 0) 用户可调参数
# ============================================================
DATA_PATH = "data_preprocessed_panel.csv"
# 若你在本对话环境的 sandbox 运行，可改为：
# DATA_PATH = "/mnt/data/data_preprocessed_panel.csv"

OUT_DIR = os.getcwd()

# 运行模式：
# - "fast": Percent/Rank 都用可行采样估计区间，点估计用“最大熵可行样本”
# - "exact_percent": Percent 用 LP 求精确区间（较慢），Rank 仍用采样
MODE = "exact_percent"

# 采样参数（fast 模式下关键）
N_SAMPLES_PERCENT = 1500     # 每个 (season, week) 可行样本数目标（Percent）
N_SAMPLES_RANK = 500         # 每个 (season, week) 可行样本数目标（Rank）
MAX_TRIALS_RANK = 280000     # Rank 周次最大尝试次数（命中可行可能较难）
DIRICHLET_ALPHA = 1.0        # Dirichlet 参数：小 -> 更尖峰；大 -> 更均匀
RANDOM_SEED = 20260129

# 示例可视化赛季：None 表示自动选一个 percent 赛季
EXAMPLE_SEASON = None

# ============================================================
# 1) 绘图配置（中文字体 + 负号）
# ============================================================
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 2) notebook / script 双兼容展示
# ============================================================
def show_df(df: pd.DataFrame, n: int = 10, title: str = ""):
    """在 notebook 中用 display，否则用 print。"""
    if title:
        print(title)
    try:
        from IPython.display import display
        display(df.head(n))
    except Exception:
        print(df.head(n).to_string(index=False))

# ============================================================
# 3) 基础数学工具
# ============================================================
def entropy(x: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy（自然对数）。"""
    x = np.clip(x, eps, 1.0)
    return float(-(x * np.log(x)).sum())

def rank_desc(values: np.ndarray) -> np.ndarray:
    """降序排名，1 为最好；并列用平均名次。"""
    return pd.Series(values).rank(method="average", ascending=False).to_numpy()

def scheme_for_season(season: int) -> str:
    """
    赛制假设（与题面一致的合理设定）：
    - rank：赛季 1-2，及 28-34
    - percent：赛季 3-27
    """
    if season in (1, 2) or (28 <= season <= 34):
        return "rank"
    return "percent"

@dataclass
class WeekContext:
    season: int
    week: int
    scheme: str
    names: List[str]
    judge_total: np.ndarray
    judge_percent: np.ndarray
    judge_rank: np.ndarray
    eliminated_mask: np.ndarray
    n_active: int
    n_elim: int

# ============================================================
# 4) 读取预处理面板数据，并构造每周上下文
# ============================================================
panel = pd.read_csv(DATA_PATH)

# 关键列存在性检查（你的预处理规范应满足）
required_cols = [
    "celebrity_name", "ballroom_partner",
    "season", "week", "active", "last_active_week",
    "total_judge_score", "judge_rank", "judge_percent",
    "n_judges_present"
]
missing = [c for c in required_cols if c not in panel.columns]
if missing:
    raise ValueError(f"面板数据缺少必要列：{missing}")

# 强制类型（防止读入为 object）
panel["season"] = pd.to_numeric(panel["season"], errors="coerce").astype(int)
panel["week"] = pd.to_numeric(panel["week"], errors="coerce").astype(int)
panel["active"] = pd.to_numeric(panel["active"], errors="coerce").fillna(0).astype(int)
panel["last_active_week"] = pd.to_numeric(panel["last_active_week"], errors="coerce")
panel["total_judge_score"] = pd.to_numeric(panel["total_judge_score"], errors="coerce")
panel["judge_rank"] = pd.to_numeric(panel["judge_rank"], errors="coerce")
panel["judge_percent"] = pd.to_numeric(panel["judge_percent"], errors="coerce")

show_df(panel, 8, "面板数据预览：")

# 仅保留 active==1 的“实际参赛周”（你已做了占位剔除，此处再次确保）
panel_active = panel[panel["active"] == 1].copy()

# 赛季真实末周（基于 active 数据）
season_last_week = panel_active.groupby("season")["week"].max().to_dict()

# 构造每个 (season, week) 的 WeekContext
contexts: List[WeekContext] = []

for (season, week), g in panel_active.groupby(["season", "week"]):
    sch = scheme_for_season(int(season))
    # 当周在场选手
    names = g["celebrity_name"].astype(str).tolist()
    judge_total = g["total_judge_score"].to_numpy(dtype=float)
    judge_percent = g["judge_percent"].to_numpy(dtype=float)
    judge_rank = g["judge_rank"].to_numpy(dtype=float)

    # 淘汰掩码：若该选手的 last_active_week == 当前 week，且当前 week < 赛季末周，则视为“本周后淘汰”
    # 解释：赛季末周是自然终止，不应视为淘汰事件
    last_w = g["last_active_week"].to_numpy(dtype=float)
    is_last_week = (week >= season_last_week[int(season)])
    eliminated_mask = (last_w == float(week)) & (~is_last_week)

    n_active = len(g)
    n_elim = int(np.sum(eliminated_mask))

    # 过滤极小规模周（无法定义淘汰比较）
    if n_active <= 1:
        continue

    contexts.append(
        WeekContext(
            season=int(season),
            week=int(week),
            scheme=sch,
            names=names,
            judge_total=judge_total,
            judge_percent=judge_percent,
            judge_rank=judge_rank,
            eliminated_mask=eliminated_mask.astype(bool),
            n_active=n_active,
            n_elim=n_elim
        )
    )

print(f"\n构造 WeekContext 完成：共 {len(contexts)} 个 (season, week) 截面。")

# ============================================================
# 5) Percent 赛制：约束 +（采样 or LP）求解
# ============================================================
def percent_constraints(judge_percent: np.ndarray, elim_mask: np.ndarray, margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Percent 合成：
      combined_share = judge_percent + vote_share
    淘汰一致性（基本形式）：
      对任意被淘汰者 e 与任意未淘汰者 s：
      judge_percent[e] + vote[e] <= judge_percent[s] + vote[s]
    转换为线性不等式：
      vote[e] - vote[s] <= judge_percent[s] - judge_percent[e] - margin
    返回 A, b，使 A x <= b
    """
    n = len(judge_percent)
    E = np.where(elim_mask)[0]
    S = np.where(~elim_mask)[0]
    if len(E) == 0 or len(S) == 0:
        return np.zeros((0, n)), np.zeros((0,))
    A, b = [], []
    for e in E:
        for s in S:
            row = np.zeros(n)
            row[e] = 1.0
            row[s] = -1.0
            A.append(row)
            b.append((judge_percent[s] - judge_percent[e]) - margin)
    return np.array(A), np.array(b)

def percent_feasible(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> bool:
    """检查 A x <= b 与单纯形约束（此处假设 x 已是单纯形上的份额向量）。"""
    if np.any(x < -1e-12):
        return False
    if abs(np.sum(x) - 1.0) > 1e-8:
        return False
    if A.size and (np.any(A.dot(x) > b + 1e-12)):
        return False
    return True

def percent_point_from_samples(X: np.ndarray) -> np.ndarray:
    """从可行样本中选熵最大的样本作为代表点。"""
    ent = np.array([entropy(x) for x in X])
    return X[np.argmax(ent)]

def percent_sampling_interval(judge_percent: np.ndarray, elim_mask: np.ndarray,
                              rng: np.random.Generator,
                              n_samples_target: int,
                              alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """
    Percent：Dirichlet 采样 -> 过滤可行 -> 得到区间（min/max）与代表点（最大熵样本）
    """
    n = len(judge_percent)
    A, b = percent_constraints(judge_percent, elim_mask, margin=0.0)

    # 无淘汰约束：整个单纯形为可行集
    if elim_mask.sum() == 0:
        x_point = np.ones(n) / n
        bounds = np.vstack([np.zeros(n), np.ones(n)])
        return x_point, bounds, True, "no_elim_uniform"

    alpha_vec = np.ones(n) * alpha
    feas = []
    trials = 0
    # 经验上，percent 约束的可行命中率通常较高，因此只做有限倍数尝试
    max_trials = n_samples_target * 35

    while len(feas) < n_samples_target and trials < max_trials:
        trials += 1
        x = rng.dirichlet(alpha_vec)
        if A.size == 0 or np.all(A.dot(x) <= b + 1e-12):
            feas.append(x)

    if len(feas) == 0:
        return np.full(n, np.nan), np.full((2, n), np.nan), False, "no_feasible_samples"

    X = np.vstack(feas)
    x_point = percent_point_from_samples(X)
    bounds = np.vstack([X.min(axis=0), X.max(axis=0)])
    return x_point, bounds, True, f"feasible={len(feas)} trials={trials}"

def percent_exact_interval_lp(judge_percent: np.ndarray, elim_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """
    Percent：对每个变量求精确 min/max（LP），严格但较慢。
    """
    n = len(judge_percent)
    A_ub, b_ub = percent_constraints(judge_percent, elim_mask, margin=0.0)
    A_eq, b_eq = np.ones((1, n)), np.array([1.0])
    bounds = [(0.0, 1.0) for _ in range(n)]

    # 可行性检查
    feas = linprog(c=np.zeros(n),
                   A_ub=A_ub if A_ub.size else None, b_ub=b_ub if b_ub.size else None,
                   A_eq=A_eq, b_eq=b_eq,
                   bounds=bounds, method="highs")
    if not feas.success:
        return np.full(n, np.nan), np.full(n, np.nan), False, "infeasible_lp"

    x_min = np.zeros(n)
    x_max = np.zeros(n)

    for i in range(n):
        c = np.zeros(n); c[i] = 1.0
        res_min = linprog(c=c,
                          A_ub=A_ub if A_ub.size else None, b_ub=b_ub if b_ub.size else None,
                          A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs")
        x_min[i] = res_min.fun if res_min.success else np.nan

        c2 = np.zeros(n); c2[i] = -1.0
        res_max = linprog(c=c2,
                          A_ub=A_ub if A_ub.size else None, b_ub=b_ub if b_ub.size else None,
                          A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method="highs")
        x_max[i] = (-res_max.fun) if res_max.success else np.nan

    return x_min, x_max, True, "ok"

# ============================================================
# 6) Rank 赛制：一致性检查 + 采样
# ============================================================
def rank_check_elimination(judge_rank: np.ndarray, vote_share: np.ndarray, elim_mask: np.ndarray) -> bool:
    """
    Rank 合成：
      combined_rank = judge_rank + fan_rank
    其中 fan_rank 为 vote_share 的降序名次（1 为最好）
    淘汰一致性：
      被淘汰者应属于 combined_rank 最差的 k 个
    边界并列：
      允许被淘汰集合落在“边界并列的最差组”内
    """
    n = len(judge_rank)
    k = int(np.sum(elim_mask))
    if k == 0:
        return True
    if k >= n:
        return False

    fan_rank = rank_desc(vote_share)
    combined = judge_rank + fan_rank

    order = np.argsort(combined)  # 小为好
    worst_k = set(order[-k:].tolist())
    elim_set = set(np.where(elim_mask)[0].tolist())
    if elim_set == worst_k:
        return True

    cutoff = np.sort(combined)[-k]
    worst_tied = set(np.where(combined >= cutoff - 1e-12)[0].tolist())
    return elim_set.issubset(worst_tied)

def rank_sampling_interval(judge_rank: np.ndarray, elim_mask: np.ndarray,
                           rng: np.random.Generator,
                           n_samples_target: int,
                           max_trials: int,
                           alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """
    Rank：Dirichlet 采样 vote_share -> 过滤一致性 -> 得到区间与代表点（最大熵样本）
    """
    n = len(judge_rank)
    k = int(np.sum(elim_mask))

    if k == 0:
        x_point = np.ones(n) / n
        bounds = np.vstack([np.zeros(n), np.ones(n)])
        return x_point, bounds, True, "no_elim_uniform"

    alpha_vec = np.ones(n) * alpha
    feas = []
    trials = 0
    local_alpha = alpha

    while len(feas) < n_samples_target and trials < max_trials:
        trials += 1
        x = rng.dirichlet(alpha_vec)
        if rank_check_elimination(judge_rank, x, elim_mask):
            feas.append(x)

        # 轻度自适应：若长时间命中不足，使分布更尖峰以提升命中概率
        if trials in (20000, 60000, 120000, 200000):
            local_alpha = max(0.25, local_alpha / 2)
            alpha_vec = np.ones(n) * local_alpha

    if len(feas) == 0:
        return np.full(n, np.nan), np.full((2, n), np.nan), False, f"no_feasible_trials={trials}"

    X = np.vstack(feas)
    x_point = percent_point_from_samples(X)  # 仍用最大熵样本作为代表点
    bounds = np.vstack([X.min(axis=0), X.max(axis=0)])
    return x_point, bounds, True, f"feasible={len(feas)} trials={trials} alpha_end={local_alpha}"

# ============================================================
# 7) 主循环：逐 (season, week) 估计 vote_share
# ============================================================
rng = np.random.default_rng(RANDOM_SEED)

estimate_rows = []
feas_rows = []

t0 = time.time()

for ctx in contexts:
    n = ctx.n_active
    k = ctx.n_elim

    # 由于你已面板化并计算 judge_rank/judge_percent，这里直接使用
    if ctx.scheme == "percent":
        # --- 可行域求解：区间 + 代表点 ---
        if MODE == "exact_percent":
            x_min, x_max, ok, msg = percent_exact_interval_lp(ctx.judge_percent, ctx.eliminated_mask)
            if ok:
                # LP 只给区间，不给代表点；代表点取区间中心并再投影到单纯形（简洁且稳定）
                x_point = (x_min + x_max) / 2.0
                x_point = np.clip(x_point, 0, None)
                if x_point.sum() > 0:
                    x_point = x_point / x_point.sum()
                else:
                    x_point = np.ones(n) / n
                bounds = np.vstack([x_min, x_max])
        else:
            x_point, bounds, ok, msg = percent_sampling_interval(
                ctx.judge_percent, ctx.eliminated_mask,
                rng=rng, n_samples_target=N_SAMPLES_PERCENT, alpha=DIRICHLET_ALPHA
            )

        feas_rows.append({
            "season": ctx.season,
            "week": ctx.week,
            "scheme": ctx.scheme,
            "n_active": n,
            "n_eliminated": k,
            "feasible": bool(ok),
            "detail": msg
        })

        if ok:
            for i, name in enumerate(ctx.names):
                estimate_rows.append({
                    "season": ctx.season,
                    "week": ctx.week,
                    "scheme": ctx.scheme,
                    "celebrity_name": name,
                    "total_judge_score": float(ctx.judge_total[i]),
                    "judge_percent": float(ctx.judge_percent[i]),
                    "judge_rank": float(ctx.judge_rank[i]),
                    "eliminated_after_week": bool(ctx.eliminated_mask[i]),
                    "vote_share_point": float(x_point[i]),
                    "vote_share_min": float(bounds[0, i]),
                    "vote_share_max": float(bounds[1, i]),
                    "vote_share_width": float(bounds[1, i] - bounds[0, i]),
                })

    else:
        # --- rank scheme ---
        x_point, bounds, ok, msg = rank_sampling_interval(
            ctx.judge_rank, ctx.eliminated_mask,
            rng=rng, n_samples_target=N_SAMPLES_RANK, max_trials=MAX_TRIALS_RANK, alpha=DIRICHLET_ALPHA
        )

        feas_rows.append({
            "season": ctx.season,
            "week": ctx.week,
            "scheme": ctx.scheme,
            "n_active": n,
            "n_eliminated": k,
            "feasible": bool(ok),
            "detail": msg
        })

        if ok:
            for i, name in enumerate(ctx.names):
                estimate_rows.append({
                    "season": ctx.season,
                    "week": ctx.week,
                    "scheme": ctx.scheme,
                    "celebrity_name": name,
                    "total_judge_score": float(ctx.judge_total[i]),
                    "judge_percent": float(ctx.judge_percent[i]),
                    "judge_rank": float(ctx.judge_rank[i]),
                    "eliminated_after_week": bool(ctx.eliminated_mask[i]),
                    "vote_share_point": float(x_point[i]),
                    "vote_share_min": float(bounds[0, i]),
                    "vote_share_max": float(bounds[1, i]),
                    "vote_share_width": float(bounds[1, i] - bounds[0, i]),
                })

t1 = time.time()
print(f"\n问题一估计完成，用时 {t1 - t0:.2f} 秒。")

est_df = pd.DataFrame(estimate_rows)
feas_df = pd.DataFrame(feas_rows)

print("估计输出维度：", est_df.shape)
print("可行性输出维度：", feas_df.shape)

show_df(feas_df.groupby("scheme")["feasible"].mean().reset_index().rename(columns={"feasible": "feasible_rate"}),
        10, "\n按赛制汇总可行率：")

show_df(est_df, 12, "\n估计结果预览：")

# ============================================================
# 8) 保存结果文件
# ============================================================
est_csv = os.path.join(OUT_DIR, "q1_vote_share_estimates.csv")
feas_csv = os.path.join(OUT_DIR, "q1_feasibility_summary.csv")

est_df.to_csv(est_csv, index=False, encoding="utf-8-sig")
feas_df.to_csv(feas_csv, index=False, encoding="utf-8-sig")

print("\n已保存结果文件：")
print(" -", est_csv)
print(" -", feas_csv)

# ============================================================
# 9) 可视化：示例赛季热图（份额点估计 + 不确定性宽度）
# ============================================================
# 选择示例赛季：优先 percent
if EXAMPLE_SEASON is None:
    percent_seasons = sorted([s for s in est_df["season"].unique() if scheme_for_season(int(s)) == "percent"])
    EXAMPLE_SEASON = percent_seasons[len(percent_seasons)//2] if percent_seasons else int(sorted(est_df["season"].unique())[0])

ex = est_df[est_df["season"] == int(EXAMPLE_SEASON)].copy()
if ex.empty:
    EXAMPLE_SEASON = int(sorted(est_df["season"].unique())[0])
    ex = est_df[est_df["season"] == EXAMPLE_SEASON].copy()

print(f"\n示例可视化赛季：{EXAMPLE_SEASON}（scheme={scheme_for_season(EXAMPLE_SEASON)}）")

# pivot：行选手，列周次
pivot_point = ex.pivot_table(index="celebrity_name", columns="week", values="vote_share_point", aggfunc="mean")
pivot_width = ex.pivot_table(index="celebrity_name", columns="week", values="vote_share_width", aggfunc="mean")

# 排序：按平均投票份额降序
order = pivot_point.mean(axis=1).sort_values(ascending=False).index
pivot_point = pivot_point.loc[order]
pivot_width = pivot_width.loc[order]

def nice_heatmap(mat: pd.DataFrame, title: str, cbar_label: str, save_path: str):
    fig = plt.figure(figsize=(10.8, 6.8))
    ax = plt.gca()

    img = ax.imshow(mat.fillna(np.nan), aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("周次")
    ax.set_ylabel("选手")

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.tolist(), rotation=0)

    ax.set_yticks(np.arange(mat.shape[0]))
    # 名字过多时稀疏显示，保持图面整洁
    if mat.shape[0] > 26:
        step = math.ceil(mat.shape[0] / 26)
        ylab = [nm if (i % step == 0) else "" for i, nm in enumerate(mat.index)]
    else:
        ylab = mat.index.tolist()
    ax.set_yticklabels(ylab)

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# 图 1：点估计热图
heatmap_path = os.path.join(OUT_DIR, "q1_season_example_heatmap.png")
nice_heatmap(
    pivot_point,
    title=f"问题一 粉丝投票份额点估计热图 示例赛季 {EXAMPLE_SEASON}",
    cbar_label="投票份额",
    save_path=heatmap_path
)
print("已保存图像：", heatmap_path)

# 图 2：不确定性宽度热图
unc_path = os.path.join(OUT_DIR, "q1_season_example_uncertainty.png")
nice_heatmap(
    pivot_width,
    title=f"问题一 投票份额不确定性宽度热图 示例赛季 {EXAMPLE_SEASON}",
    cbar_label="区间宽度",
    save_path=unc_path
)
print("已保存图像：", unc_path)

# ============================================================
# 10) 可视化：各赛季周次可行率概览
# ============================================================
season_feas = feas_df.groupby(["season", "scheme"])["feasible"].mean().reset_index()

fig = plt.figure(figsize=(10.2, 4.8))
ax = plt.gca()
for sch, g in season_feas.groupby("scheme"):
    ax.plot(g["season"], g["feasible"], marker="o", linestyle="-", label=f"{sch} 周可行率")

ax.set_title("问题一 各赛季周次可行率概览")
ax.set_xlabel("赛季")
ax.set_ylabel("周可行率")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.25)
ax.legend()

plt.tight_layout()
feas_plot_path = os.path.join(OUT_DIR, "q1_feasibility_overview.png")
plt.savefig(feas_plot_path, bbox_inches="tight")
plt.show()

print("已保存图像：", feas_plot_path)

# ============================================================
# 11) 过程诊断展示：不确定性最大记录（信息约束弱的情形）
# ============================================================
print("\n不确定性最大的若干记录（用于定位约束弱的周次与选手）：")
show_df(est_df.sort_values("vote_share_width", ascending=False), 20)

print("\n生成文件清单：")
for p in [est_csv, feas_csv, heatmap_path, unc_path, feas_plot_path]:
    print(" -", p)
