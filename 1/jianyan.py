# -*- coding: utf-8 -*-
"""
问题一：一致性 + 确定性 + 灵敏度 + 鲁棒性 诊断代码
---------------------------------------------------
依赖文件（需已存在）：
- q1_vote_share_estimates.csv
- q1_feasibility_summary.csv
可选（用于灵敏度重估的抽样复算）：
- data_preprocessed_panel.csv

输出（保存当前目录，同时展示）：
- q1_diag_consistency_summary.csv
- q1_diag_certainty_summary.csv
- q1_diag_sensitivity_summary.csv
- q1_diag_robustness_summary.csv
- Figures:
  * q1_diag_width_hist.png
  * q1_diag_width_box_elim.png
  * q1_diag_feasible_by_season.png
  * q1_diag_sensitivity_alpha.png
  * q1_diag_robust_seed.png
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

# ============== 图形参数（中文 + 负号） ==============
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140

OUT_DIR = os.getcwd()

EST_PATH  = "q1_vote_share_estimates.csv"
FEAS_PATH = "q1_feasibility_summary.csv"
PANEL_PATH = "data_preprocessed_panel.csv"  # 若不做灵敏度复算也可不提供

# ---------- 读取 ----------
est = pd.read_csv(EST_PATH)
feas = pd.read_csv(FEAS_PATH)

# ============== A. 一致性诊断（Consistency） ==============
overall_feasible_rate = float(feas["feasible"].mean())
feasible_by_scheme = feas.groupby("scheme")["feasible"].mean().reset_index()
feasible_by_season = feas.groupby(["season", "scheme"])["feasible"].mean().reset_index()

infeasible_weeks = feas.loc[~feas["feasible"]].copy()

# 绘图：按赛季可行率
fig = plt.figure(figsize=(10, 4.6))
ax = plt.gca()
for sch, g in feasible_by_season.groupby("scheme"):
    ax.plot(g["season"], g["feasible"], marker="o", linestyle="-", label=f"{sch} 周可行率")
ax.set_title("问题一 一致性检验 各赛季周可行率")
ax.set_xlabel("赛季")
ax.set_ylabel("周可行率")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.25)
ax.legend()
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "q1_diag_feasible_by_season.png")
plt.savefig(fig_path, bbox_inches="tight")
plt.show()

# ============== B. 结构一致性与基本约束（Validity） ==============
# 点估计是否落在区间内
est["point_in_interval"] = (
    (est["vote_share_point"] >= est["vote_share_min"] - 1e-12) &
    (est["vote_share_point"] <= est["vote_share_max"] + 1e-12)
)
interval_coverage_rate = float(est["point_in_interval"].mean())

# 单纯形约束：每个 (season, week, scheme) 上 vote_share_point 是否和为 1（允许微小数值误差）
group_sum = est.groupby(["season", "week", "scheme"])["vote_share_point"].sum().reset_index(name="sum_vote_share_point")
group_sum["sum_close_1"] = (group_sum["sum_vote_share_point"] - 1.0).abs() <= 1e-6
simplex_rate = float(group_sum["sum_close_1"].mean())

# ============== C. 确定性（Certainty） ==============
width_desc = est["vote_share_width"].describe()

width_by_elim = (
    est.groupby("eliminated_after_week")["vote_share_width"]
       .agg(["mean", "median", "std", "count"])
       .reset_index()
)

# 图：宽度直方图
fig = plt.figure(figsize=(6.8, 4.6))
ax = plt.gca()
ax.hist(est["vote_share_width"].dropna().to_numpy(), bins=35)
ax.set_title("问题一 确定性诊断 投票份额区间宽度分布")
ax.set_xlabel("区间宽度")
ax.set_ylabel("频数")
ax.grid(True, alpha=0.2)
plt.tight_layout()
fig_path2 = os.path.join(OUT_DIR, "q1_diag_width_hist.png")
plt.savefig(fig_path2, bbox_inches="tight")
plt.show()

# 图：淘汰者 vs 未淘汰者 箱线图
fig = plt.figure(figsize=(6.8, 4.6))
ax = plt.gca()
est.boxplot(column="vote_share_width", by="eliminated_after_week", grid=False, ax=ax)
ax.set_title("问题一 确定性诊断 淘汰者与未淘汰者区间宽度对比")
plt.suptitle("")
ax.set_xlabel("是否在该周后被淘汰")
ax.set_ylabel("区间宽度")
plt.tight_layout()
fig_path3 = os.path.join(OUT_DIR, "q1_diag_width_box_elim.png")
plt.savefig(fig_path3, bbox_inches="tight")
plt.show()

# ============== D. 灵敏度分析（Sensitivity） ==============
# 说明：灵敏度复算需要原面板数据。为控制耗时，抽取若干 (season,week) 子集复算。
sensitivity_rows = []

def percent_constraints(judge_percent, elim_mask):
    n = len(judge_percent)
    E = np.where(elim_mask)[0]
    S = np.where(~elim_mask)[0]
    if len(E) == 0 or len(S) == 0:
        return np.zeros((0, n)), np.zeros((0,))
    A, b = [], []
    for e in E:
        for s in S:
            row = np.zeros(n); row[e] = 1.0; row[s] = -1.0
            A.append(row); b.append(judge_percent[s] - judge_percent[e])
    return np.array(A), np.array(b)

def rank_check_elimination(judge_rank, vote_share, elim_mask):
    n = len(judge_rank); k = int(elim_mask.sum())
    if k == 0: return True
    if k >= n: return False
    fan_rank = rankdata(-vote_share, method="average")
    combined = judge_rank + fan_rank
    order = np.argsort(combined)
    worst_k = set(order[-k:].tolist())
    elim_set = set(np.where(elim_mask)[0].tolist())
    if elim_set == worst_k:
        return True
    cutoff = np.sort(combined)[-k]
    worst_tied = set(np.where(combined >= cutoff - 1e-12)[0].tolist())
    return elim_set.issubset(worst_tied)

def resample_week(judge_percent, judge_rank, elim_mask, scheme, rng, N, alpha, max_trials=60000):
    n = len(elim_mask)
    if elim_mask.sum() == 0:
        x = np.ones(n) / n
        return x, np.zeros(n), np.ones(n), True
    alpha_vec = np.ones(n) * alpha
    got = 0
    x_sum = np.zeros(n)
    x_min = np.ones(n)
    x_max = np.zeros(n)
    trials = 0
    A, b = (None, None)
    if scheme == "percent":
        A, b = percent_constraints(judge_percent, elim_mask)

    while got < N and trials < max_trials:
        trials += 1
        x = rng.dirichlet(alpha_vec)
        ok = False
        if scheme == "percent":
            ok = (A.size == 0) or np.all(A.dot(x) <= b + 1e-12)
        else:
            ok = rank_check_elimination(judge_rank, x, elim_mask)
        if ok:
            got += 1
            x_sum += x
            x_min = np.minimum(x_min, x)
            x_max = np.maximum(x_max, x)

    if got == 0:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), False
    x_point = x_sum / got
    return x_point, x_min, x_max, True

if os.path.exists(PANEL_PATH):
    panel = pd.read_csv(PANEL_PATH)
    # 基础类型
    for c in ["season","week","active"]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel["season"] = panel["season"].astype(int)
    panel["week"] = panel["week"].astype(int)
    panel["active"] = panel["active"].fillna(0).astype(int)
    for c in ["last_active_week","judge_percent","judge_rank"]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

    panel = panel[panel["active"] == 1].copy()
    season_last_week = panel.groupby("season")["week"].max().to_dict()

    # 抽取子集：随机抽 25 个周次截面（保证包含 rank/percent）
    base_weeks = feas.sample(n=min(25, len(feas)), random_state=2026)[["season","week","scheme"]].to_records(index=False)

    alphas = [0.4, 0.7, 1.0, 1.5]
    Ns = [80, 160, 320]

    for (season, week, scheme) in base_weeks:
        g = panel[(panel["season"] == int(season)) & (panel["week"] == int(week))].copy()
        if len(g) <= 1:
            continue
        is_last_week = int(week) >= season_last_week[int(season)]
        elim_mask = (g["last_active_week"].to_numpy(float) == float(week)) & (~is_last_week)
        judge_percent = g["judge_percent"].to_numpy(float)
        judge_rank = g["judge_rank"].to_numpy(float)

        # 该周基准结果（来自已生成文件）
        base = est[(est["season"]==int(season)) & (est["week"]==int(week)) & (est["scheme"]==scheme)]
        if base.empty:
            continue
        base_point = base.sort_values("celebrity_name")["vote_share_point"].to_numpy()
        base_width = base.sort_values("celebrity_name")["vote_share_width"].to_numpy()

        for alpha in alphas:
            for N in Ns:
                rng = np.random.default_rng(100000 + int(season)*100 + int(week) + int(alpha*10) + N)
                x_point, x_min, x_max, ok = resample_week(judge_percent, judge_rank, elim_mask.astype(bool),
                                                          scheme, rng, N=N, alpha=alpha)
                if not ok:
                    continue
                width = x_max - x_min
                # 与基准点估计/宽度的一致性（排序一致性 + 平均绝对差）
                # 为保证对齐：按名字排序
                names = g["celebrity_name"].astype(str).tolist()
                order = np.argsort(names)
                x_point = x_point[order]
                width = width[order]
                # 基准也按同顺序（基准已按 celebrity_name 排序）
                mad_point = float(np.mean(np.abs(x_point - base_point)))
                mad_width = float(np.mean(np.abs(width - base_width)))
                rho_point = spearmanr(x_point, base_point).correlation
                rho_width = spearmanr(width, base_width).correlation
                sensitivity_rows.append({
                    "season": int(season), "week": int(week), "scheme": scheme,
                    "alpha": alpha, "N": N,
                    "mad_point": mad_point, "mad_width": mad_width,
                    "spearman_point": rho_point, "spearman_width": rho_width
                })

sensitivity_df = pd.DataFrame(sensitivity_rows)

# 图：alpha 灵敏度（若有数据）
if not sensitivity_df.empty:
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = plt.gca()
    tmp = sensitivity_df.groupby("alpha")["mad_point"].mean().reset_index()
    ax.plot(tmp["alpha"], tmp["mad_point"], marker="o", linestyle="-")
    ax.set_title("问题一 灵敏度分析 alpha 对点估计平均绝对差的影响")
    ax.set_xlabel("alpha")
    ax.set_ylabel("平均绝对差 mad_point")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig_path4 = os.path.join(OUT_DIR, "q1_diag_sensitivity_alpha.png")
    plt.savefig(fig_path4, bbox_inches="tight")
    plt.show()

# ============== E. 鲁棒性分析（Robustness） ==============
# 多随机种子复算子集周次，衡量点估计与宽度的变异（需要面板）
rob_rows = []

if os.path.exists(PANEL_PATH):
    # 选择 15 个周次做种子复算
    test_weeks = feas.sample(n=min(15, len(feas)), random_state=77)[["season","week","scheme"]].to_records(index=False)
    seeds = [11, 101, 1001, 2026]

    for (season, week, scheme) in test_weeks:
        g = panel[(panel["season"]==int(season)) & (panel["week"]==int(week))].copy()
        if len(g) <= 1:
            continue
        is_last_week = int(week) >= season_last_week[int(season)]
        elim_mask = (g["last_active_week"].to_numpy(float) == float(week)) & (~is_last_week)
        judge_percent = g["judge_percent"].to_numpy(float)
        judge_rank = g["judge_rank"].to_numpy(float)

        names = g["celebrity_name"].astype(str).tolist()
        order = np.argsort(names)

        points = []
        widths = []
        for sd in seeds:
            rng = np.random.default_rng(sd + int(season)*100 + int(week))
            N = 200 if scheme=="percent" else 90
            x_point, x_min, x_max, ok = resample_week(judge_percent, judge_rank, elim_mask.astype(bool),
                                                      scheme, rng, N=N, alpha=0.8)
            if not ok:
                continue
            x_point = x_point[order]
            width = (x_max - x_min)[order]
            points.append(x_point)
            widths.append(width)

        if len(points) >= 2:
            P = np.vstack(points)    # [nseed, ncontest]
            W = np.vstack(widths)
            # 以周次为单位：点估计的平均标准差、宽度的平均标准差
            rob_rows.append({
                "season": int(season), "week": int(week), "scheme": scheme,
                "mean_sd_point": float(np.mean(P.std(axis=0))),
                "mean_sd_width": float(np.mean(W.std(axis=0))),
                "n_seeds_used": int(len(points))
            })

rob_df = pd.DataFrame(rob_rows)

# 图：鲁棒性散点（若有）
if not rob_df.empty:
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = plt.gca()
    ax.scatter(rob_df["mean_sd_point"], rob_df["mean_sd_width"])
    ax.set_title("问题一 鲁棒性诊断 不同周次的点估计与宽度变异性")
    ax.set_xlabel("点估计平均标准差")
    ax.set_ylabel("区间宽度平均标准差")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig_path5 = os.path.join(OUT_DIR, "q1_diag_robust_seed.png")
    plt.savefig(fig_path5, bbox_inches="tight")
    plt.show()

# ============== F. 汇总输出并保存 ==============
consistency_summary = pd.DataFrame([{
    "overall_feasible_rate": overall_feasible_rate,
    "interval_coverage_rate": interval_coverage_rate,
    "simplex_sum_close_rate": simplex_rate,
    "n_infeasible_weeks": int((~feas["feasible"]).sum()),
    "n_weeks_total": int(len(feas)),
    "n_est_rows": int(len(est))
}])

certainty_summary = pd.DataFrame([{
    "width_mean": float(est["vote_share_width"].mean()),
    "width_median": float(est["vote_share_width"].median()),
    "width_std": float(est["vote_share_width"].std()),
    "width_q25": float(est["vote_share_width"].quantile(0.25)),
    "width_q75": float(est["vote_share_width"].quantile(0.75)),
}])

# 保存
cons_path = os.path.join(OUT_DIR, "q1_diag_consistency_summary.csv")
cert_path = os.path.join(OUT_DIR, "q1_diag_certainty_summary.csv")
sens_path = os.path.join(OUT_DIR, "q1_diag_sensitivity_summary.csv")
rob_path  = os.path.join(OUT_DIR, "q1_diag_robustness_summary.csv")

consistency_summary.to_csv(cons_path, index=False, encoding="utf-8-sig")
certainty_summary.to_csv(cert_path, index=False, encoding="utf-8-sig")
sensitivity_df.to_csv(sens_path, index=False, encoding="utf-8-sig")
rob_df.to_csv(rob_path, index=False, encoding="utf-8-sig")

print("==== 一致性与有效性核心指标 ====")
print(consistency_summary.to_string(index=False))
print("\n==== 确定性核心指标 ====")
print(certainty_summary.to_string(index=False))

print("\n已保存：")
for p in [cons_path, cert_path, sens_path, rob_path,
          "q1_diag_feasible_by_season.png", "q1_diag_width_hist.png", "q1_diag_width_box_elim.png"]:
    print(" -", os.path.join(OUT_DIR, p))
