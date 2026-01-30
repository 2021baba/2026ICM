# ============================================================
# 2026 MCM Problem C (Data With The Stars) —— 数据预处理（Python）
# 可直接在 PyCharm 运行：Python 3.8-3.11
#
# 目标（严格贴合赛题）：
# 1) 识别并“保留语义”：
#    - N/A：结构性缺失（该周无该评委/该季无该周），不能随意填补
#    - 0：淘汰后的占位符（选手已不再参赛），不能当作真实评分参与建模
# 2) 构建后续“反推粉丝票数”所需的周内比较量：
#    - total_judge_score（周内评委总分）
#    - judge_rank（周内总分排名，支持 rank-based 合成机制）
#    - judge_percent（周内总分占比，支持 percent-based 合成机制）
# 3) 输出“选手-赛季-周”的面板数据（tidy/panel），并生成可视化与报告
#
# 输出（保存在当前目录）：
# - data_preprocessed_panel.csv
# - preprocess_report.json
# - 多张高质量 PNG 图（缺失结构、赛季周数、评委数量分布、示例赛季趋势等）
#
# 可视化要求：
# - 解决中文显示与负号显示问题
# - 图像保存 + 也要弹窗显示（plt.show）
#
# 依赖（Anaconda 常见默认库）：
# pip install pandas numpy matplotlib
# ============================================================

import os
import re
import json
import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# 0. 全局显示与绘图设置
# ---------------------------
def setup_plotting() -> None:
    """
    解决：
    1) Matplotlib 中文字体显示问题
    2) 负号显示问题（unicode_minus）
    3) 图像更美观（统一 DPI、网格、布局）
    """
    # 尽量使用系统常见中文字体；如果没有，会自动回退到默认字体
    # Windows 常见：SimHei / Microsoft YaHei
    # macOS 常见：PingFang SC / Heiti SC
    # Linux 常见：Noto Sans CJK SC / WenQuanYi
    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "PingFang SC", "Heiti SC",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.titleweight"] = "bold"


# ---------------------------
# 1. 路径设置（学生只需要改这里）
# ---------------------------
# ========== 本地数据路径（学生需修改） ==========
# Windows 示例：data_path = r"C:\Users\XXX\Desktop\MCM\2026_MCM_Problem_C_Data.csv"
# Mac/Linux 示例：data_path = "/Users/XXX/Desktop/MCM/2026_MCM_Problem_C_Data.csv"
data_path = "2026_MCM_Problem_C_Data.csv"

# 输出目录（题目要求：保存在当前目录）
output_dir = "."


# ---------------------------
# 2. 安全读取与基础工具函数
# ---------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    """
    安全读取 CSV：
    - 自动识别常见编码
    - 明确报错与排查建议
    - 将 N/A 等识别为 NaN（结构性缺失先保留，不填补）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ 找不到数据文件：{path}\n\n"
            "排查建议：\n"
            "1) 检查 data_path 是否为正确的绝对路径或相对路径；\n"
            "2) Windows 路径建议使用 r\"C:\\\\...\\\\file.csv\" 或使用 /；\n"
            "3) 路径尽量避免中文与空格；\n"
            "4) 确保文件确实存在于该目录。"
        )

    # 题目说明中 N/A 需要识别为缺失（NaN），但不做数值填补
    na_tokens = ["N/A", "NA", "NaN", "nan", "", " ", "NULL", "null"]

    encodings_to_try = ["utf-8", "utf-8-sig", "gbk", "latin1"]
    last_error = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc, na_values=na_tokens)
            return df
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"❌ CSV 读取失败：{last_error}\n\n"
        "排查建议：\n"
        "1) 用 Excel 打开后另存为 UTF-8 CSV；\n"
        "2) 检查分隔符是否为逗号（,）；\n"
        "3) 文件是否损坏或被占用；\n"
        "4) 若文件很大可先抽样检查。"
    )


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------
# 3. 解析评分列：weekX_judgeY_score
# ---------------------------
SCORE_COL_PATTERN = re.compile(r"^week(\d+)_judge(\d+)_score$", re.IGNORECASE)


def find_score_columns(df: pd.DataFrame) -> List[str]:
    """
    找出所有 weekX_judgeY_score 形式的评分列
    """
    score_cols = []
    for c in df.columns:
        if SCORE_COL_PATTERN.match(str(c).strip()):
            score_cols.append(c)
    if not score_cols:
        raise ValueError(
            "❌ 未找到评分列 weekX_judgeY_score。\n"
            "请确认数据是否为官方 2026_MCM_Problem_C_Data.csv，或列名是否被修改。"
        )
    return sorted(score_cols)


def parse_week_judge_from_col(col: str) -> Tuple[int, int]:
    """
    将列名 weekX_judgeY_score 解析为 (week=X, judge=Y)
    """
    m = SCORE_COL_PATTERN.match(str(col).strip())
    if not m:
        raise ValueError(f"非法评分列名：{col}")
    return int(m.group(1)), int(m.group(2))


def coerce_score_columns_to_numeric(df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    """
    将评分列转为数值型：
    - N/A -> NaN
    - 非法字符串 -> NaN（保留缺失语义，后续结构性识别）
    """
    out = df.copy()
    for c in score_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ---------------------------
# 4. 结构性缺失识别：赛季周数 & 每周评委集合
# ---------------------------
def identify_season_weeks_and_judges(
    df: pd.DataFrame,
    score_cols: List[str],
    season_col: str = "season"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    输出两个表：
    1) season_week_info: 每个 (season, week) 的结构信息
       - has_any_score: 该周是否有任何评分（决定赛季真实周数）
       - n_judges_present: 该周实际存在评分的评委数量（结构性缺失：某周没有第4评委）
    2) judge_presence: 每个 (season, week, judge) 是否“存在”（存在=该 judge 在该周至少给某人打分）

    说明（贴合题面）：
    - N/A 可能来自“该周无该评委”或“该季无该周”
    - 该步骤将二者区分为“结构信息”，避免数值填补导致规则扭曲
    """
    # 构建 (week, judge) -> col 映射
    wj_map = {}
    weeks = set()
    judges = set()
    for c in score_cols:
        w, j = parse_week_judge_from_col(c)
        wj_map[(w, j)] = c
        weeks.add(w)
        judges.add(j)

    weeks = sorted(list(weeks))
    judges = sorted(list(judges))

    seasons = sorted(df[season_col].dropna().unique().astype(int).tolist())

    judge_presence_rows = []
    season_week_rows = []

    for s in seasons:
        df_s = df[df[season_col] == s]
        for w in weeks:
            # 判断该 (s,w) 是否“真实存在”（至少有一个评分非 NaN）
            # 该判断相当于：该季是否跑到这周
            cols_w = [wj_map[(w, j)] for j in judges if (w, j) in wj_map]
            sub = df_s[cols_w]
            has_any = sub.notna().any(axis=None)

            # 判断本周存在的评委数（至少对某个选手打了分）
            n_judges_present = 0
            for j in judges:
                col = wj_map.get((w, j))
                if col is None:
                    continue
                present = df_s[col].notna().any()
                judge_presence_rows.append({
                    "season": s,
                    "week": w,
                    "judge": j,
                    "judge_present": int(bool(present))
                })
                if present:
                    n_judges_present += 1

            season_week_rows.append({
                "season": s,
                "week": w,
                "has_any_score": int(bool(has_any)),
                "n_judges_present": int(n_judges_present)
            })

    judge_presence = pd.DataFrame(judge_presence_rows)
    season_week_info = pd.DataFrame(season_week_rows)

    return season_week_info, judge_presence


def compute_season_max_week(season_week_info: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 season 计算最大有效周次：
    max_week = max{week | has_any_score==1}

    贴合题面：
    - 不同赛季周数不同
    - 超过赛季长度的周次在原数据为 N/A（结构性缺失）
    """
    tmp = season_week_info[season_week_info["has_any_score"] == 1].copy()
    max_week = tmp.groupby("season")["week"].max().reset_index()
    max_week = max_week.rename(columns={"week": "season_max_week"})
    return max_week


# ---------------------------
# 5. 淘汰掩码：0 分占位 -> “参赛窗口”
# ---------------------------
def build_panel_and_elimination_mask(
    df: pd.DataFrame,
    score_cols: List[str],
    season_week_info: pd.DataFrame,
    season_max_week_df: pd.DataFrame,
    season_col: str = "season",
    name_col: str = "celebrity_name"
) -> pd.DataFrame:
    """
    将宽表转为“选手-赛季-周”面板数据，并构造淘汰掩码。

    核心逻辑（严格根据题面数据说明）：
    - N/A：不填补，表示该周/该评委“不存在”
    - 0：淘汰后占位 -> 不应参与后续周的任何建模、排序、占比计算

    输出 panel 的关键列：
    - season, week, celebrity_name
    - total_judge_score (sum of existing judges that week)
    - n_judges_present (结构信息)
    - active (是否仍在赛：total_judge_score>0 且 week<=season_max_week)
    - last_active_week（每位选手该季最后参赛周）
    """
    # 预先建立 week->judge cols 映射
    wj_map = {}
    weeks = set()
    judges = set()
    for c in score_cols:
        w, j = parse_week_judge_from_col(c)
        wj_map[(w, j)] = c
        weeks.add(w)
        judges.add(j)
    weeks = sorted(list(weeks))
    judges = sorted(list(judges))

    # 合并赛季最大周信息（结构性：该季真实存在周数）
    df2 = df.merge(season_max_week_df, on="season", how="left")

    panel_rows = []
    static_cols = [c for c in df.columns if c not in score_cols]

    for idx, row in df2.iterrows():
        season = int(row[season_col])
        celeb = row.get(name_col, None)

        # 若某赛季 max_week 缺失，说明该赛季数据异常；这里保守处理
        season_max_week = row.get("season_max_week")
        if pd.isna(season_max_week):
            season_max_week = max(weeks)

        # 取静态属性
        static_part = {c: row.get(c) for c in static_cols}

        for w in weeks:
            # 超过真实赛季长度的周：结构性不存在
            if w > int(season_max_week):
                continue

            # 汇总该周评委总分：只加“存在的评委分数”，NaN 不加（不填补）
            scores = []
            n_present = 0
            for j in judges:
                col = wj_map.get((w, j))
                if col is None:
                    continue
                v = row.get(col)
                if not pd.isna(v):
                    n_present += 1
                    scores.append(float(v))

            total = float(np.nansum(scores)) if scores else 0.0

            panel_rows.append({
                **static_part,
                "season": season,
                "week": int(w),
                "n_judges_present": int(n_present),
                "total_judge_score": total
            })

    panel = pd.DataFrame(panel_rows)

    # --- 淘汰掩码：last_active_week
    # 注意：total_judge_score==0 通常意味着淘汰后占位（题面说明）
    # 我们定义 active_week 为 total>0 的周；last_active_week 为 active_week 的最大周
    last_active = (
        panel[panel["total_judge_score"] > 0]
        .groupby(["season", name_col])["week"]
        .max()
        .reset_index()
        .rename(columns={"week": "last_active_week"})
    )

    panel = panel.merge(last_active, on=["season", name_col], how="left")

    # 若某选手所有周 total==0（极少，数据异常或字段缺失），last_active_week 为空
    # 这里保守：设为 0，后续会全部剔除
    panel["last_active_week"] = panel["last_active_week"].fillna(0).astype(int)

    # active: week <= last_active_week 且 total>0（确保淘汰后周不进入建模）
    panel["active"] = ((panel["week"] <= panel["last_active_week"]) &
                       (panel["total_judge_score"] > 0)).astype(int)

    return panel


# ---------------------------
# 6. 生成 rank / percent（直接为题目两种合成机制服务）
# ---------------------------
def add_rank_and_percent(panel: pd.DataFrame) -> pd.DataFrame:
    """
    在每个 (season, week) 的“仍在赛(active==1)”选手集合内：
    - judge_rank：total_judge_score 降序排名（1=最高）
    - judge_percent：total / sum(total) （用于 percent-based 合成机制）

    注意：这一步的“比较集合”必须是 active==1 的选手，否则会被淘汰占位 0 污染。
    """
    out = panel.copy()

    # 只在 active==1 的行上计算 rank/percent，其余设为 NaN
    out["judge_rank"] = np.nan
    out["judge_percent"] = np.nan

    # 分组计算
    for (s, w), grp in out.groupby(["season", "week"]):
        grp_active = grp[grp["active"] == 1].copy()
        if grp_active.empty:
            continue

        # rank：降序，method="min" 保证并列同名次（更保守、可解释）
        ranks = grp_active["total_judge_score"].rank(
            ascending=False, method="min"
        )

        total_sum = grp_active["total_judge_score"].sum()
        if total_sum <= 0:
            percents = pd.Series(np.nan, index=grp_active.index)
        else:
            percents = grp_active["total_judge_score"] / total_sum

        out.loc[grp_active.index, "judge_rank"] = ranks
        out.loc[grp_active.index, "judge_percent"] = percents

    return out


# ---------------------------
# 7. 静态属性清洗（保持可解释性）
# ---------------------------
def clean_static_fields(panel: pd.DataFrame) -> pd.DataFrame:
    """
    对静态字段做“可解释、保守”的清洗：
    - 字符串去除首尾空格
    - 缺失类别填为 'Unknown'
    - 年龄转数值
    """
    out = panel.copy()

    # 常见静态字段（以题面表格为准）
    cat_cols = [
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "results"
    ]
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
            # pandas 会把 NaN 转成 'nan' 字符串，这里矫正
            out.loc[out[c].isin(["nan", "NaN", "None"]), c] = np.nan
            out[c] = out[c].fillna("Unknown")

    # 年龄字段
    age_col = "celebrity_age_during_season"
    if age_col in out.columns:
        out[age_col] = pd.to_numeric(out[age_col], errors="coerce")

    # placement 也转为数值（有利于后续统计）
    if "placement" in out.columns:
        out["placement"] = pd.to_numeric(out["placement"], errors="coerce")

    return out


# ---------------------------
# 8. 可视化（保存 + 显示）
# ---------------------------
def plot_and_save(fig_name: str) -> None:
    """
    保存当前图并显示；保存到当前目录
    """
    out_path = os.path.join(output_dir, fig_name)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"✅ 图已保存：{out_path}")
    plt.show()


def visualize_missing_structure(season_week_info: pd.DataFrame) -> None:
    """
    可视化：每季每周是否存在（has_any_score）
    用热力图展示赛季长度差异（结构性缺失）
    """
    pivot = season_week_info.pivot(index="season", columns="week", values="has_any_score").fillna(0)

    plt.figure(figsize=(10, 7))
    plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
    plt.title("结构性缺失可视化：各赛季各周是否存在（1=存在, 0=不存在）")
    plt.xlabel("周次 week")
    plt.ylabel("赛季 season")
    plt.colorbar(label="是否存在")
    plt.xticks(ticks=np.arange(pivot.shape[1]), labels=pivot.columns)
    plt.yticks(ticks=np.arange(pivot.shape[0]), labels=pivot.index)
    plot_and_save("plot_structural_missing_season_week.png")


def visualize_judges_per_week(season_week_info: pd.DataFrame) -> None:
    """
    可视化：每个 (season, week) 的实际评委人数 n_judges_present
    """
    pivot = season_week_info.pivot(index="season", columns="week", values="n_judges_present").fillna(0)

    plt.figure(figsize=(10, 7))
    plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
    plt.title("每季每周实际评委人数（结构信息）")
    plt.xlabel("周次 week")
    plt.ylabel("赛季 season")
    plt.colorbar(label="评委人数")
    plt.xticks(ticks=np.arange(pivot.shape[1]), labels=pivot.columns)
    plt.yticks(ticks=np.arange(pivot.shape[0]), labels=pivot.index)
    plot_and_save("plot_judges_count_season_week.png")


def visualize_season_length(season_max_week_df: pd.DataFrame) -> None:
    """
    可视化：各赛季真实周数 season_max_week
    """
    plt.figure(figsize=(10, 4))
    plt.plot(season_max_week_df["season"], season_max_week_df["season_max_week"], marker="o")
    plt.title("各赛季真实周数（season_max_week）")
    plt.xlabel("赛季 season")
    plt.ylabel("最大有效周次")
    plot_and_save("plot_season_length.png")


def visualize_total_score_distribution(panel: pd.DataFrame) -> None:
    """
    可视化：total_judge_score 的分布（仅 active==1）
    """
    x = panel.loc[panel["active"] == 1, "total_judge_score"].dropna().values
    plt.figure(figsize=(8, 4))
    plt.hist(x, bins=40)
    plt.title("仍在赛(active==1)的评委总分分布（total_judge_score）")
    plt.xlabel("total_judge_score")
    plt.ylabel("频数")
    plot_and_save("plot_total_judge_score_hist.png")


def visualize_example_season(panel: pd.DataFrame, season: int = 1) -> None:
    """
    示例图：选取一个赛季，展示 top-3 选手在赛期间的 total_judge_score 随周变化
    （用于论文展示“预处理后可进行时序比较”）
    """
    df_s = panel[(panel["season"] == season) & (panel["active"] == 1)].copy()
    if df_s.empty:
        print(f"⚠️ season={season} 没有可用的 active 数据，跳过示例图。")
        return

    # 选取该季最终名次最好（placement 最小）或总分最高的前三名进行展示
    if "placement" in df_s.columns and df_s["placement"].notna().any():
        top_names = (
            df_s.groupby("celebrity_name")["placement"]
            .min()
            .sort_values()
            .head(3)
            .index
            .tolist()
        )
    else:
        top_names = (
            df_s.groupby("celebrity_name")["total_judge_score"]
            .sum()
            .sort_values(ascending=False)
            .head(3)
            .index
            .tolist()
        )

    plt.figure(figsize=(10, 5))
    for name in top_names:
        tmp = df_s[df_s["celebrity_name"] == name].sort_values("week")
        plt.plot(tmp["week"], tmp["total_judge_score"], marker="o", label=name)

    plt.title(f"示例赛季 Season {season}：Top-3 选手评委总分随周变化（仅 active 周）")
    plt.xlabel("周次 week")
    plt.ylabel("total_judge_score")
    plt.legend()
    plot_and_save(f"plot_example_season_{season}_top3_trend.png")


# ---------------------------
# 9. 主流程：预处理 + 输出 + 展示
# ---------------------------
def main() -> None:
    warnings.filterwarnings("ignore")
    setup_plotting()
    ensure_dir(output_dir)

    print_section("Step 1) 读取数据并识别评分列")
    df_raw = safe_read_csv(data_path)
    print(f"原始数据维度：{df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")
    print("原始数据前 5 行：")
    print(df_raw.head())

    score_cols = find_score_columns(df_raw)
    print(f"识别到评分列数量：{len(score_cols)}（示例前 8 列）")
    print(score_cols[:8])

    print_section("Step 2) 评分列转数值（N/A -> NaN，保留结构性缺失语义）")
    df = coerce_score_columns_to_numeric(df_raw, score_cols)

    # 缺失统计（评分列）
    miss_rate_scores = df[score_cols].isna().mean().sort_values(ascending=False)
    print("评分列缺失率 Top 10：")
    print(miss_rate_scores.head(10))

    print_section("Step 3) 结构性缺失识别：赛季周存在性 + 每周评委人数")
    season_week_info, judge_presence = identify_season_weeks_and_judges(df, score_cols, season_col="season")
    season_max_week_df = compute_season_max_week(season_week_info)

    print("赛季真实周数（前 10 个赛季）：")
    print(season_max_week_df.head(10))

    # 可视化：结构性缺失/评委人数/赛季长度
    visualize_missing_structure(season_week_info)
    visualize_judges_per_week(season_week_info)
    visualize_season_length(season_max_week_df)

    print_section("Step 4) 宽表转面板 + 淘汰掩码（0 分占位 -> active 窗口）")
    panel = build_panel_and_elimination_mask(
        df=df,
        score_cols=score_cols,
        season_week_info=season_week_info,
        season_max_week_df=season_max_week_df,
        season_col="season",
        name_col="celebrity_name"
    )

    # 展示淘汰逻辑的关键统计
    active_ratio = panel["active"].mean()
    print(f"面板数据维度（未加 rank/percent）：{panel.shape[0]} 行 × {panel.shape[1]} 列")
    print(f"active==1 的比例：{active_ratio:.3f}")

    # 每季 active 观测数（示例）
    print("每季 active 观测数（前 10 个赛季）：")
    print(panel[panel["active"] == 1].groupby("season").size().head(10))

    print_section("Step 5) 计算 judge_rank 与 judge_percent（周内比较，服务两种合成机制）")
    panel = add_rank_and_percent(panel)

    # 清洗静态字段（可解释性、后续影响因素模型）
    panel = clean_static_fields(panel)

    # 关键列检查
    check_cols = ["season", "week", "celebrity_name", "total_judge_score",
                  "n_judges_present", "active", "judge_rank", "judge_percent"]
    print("关键列缺失情况（仅列出 check_cols）：")
    print(panel[check_cols].isna().mean())

    print_section("Step 6) 可视化：总分分布 + 示例赛季趋势图")
    visualize_total_score_distribution(panel)
    visualize_example_season(panel, season=1)

    print_section("Step 7) 输出结果文件（当前目录） + 同时展示样例数据")
    # 输出：只保留 active==1 的观测作为后续建模的主数据集
    panel_model = panel[panel["active"] == 1].copy()

    out_csv = os.path.join(output_dir, "data_preprocessed_panel.csv")
    panel_model.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存预处理面板数据：{out_csv}")
    print(f"保存后数据维度：{panel_model.shape[0]} 行 × {panel_model.shape[1]} 列")

    # 输出报告（便于写论文“预处理说明”）
    report = {
        "raw_shape": [int(df_raw.shape[0]), int(df_raw.shape[1])],
        "panel_shape_all_weeks": [int(panel.shape[0]), int(panel.shape[1])],
        "panel_shape_active_only": [int(panel_model.shape[0]), int(panel_model.shape[1])],
        "score_columns_count": int(len(score_cols)),
        "missing_rate_scores_top10": miss_rate_scores.head(10).to_dict(),
        "season_max_week": season_max_week_df.set_index("season")["season_max_week"].to_dict(),
        "notes": {
            "NA_semantics": "N/A treated as structural missingness (not applicable). Not imputed.",
            "Zero_semantics": "0 indicates eliminated-after placeholder; excluded via active mask.",
            "Rank_percent": "Computed within (season,week) among active contestants only."
        }
    }
    report_path = os.path.join(output_dir, "preprocess_report.json")
    save_json(report, report_path)
    print(f"✅ 已保存预处理报告：{report_path}")

    # 按题目要求：显示前10行与后5行
    print("\n处理后数据（用于建模）前10行：")
    print(panel_model[check_cols].head(10))

    print("\n处理后数据（用于建模）后5行：")
    print(panel_model[check_cols].tail(5))

    print_section("完成：你已获得可直接用于 fan votes 反推与 rank/percent 对比的面板数据")
    print("建议下一步建模：")
    print("1) 在每个 (season,week) 内，根据淘汰结果建立约束，反推 fan_rank / fan_percent；")
    print("2) 用反推结果分别套入 rank-based 与 percent-based 合成机制，计算淘汰一致性指标；")
    print("3) 结合 industry/age/partner 等静态变量，建立 fan vote 的解释性模型（如分层回归/随机效应）。")


if __name__ == "__main__":
    main()