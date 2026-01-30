# -*- coding: utf-8 -*-
"""
2026 MCM Problem C - Question 2: Ultimate Robust Counterfactual Simulator
Fixes: MC Normalization Dimension, Uncertainty Compatibility, Local JFI Copying.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr

try:
    import seaborn as sns

    HAS_SNS = True
except ImportError:
    HAS_SNS = False


# ==========================================
# 1. 环境与数据预处理
# ==========================================
def setup_analysis():
    plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 读入数据
    df = pd.read_csv('mcmc_results_for_paper.csv')

    # 初始化清洗：强制 (season, week) 归一化，确保份额逻辑严谨
    df['est_fan_share'] = df['est_fan_share'].clip(1e-6, None)
    df['est_fan_share'] = df.groupby(['season', 'week'])['est_fan_share'].transform(lambda x: x / x.sum())

    # 模拟法官分 (如果缺失)
    if 'judge_percent' not in df.columns:
        np.random.seed(42)
        df['judge_percent'] = np.random.uniform(0.05, 0.20, len(df))

    return df


# ==========================================
# 2. 核心赛季模拟逻辑 (Stable Tie-break)
# ==========================================
def simulate_path(season_df, scheme='percent', use_save=False):
    weeks = sorted(season_df['week'].unique())
    alive = set(season_df['name'].unique())
    elim_record = {}
    ranks = {}
    curr_rank = 1

    for w in weeks:
        w_data = season_df[(season_df['week'] == w) & (season_df['name'].isin(alive))].copy()
        if w_data.empty: continue

        k = int(w_data['num_elim'].iloc[0])
        if k <= 0: continue

        n = len(w_data)
        if scheme == 'percent':
            w_data['score'] = 0.5 * w_data['judge_percent'] + 0.5 * w_data['est_fan_share']
        else:
            j_p = n - w_data['judge_percent'].rank(ascending=False, method='min') + 1
            f_p = n - w_data['est_fan_share'].rank(ascending=False, method='min') + 1
            w_data['score'] = 0.5 * j_p + 0.5 * f_p

        # 稳定排序优先级：综合分 -> 法官分(升序，即低分先汰) -> 份额 -> 姓名
        w_data = w_data.sort_values(
            ['score', 'judge_percent', 'est_fan_share', 'name'],
            ascending=[True, True, True, True]
        )

        potentials = w_data.head(k + 1 if use_save else k)
        if use_save and k == 1 and len(potentials) >= 2:
            b1, b2 = potentials.iloc[0], potentials.iloc[1]
            final_elim = b1['name'] if b1['judge_percent'] <= b2['judge_percent'] else b2['name']
            eliminated = [final_elim]
        else:
            eliminated = potentials.head(k)['name'].tolist()

        elim_record[w] = eliminated
        for name in eliminated:
            ranks[name] = curr_rank
            alive.remove(name)
            curr_rank += 1

    for name in alive: ranks[name] = curr_rank
    return ranks, elim_record


# ==========================================
# 3. 反事实运行与交叉指标 (修正 3: 局部计算逻辑)
# ==========================================
def run_full_cross_val(df):
    results = []
    for s, group in df.groupby('season'):
        orig_s = 'percent' if s <= 27 else 'rank'
        cf_s = 'rank' if orig_s == 'percent' else 'percent'
        use_save = (s >= 28)

        o_ranks, o_elims = simulate_path(group, orig_s, use_save)
        c_ranks, c_elims = simulate_path(group, cf_s, use_save)

        # 准备指标计算基础数据
        latest = group.sort_values('week').groupby('name').tail(1).copy()

        # ✅ 修正 3: 局部 copy 避免 SettingWithCopy 警告与副作用
        def calc_jfi(r_dict):
            stats = latest.copy()
            stats['tmp_r'] = stats['name'].map(r_dict)
            cj, _ = spearmanr(stats['tmp_r'], stats['judge_percent'])
            cf, _ = spearmanr(stats['tmp_r'], stats['est_fan_share'])
            return np.clip(cj / (cf + 1e-6), -10, 10)

        # 交叉验证指标 (Weekly Consistency)
        weeks_u = sorted(set(o_elims.keys()) | set(c_elims.keys()))
        disagree = sum([set(o_elims.get(w, [])) != set(c_elims.get(w, [])) for w in weeks_u])

        results.append({
            'season': s, 'regime': orig_s,
            'JFI_Orig': calc_jfi(o_ranks),
            'JFI_CF': calc_jfi(c_ranks),
            'Agreement': spearmanr([o_ranks[n] for n in sorted(o_ranks.keys())],
                                   [c_ranks[n] for n in sorted(o_ranks.keys())])[0],
            'WeeklyDisagree': disagree / len(weeks_u) if weeks_u else 0.0
        })
    return pd.DataFrame(results)


# ==========================================
# 4. Monte Carlo 分析 (修正 1 & 2: 分组与兼容性)
# ==========================================
def mc_survival(df, target_name, n_sim=100):
    td = df[df['name'] == target_name]
    if td.empty: return None
    s_id = int(td['season'].iloc[0])
    season_df = df[df['season'] == s_id].copy()

    orig_s = 'percent' if s_id <= 27 else 'rank'
    cf_s = 'rank' if orig_s == 'percent' else 'percent'
    use_save = (s_id >= 28)

    cnt_o, cnt_c = 0, 0
    for _ in range(n_sim):
        noise = season_df.copy()

        # ✅ 修正 2: 兼容性字段检查与广播稳定
        if 'uncertainty' in noise.columns:
            sigma = noise['uncertainty'].fillna(0.01) / 4
        else:
            sigma = 0.01 / 4

        noise['est_fan_share'] += np.random.normal(0, sigma, size=len(noise))
        noise['est_fan_share'] = noise['est_fan_share'].clip(1e-6, None)

        # ✅ 修正 1: 归一化分组维度对齐
        noise['est_fan_share'] = noise.groupby(['season', 'week'])['est_fan_share'].transform(lambda x: x / x.sum())

        r_o, _ = simulate_path(noise, orig_s, use_save)
        r_c, _ = simulate_path(noise, cf_s, use_save)

        # 判定是否进入决赛 (前3名)
        mo, mc = max(r_o.values()), max(r_c.values())
        if r_o.get(target_name, 0) >= mo - 2: cnt_o += 1
        if r_c.get(target_name, 0) >= mc - 2: cnt_c += 1

    return cnt_o / n_sim, cnt_c / n_sim


# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    # ✅ 修正补丁: 调用一次，承接数据
    df_raw = setup_analysis()

    summary = run_full_cross_val(df_raw)

    print("\n" + "=" * 60)
    print("COUNTERFACTUAL SUMMARY: POLICY BIAS REPORT")
    print("-" * 60)
    print(summary.groupby('regime')[['JFI_Orig', 'JFI_CF', 'Agreement', 'WeeklyDisagree']].mean())
    print("=" * 60)

    # 典型个案：Bobby Bones (S27)
    bb_res = mc_survival(df_raw, "Bobby Bones", n_sim=200)
    if bb_res:
        print(f"\n[Validation] Bobby Bones Finale Probability:")
        print(f" - Actual Scheme (Percent): {bb_res[0]:.1%}")
        print(f" - Counterfactual (Rank):   {bb_res[1]:.1%}")