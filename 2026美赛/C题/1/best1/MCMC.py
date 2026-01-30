# -*- coding: utf-8 -*-
"""
2026 MCM Problem C - Question 1 & 2
Academic Production Version: Bayesian MCMC with Regularization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ==========================================
# 1. 环境配置 (适配美赛英文要求，解决字体报错)
# ==========================================
def setup_env():
    # 优先使用英文，彻底避免中文字体缺失导致的 UserWarning 和方块图
    sns.set_theme(style="whitegrid")
    plt.rcParams['axes.unicode_minus'] = False
    print(">>> Environment configured. Visualization labels set to English.")


# ==========================================
# 2. 核心 MCMC 引擎 (Metropolis-Hastings)
# ==========================================
def run_mcmc_final(group, s, w):
    rng = np.random.default_rng(seed=int(2026 * s + w))
    n = len(group)
    judge = group['judge_percent'].to_numpy(dtype=float)
    elim_mask = group['elim_this_week'].values
    elim_indices = np.where(elim_mask)[0]
    num_elim = len(elim_indices)

    # 过滤决赛周或无淘汰信息的周（作为控制变量）
    if num_elim == 0 or group['is_final_week'].iloc[0]:
        return np.full(n, 1.0 / n), np.full(n, np.nan)

    # 模型参数：L2正则化强度与K值权重因子
    lam = 5.0
    like_weight = 1.0 if num_elim > 1 else 0.5  # 减少单一淘汰约束的虚假确定性

    def get_log_posterior(votes):
        scores = 0.5 * judge + 0.5 * votes
        # 1. 第K位 Gap 连续梯度
        threshold = np.partition(scores, num_elim - 1)[num_elim - 1]
        gap = np.max(scores[elim_indices]) - threshold

        # 2. 离散命中项
        rank_indices = np.argsort(scores)
        hits = len(set(elim_indices).intersection(set(rank_indices[:num_elim])))
        miss = num_elim - hits

        # 3. 似然 (Likelihood) + 弱先验 (L2 Regularization)
        log_like = like_weight * (35.0 * hits - 70.0 * miss - 450.0 * max(gap, 0))
        log_prior = -lam * np.sum((votes - 1.0 / n) ** 2)

        return log_like + log_prior

    # Dirichlet 空间采样
    n_iter, burn_in = 5000, 2000
    current_votes = np.ones(n) / n
    current_lp = get_log_posterior(current_votes)
    samples = []

    for _ in range(n_iter):
        alpha = current_votes * 300.0 + 1e-3
        proposal = rng.dirichlet(alpha)
        proposal_lp = get_log_posterior(proposal)

        if np.log(rng.random()) < (proposal_lp - current_lp):
            current_votes = proposal
            current_lp = proposal_lp
        samples.append(current_votes)

    samples_arr = np.array(samples[burn_in:])
    return samples_arr.mean(axis=0), (
                np.percentile(samples_arr, 97.5, axis=0) - np.percentile(samples_arr, 2.5, axis=0))


# ==========================================
# 3. 数据管线
# ==========================================
def execute_pipeline():
    if not os.path.exists('data_preprocessed_panel.csv'):
        raise FileNotFoundError("Input file 'data_preprocessed_panel.csv' not found.")

    df = pd.read_csv('data_preprocessed_panel.csv')
    df = df.sort_values(['season', 'week', 'celebrity_name']).copy()

    # 预处理淘汰逻辑
    df['end_this_week'] = (df['week'] == df['last_active_week'])
    df['elim_this_week'] = df['end_this_week'] & df['results'].str.startswith('Eliminated', na=False)
    df['is_final_week'] = (df['week'] == df.groupby('season')['week'].transform('max'))

    results_list = []
    print(">>> Starting MCMC Inversion...")

    for (s, w), group in df.groupby(['season', 'week']):
        group = group.sort_values('celebrity_name').reset_index(drop=True)
        means, uncertainties = run_mcmc_final(group, s, w)

        k = group['elim_this_week'].sum()
        for i, row in enumerate(group.itertuples()):
            results_list.append({
                'season': s, 'week': w, 'name': row.celebrity_name,
                'industry': row.celebrity_industry,
                'est_fan_share': means[i],
                'uncertainty': uncertainties[i],
                'num_elim': k,
                'is_eliminated': row.elim_this_week,
                'is_valid_signal': (not row.is_final_week) and (k > 0) and (uncertainties[i] > 1e-4)
            })

    res_df = pd.DataFrame(results_list)
    res_df.to_csv('mcmc_results_for_paper.csv', index=False)
    return res_df


# ==========================================
# 4. 可视化函数
# ==========================================
def plot_academic_figures(res_df):
    valid_data = res_df[res_df['is_valid_signal']].copy()

    # FIG 1: Industry Support
    plt.figure(figsize=(12, 6))
    order = valid_data.groupby('industry')['est_fan_share'].median().sort_values(ascending=False).index[:10]
    sns.boxplot(data=valid_data, x='industry', y='est_fan_share', order=order, palette='Set2')
    plt.title("Figure 1: Estimated Fan Support Distribution by Industry", fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel("Estimated Vote Share")
    plt.tight_layout()
    plt.savefig('fig1_industry_fan_share.png', dpi=300)

    # FIG 2: Uncertainty Eliminated vs Survived
    plt.figure(figsize=(10, 6))
    valid_data['Status'] = valid_data['is_eliminated'].map({True: 'Eliminated', False: 'Survived'})
    sns.kdeplot(data=valid_data, x='uncertainty', hue='Status', fill=True, palette='coolwarm')
    plt.title("Figure 2: Distribution of Uncertainty (Eliminated vs Survived)", fontsize=14)
    plt.xlabel("95% HDI Width (Uncertainty)")
    plt.tight_layout()
    plt.savefig('fig2_uncertainty_density.png', dpi=300)

    # FIG 3: K vs Uncertainty
    plt.figure(figsize=(10, 6))
    k_summary = valid_data.groupby(['season', 'week', 'num_elim'])['uncertainty'].mean().reset_index()
    sns.pointplot(data=k_summary, x='num_elim', y='uncertainty', color='black', capsize=.2)
    plt.title("Figure 3: Mean Posterior Uncertainty vs Number of Eliminations (K)", fontsize=14)
    plt.xlabel("Elimination Count (K)")
    plt.ylabel("Average Week Uncertainty")
    plt.tight_layout()
    plt.savefig('fig3_k_sensitivity.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    setup_env()
    results = execute_pipeline()
    plot_academic_figures(results)
    print(">>> Task complete. Check working directory for Figures 1-3.")