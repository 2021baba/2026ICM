# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os


def setup_env():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 适配中文显示
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font='SimHei')
    print(">>> 环境配置完成：多维度量化分析模型已就绪。")


def analyze_system_impact():
    # 1. 加载第一问生成的最佳数据集（建议使用贝叶斯或newq1版本）
    file_path = 'newq1_vote_share_estimates.csv'
    if not os.path.exists(file_path):
        print(f"错误：未找到 {file_path}")
        return

    df = pd.read_csv(file_path)

    # 2. 定义量化指标：计算“专业 vs 人气”的对冲程度
    # 定义：Upsets (逆袭) = 法官排名靠前(前3)但最终淘汰，或法官排名靠后(后3)但生存
    def identify_upsets(group):
        group = group.sort_values('judge_percent', ascending=False)
        group['judge_rank_internal'] = range(1, len(group) + 1)

        # 识别法官心目中的优胜者被粉丝票“做掉”的情况
        is_top_judge = group['judge_rank_internal'] <= 3
        is_bottom_judge = group['judge_rank_internal'] > (len(group) - 3)

        # 逆袭定义
        group['is_upset'] = ((is_top_judge & group['eliminated_after_week']) |
                             (is_bottom_judge & ~group['eliminated_after_week']))
        return group

    df = df.groupby(['season', 'week']).apply(identify_upsets)

    # 3. 按制度（Scheme）进行分组对比
    analysis = df.groupby('scheme').agg({
        'judge_percent': ['std', 'mean'],
        'vote_share_point': 'std',
        'is_upset': 'mean'  # 逆袭率
    }).reset_index()

    analysis.columns = ['Scheme', 'Judge_Dispersion', 'Judge_Avg', 'Fan_Dispersion', 'Upset_Rate']

    print("\n--- 两种制度量化指标对比 ---")
    print(analysis)

    # 4. 统计显著性检验 (Mann-Whitney U test)
    rank_upsets = df[df['scheme'] == 'rank']['is_upset']
    perc_upsets = df[df['scheme'] == 'percentage']['is_upset']
    u_stat, p_val = mannwhitneyu(rank_upsets, perc_upsets)
    print(f"\n显著性检验 (P-value): {p_val:.4f}")

    return df, analysis


def plot_enhanced_results(df, analysis):
    # 图1：逆袭率对比 (反映系统公平性)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=analysis, x='Scheme', y='Upset_Rate', palette='viridis')
    plt.title('图1：不同计分体制下的“意外淘汰/逆袭”概率对比', fontsize=14)
    plt.ylabel('逆袭率 (Upset Rate)')
    plt.savefig('q2_upset_comparison.png')

    # 图2：法官分与粉丝分的影响力分布
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df[df['scheme'] == 'rank'], x='vote_share_point', label='Rank System', fill=True)
    sns.kdeplot(data=df[df['scheme'] == 'percentage'], x='vote_share_point', label='Percentage System', fill=True)
    plt.title('图2：粉丝投票份额在不同制度下的分布密度', fontsize=14)
    plt.xlabel('粉丝投票份额 (Fan Vote Share)')
    plt.legend()
    plt.savefig('q2_distribution_density.png')
    plt.show()


if __name__ == "__main__":
    setup_env()
    processed_df, summary = analyze_system_impact()
    plot_enhanced_results(processed_df, summary)