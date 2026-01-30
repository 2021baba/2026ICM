# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# -----------------------------
# 1. 环境配置：解决中文显示
# -----------------------------
def setup_matplotlib():
    # 尝试设置常用中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font=plt.rcParams['font.sans-serif'][0])


# -----------------------------
# 2. 核心分析函数
# -----------------------------
def run_comparison_analysis(file_path):
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    df = pd.read_csv(file_path)

    # 关键步骤：检查 scheme 列
    if 'scheme' not in df.columns:
        print("警告：数据中缺少 'scheme' 列，无法进行方案对比。")
        return

    schemes = df['scheme'].unique()
    print(f">>> 检测到以下方案: {schemes}")

    # --- 比较 1: 法官影响力 (Judge vs Final) ---
    # 我们计算法官分百分数与估算票数的相关性
    # 意义：相关性越高，说明该规则越“尊重”专业打分

    correlation_results = []
    for s in schemes:
        sub = df[df['scheme'] == s]
        # 计算每一周内，法官分与最终估计票数的相关性均值
        weekly_corr = sub.groupby(['season', 'week']).apply(
            lambda x: x['judge_percent'].corr(x['vote_share_point'])
        ).mean()

        correlation_results.append({'Scheme': s, 'Avg_Judge_Influence': weekly_corr})

    corr_df = pd.DataFrame(correlation_results)

    # --- 比较 2: 得分区分度 (Score Volatility) ---
    # 意义：波动率越高，说明规则越能拉开选手差距，减少“平局”
    volatility_df = df.groupby(['season', 'week', 'scheme'])['vote_share_point'].std().reset_index()

    # -----------------------------
    # 3. 美化可视化
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # A. 条形图：平均影响力对比
    sns.barplot(data=corr_df, x='Scheme', y='Avg_Judge_Influence', ax=axes[0], palette='viridis')
    axes[0].set_title('图1：不同计分方案下法官权重的“有效性”对比', fontsize=13)
    axes[0].set_ylabel('相关系数 (法官分 vs 最终票数)')

    # B. 小提琴图：得分区分度分布
    sns.violinplot(data=volatility_df, x='scheme', y='vote_share_point', ax=axes[1], inner="quartile")
    axes[1].set_title('图2：两种方案对选手差距的“拉开程度”分布', fontsize=13)
    axes[1].set_ylabel('得分标准差 (Volatility)')

    plt.tight_layout()
    plt.savefig('problem2_comprehensive_comparison.png', dpi=300)
    plt.show()

    # 输出结果表
    print("\n" + "=" * 40)
    print("问题二：量化对比结果表")
    print("=" * 40)
    print(corr_df.to_string(index=False))
    print("-" * 40)

    # 保存结果
    corr_df.to_csv('problem2_analysis_results.csv', index=False)


# -----------------------------
# 4. 执行
# -----------------------------
if __name__ == "__main__":
    setup_matplotlib()
    run_comparison_analysis('q1_vote_share_estimates.csv')