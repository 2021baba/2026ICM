# -*- coding: utf-8 -*-
"""
2026 MCM Problem C - Problem 3: Structural Impact Analysis
分析规则修改（Rank vs Percentage & Judge's Save）对竞赛生态的影响。
核心：构建 Logit 回归模型评估法官分对存活概率的边际贡献。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
import os


# ---------------------------------------------------------
# 1. 环境配置：解决可视化中文、负号与美观度
# ---------------------------------------------------------
def setup_graphics():
    # 自动搜索系统中可用的中文字体
    available_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS']
    selected_font = 'sans-serif'
    for f in available_fonts:
        plt.rcParams['font.sans-serif'] = [f]
        try:
            plt.figure(figsize=(1, 1))
            plt.text(0.5, 0.5, '测试')
            plt.close()
            selected_font = f
            break
        except:
            continue

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示
    sns.set_theme(style="whitegrid", font=selected_font)
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['figure.dpi'] = 150
    print(f">>> 环境配置完成，使用字体: {selected_font}")


# ---------------------------------------------------------
# 2. 数据集成与结构化处理
# ---------------------------------------------------------
def prepare_causal_data():
    # 读取第一问生成的反演结果
    if not os.path.exists('q1_vote_share_estimates.csv'):
        raise FileNotFoundError("错误：需要 q1_vote_share_estimates.csv。请确保已完成第一问数据生成。")

    df = pd.read_csv('q1_vote_share_estimates.csv')

    # 定义因变量：是否在该周生存 (未淘汰 = 1, 已淘汰 = 0)
    df['is_survived'] = (~df['eliminated_after_week']).astype(int)

    # 定义规则指示变量 (假设以 Season 10 为界，DWTS 开始深度引入 Percentage 与 Save 机制)
    # 注：实际界限可根据赛题描述调整，此处假设早期为 Rank 统治期，后期为平衡期
    df['is_modern_rule'] = (df['season'] >= 18).astype(int)

    return df


# ---------------------------------------------------------
# 3. 建模分析：结构性变动 Logit 回归
# ---------------------------------------------------------
def run_structural_analysis(df):
    """
    通过交互项回归分析规则变动对法官分影响力的改变
    Logit(P) = beta0 + beta1*JudgeScore + beta2*RuleShift + beta3*(JudgeScore * RuleShift)
    """
    # 提取特征：法官评分百分比 (标准化)
    X = df[['judge_percent', 'is_modern_rule']]
    # 构建交互项：观察规则变动对法官分权重的调节效应
    X['interaction'] = X['judge_percent'] * X['is_modern_rule']
    X = sm.add_constant(X)

    y = df['is_survived']

    model = sm.Logit(y, X).fit(disp=False)
    print("\n>>> 结构性变动回归分析摘要：")
    print(model.summary())

    return model


# ---------------------------------------------------------
# 4. 可视化呈现：因果效应的可视化证据
# ---------------------------------------------------------
def plot_rule_impact(df, model):
    # 图 1: 法官分 vs 生存概率的逻辑回归曲线对比
    plt.figure(figsize=(10, 6))

    # 模拟数据绘制回归平滑曲线
    judge_range = np.linspace(df['judge_percent'].min(), df['judge_percent'].max(), 100)

    # 早期规则预测 (is_modern_rule = 0)
    pred_old = 1 / (1 + np.exp(-(model.params['const'] + model.params['judge_percent'] * judge_range)))
    # 现代规则预测 (is_modern_rule = 1)
    pred_new = 1 / (1 + np.exp(-(model.params['const'] + model.params['is_modern_rule'] +
                                 (model.params['judge_percent'] + model.params['interaction']) * judge_range)))

    plt.plot(judge_range, pred_old, label='Rank 制度时期 (早期)', color='gray', linestyle='--', linewidth=2)
    plt.plot(judge_range, pred_new, label='Percentage 制度时期 (后期)', color='#d62728', linewidth=3)

    plt.title('图1：规则变动前后法官评分对存活概率的影响（逻辑回归映射）', fontsize=14)
    plt.xlabel('法官评分占比 (Judge Score Percentage)', fontsize=12)
    plt.ylabel('预测生存概率 (Predicted Probability of Survival)', fontsize=12)
    plt.legend()
    plt.savefig('q3_logistic_impact.png')
    plt.show()

    # 图 2: 赛季维度的“异常淘汰（Upsets）”频率分析
    # 定义“异常淘汰”：法官分排名前 30% 但被淘汰
    df['is_upset'] = (df['judge_rank'] <= (df.groupby(['season', 'week'])['celebrity_name'].transform('count') * 0.3)) & \
                     (df['eliminated_after_week'] == True)

    upset_trend = df.groupby('season')['is_upset'].mean().reset_index()

    plt.figure(figsize=(12, 5))
    sns.regplot(data=upset_trend, x='season', y='is_upset', lowess=True,
                line_kws={'color': 'red', 'label': '平滑趋势线'}, scatter_kws={'alpha': 0.6})
    plt.axvline(x=18, color='green', linestyle=':', label='重大规则切换点')
    plt.title('图2：跨赛季“技术型选手”意外淘汰频率演变', fontsize=14)
    plt.xlabel('赛季 (Season)', fontsize=12)
    plt.ylabel('意外淘汰率 (Upset Rate)', fontsize=12)
    plt.legend()
    plt.savefig('q3_upset_trend.png')
    plt.show()


# ---------------------------------------------------------
# 5. 主程序运行
# ---------------------------------------------------------
if __name__ == "__main__":
    setup_graphics()

    try:
        # 1. 准备数据
        analysis_df = prepare_causal_data()

        # 2. 执行回归建模
        logit_model = run_structural_analysis(analysis_df)

        # 3. 结果分析与可视化
        plot_rule_impact(analysis_df, logit_model)

        # 4. 统计指标保存
        with open('q3_model_analysis.txt', 'w', encoding='utf-8') as f:
            f.write("Problem 3 Structural Analysis Report\n")
            f.write("=" * 40 + "\n")
            f.write(str(logit_model.summary()))

        print("\n>>> 分析完成！")
        print(">>> 关键证据已生成：")
        print("    - q3_logistic_impact.png (展示了规则如何保护高分选手)")
        print("    - q3_upset_trend.png (展示了系统稳定性的历史演变)")
        print("    - q3_model_analysis.txt (包含回归系数的统计显著性证据)")

    except Exception as e:
        print(f"运行时发生错误: {e}")