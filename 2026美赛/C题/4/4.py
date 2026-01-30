# -*- coding: utf-8 -*-
"""
2026 MCM Problem C - Question 4: Individual Heterogeneity & Fairness Analysis
任务：分析选手背景特征（行业、年龄、搭档）对成绩的影响，识别系统偏见。
模型：随机森林回归 (Random Forest) + 特征重要性分析 + SHAP 解释。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os


# ---------------------------------------------------------
# 1. 环境配置：解决可视化中文与负号
# ---------------------------------------------------------
def setup_graphics():
    # 尝试加载常见中文字体
    fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'PingFang SC']
    for font in fonts:
        plt.rcParams['font.sans-serif'] = [font]
        try:
            plt.figure(figsize=(1, 1))
            plt.text(0.5, 0.5, '测试')
            plt.close()
            break
        except:
            continue
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font=plt.rcParams['font.sans-serif'][0])
    print(">>> 环境配置完成。")


# ---------------------------------------------------------
# 2. 数据准备与特征工程
# ---------------------------------------------------------
def prepare_feature_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到数据文件: {file_path}")

    df = pd.read_csv(file_path)

    # 2.1 提取关键特征
    # 目标：预测最终排名 (placement)，排名越小代表成绩越好
    # 特征：年龄、行业、是否处于后期赛季(规则变动)

    # 行业编码
    le_industry = LabelEncoder()
    df['industry_code'] = le_industry.fit_transform(df['celebrity_industry'].astype(str))

    # 提取选手的聚合特征 (按每位选手在每个赛季的表现聚合)
    contestant_stats = df.groupby(['celebrity_name', 'season']).agg({
        'placement': 'first',
        'celebrity_age_during_season': 'first',
        'industry_code': 'first',
        'celebrity_industry': 'first',
        'judge_percent': 'mean',
        'active': 'sum'  # 存活周数
    }).reset_index()

    return contestant_stats, le_industry


# ---------------------------------------------------------
# 3. 建模：随机森林特征贡献度分析
# ---------------------------------------------------------
def analyze_feature_importance(df):
    # 定义特征 X 和 目标 y
    X = df[['celebrity_age_during_season', 'industry_code', 'judge_percent']]
    y = df['placement']

    # 构建随机森林模型
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)

    # 提取特征重要性
    importances = rf.feature_importances_
    feature_names = ['年龄 (Age)', '行业背景 (Industry)', '法官评分 (Judge Skill)']

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return rf, importance_df


# ---------------------------------------------------------
# 4. 可视化呈现：因果与关联分析
# ---------------------------------------------------------
def plot_q4_results(df, importance_df, le_industry):
    # 图 1: 特征重要性条形图
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='magma')
    plt.title('图1：各先验特征对最终排名影响的贡献度 (随机森林模型)', fontsize=14)
    plt.xlabel('贡献度权重 (Importance Score)')
    plt.savefig('q4_feature_importance.png', dpi=300)
    plt.show()

    # 图 2: 年龄与成绩的非线性关系 (带回归线)
    plt.figure(figsize=(12, 6))
    sns.regplot(data=df, x='celebrity_age_during_season', y='placement',
                lowess=True, scatter_kws={'alpha': 0.4, 'color': 'teal'}, line_kws={'color': 'red'})
    plt.gca().invert_yaxis()  # 排名 1 在最上方
    plt.title('图2：选手年龄与最终成绩的相关性趋势 (Lowess 平滑)', fontsize=14)
    plt.xlabel('参赛年龄 (Age during season)')
    plt.ylabel('最终排名 (Placement, 1为冠军)')
    plt.savefig('q4_age_impact.png', dpi=300)
    plt.show()

    # 图 3: 行业表现差异分析
    plt.figure(figsize=(14, 7))
    # 选取样本较多的行业进行对比
    top_industries = df['celebrity_industry'].value_counts().nlargest(8).index
    sub_df = df[df['celebrity_industry'].isin(top_industries)]

    sns.boxplot(data=sub_df, x='celebrity_industry', y='placement', palette='Set3')
    plt.gca().invert_yaxis()
    plt.xticks(rotation=45)
    plt.title('图3：不同行业背景选手的成绩分布对比 (箱线图)', fontsize=14)
    plt.ylabel('最终排名 (Placement)')
    plt.xlabel('行业 (Industry)')
    plt.savefig('q4_industry_comparison.png', dpi=300)
    plt.show()


# ---------------------------------------------------------
# 5. 主程序执行
# ---------------------------------------------------------
if __name__ == "__main__":
    setup_graphics()

    data_path = 'data_preprocessed_panel.csv'

    try:
        # 1. 准备数据
        c_stats, encoder = prepare_feature_data(data_path)

        # 2. 建模分析
        model, importance = analyze_feature_importance(c_stats)

        print("\n>>> 特征重要性分析结果：")
        print(importance)

        # 3. 可视化
        plot_q4_results(c_stats, importance, encoder)

        # 4. 保存统计结果
        importance.to_csv('q4_feature_importance_results.csv', index=False)

        print("\n>>> 分析完成！")
        print(">>> 关键证据文件已生成：")
        print("    - q4_feature_importance.png (特征权重图)")
        print("    - q4_age_impact.png (年龄与成绩趋势图)")
        print("    - q4_industry_comparison.png (行业表现对比图)")
        print("    - q4_feature_importance_results.csv (量化指标表)")

    except Exception as e:
        print(f"运行时发生错误: {e}")