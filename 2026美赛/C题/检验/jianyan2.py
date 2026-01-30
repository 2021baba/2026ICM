# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os


def setup_graphics():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号
    sns.set_theme(style="whitegrid", font='SimHei')


# 修改函数名，避免 pytest 干扰
def perform_robustness_analysis(data, noise_level=0.05):
    """
    通过对法官评分施加扰动，测试模型输出的稳定性 [cite: 6]
    """
    print(f"\n>>> 启动鲁棒性检验 (噪声水平: {noise_level * 100}%)")

    # 模拟数据扰动 [cite: 6]
    perturbed_data = data.copy()
    perturbed_data['judge_percent'] *= (1 + np.random.normal(0, noise_level, len(data)))

    # 计算扰动前后的参数相关性 [cite: 6]
    correlation = np.corrcoef(data['judge_percent'], perturbed_data['judge_percent'])[0, 1]
    print(f"数据扰动前后相关性: {correlation:.4f}")
    return correlation


def plot_sensitivity_curve():
    """基于回归报告参数绘制灵敏度曲线 [cite: 1, 6]"""
    scores = np.linspace(0, 0.5, 100)
    # 使用报告中的 coef 参数: const=1.7943, judge_percent=1.7568 [cite: 6]
    prob_rank = 1 / (1 + np.exp(-(1.7943 + 1.7568 * scores)))

    plt.figure(figsize=(10, 6))
    plt.plot(scores, prob_rank, 'r-', label='Percentage System (Modern)')
    plt.title('模型灵敏度检验：法官评分对生存概率的影响', fontsize=14)
    plt.xlabel('法官评分占比 (Judge Score Percentage)')
    plt.ylabel('预测生存概率 (Survival Probability)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    setup_graphics()

    # 尝试读取之前生成的数据
    data_path = 'q1_vote_share_estimates.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # 执行修正后的函数名
        perform_robustness_analysis(df)
        plot_sensitivity_curve()
    else:
        print(f"未找到数据文件 {data_path}，请先运行问题一或三的代码。")