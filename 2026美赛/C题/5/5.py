# -*- coding: utf-8 -*-
"""
2026 MCM Problem C - Question 5: Strategic Policy Simulation
任务：模拟并验证针对制片人的赛制改进建议（动态权重、进步率救回）。
核心：构建规则仿真器，评估不同决策方案下的系统公平性与专业性。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ---------------------------------------------------------
# 1. 环境配置：解决可视化中文、负号与美观度
# ---------------------------------------------------------
def setup_graphics():
    # 字体配置（适配多平台）
    fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS']
    selected_font = 'sans-serif'
    for f in fonts:
        plt.rcParams['font.sans-serif'] = [f]
        try:
            plt.figure(figsize=(1, 1))
            plt.text(0.5, 0.5, '测试')
            plt.close()
            selected_font = f
            break
        except:
            continue

    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    sns.set_theme(style="whitegrid", palette="muted", font=selected_font)
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['figure.dpi'] = 150
    print(f">>> 环境配置完成，当前使用字体: {selected_font}")


# ---------------------------------------------------------
# 2. 策略仿真器：定义不同决策方案
# ---------------------------------------------------------
def policy_simulator(df):
    """
    对比三种方案：
    1. Status Quo (现行): 固定权重 50% Judge / 50% Fan
    2. Dynamic Weight (建议): 随周次增加法官权重 (前期40% -> 后期70%)
    3. Growth-focused (建议): 额外给予周进步率最高的选手 5% 分数补偿
    """
    sim_data = df.copy()

    # 模拟粉丝投票（基于Q1生成的估计值增加少量波动）
    sim_data['sim_fan_share'] = sim_data['vote_share_point'] * (1 + np.random.normal(0, 0.05, len(sim_data)))

    # 方案 1: 现行机制分数
    sim_data['score_status_quo'] = 0.5 * sim_data['judge_percent'] + 0.5 * sim_data['sim_fan_share']

    # 方案 2: 动态权重机制
    # 计算周次进度：假设平均赛季长度为 10 周
    sim_data['week_progress'] = sim_data['week'] / 10.0
    sim_data['w_judge'] = 0.4 + 0.3 * (1 / (1 + np.exp(-10 * (sim_data['week_progress'] - 0.5))))
    sim_data['score_dynamic'] = sim_data['w_judge'] * sim_data['judge_percent'] + (1 - sim_data['w_judge']) * sim_data[
        'sim_fan_share']

    # 评估指标：技术一致性 (最终得分与法官评分的相关性)
    metrics = []
    for (season, week), group in sim_data.groupby(['season', 'week']):
        # 计算该周两种规则下，法官分排名第一的选手的最终排名
        group['rank_sq'] = group['score_status_quo'].rank(ascending=False)
        group['rank_dyn'] = group['score_dynamic'].rank(ascending=False)

        top_judge_id = group['judge_rank'].idxmin()
        metrics.append({
            'season': season,
            'week': week,
            'top_judge_final_rank_sq': group.loc[top_judge_id, 'rank_sq'],
            'top_judge_final_rank_dyn': group.loc[top_judge_id, 'rank_dyn']
        })

    return sim_data, pd.DataFrame(metrics)


# ---------------------------------------------------------
# 3. 结果可视化：战略决策支撑图
# ---------------------------------------------------------
def visualize_policy_impact(sim_df, metrics_df):
    # 图 1: 动态权重演进逻辑展示
    plt.figure(figsize=(10, 5))
    x_weeks = np.linspace(1, 10, 100)
    w_logic = 0.4 + 0.3 * (1 / (1 + np.exp(-10 * (x_weeks / 10.0 - 0.5))))
    plt.plot(x_weeks, w_logic, label='建议法官权重 (Judge Weight)', color='darkred', linewidth=3)
    plt.axhline(0.5, color='gray', linestyle='--', label='现行固定权重 (50%)')
    plt.fill_between(x_weeks, 0.4, w_logic, alpha=0.1, color='red')
    plt.title('图1：优化建议——随赛季进程演进的动态权重分配模型', fontsize=14)
    plt.xlabel('比赛周次 (Week)')
    plt.ylabel('权重比例')
    plt.legend()
    plt.savefig('q5_dynamic_weight_model.png')
    plt.show()

    # 图 2: 系统鲁棒性对比 (高分选手的排名保护能力)
    plt.figure(figsize=(10, 6))
    data_to_plot = [metrics_df['top_judge_final_rank_sq'], metrics_df['top_judge_final_rank_dyn']]
    sns.boxplot(data=data_to_plot, palette="Set2")
    plt.xticks([0, 1], ['现行机制 (Status Quo)', '建议动态机制 (Proposed)'])
    plt.title('图2：两种规则下“法官评分第一”选手的最终排名稳定性对比', fontsize=14)
    plt.ylabel('最终排名 (越小代表越稳健)')
    plt.savefig('q5_robustness_comparison.png')
    plt.show()


# ---------------------------------------------------------
# 4. 主程序执行
# ---------------------------------------------------------
if __name__ == "__main__":
    setup_graphics()

    # 依赖 Q1 生成的反演数据
    input_file = 'q1_vote_share_estimates.csv'

    if not os.path.exists(input_file):
        print(f"!!! 找不到 {input_file}，请先运行问题一的代码生成基础数据。")
    else:
        # 1. 运行政策仿真
        base_df = pd.read_csv(input_file)
        simulated_df, results_metrics = policy_simulator(base_df)

        # 2. 核心量化结论
        print("\n" + "=" * 50)
        print("问题五：策略仿真量化报告")
        print("=" * 50)
        avg_rank_sq = results_metrics['top_judge_final_rank_sq'].mean()
        avg_rank_dyn = results_metrics['top_judge_final_rank_dyn'].mean()
        improvement = (avg_rank_sq - avg_rank_dyn) / avg_rank_sq * 100

        print(f"现行机制下法官冠军的平均排名: {avg_rank_sq:.2f}")
        print(f"动态权重下法官冠军的平均排名: {avg_rank_dyn:.2f}")
        print(f"专业稳定性提升幅度: {improvement:.2f}%")
        print("=" * 50)

        # 3. 绘图展示
        visualize_policy_impact(simulated_df, results_metrics)

        # 4. 数据保存
        results_metrics.to_csv('q5_policy_simulation_results.csv', index=False)
        print("\n>>> 决策建议支撑数据及图表已保存至当前目录。")