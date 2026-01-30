import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 加载数据
df = pd.read_csv("data_preprocessed_panel.csv")


# 2. 核心：蒙特卡洛模拟函数
def monte_carlo_fan_votes(group, num_samples=10000):
    """
    针对每一周的数据，模拟数万种可能的粉丝投票组合，
    筛选出满足“淘汰者总分最低”条件的样本。
    """
    n = len(group)
    judge_percents = group['judge_percent'].values * 100  # 转换为百分比制

    # 判定本周谁被淘汰
    # 逻辑：结果包含 'Eliminated' 且 last_active_week 等于当前 week
    is_elim = group['results'].str.contains('Eliminated', na=False) & \
              (group['last_active_week'] == group['week'])

    elim_idx_list = np.where(is_elim)[0]
    if len(elim_idx_list) == 0:
        return None  # 如果没有淘汰数据（如决赛），跳过

    elim_idx = elim_idx_list[0]

    # 随机生成符合 Dirichlet 分布的样本 (Sum=100, 且全为正数)
    # 这比纯随机更符合“比例分配”的特性
    samples = np.random.dirichlet(np.ones(n), num_samples) * 100

    # 计算总分组合: Total = Judge_Percent + Fan_Percent
    # 筛选条件：淘汰者的总分必须是全场最低
    # 判定矩阵: samples + judge_percents
    total_scores = samples + judge_percents

    # 检查淘汰者(elim_idx)是否在每一行中都是最小值
    # 允许极小的误差 epsilon
    is_valid = np.all(total_scores[:, [elim_idx]] <= total_scores + 0.01, axis=1)

    valid_samples = samples[is_valid]

    if len(valid_samples) < 10:
        # 如果样本太少，说明评委分差距太大，随机采样难以触碰边界，返回均值
        return None

    # 计算均值作为估计值，标准差作为不确定度
    mean_estimates = np.mean(valid_samples, axis=0)
    std_estimates = np.std(valid_samples, axis=0)

    return mean_estimates, std_estimates


# 3. 遍历所有赛季和周进行计算
results_list = []
unique_times = df[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])

print(f"开始蒙特卡洛模拟，总计 {len(unique_times)} 个比赛周...")

for _, row in unique_times.iterrows():
    s, w = row['season'], row['week']
    group = df[(df['season'] == s) & (df['week'] == w)].copy()

    if len(group) <= 1: continue

    sim_res = monte_carlo_fan_votes(group)

    if sim_res:
        means, stds = sim_res
        for i, (idx, r) in enumerate(group.iterrows()):
            results_list.append({
                'season': s,
                'week': w,
                'celebrity_name': r['celebrity_name'],
                'judge_percent': r['judge_percent'] * 100,
                'est_fan_percent': means[i],
                'uncertainty': stds[i],
                'is_eliminated': (r['results'].find('Eliminated') != -1 and r['last_active_week'] == w)
            })

# 4. 结果导出
df_fan_estimates = pd.DataFrame(results_list)
df_fan_estimates.to_csv("Task1_MonteCarlo_FanVotes.csv", index=False)
print(f"估算完成！结果已保存至 Task1_MonteCarlo_FanVotes.csv，共 {len(df_fan_estimates)} 条记录。")

# 5. 可视化：示例展示一个赛季的趋势
sample_season = df_fan_estimates[df_fan_estimates['season'] == 1]
if not sample_season.empty:
    import seaborn as sns

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=sample_season, x='week', y='est_fan_percent', hue='celebrity_name', marker='o')
    plt.title("Season 1: Estimated Fan Vote Trend (Monte Carlo)")
    plt.ylabel("Estimated Fan Percentage (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()