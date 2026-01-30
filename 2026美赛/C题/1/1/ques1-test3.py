import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. 定义行业先验人气权重 (基于行业平均影响力的初始假设)
# mu: 初始人气均值, sigma: 确定性（值越小表示该行业内人气分布越集中）
PRIOR_CONFIG = {
    'Musician': {'mu': 30.0, 'sigma': 8.0},
    'Singer': {'mu': 30.0, 'sigma': 8.0},
    'Actor/Actress': {'mu': 25.0, 'sigma': 6.0},
    'Athlete': {'mu': 28.0, 'sigma': 5.0},
    'Reality TV': {'mu': 22.0, 'sigma': 10.0},
    'Model': {'mu': 20.0, 'sigma': 7.0},
    'TV Personality': {'mu': 24.0, 'sigma': 6.0},
    'Default': {'mu': 22.0, 'sigma': 8.0}
}


def bayesian_map_estimation(group):
    """
    贝叶斯最大后验估计：在给定行业先验的情况下，找到最可能的粉丝投票比例
    """
    n = len(group)
    judge_percents = group['judge_percent'].values * 100  # 转换为百分比

    # 自动识别淘汰者
    elim_mask = group['results'].str.contains('Eliminated', na=False) & \
                (group['last_active_week'] == group['week'])

    if not elim_mask.any():
        return None

    elim_idx = np.where(elim_mask)[0][0]

    # 提取并归一化先验
    industries = group['celebrity_industry'].tolist()
    mus = np.array([PRIOR_CONFIG.get(ind, PRIOR_CONFIG['Default'])['mu'] for ind in industries])
    sigmas = np.array([PRIOR_CONFIG.get(ind, PRIOR_CONFIG['Default'])['sigma'] for ind in industries])

    # 归一化 mu 使其总和为 100
    mus = (mus / mus.sum()) * 100

    # 目标函数：负对数后验 (我们要最小化它)
    # Loss = Sum( (Fan_i - Prior_mu_i)^2 / (2 * sigma_i^2) )
    def objective(f_votes):
        return np.sum(((f_votes - mus) ** 2) / (2 * sigmas ** 2))

    # 约束条件 1: 总和等于 100
    cons = [{'type': 'eq', 'fun': lambda f: np.sum(f) - 100}]

    # 约束条件 2: 淘汰者总分最低 (Judge + Fan)
    # 对于所有幸存者 i: (Judge_e + Fan_e) <= (Judge_i + Fan_i)
    for i in range(n):
        if i == elim_idx: continue
        # 闭包捕获索引 i
        cons.append({
            'type': 'ineq',
            'fun': lambda f, idx=i: (judge_percents[idx] + f[idx]) - (judge_percents[elim_idx] + f[elim_idx])
        })

    # 变量边界: 粉丝票在 0-100 之间
    bounds = [(0, 100) for _ in range(n)]

    # 求解
    res = minimize(objective, x0=mus, method='SLSQP', bounds=bounds, constraints=cons)

    if res.success:
        return res.x
    return None


# 2. 加载数据
df = pd.read_csv("data_preprocessed_panel.csv")
results_list = []

# 3. 遍历所有赛季和周
unique_times = df[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])

print("开始贝叶斯 MAP 估算...")

for _, row in unique_times.iterrows():
    s, w = row['season'], row['week']
    group = df[(df['season'] == s) & (df['week'] == w)].copy()

    if len(group) <= 1: continue

    est_fans = bayesian_map_estimation(group)

    if est_fans is not None:
        for i, (idx, r) in enumerate(group.iterrows()):
            results_list.append({
                'season': s,
                'week': w,
                'celebrity_name': r['celebrity_name'],
                'industry': r['celebrity_industry'],
                'judge_percent': r['judge_percent'] * 100,
                'est_fan_percent': est_fans[i],
                'is_eliminated': (i == np.where(group['results'].str.contains('Eliminated', na=False) & \
                                                (group['last_active_week'] == w))[0][0])
            })

# 4. 保存与展示
df_output = pd.DataFrame(results_list)
df_output.to_csv("Task1_Bayesian_FanVotes.csv", index=False)
print(f"处理完成，结果已保存至 Task1_Bayesian_FanVotes.csv。")

# 5. 展示一个有趣的分析：行业平均粉丝溢出值 (Fan % - Judge %)
df_output['popularity_bias'] = df_output['est_fan_percent'] - df_output['judge_percent']
industry_bias = df_output.groupby('industry')['popularity_bias'].mean().sort_values()

plt.figure(figsize=(10, 6))
industry_bias.plot(kind='barh', color='salmon')
plt.title("Popularity Bias by Industry (Fan% minus Judge%)")
plt.xlabel("Average Bias (%)")
plt.tight_layout()
plt.show()