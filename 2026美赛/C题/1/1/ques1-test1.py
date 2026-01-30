import numpy as np
import pandas as pd
from scipy.optimize import linprog


def analyze_all_weeks(csv_path):
    df = pd.read_csv(csv_path)
    all_results = []

    # 获取数据中所有的赛季和周的组合
    # 按照赛季和周排序，确保分析顺序
    time_points = df[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])

    print(f"开始处理数据，共检测到 {len(time_points)} 个比赛周...")

    for _, row in time_points.iterrows():
        s, w = row['season'], row['week']

        # 1. 筛选当周数据
        current_data = df[(df['season'] == s) & (df['week'] == w)].copy()
        if len(current_data) <= 1: continue  # 只有一个人没法比

        # 2. 提取信息
        contestants = current_data['celebrity_name'].tolist()
        j_percents = (current_data['judge_percent'] * 100).values

        # 识别被淘汰者 (同前逻辑)
        elim_mask = current_data['results'].str.contains('Eliminated', na=False) & \
                    (current_data['last_active_week'] == w)

        if not elim_mask.any():
            continue  # 如果这周没人淘汰，线性规划缺乏“上限”约束，跳过或特殊处理

        e_idx = np.where(elim_mask)[0][0]

        # 3. 线性规划核心逻辑
        num_c = len(j_percents)
        A_ub, b_ub = [], []
        epsilon = 0.01

        # 核心约束：淘汰者的总分(J+F)必须小于其他所有人
        for i in range(num_c):
            if i == e_idx: continue
            r = np.zeros(num_c)
            r[e_idx], r[i] = 1, -1
            A_ub.append(r)
            b_ub.append(j_percents[i] - j_percents[e_idx] - epsilon)

        A_eq, b_eq = [np.ones(num_c)], [100]
        bounds = [(0, 100) for _ in range(num_c)]

        # 4. 计算每个人的区间
        for i in range(num_c):
            res_min = linprog(c=[1 if k == i else 0 for k in range(num_c)],
                              A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            res_max = linprog(c=[-1 if k == i else 0 for k in range(num_c)],
                              A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

            if res_min.success and res_max.success:
                f_min, f_max = res_min.x[i], -res_max.fun
                # 记录结果
                all_results.append({
                    'Season': s,
                    'Week': w,
                    'Name': contestants[i],
                    'Judge_Percent': j_percents[i],
                    'Fan_Min': f_min,
                    'Fan_Max': f_max,
                    'Fan_Est': (f_min + f_max) / 2,
                    'Is_Eliminated': (i == e_idx)
                })

    # 5. 转换为结果表格
    results_df = pd.DataFrame(all_results)
    return results_df


# 执行全局分析
final_results = analyze_all_weeks('data_preprocessed_panel.csv')

# 查看前几行
print("\n--- 分析完成（前10条结果） ---")
print(final_results.head(10))

# 保存到本地
final_results.to_csv('fan_vote_estimates_all.csv', index=False)
print("\n结果已保存至: fan_vote_estimates_all.csv")