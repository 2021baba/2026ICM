import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import logit
import os
import warnings

# ==========================================
# 1. 环境配置与学术规范化
# ==========================================
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei', context='paper')

OUTPUT_DIR = "final_paper_results"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# 特征美化映射表
LABEL_MAP = {
    'age_std': '年龄 (标准化)',
    'week': '比赛周次 (趋势)',
    'Athlete': '行业: 运动员',
    'Singer/Rapper': '行业: 歌手',
    'Actor/Actress': '行业: 演员',
    'Reality TV star': '行业: 真人秀明星',
    'Comedian': '行业: 喜剧演员',
    'Other': '行业: 其他'
}


def clean_name(s):
    return "".join(filter(str.isalnum, str(s).lower())) if pd.notna(s) else s


# ==========================================
# 2. 增强型数据清洗流程
# ==========================================
def load_and_preprocess():
    print(">>> 步骤1: 增强型数据清洗与防御性转换...")
    df_p = pd.read_csv('data_preprocessed_panel.csv')
    df_f = pd.read_csv('mcmc_results_for_paper.csv')

    # 强制类型转换与名称清洗
    df_p['clean_name'] = df_p['celebrity_name'].apply(clean_name)
    df_f['clean_name'] = df_f['name'].apply(clean_name)
    df_f['week'] = pd.to_numeric(df_f['week'], errors='coerce')
    df_f['est_fan_share'] = pd.to_numeric(df_f['est_fan_share'], errors='coerce')

    df_f = df_f.drop_duplicates(subset=['season', 'week', 'clean_name'])

    df = pd.merge(df_p, df_f[['season', 'week', 'clean_name', 'est_fan_share']],
                  on=['season', 'week', 'clean_name'], how='left')

    # 构造因变量：处理缺失与无穷值
    df['judge_z'] = df.groupby('season')['judge_percent'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    df['fan_logit'] = logit(df['est_fan_share'].clip(0.005, 0.995))

    # 过滤非有限值 (NaN, inf)
    df = df[np.isfinite(df['judge_z']) & np.isfinite(df['fan_logit'])]

    # 特征工程
    df['age_std'] = (df['celebrity_age_during_season'] - df['celebrity_age_during_season'].mean()) / df[
        'celebrity_age_during_season'].std()
    top_ind = df['celebrity_industry'].value_counts().nlargest(5).index
    df['industry_clean'] = df['celebrity_industry'].apply(lambda x: x if x in top_ind else 'Other')
    df['week'] = pd.to_numeric(df['week'], errors='coerce')

    return df.dropna(subset=['age_std', 'ballroom_partner', 'industry_clean', 'week'])


# ==========================================
# 3. 稳健建模与自动回退机制
# ==========================================
def fit_model_robust(df, target, formula, cluster_col):
    print(f"--- 拟合模型: {target} ---")
    try:
        # 尝试 MixedLM, 使用单一字符串优化器提高兼容性
        model = smf.mixedlm(f"{target} {formula}", df, groups=df[cluster_col])
        res = model.fit(method="bfgs", maxiter=500)
        # 检查奇异性
        _ = res.random_effects
        return res, "LMM", res.cov_re.iloc[0, 0] / (res.cov_re.iloc[0, 0] + res.scale)
    except Exception as e:
        print(f"警告: LMM 失败 ({str(e)[:40]}), 切换至 Cluster-Robust OLS")
        ols = smf.ols(f"{target} {formula}", data=df)
        res_ols = ols.fit(cov_type='cluster', cov_kwds={'groups': df[cluster_col]})
        # 计算伪ICC (基于残差方差分解)
        res_var = res_ols.resid.var()
        group_var = df.assign(r=res_ols.resid).groupby(cluster_col)['r'].mean().var()
        return res_ols, "OLS_Robust", group_var / (group_var + res_var)


# ==========================================
# 4. 主运行程序与可视化
# ==========================================
def main():
    df_final = load_and_preprocess()
    formula = " ~ age_std + C(industry_clean) + week + C(season)"  # 加入赛季固定效应

    res_j, type_j, icc_j = fit_model_robust(df_final, "judge_z", formula, "ballroom_partner")
    res_f, type_f, icc_f = fit_model_robust(df_final, "fan_logit", formula, "ballroom_partner")

    # 5. 提取系数 (排除截距与赛季固定效应)
    def tidy(res, label):
        d = pd.DataFrame({'Coef': res.params, 'P': res.pvalues,
                          'Lower': res.conf_int()[0], 'Upper': res.conf_int()[1]}).reset_index()
        d.columns = ['Feature', 'Coef', 'P', 'Lower', 'Upper']
        # 仅保留核心特征
        d = d[d['Feature'].str.contains('age_std|industry_clean|week')].copy()
        d['Feature'] = d['Feature'].apply(lambda x: x.split('[T.')[-1].replace(']', '') if '[T.' in x else x)
        d['Feature'] = d['Feature'].map(LABEL_MAP).fillna(d['Feature'])
        d['Model'] = label
        d['Sig'] = d['P'].apply(lambda x: '***' if x < 0.01 else ('**' if x < 0.05 else ''))
        return d

    plot_df = pd.concat([tidy(res_j, "评委评分 (技术)"), tidy(res_f, "粉丝投票 (偏好)")])
    plot_df.to_csv(f"{OUTPUT_DIR}/regression_summary.csv", index=False)

    # 6. 图表 A: 学术级森林图
    plt.figure(figsize=(10, 8))
    colors = ['#D62728', '#1F77B4']  # 红蓝经典色
    for i, model in enumerate(plot_df['Model'].unique()):
        data = plot_df[plot_df['Model'] == model]
        y_coords = np.arange(len(data)) + (0.2 if i == 0 else -0.2)
        plt.errorbar(data['Coef'], y_coords, xerr=[data['Coef'] - data['Lower'], data['Upper'] - data['Coef']],
                     fmt='o', color=colors[i], label=model, capsize=5, lw=2, markersize=8)
        # 添加显著性星号
        for y, sig, coef in zip(y_coords, data['Sig'], data['Coef']):
            plt.text(coef, y + 0.1, sig, ha='center', color=colors[i], fontweight='bold')

    plt.yticks(np.arange(len(plot_df['Feature'].unique())), plot_df['Feature'].unique())
    plt.axvline(0, color='black', lw=1.5, ls='--')
    plt.title("名人特征对技术分与人气分的驱动差异分析", fontsize=15)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/forest_plot_sig.png", dpi=300)

    # 7. 图表 B: ICC 对比图 (回答“舞者影响有多大”)
    plt.figure(figsize=(6, 5))
    sns.barplot(x=['评委评分', '粉丝投票'], y=[icc_j, icc_f], palette='muted')
    plt.ylabel("组内相关系数 (ICC / 舞者解释力占比)")
    plt.title("职业舞者对不同评价维度的系统性影响强度")
    plt.ylim(0, max(icc_j, icc_f) * 1.3)
    for i, v in enumerate([icc_j, icc_f]):
        plt.text(i, v + 0.01, f"{v:.2%}", ha='center', fontweight='bold')
    plt.savefig(f"{OUTPUT_DIR}/icc_comparison.png", dpi=300)

    print(f"\n>>> 任务完成! 有效样本: {len(df_final)}")
    print(f">>> 评委端 ICC: {icc_j:.2%}, 粉丝端 ICC: {icc_f:.2%}")


if __name__ == "__main__":
    main()