# -*- coding: utf-8 -*-
"""
2026 MCM Problem C: Integrated Simulation System (Fixed & Academic English Edition)
Features: Monte Carlo Robustness, JFI Bias Calculation, Policy Regime Comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import difflib
import os

# ==========================================
# 0. Environment Configuration
# ==========================================
# Using standard sans-serif for better academic compatibility
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

output_dir = "./results_fixed/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# ==========================================
# 1. Function: Data Preprocessing
# ==========================================
def load_and_preprocess(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Simulate Professional Scores (Judge Percent) based on valid signal logic
    if 'judge_percent' not in df.columns:
        np.random.seed(42)
        df['judge_percent'] = np.where(df['is_valid_signal'] == True,
                                       np.random.uniform(0.12, 0.22, len(df)),
                                       np.random.uniform(0.04, 0.14, len(df)))

    df['uncertainty'] = df['uncertainty'].fillna(0.01)
    return df


# ==========================================
# 2. Function: Core Simulation Engine (Fixed Logic)
# ==========================================
def simulate_season_path_fixed(season_df, scheme='percent', use_save=False):
    """
    Simulates elimination path.
    Returns ranks where 1 = first eliminated, total_n = winner.
    """
    weeks = sorted(season_df['week'].unique())
    names = season_df['name'].unique()
    alive = set(names)
    total_n = len(names)

    ranks = {}
    curr_rank_pos = 1
    last_performance_scores = {n: 0.0 for n in names}

    for w in weeks:
        w_data = season_df[(season_df['week'] == w) & (season_df['name'].isin(alive))].copy()
        if w_data.empty: continue

        n_curr = len(w_data)
        k = min(int(w_data['num_elim'].max()), n_curr)

        # Scoring Logic (Higher Score = Better Performance)
        if scheme == 'percent':
            w_data['score'] = 0.5 * w_data['judge_percent'] + 0.5 * w_data['est_fan_share']
        else:
            # Rank-based: Higher points for higher percentiles
            j_pts = w_data['judge_percent'].rank(method='min', ascending=True)
            f_pts = w_data['est_fan_share'].rank(method='min', ascending=True)
            w_data['score'] = 0.5 * j_pts + 0.5 * f_pts

        for _, row in w_data.iterrows():
            last_performance_scores[row['name']] = row['score']

        # Sort: Worst performance (lowest score) at the top for elimination
        w_data = w_data.sort_values(by=['score', 'judge_percent'], ascending=True)

        if k > 0:
            if use_save and k == 1 and n_curr >= 2:
                # Judges' Save Logic: Save the one with higher professional merit in Bottom 2
                bottom_2 = w_data.head(2)
                elim_name = bottom_2.sort_values('judge_percent').iloc[0]['name']
                eliminated = [elim_name]
            else:
                eliminated = w_data.head(k)['name'].tolist()

            for name in eliminated:
                ranks[name] = curr_rank_pos
                alive.discard(name)
                curr_rank_pos += 1

    # Survivors (Finalists) ranked by their final round performance
    survivors = sorted(list(alive), key=lambda x: last_performance_scores[x])
    for s in survivors:
        ranks[s] = curr_rank_pos
        curr_rank_pos += 1

    return ranks


# ==========================================
# 3. Main Execution Logic
# ==========================================
if __name__ == "__main__":
    # A. Load Data
    raw_df = load_and_preprocess('mcmc_results_for_paper.csv')

    # B. Macro-Level Analysis (JFI Bias)
    seasons = raw_df['season'].unique()
    macro_list = []

    # C. Monte Carlo Case Studies
    target_cases = [("Jerry Rice", 2), ("Billy Ray Cyrus", 4), ("Bristol Palin", 11), ("Bobby Bones", 27)]
    case_results = []

    print("Executing Comprehensive Simulation and Policy Impact Analysis...")

    for s_id in seasons:
        s_df = raw_df[raw_df['season'] == s_id].copy()

        # Baseline Simulations
        r_p = simulate_season_path_fixed(s_df, 'percent')
        r_r = simulate_season_path_fixed(s_df, 'rank')

        names = sorted(r_p.keys())
        stats = s_df.groupby('name')[['judge_percent', 'est_fan_share']].mean().reindex(names)

        # Performance vectors (Higher value = Higher rank)
        perf_p = [r_p[n] for n in names]
        perf_r = [r_r[n] for n in names]

        # Calculate Judicial Bias Index (JBI): Corr(Perf, Judge) - Corr(Perf, Fan)
        bias_p = spearmanr(perf_p, stats['judge_percent'])[0] - spearmanr(perf_p, stats['est_fan_share'])[0]
        bias_r = spearmanr(perf_r, stats['judge_percent'])[0] - spearmanr(perf_r, stats['est_fan_share'])[0]

        macro_list.append({'Season': s_id, 'Bias_Percent': bias_p, 'Bias_Rank': bias_r})

        # Process Controversial Cases
        for c_name, c_s in target_cases:
            if s_id == c_s:
                actual = s_df['name'].unique()
                match_res = difflib.get_close_matches(c_name, actual, n=1)
                if not match_res: continue
                t_name = match_res[0]

                sim_scores = {'P': [], 'R': [], 'RS': []}
                for _ in range(100):
                    # Monte Carlo: Inject Noise and Re-normalize
                    m_df = s_df.copy()
                    m_df['est_fan_share'] *= np.random.normal(1, 0.1, len(m_df))
                    m_df['est_fan_share'] = m_df.groupby('week')['est_fan_share'].transform(lambda x: x / x.sum())

                    rp = simulate_season_path_fixed(m_df, 'percent')
                    rr = simulate_season_path_fixed(m_df, 'rank')
                    rs = simulate_season_path_fixed(m_df, 'rank', use_save=True)

                    n_div = len(rp) - 1
                    sim_scores['P'].append((rp[t_name] - 1) / n_div * 100)
                    sim_scores['R'].append((rr[t_name] - 1) / n_div * 100)
                    sim_scores['RS'].append((rs[t_name] - 1) / n_div * 100)

                case_results.append({
                    'Contestant': t_name, 'Season': s_id,
                    'Percent_Regime_%': np.mean(sim_scores['P']),
                    'Rank_Regime_%': np.mean(sim_scores['R']),
                    'Rank+Save_Regime_%': np.mean(sim_scores['RS'])
                })

    # D. Results Compilation and Visualization
    macro_df = pd.DataFrame(macro_list)
    case_df = pd.DataFrame(case_results)

    print("\n--- Monte Carlo Case Study Results Summary ---")
    print(case_df)

    # Visualization 1: Judicial Bias Index (JBI) Comparison
    # JBI > 0 indicates outcome favors professional merit over fan popularity.

    plt.figure(figsize=(10, 6))
    m_melt = macro_df.melt(id_vars='Season', value_vars=['Bias_Percent', 'Bias_Rank'],
                           var_name='Voting Regime', value_name='Judicial Bias Index (JBI)')
    sns.boxplot(data=m_melt, x='Voting Regime', y='Judicial Bias Index (JBI)', palette="vlag")
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.title('Figure 1: Comparison of Judicial Bias Across Voting Regimes (JBI > 0 Favors Judges)')
    plt.savefig(os.path.join(output_dir, "jbi_comparison_academic.png"), dpi=300)

    # Visualization 2: Contestant Sensitivity Analysis

    plt.figure(figsize=(12, 7))
    c_melt = case_df.melt(id_vars='Contestant', value_vars=['Percent_Regime_%', 'Rank_Regime_%', 'Rank+Save_Regime_%'],
                          var_name='Regime', value_name='Survival Percentile (0-100)')
    sns.barplot(data=c_melt, x='Contestant', y='Survival Percentile (0-100)', hue='Regime', palette="muted")
    plt.title('Figure 2: Survival Sensitivity of Controversial Contestants to Policy Changes')
    plt.ylabel('Expected Survival Percentile (100 = Winner)')
    plt.savefig(os.path.join(output_dir, "case_impact_academic.png"), dpi=300)

    plt.show()
    print(f"\n✅ Analysis Completed. Outputs saved to: {output_dir}")

    # E. Scientific Tables for Latex
    # You can copy the printed dataframes below into a converter for your paper.
    case_df.to_csv(os.path.join(output_dir, "policy_impact_table.csv"), index=False)