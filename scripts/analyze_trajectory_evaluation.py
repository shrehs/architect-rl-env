#!/usr/bin/env python3
"""
Trajectory-Level Evaluation Analysis
Shows the new step-wise reasoning quality metrics
"""

import pandas as pd

csv = pd.read_csv('artifacts/trajectory_evaluation_test/episode_metrics.csv')

print("\n" + "="*110)
print("TRAJECTORY-LEVEL EVALUATION METRICS")
print("="*110)

# Show sample rows
print("\nSample Episodes with Trajectory Metrics:")
print("-" * 110)

display_cols = ['agent', 'steps', 'oracle_score', 'information_gain_score', 
                'utilization_score', 'redundancy_score', 'trajectory_quality_bonus']

sample = csv[display_cols].head(10)
for idx, row in sample.iterrows():
    print(f"Agent: {row['agent']:10s} | Steps: {row['steps']:2.0f} | Oracle: {row['oracle_score']:.2f} | "
          f"InfoGain: {row['information_gain_score']:.2f} | Utilization: {row['utilization_score']:.2f} | "
          f"Redundancy: {row['redundancy_score']:.2f} | Quality: {row['trajectory_quality_bonus']:.3f}")

# Summary by agent
print("\n" + "="*110)
print("Summary by Agent Type:")
print("-" * 110)

for agent in sorted(csv['agent'].unique()):
    agent_data = csv[csv['agent'] == agent]
    print(f"\n{agent.upper()}:")
    print(f"  Information Gain Score:  {agent_data['information_gain_score'].mean():.3f} "
          f"(measures early discovery of constraints)")
    print(f"  Utilization Score:       {agent_data['utilization_score'].mean():.3f} "
          f"(% of discovered constraints used in decision)")
    print(f"  Redundancy Score:        {agent_data['redundancy_score'].mean():.3f} "
          f"(1.0 = no repeated questions, 0.0 = all repeats)")
    print(f"  Trajectory Quality:      {agent_data['trajectory_quality_bonus'].mean():.4f} "
          f"(terminal reward bonus)")
    print(f"  Average Oracle Score:    {agent_data['oracle_score'].mean():.3f}")

print("\n" + "="*110)
print("INTERPRETATION:")
print("="*110)
print("""
A. Information Gain (Delta Information Gain)
   - High (1.0): Discovered constraints early, used rest for confirmation
   - Low (0.0): Discovered constraints too late or in poor order
   - Measures the step-wise quality of problem exploration

B. Utilization Score (Constraint Utilization)
   - High (1.0): Final decision uses ALL discovered constraints
   - Low (0.0): Agent ignores discovered constraints in final decision
   - Prevents agents from collecting info then ignoring it

C. Redundancy Score (Trajectory Efficiency)
   - High (1.0): No repeated questions, efficient exploration
   - Low (0.0): Many repeated questions, wasted steps
   - Directly correlates with step efficiency

D. Trajectory Quality Bonus
   - Sum of: 0.05×InformationGain + 0.05×Utilization + 0.05×Redundancy
   - Maximum +0.15 reward for perfect trajectory
   - Added at terminal step for high-quality reasoning paths
""")
