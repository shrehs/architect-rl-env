#!/usr/bin/env python3
"""
Behavioral Differentiation Analysis
Compares improved vs heuristic agents across key metrics
"""

import pandas as pd

csv = pd.read_csv('artifacts/agent_comparison_comprehensive/episode_metrics.csv')

print('\n' + '='*80)
print('BEHAVIORAL DIFFERENTIATION EXPERIMENT RESULTS')
print('='*80)

# Overall comparison
print('\n[1] OVERALL AVERAGES (all modes combined)')
print('-' * 80)
for agent in ['heuristic', 'improved']:
    data = csv[csv['agent'] == agent]
    print(f'\n{agent.upper()}:')
    print(f'  Oracle Score:      {data["oracle_score"].mean():.3f}')
    print(f'  Trajectory Score:  {data["trajectory_score"].mean():.3f}')
    print(f'  Recovery Score:    {data["recovery_score"].mean():.3f}')
    print(f'  Efficiency Score:  {data["global_efficiency_score"].mean():.3f}')
    print(f'  Avg Steps:         {data["steps"].mean():.1f}')
    print(f'  Success Rate:      {(data["success"].astype(int).mean()):.1%}')

# Mode-specific
print('\n[2] ORACLE SCORE BY MODE (resilience to adverse conditions)')
print('-' * 80)
for mode in ['clean', 'noisy', 'adversarial']:
    print(f'\n{mode.upper()}:')
    for agent in ['heuristic', 'improved']:
        data = csv[(csv['agent'] == agent) & (csv['mode'] == mode)]
        oracle = data['oracle_score'].mean()
        print(f'  {agent:10s}: {oracle:.2f}')

# Recovery analysis
print('\n[3] RECOVERY SCORE BY MODE')
print('-' * 80)
for mode in ['clean', 'noisy', 'adversarial']:
    print(f'\n{mode.upper()}:')
    for agent in ['heuristic', 'improved']:
        data = csv[(csv['agent'] == agent) & (csv['mode'] == mode)]
        recovery = data['recovery_score'].mean()
        print(f'  {agent:10s}: {recovery:.2f}')

# Efficiency analysis
print('\n[4] EFFICIENCY BY MODE (step counts)')
print('-' * 80)
for mode in ['clean', 'noisy', 'adversarial']:
    print(f'\n{mode.upper()}:')
    for agent in ['heuristic', 'improved']:
        data = csv[(csv['agent'] == agent) & (csv['mode'] == mode)]
        steps = data['steps'].mean()
        print(f'  {agent:10s}: {steps:.1f} steps')

print('\n' + '='*80)
print('VALIDATION AGAINST EXPECTATIONS:')
print('='*80)
heur = csv[csv['agent'] == 'heuristic']
impr = csv[csv['agent'] == 'improved']

oracle_h = heur['oracle_score'].mean()
oracle_i = impr['oracle_score'].mean()
traj_h = heur['trajectory_score'].mean()
traj_i = impr['trajectory_score'].mean()
recov_h = heur['recovery_score'].mean()
recov_i = impr['recovery_score'].mean()
eff_h = heur['global_efficiency_score'].mean()
eff_i = impr['global_efficiency_score'].mean()

oracle_similar = abs(oracle_h - oracle_i) < 0.3
traj_better = traj_i > traj_h
recov_better = recov_i > recov_h
eff_better = eff_i >= eff_h

print(f'\n✓ Oracle ~ similar?           {str(oracle_similar): <25} (H={oracle_h:.2f}, I={oracle_i:.2f})')
print(f'✓ Trajectory: improved > heur? {str(traj_better): <25} (H={traj_h:.2f}, I={traj_i:.2f})')
print(f'✓ Recovery: improved >> heur?  {str(recov_better): <25} (H={recov_h:.2f}, I={recov_i:.2f})')
print(f'✓ Efficiency: impr >= heur?   {str(eff_better): <25} (H={eff_h:.2f}, I={eff_i:.2f})')

all_pass = oracle_similar and traj_better and recov_better and eff_better
print(f'\n🎯 ALL EXPECTATIONS MET: {all_pass}')
print()
