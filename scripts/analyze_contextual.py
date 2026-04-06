import csv

rows = list(csv.DictReader(open('artifacts/evaluation/episode_metrics.csv')))

print('=== Contextual Diversity Bonus Dynamics ===\n')

# Show progression of bonus over time as paths accumulate
print('Path progression (first 20 episodes):')
print(f'{"Ep":>3} {"Path":>15} {"Freq":>7} {"Scale":>7} {"Bonus":>7}')
print('-' * 45)

for i in range(min(20, len(rows))):
    path = str(rows[i].get("matched_trajectory", "unknown"))
    freq = float(rows[i].get("path_frequency", 0.0))
    scale = float(rows[i].get("contextual_bonus_scale", 0.0))
    bonus = float(rows[i].get("trajectory_diversity_bonus", 0.0))
    print(f'{i:3d} {path:>15} {freq:7.1%} {scale:7.2f}x {bonus:7.4f}')

print('\n\nPath progression (episodes 400-450, near end):')
print(f'{"Ep":>3} {"Path":>15} {"Freq":>7} {"Scale":>7} {"Bonus":>7}')
print('-' * 45)

start = max(0, len(rows) - 50)
for i in range(start, len(rows)):
    path = str(rows[i].get("matched_trajectory", "unknown"))
    freq = float(rows[i].get("path_frequency", 0.0))
    scale = float(rows[i].get("contextual_bonus_scale", 0.0))
    bonus = float(rows[i].get("trajectory_diversity_bonus", 0.0))
    ep_num = i - start + 1
    print(f'{i:3d} {path:>15} {freq:7.1%} {scale:7.2f}x {bonus:7.4f}')

print('\n\nAnalysis:')
print('Expected behavior:')
print('- As alternative_2 gets used more, its scale should decrease (lower bonus)')
print('- Rare paths (alternative_1) should keep high scale and high bonus')
print('- Primary never gets bonus (matched_path_idx must be > 0)\n')

# Compute final frequencies
path_usage = {}
for row in rows:
    path = str(row.get("matched_trajectory", "unknown"))
    if path != "unknown":
        path_usage[path] = path_usage.get(path, 0) + 1

print('Overall path usage in this evaluation (450 episodes):')
for path, count in sorted(path_usage.items(), key=lambda x: -x[1]):
    pct = 100 * count / len(rows)
    print(f'  {path:20s}: {count:3d} ({pct:5.1f}%)')
